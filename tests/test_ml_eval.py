"""Offline tests for the machine-learning evaluation (`selection/ml_eval.py`).

The module's job is to tell a model that learned something from one that memorised
its training window, so the tests build both cases explicitly: a panel where one
feature drives the target, and one where the target is pure noise. Everything runs
on synthetic frames, with no prices and no network.
"""
import numpy as np
import pandas as pd
import pytest

from src.selection import hurdle as H
from src.selection import ml_eval as M

_DATES = list(pd.date_range("2020-01-31", periods=24, freq="ME"))
_TICKERS = [f"T{i}" for i in range(40)]


def _panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([_DATES, _TICKERS],
                                       names=["rebalance_date", "ticker"])
    features = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index,
                            columns=["f1", "f2", "f3", "f4"])
    signal = (0.8 * features["f1"] + 0.2 * rng.normal(size=len(index))).rename("target")
    noise = pd.Series(rng.normal(size=len(index)), index=index, name="target")
    return features, signal, noise



def test_r2_is_zero_when_the_prediction_is_the_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M._r2(y, np.full_like(y, y.mean()), float(y.mean())) == pytest.approx(0.0)


def test_r2_is_negative_when_the_model_is_worse_than_the_constant():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M._r2(y, np.array([4.0, 3.0, 2.0, 1.0]), float(y.mean())) < 0


def test_r2_is_nan_for_a_flat_target_where_the_ratio_is_undefined():
    y = np.array([2.0, 2.0, 2.0])
    assert np.isnan(M._r2(y, np.array([1.0, 2.0, 3.0]), 2.0))


def test_r2_of_an_empty_sample_is_nan():
    assert np.isnan(M._r2(np.array([]), np.array([]), 0.0))



def test_target_at_aligns_a_two_level_target_to_a_ticker_indexed_fold():
    features, signal, _ = _panel()
    date = _DATES[3]
    tickers = pd.Index(_TICKERS[:5], name="ticker")
    got = M._target_at(signal, date, tickers)
    assert list(got.index) == list(tickers)
    assert got.notna().all()
    assert got.iloc[0] == pytest.approx(signal.loc[(date, tickers[0])])


def test_target_at_returns_nan_for_a_date_the_target_does_not_cover():
    _, signal, _ = _panel()
    got = M._target_at(signal, pd.Timestamp("1999-12-31"), pd.Index(_TICKERS[:3]))
    assert got.isna().all()



def test_fold_scores_cover_every_fold_after_the_warm_up():
    features, signal, _ = _panel()
    scores = M.fold_scores(features, signal, _DATES, "xgb", min_train_dates=6)
    assert len(scores) == len(_DATES) - 6
    assert all(s.n_train > 0 and s.n_test == len(_TICKERS) for s in scores)


def test_fold_scores_use_a_rolling_window_of_the_requested_length():
    features, signal, _ = _panel()
    scores = M.fold_scores(features, signal, _DATES, "xgb", 6, train_window=3)
    assert all(s.n_train == 3 * len(_TICKERS) for s in scores)


def test_a_learnable_target_scores_well_out_of_sample():
    features, signal, _ = _panel()
    pooled = M.pooled_metrics(M.fold_scores(features, signal, _DATES, "xgb", 6))
    assert pooled["r2_oos"] > 0.5
    assert pooled["ic"] > 0.5
    assert pooled["gap"] < 0.2


def test_a_noise_target_is_beaten_by_the_constant_and_shows_a_large_gap():
    features, _, noise = _panel()
    pooled = M.pooled_metrics(M.fold_scores(features, noise, _DATES, "xgb", 6))
    assert pooled["r2_oos"] < 0
    assert pooled["r2_train"] > 0.3
    assert pooled["gap"] > 0.3
    assert abs(pooled["ic_t"]) < 3


def test_pooled_metrics_of_no_folds_are_all_nan():
    pooled = M.pooled_metrics([])
    assert all(np.isnan(v) for v in pooled.values())


def test_pooled_r2_weights_a_fold_by_its_test_size():
    wide = M.FoldScore(_DATES[0], 10, 90, 0.5, 0.0, 1.0, 1.0, 0.1)
    thin = M.FoldScore(_DATES[1], 10, 10, 0.5, 1.0, 1.0, 1.0, 0.1)
    pooled = M.pooled_metrics([wide, thin])
    assert pooled["r2_oos"] == pytest.approx((90 * 0.0 + 10 * 1.0) / 100)



def test_learning_curve_reports_one_row_per_window_including_expanding():
    features, signal, _ = _panel()
    curve = M.learning_curve(features, signal, _DATES, "xgb", 6, windows=(3, 6, None))
    assert list(curve.index) == ["3m", "6m", "expanding"]
    assert (curve["folds"] == len(_DATES) - 6).all()


def test_learning_curve_shrinks_the_train_test_gap_as_the_window_grows():
    features, _, noise = _panel()
    curve = M.learning_curve(features, noise, _DATES, "xgb", 6, windows=(3, None))
    assert curve.loc["expanding", "gap"] < curve.loc["3m", "gap"]



def test_prediction_stability_reports_one_pair_per_adjacent_month():
    features, signal, _ = _panel()
    out = M.prediction_stability(features, signal, _DATES, "xgb", 6)
    assert out["pairs"] == len(_DATES) - 6 - 1
    assert -1.0 <= out["prediction_rho"] <= 1.0


def test_a_persistent_feature_produces_persistent_predictions():
    """A feature that barely moves month to month should be ranked the same way twice."""
    rng = np.random.default_rng(1)
    index = pd.MultiIndex.from_product([_DATES, _TICKERS],
                                       names=["rebalance_date", "ticker"])
    fixed = np.tile(rng.normal(size=len(_TICKERS)), len(_DATES))
    features = pd.DataFrame({"f1": fixed,
                             "f2": rng.normal(size=len(index))}, index=index)
    target = pd.Series(fixed + 0.05 * rng.normal(size=len(index)), index=index)
    out = M.prediction_stability(features, target, _DATES, "xgb", 6)
    assert out["prediction_rho"] > 0.8



def test_error_analysis_splits_by_year_and_volatility_bucket():
    features, _, noise = _panel()
    vol = pd.Series(np.random.default_rng(2).random(len(features)), index=features.index)
    out = M.error_analysis(features, noise, _DATES, vol, "xgb", 6)
    assert set(out["by_year"].index) == {2020, 2021}
    assert out["by_year"]["n"].sum() == (len(_DATES) - 6) * len(_TICKERS)
    assert len(out["by_volatility"]) == M.VOL_BUCKETS


def test_error_analysis_without_volatility_returns_only_the_yearly_table():
    features, _, noise = _panel()
    out = M.error_analysis(features, noise, _DATES, None, "xgb", 6)
    assert "by_volatility" not in out



def test_walk_forward_folds_and_predictions_agree_on_every_fold():
    """The diagnostics must see exactly the folds the portfolio run saw."""
    features, signal, _ = _panel()
    folds = list(H.walk_forward_folds(features, signal, _DATES, "xgb", 6))
    preds = H.walk_forward_predictions(features, signal, _DATES, "xgb", 6)
    assert [f.date for f in folds] == [p.date for p in preds]
    for fold, prediction in zip(folds, preds):
        pd.testing.assert_series_equal(fold.model.predict_scores(fold.test),
                                       prediction.scores)

"""Offline tests for the continuous excess-return target and the ReturnModel.

Deterministic synthetic prices; no network. The target is
``(r_stock - beta * r_spy) / vol`` with beta and vol estimated only from data on or
before the rebalance date, so the tests check the arithmetic, the leakage
guarantee and the winsorisation.
"""
import numpy as np
import pandas as pd
import pytest

from src.selection import labels as L
from src.selection.model import ReturnModel

_IDX = pd.date_range("2019-01-01", periods=600, freq="B")


def _spy_returns(seed=0):
    return np.random.default_rng(seed).normal(0.0004, 0.01, len(_IDX))


def _frame(returns):
    return pd.DataFrame({"Close": 100.0 * np.cumprod(1.0 + np.asarray(returns))}, index=_IDX)


def _dates():
    return list(pd.date_range("2020-03-31", periods=12, freq="ME"))


def test_pure_beta_stock_has_near_zero_excess():
    spy = _spy_returns()
    prices = {"LEV": _frame(2.0 * spy)}
    targets = L.make_excess_return_targets(
        prices, _frame(spy), _dates(), beta_shrinkage=0.0, winsorize=None,
    )
    assert not targets.empty
    assert targets["beta"].to_numpy() == pytest.approx(2.0, abs=1e-9)
    assert targets["excess_beta"].abs().max() < 0.01


def test_blume_shrinkage_pulls_beta_towards_one():
    spy = _spy_returns()
    prices = {"LEV": _frame(2.0 * spy)}
    targets = L.make_excess_return_targets(prices, _frame(spy), _dates(), winsorize=None)
    assert targets["beta"].to_numpy() == pytest.approx(2.0 * 2 / 3 + 1 / 3, abs=1e-9)


def test_target_is_beta_adjusted_excess_over_vol():
    spy = _spy_returns()
    idio = np.random.default_rng(1).normal(0.0, 0.01, len(_IDX))
    prices = {"A": _frame(1.2 * spy + idio)}
    targets = L.make_excess_return_targets(prices, _frame(spy), _dates(), winsorize=None)
    row = targets.iloc[0]
    assert row["target"] == pytest.approx(row["excess_beta"] / row["vol"])
    assert row["excess_beta"] == pytest.approx(
        row["fwd_return"] - row["beta"] * (row["fwd_return"] - row["excess_spy"])
    )


def test_beta_and_vol_do_not_see_the_future():
    spy = _spy_returns()
    idio = np.random.default_rng(2).normal(0.0, 0.01, len(_IDX))
    base = 1.1 * spy + idio
    t = pd.Timestamp("2020-06-30")

    shocked = base.copy()
    shocked[_IDX > t] *= 5.0

    a = L.make_excess_return_targets({"A": _frame(base)}, _frame(spy), [t], winsorize=None)
    b = L.make_excess_return_targets({"A": _frame(shocked)}, _frame(spy), [t], winsorize=None)
    assert a.loc[(t, "A"), "beta"] == pytest.approx(b.loc[(t, "A"), "beta"])
    assert a.loc[(t, "A"), "vol"] == pytest.approx(b.loc[(t, "A"), "vol"])
    assert a.loc[(t, "A"), "fwd_return"] != pytest.approx(b.loc[(t, "A"), "fwd_return"])


def test_too_little_history_drops_the_row():
    spy = _spy_returns()
    early = pd.Timestamp("2019-02-28")
    targets = L.make_excess_return_targets({"A": _frame(spy)}, _frame(spy), [early])
    assert targets.empty


def test_winsorisation_clips_within_each_date():
    spy = _spy_returns()
    rng = np.random.default_rng(3)
    prices = {f"S{i}": _frame(spy + rng.normal(0, 0.01, len(_IDX))) for i in range(30)}
    raw = L.make_excess_return_targets(prices, _frame(spy), _dates(), winsorize=None)
    clipped = L.make_excess_return_targets(prices, _frame(spy), _dates(), winsorize=(0.1, 0.9))
    for date, group in raw.groupby(level="rebalance_date"):
        lo, hi = group["target"].quantile([0.1, 0.9])
        c = clipped.xs(date, level="rebalance_date")["target"]
        assert c.min() >= lo - 1e-12 and c.max() <= hi + 1e-12
    pd.testing.assert_series_equal(raw["excess_spy"], clipped["excess_spy"])


def test_missing_benchmark_close_raises():
    with pytest.raises(ValueError):
        L.make_excess_return_targets({"A": _frame(_spy_returns())}, pd.DataFrame(), _dates())


@pytest.mark.parametrize("kind", ["xgb", "rf", "ridge"])
def test_return_model_learns_a_linear_signal(kind):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=400), "f2": rng.normal(size=400)})
    y = pd.Series(2.0 * X["f1"] + rng.normal(scale=0.1, size=400))
    X.loc[::7, "f2"] = np.nan
    pred = ReturnModel(kind=kind).fit(X, y).predict_scores(X)
    assert pred.index.equals(X.index)
    assert np.corrcoef(pred, y)[0, 1] > 0.8


def test_return_model_degenerate_fit_predicts_the_mean():
    X = pd.DataFrame({"f": [1.0, 2.0]})
    pred = ReturnModel(kind="xgb").fit(X, pd.Series([0.3, np.nan])).predict_scores(X)
    assert pred.tolist() == [0.3, 0.3]


def test_return_model_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ReturnModel(kind="logistic")


def test_ex_ante_risk_keeps_names_without_a_future_price():
    spy = _spy_returns()
    t = pd.Timestamp("2020-06-30")
    alive = _frame(1.1 * spy)
    delisted_next_week = alive[alive.index <= t + pd.Timedelta(days=5)]
    long_gone = alive[alive.index <= t - pd.Timedelta(days=60)]
    risk = L.ex_ante_risk({"ALIVE": alive, "DELIST": delisted_next_week, "GONE": long_gone},
                          _frame(spy), [t])
    names = set(risk.xs(t, level="rebalance_date").index)
    assert names == {"ALIVE", "DELIST"}
    targets = L.make_excess_return_targets({"ALIVE": alive}, _frame(spy), [t], winsorize=None)
    assert risk.loc[(t, "ALIVE"), "beta"] == pytest.approx(targets.loc[(t, "ALIVE"), "beta"])
    assert risk.loc[(t, "ALIVE"), "vol"] == pytest.approx(targets.loc[(t, "ALIVE"), "vol"])

"""Offline tests for the model-explanation tools (`selection/explain.py`)."""
import numpy as np
import pandas as pd
import pytest

from src.selection import explain as X
from src.selection.model import ReturnModel


def _panel(n_months=20, n_stocks=60, seed=0):
    """``good`` drives the target upwards, ``bad`` drives it down, ``noise`` does nothing."""
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2018-01-31", periods=n_months, freq="ME"))
    idx = pd.MultiIndex.from_product([dates, [f"T{i:02d}" for i in range(n_stocks)]],
                                     names=["rebalance_date", "ticker"])
    fm = pd.DataFrame(rng.uniform(size=(len(idx), 3)), index=idx, columns=["good", "bad", "noise"])
    target = 2.0 * fm["good"] - 1.0 * fm["bad"] + rng.normal(0, 0.1, len(idx))
    return fm, target.rename("target"), dates


def test_contributions_sum_to_the_prediction():
    fm, target, _ = _panel()
    model = ReturnModel(kind="xgb").fit(fm, target)
    c = model.contributions(fm)
    assert list(c.columns) == ["good", "bad", "noise", "bias"]
    np.testing.assert_allclose(c.sum(axis=1), model.predict_scores(fm), atol=1e-4)


def test_contributions_only_for_xgb():
    fm, target, _ = _panel(n_months=2)
    with pytest.raises(NotImplementedError):
        ReturnModel(kind="ridge").fit(fm, target).contributions(fm)


def test_importance_finds_the_drivers_their_direction_and_their_own_ic():
    fm, target, dates = _panel()
    contribs, preds = X.walk_forward_contributions(fm, target, dates, min_train_dates=6)
    assert len(preds) == len(dates) - 6
    table = X.importance_table(contribs, fm, target)
    assert list(table.index[:2]) == ["good", "bad"]
    assert table.loc["good", "direction"] > 0.5
    assert table.loc["bad", "direction"] < -0.5
    assert table.loc["good", "own_ic"] > 0 and table.loc["good", "own_ic_t"] > 2
    assert table.loc["bad", "own_ic"] < 0
    assert table["share"].sum() == pytest.approx(1.0)


def test_dependence_is_monotone_for_a_linear_driver():
    fm, target, dates = _panel()
    contribs, _ = X.walk_forward_contributions(fm, target, dates, min_train_dates=6)
    dep = X.dependence(contribs, fm, "good")
    assert list(dep.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert dep.is_monotonic_increasing


def test_importance_by_period_shares_sum_to_one():
    fm, target, dates = _panel()
    contribs, _ = X.walk_forward_contributions(fm, target, dates, min_train_dates=6)
    per = X.importance_by_period(contribs, [("A", "2018-01-01", "2018-12-31"),
                                            ("B", "2019-01-01", "2019-12-31")])
    assert per.sum().to_numpy() == pytest.approx([1.0, 1.0])

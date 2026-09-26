"""Tests for the per-pick model explanation shown on the app's cards."""
import numpy as np
import pandas as pd
import pytest

from src.selection import explain as X
from src.selection.model import RankerModel
from src.ui import data_access as D
from src.ui import evidence as E


def _panel(n=120, seed=0):
    """``good`` decides the label; ``noise`` does not."""
    rng = np.random.default_rng(seed)
    X_ = pd.DataFrame({"mom_6m": rng.uniform(size=n), "volatility": rng.uniform(size=n)},
                      index=[f"T{i:03d}" for i in range(n)])
    y = pd.Series((X_["mom_6m"] > 0.5).astype(int), index=X_.index)
    return X_, y


def test_ranker_contributions_are_exact_and_xgb_only():
    X_, y = _panel()
    model = RankerModel(kind="xgb").fit(X_, y)
    c = model.contributions(X_)
    assert list(c.columns) == ["mom_6m", "volatility", "bias"]
    top = X_["mom_6m"].idxmax()
    assert abs(c.loc[top, "mom_6m"]) > abs(c.loc[top, "volatility"])
    with pytest.raises(NotImplementedError):
        RankerModel(kind="rf").fit(X_, y).contributions(X_)


def test_single_class_fold_explains_nothing_rather_than_failing():
    X_, _ = _panel(n=10)
    ones = pd.Series(1, index=X_.index)
    c = RankerModel(kind="xgb").fit(X_, ones).contributions(X_)
    assert (c == 0).all().all()


def test_pick_reasons_rank_the_biggest_movers_with_direction_and_percentile():
    X_, y = _panel()
    model = RankerModel(kind="xgb").fit(X_, y)
    contributions = model.contributions(X_)
    top = X_["mom_6m"].idxmax()
    reasons = X.pick_reasons(contributions, X_, [top, "missing"], top_n=2)

    assert "missing" not in reasons
    first = reasons[top][0]
    assert first["feature"] == "mom_6m" and first["label"] == "6-month return"
    assert first["direction"] == "raised" and first["contribution"] > 0
    assert first["percentile"] == pytest.approx(1.0)
    assert len(reasons[top]) == 2
    assert abs(reasons[top][0]["contribution"]) >= abs(reasons[top][1]["contribution"])


def test_reason_sentences_say_where_the_stock_sat():
    assert E.pick_reason_text({"label": "6-month return", "direction": "raised", "percentile": 0.93}) == \
        "6-month return (top 7% this month) raised its score"
    assert E.pick_reason_text({"label": "21-day volatility", "direction": "lowered", "percentile": 0.1}) == \
        "21-day volatility (bottom 10% this month) lowered its score"
    assert E.pick_reason_text({"label": "MACD", "direction": "raised", "percentile": None}) == \
        "MACD raised its score"
    assert "the highest this month" in E.pick_reason_text(
        {"label": "12-1 momentum / volatility", "direction": "lowered", "percentile": 1.0})
    assert "the lowest this month" in E.pick_reason_text(
        {"label": "price / book", "direction": "raised", "percentile": 0.0})


def test_explain_picks_is_skipped_when_the_model_cannot_explain_itself():
    X_, y = _panel(n=20)
    date = pd.Timestamp("2026-08-31")
    rf = {"date": date, "model": RankerModel(kind="rf").fit(X_, y), "scored": X_}
    assert D._explain_picks(rf, date, ["T001"]) == {}
    assert D._explain_picks({}, date, ["T001"]) == {}
    xgb = {"date": date, "model": RankerModel(kind="xgb").fit(X_, y), "scored": X_}
    assert D._explain_picks(xgb, pd.Timestamp("2026-07-31"), ["T001"]) == {}
    assert list(D._explain_picks(xgb, date, ["T001"])) == ["T001"]


def test_api_pick_reasons_carry_both_sentence_and_numbers():
    from src.api.main import _model_reasons

    result = D.SelectionResult(
        result=None, latest_date=None, latest_picks=["AAA"], universe=["AAA"], n_rebalances=1,
        pick_reasons={"AAA": [{"feature": "mom_6m", "label": "6-month return",
                               "contribution": 0.42, "direction": "raised", "percentile": 0.93}]},
    )
    reason = _model_reasons(result, "AAA")[0]
    assert reason["text"] == "6-month return (top 7% this month) raised its score"
    assert reason["contribution"] == 0.42
    assert _model_reasons(result, "ZZZ") == []



def _backtest_stub(dates, picks_by_date, returns):
    """Minimal stand-in for SelectionBacktest: a weights frame and a return per entry date."""
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=["AAA", "BBB", "CCC"])
    for d, names in picks_by_date.items():
        for n in names:
            weights.loc[d, n] = 1.0 / len(names)
    return type("Stub", (), {"weights": weights,
                             "strategy_returns": pd.Series(returns, index=pd.DatetimeIndex(dates))})


def _spy(dates, closes):
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))


def test_previous_issue_reports_last_months_picks_against_the_index():
    dates = ["2026-06-30", "2026-07-31", "2026-08-31"]
    stub = _backtest_stub(dates, {dates[1]: ["AAA", "BBB"]}, [0.01, 0.0542, 0.0])
    spy = _spy(dates, [100.0, 100.0, 102.68])

    issue = D._previous_issue(stub, spy)
    assert issue["entry"] == pd.Timestamp("2026-07-31") and issue["exit"] == pd.Timestamp("2026-08-31")
    assert issue["picks"] == ["AAA", "BBB"]
    assert issue["strategy_return"] == pytest.approx(0.0542)
    assert issue["spy_return"] == pytest.approx(0.0268)
    assert issue["excess"] == pytest.approx(0.0274)


def test_previous_issue_is_absent_when_there_is_no_completed_month():
    one = _backtest_stub(["2026-08-31"], {"2026-08-31": ["AAA"]}, [0.0])
    assert D._previous_issue(one, _spy(["2026-08-31"], [100.0])) is None


def test_previous_issue_survives_a_missing_benchmark():
    dates = ["2026-07-31", "2026-08-31"]
    stub = _backtest_stub(dates, {dates[0]: ["AAA"]}, [0.03, 0.0])
    issue = D._previous_issue(stub, None)
    assert issue["strategy_return"] == pytest.approx(0.03)
    assert issue["spy_return"] is None and issue["excess"] is None

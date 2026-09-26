"""Offline tests for the S&P 500 addition/removal event study (`selection/index_events.py`)."""
import numpy as np
import pandas as pd
import pytest

from src.selection import index_events as E

_DAYS = pd.bdate_range("2015-01-02", periods=900)


def _frame(values, index=_DAYS):
    return pd.DataFrame({"Close": np.asarray(values, dtype=float)}, index=index)


def _snapshots():
    return pd.Series(
        [frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"C", "D"})],
        index=pd.DatetimeIndex(["2015-01-01", "2016-06-01", "2017-06-01"]),
    )


def test_membership_changes_lists_adds_and_removes():
    ev = E.membership_changes(_snapshots())
    got = set(map(tuple, ev[["ticker", "event"]].to_numpy()))
    assert got == {("C", "add"), ("B", "remove"), ("D", "add"), ("A", "remove")}
    assert E.membership_changes(_snapshots(), start="2017-01-01")["ticker"].tolist() == ["D", "A"]


def test_window_excess_is_stock_minus_spy_over_the_same_days():
    spy = _frame(np.full(len(_DAYS), 100.0))
    stock = _frame(100.0 * 1.01 ** np.arange(len(_DAYS)))
    date = _DAYS[400]
    events = pd.DataFrame({"date": [date], "ticker": ["S"], "event": ["add"]})
    res = E.event_returns(events, {"S": stock}, spy)
    assert res.loc[0, "group"] == "added"
    assert res.loc[0, "mkt -5..-1"] == pytest.approx(1.01 ** 5 - 1)
    assert res.loc[0, "mkt 0..+19"] == pytest.approx(1.01 ** 20 - 1)


def test_recent_listing_is_separated_from_established_additions():
    spy = _frame(np.full(len(_DAYS), 100.0))
    late = _DAYS[500:]
    newco = _frame(np.full(len(late), 50.0), index=late)
    date = _DAYS[520]
    events = pd.DataFrame({"date": [date], "ticker": ["NEW"], "event": ["add"]})
    assert E.event_returns(events, {"NEW": newco}, spy).loc[0, "group"] == "added_new"


def test_removals_split_by_whether_the_stock_kept_trading():
    spy = _frame(np.full(len(_DAYS), 100.0))
    date = _DAYS[400]
    stopped = _frame(np.full(401, 10.0), index=_DAYS[:401])
    ended_early = _frame(np.full(300, 10.0), index=_DAYS[:300])
    events = pd.DataFrame({"date": [date] * 3, "ticker": ["KEEP", "STOP", "OLD"],
                           "event": ["remove"] * 3})
    res = E.event_returns(events, {"KEEP": _frame(np.full(len(_DAYS), 10.0)),
                                   "STOP": stopped, "OLD": ended_early}, spy)
    assert res.set_index("ticker")["group"].to_dict() == {
        "KEEP": "removed_trading", "STOP": "removed_stopped", "OLD": "no_data"}
    stop = res.set_index("ticker").loc["STOP"]
    assert np.isfinite(stop["mkt -20..-1"]) and np.isnan(stop["mkt 0..+19"])


def test_beta_adjusted_return_removes_market_exposure():
    rng = np.random.default_rng(0)
    m = rng.normal(0.0005, 0.01, len(_DAYS))
    spy = _frame(100 * np.cumprod(1 + m))
    stock = _frame(100 * np.cumprod(1 + 2.0 * m))
    events = pd.DataFrame({"date": [_DAYS[500]], "ticker": ["LEV"], "event": ["add"]})
    row = E.event_returns(events, {"LEV": stock}, spy).loc[0]
    assert row["beta"] == pytest.approx(2.0 * 2 / 3 + 1 / 3, abs=0.02)
    assert abs(row["mkt 0..+59"]) > 1e-3
    assert abs(row["beta 0..+59"]) < 0.5 * abs(row["mkt 0..+59"])


def test_t_stat_is_clustered_by_event_date():
    d1, d2, d3 = pd.Timestamp("2016-01-04"), pd.Timestamp("2017-01-03"), pd.Timestamp("2018-01-02")
    values = pd.Series([0.05] * 10 + [0.01, -0.02])
    dates = pd.Series([d1] * 10 + [d2, d3])
    s = E.summarise(values, dates)
    assert s["n"] == 12 and s["n_dates"] == 3
    per_date = np.array([0.05, 0.01, -0.02])
    assert s["t"] == pytest.approx(per_date.mean() / (per_date.std(ddof=1) / np.sqrt(3)))
    assert s["share_positive"] == pytest.approx(11 / 12)


def test_ticker_change_pairs_are_dropped():
    series = 100 + np.arange(len(_DAYS), dtype=float)
    date = _DAYS[300]
    events = pd.DataFrame({"date": [date] * 3, "ticker": ["OLD", "NEW", "REAL"],
                           "event": ["remove", "add", "add"]})
    prices = {"OLD": _frame(series), "NEW": _frame(series), "REAL": _frame(series * 2)}
    clean, pairs = E.drop_ticker_changes(events, prices)
    assert pairs == [("OLD", "NEW")]
    assert clean["ticker"].tolist() == ["REAL"]

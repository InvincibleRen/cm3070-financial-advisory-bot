"""Offline tests for the Phase 1 forward-return labelling.

Deterministic synthetic prices; no network. The labels are a *training target*,
so the tests focus on: correct relative-to-median logic, a clean MultiIndex, and
that an incomplete forward window drops the example rather than truncating it.
"""
import numpy as np
import pandas as pd

from src.selection import labels as L


def _prices_from_closes(closes_by_ticker, start="2020-01-01"):
    """Build ``{ticker -> OHLCV frame}`` from lists of daily closes."""
    prices = {}
    for ticker, closes in closes_by_ticker.items():
        idx = pd.date_range(start, periods=len(closes), freq="D")
        prices[ticker] = pd.DataFrame({"Close": closes}, index=idx)
    return prices


def test_labels_beat_median_are_one():
    closes = {
        "A": [100.0] * 40 + [150.0] * 40,
        "B": [100.0] * 40 + [110.0] * 40,
        "C": [100.0] * 40 + [80.0] * 40,
    }
    prices = _prices_from_closes(closes)
    t = pd.Timestamp("2020-01-15")

    labels = L.make_labels(prices, [t], horizon_months=1)

    assert labels.loc[(t, "A")] == 1
    assert labels.loc[(t, "B")] == 0
    assert labels.loc[(t, "C")] == 0


def test_label_index_is_named_multiindex():
    prices = _prices_from_closes({"A": [10.0] * 80, "B": [20.0] * 80})
    labels = L.make_labels(prices, [pd.Timestamp("2020-01-10")], horizon_months=1)
    assert list(labels.index.names) == ["rebalance_date", "ticker"]
    assert labels.name == "label"


def test_incomplete_forward_window_is_dropped():
    prices = _prices_from_closes({"A": list(np.linspace(100, 140, 40))})
    near_end = pd.Timestamp("2020-02-05")
    labels = L.make_labels(prices, [near_end], horizon_months=1)
    assert (near_end, "A") not in labels.index


def test_entry_before_history_is_dropped():
    prices = _prices_from_closes({"A": [100.0] * 80}, start="2020-06-01")
    too_early = pd.Timestamp("2020-01-01")
    labels = L.make_labels(prices, [too_early], horizon_months=1)
    assert too_early not in labels.index.get_level_values("rebalance_date")


def test_labels_are_as_of_non_trading_dates():
    closes = {"A": [100.0] * 30 + [130.0] * 30, "B": [100.0] * 60}
    prices = _prices_from_closes(closes)
    saturday = pd.Timestamp("2020-01-04")
    labels = L.make_labels(prices, [saturday], horizon_months=1)
    assert labels.loc[(saturday, "A")] == 1
    assert labels.loc[(saturday, "B")] == 0


def test_entry_is_priced_the_trading_day_after_signal():
    closes = [100.0] * 20 + [1000.0] + [100.0] * 9 + [130.0] * 40
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    series = {"A": pd.Series(closes, index=idx)}
    signal_date = idx[20]

    fwd = L._forward_returns_at(series, signal_date, horizon_months=1, execution_lag=1)

    assert fwd["A"] > 0.25


def test_default_execution_prices_at_the_signal_close():
    closes = [100.0] * 20 + [1000.0] + [100.0] * 9 + [130.0] * 40
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    series = {"A": pd.Series(closes, index=idx)}
    signal_date = idx[20]

    fwd = L._forward_returns_at(series, signal_date, horizon_months=1)

    assert fwd["A"] < -0.8


def test_next_trading_day_helper_skips_the_signal_date():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(range(10), index=idx, dtype="float64")
    assert L._price_next_trading_day(s, idx[3]) == 4.0
    assert L._price_next_trading_day(s, idx[-1]) is None
    assert L._price_next_trading_day(s, pd.Timestamp("2019-12-01")) is None



def _flat_prices(rates):
    """One constant-growth Close series per ticker, on a shared business-day index."""
    index = pd.date_range("2020-01-01", periods=200, freq="B")
    return {
        name: pd.DataFrame({"Close": 100 * np.cumprod(1 + np.full(len(index), r))}, index=index)
        for name, r in rates.items()
    }


def test_mean_benchmark_is_a_higher_bar_when_one_name_runs_away():
    prices = _flat_prices({"A": 0.0005, "B": 0.001, "C": 0.002, "D": 0.010})
    dates = [pd.Timestamp("2020-02-28")]

    by_median = L.make_labels(prices, dates, benchmark="median")
    by_mean = L.make_labels(prices, dates, benchmark="mean")

    assert by_median.xs(dates[0])["C"] == 1
    assert by_mean.xs(dates[0])["C"] == 0
    assert by_mean.xs(dates[0])["D"] == 1
    assert by_mean.sum() < by_median.sum()


def test_median_remains_the_default_benchmark():
    prices = _flat_prices({"A": 0.0005, "B": 0.001, "C": 0.002, "D": 0.010})
    dates = [pd.Timestamp("2020-02-28")]
    pd.testing.assert_series_equal(
        L.make_labels(prices, dates), L.make_labels(prices, dates, benchmark="median")
    )


def test_unknown_benchmark_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="benchmark must be"):
        L.make_labels(_flat_prices({"A": 0.001}), [pd.Timestamp("2020-02-28")], benchmark="mode")

"""Offline tests for the technical-signal study (`common/signal_study.py`)."""
import numpy as np
import pandas as pd
import pytest

from src.common import signal_study as Z

_DAYS = pd.bdate_range("2015-01-02", periods=700)


def _ohlc(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close},
                        index=_DAYS)


def _spy():
    return _ohlc(100 * 1.0003 ** np.arange(len(_DAYS)))


def test_new_buy_marks_only_the_first_day_of_a_buy_run():
    path = np.r_[np.linspace(100, 60, 350), np.linspace(60, 140, 350)]
    panel = Z.stock_panel("S", _ohlc(path), _spy()["Close"], None, horizons=(5,))
    buys = panel.index[panel["new_buy"]]
    assert len(buys) >= 1
    for day in buys:
        prev = panel.index[panel.index.get_loc(day) - 1]
        assert not panel.loc[prev, "buy_state"]
    assert panel["buy_state"].sum() > len(buys)


def test_forward_returns_start_at_the_event_close():
    path = 100 * 1.01 ** np.arange(len(_DAYS))
    panel = Z.stock_panel("S", _ohlc(path), _spy()["Close"], None, horizons=(5,))
    assert panel["r5"].dropna().to_numpy() == pytest.approx(1.01 ** 5 - 1)
    assert panel["spy5"].dropna().to_numpy() == pytest.approx(1.0003 ** 5 - 1)


def test_warm_up_days_before_sma200_are_excluded():
    panel = Z.stock_panel("S", _ohlc(np.linspace(100, 120, len(_DAYS))), _spy()["Close"], None)
    assert panel.index.min() == _DAYS[199]


def test_only_member_days_are_kept():
    snaps = pd.Series([frozenset({"S"}), frozenset()],
                      index=pd.DatetimeIndex([_DAYS[0], _DAYS[400]]))
    panel = Z.stock_panel("S", _ohlc(np.linspace(100, 120, len(_DAYS))), _spy()["Close"], snaps)
    assert panel.index.max() < _DAYS[400]


def test_universe_adjusted_excess_sums_to_zero_each_day():
    rng = np.random.default_rng(0)
    prices = {f"S{i}": _ohlc(100 * np.cumprod(1 + rng.normal(0.0003, 0.02, len(_DAYS))))
              for i in range(5)}
    panel = Z.build_panel(prices, _spy(), None, horizons=(5,))
    per_day = panel.dropna(subset=["x_uni5"]).groupby("date")["x_uni5"].sum()
    assert per_day.abs().max() < 1e-12


def test_newey_west_t_matches_the_plain_t_without_autocorrelation_lags():
    x = pd.Series([0.01, 0.03, -0.02, 0.02, 0.00, 0.04])
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert Z.newey_west_t(x, 0) == pytest.approx(plain)
    ar = pd.Series(np.repeat([0.02, -0.01, 0.03, 0.01, -0.02, 0.02], 5))
    assert abs(Z.newey_west_t(ar, 5)) < abs(Z.newey_west_t(ar, 0))


def test_summary_reports_hit_rates_and_means():
    rng = np.random.default_rng(1)
    prices = {f"S{i}": _ohlc(100 * np.cumprod(1 + rng.normal(0.0003, 0.02, len(_DAYS))))
              for i in range(8)}
    panel = Z.build_panel(prices, _spy(), None, horizons=(5,))
    table = Z.results_table(panel, events=("all_member_days", "buy_state"), horizons=(5,))
    row = table.loc[("all_member_days", 5)]
    assert row["x_uni"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(row["t_uni"])
    assert 0.0 <= row["beat_spy"] <= 1.0 and row["n"] > 0

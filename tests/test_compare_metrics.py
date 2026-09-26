"""Offline tests for the stock-vs-SPY comparison math (`api/main._perf_metrics`, `_beta_to`).

These cover the risk-comparison logic behind the S6 "buy it or SPY?" card without any
network: synthetic price/return series with known properties.
"""
import pandas as pd
import pytest

from src.api.main import _beta_to, _perf_metrics


def _bdays(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_beta_is_two_when_stock_moves_twice_the_market():
    idx = _bdays(300)
    spy = pd.Series([0.001 * (1 if i % 2 else -1) for i in range(300)], index=idx)
    stock = spy * 2.0
    assert _beta_to(stock, spy) == pytest.approx(2.0, abs=1e-9)


def test_beta_none_when_market_has_no_variance():
    idx = _bdays(300)
    spy = pd.Series([0.0] * 300, index=idx)
    stock = pd.Series([0.001] * 300, index=idx)
    assert _beta_to(stock, spy) is None


def test_perf_metrics_constant_growth_matches_cagr():
    r = 0.001
    close = pd.Series([(1 + r) ** i for i in range(253)], index=_bdays(253))
    m = _perf_metrics(close)
    assert m["n_days"] == 252
    assert m["ann_return"] == pytest.approx((1 + r) ** 252 - 1, rel=1e-6)
    assert m["volatility"] == pytest.approx(0.0, abs=1e-9)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_perf_metrics_drawdown_is_negative_after_a_fall():
    prices = [100 + i for i in range(40)] + [140 - 2 * i for i in range(40)] + [60 + i for i in range(20)]
    m = _perf_metrics(pd.Series(prices, index=_bdays(len(prices)), dtype="float64"))
    assert m["max_drawdown"] < 0.0
    assert m["volatility"] > 0.0


def test_perf_metrics_too_short_returns_empty():
    assert _perf_metrics(pd.Series(range(1, 11), index=_bdays(10), dtype="float64")) == {}

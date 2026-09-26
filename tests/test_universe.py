"""Offline tests for the Phase 0 universe loader.

No network is used: fake fetchers stand in for yfinance so the caching,
reporting-lag and ratio logic is verified deterministically.
"""
import math

import pandas as pd
import pytest

from src.selection import universe as U



def _fake_prices(ticker, start, end):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    base = {"AAPL": 100.0, "MSFT": 200.0}.get(ticker, 50.0)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1,
            "Low": base - 1,
            "Close": [base + i for i in range(5)],
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _fake_fundamentals(ticker):
    """Five quarters of raw statements indexed by period-end."""
    periods = pd.to_datetime(
        ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31"]
    )
    return pd.DataFrame(
        {
            "net_income": [10, 10, 10, 10, 20],
            "total_revenue": [100, 100, 100, 100, 100],
            "total_equity": [200, 200, 200, 200, 200],
            "total_debt": [100, 100, 100, 100, 100],
            "shares_outstanding": [1000, 1000, 1000, 1000, 1000],
        },
        index=periods,
    )



def test_load_prices_fetches_and_caches(tmp_path):
    calls = []

    def counting_fetcher(ticker, start, end):
        calls.append(ticker)
        return _fake_prices(ticker, start, end)

    out = U.load_prices(
        ["AAPL", "MSFT"], cache_dir=str(tmp_path), fetcher=counting_fetcher
    )
    assert set(out) == {"AAPL", "MSFT"}
    assert len(out["AAPL"]) == 5
    assert (tmp_path / "AAPL.csv").exists()
    assert calls == ["AAPL", "MSFT"]


def test_load_prices_uses_cache_second_time(tmp_path):
    calls = []

    def counting_fetcher(ticker, start, end):
        calls.append(ticker)
        return _fake_prices(ticker, start, end)

    U.load_prices(["AAPL"], cache_dir=str(tmp_path), fetcher=counting_fetcher)
    out = U.load_prices(["AAPL"], cache_dir=str(tmp_path), fetcher=counting_fetcher)
    assert calls == ["AAPL"]
    assert isinstance(out["AAPL"].index, pd.DatetimeIndex)
    assert list(out["AAPL"]["Close"]) == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_start_narrows_returned_window_even_from_cache(tmp_path):
    U.load_prices(["AAPL"], cache_dir=str(tmp_path), fetcher=_fake_prices)
    narrowed = U.load_prices(
        ["AAPL"], start="2020-01-03", cache_dir=str(tmp_path), fetcher=_fake_prices
    )
    assert narrowed["AAPL"].index.min() >= pd.Timestamp("2020-01-03")
    assert len(narrowed["AAPL"]) == 3


def test_end_trims_returned_window(tmp_path):
    out = U.load_prices(
        ["AAPL"], end="2020-01-02", cache_dir=str(tmp_path), fetcher=_fake_prices
    )
    assert out["AAPL"].index.max() <= pd.Timestamp("2020-01-02")
    assert len(out["AAPL"]) == 2


def test_force_refresh_bypasses_cache(tmp_path):
    calls = []

    def counting_fetcher(ticker, start, end):
        calls.append(ticker)
        return _fake_prices(ticker, start, end)

    U.load_prices(["AAPL"], cache_dir=str(tmp_path), fetcher=counting_fetcher)
    U.load_prices(
        ["AAPL"], cache_dir=str(tmp_path), fetcher=counting_fetcher, force_refresh=True
    )
    assert calls == ["AAPL", "AAPL"]


def test_failed_ticker_is_skipped_not_fatal(tmp_path):
    def flaky_fetcher(ticker, start, end):
        if ticker == "BAD":
            raise ValueError("no data")
        return _fake_prices(ticker, start, end)

    with pytest.warns(UserWarning):
        out = U.load_prices(
            ["AAPL", "BAD"], cache_dir=str(tmp_path), fetcher=flaky_fetcher
        )
    assert set(out) == {"AAPL"}



def test_available_date_applies_reporting_lag(tmp_path):
    df = U.load_fundamentals(
        ["AAPL"], cache_dir=str(tmp_path), reporting_lag_days=45,
        fetcher=_fake_fundamentals,
    )
    row = df[df["period_end"] == pd.Timestamp("2022-03-31")].iloc[0]
    assert row["available_date"] == pd.Timestamp("2022-03-31") + pd.Timedelta(days=45)


def test_fundamental_ratios_are_correct(tmp_path):
    df = U.load_fundamentals(
        ["AAPL"], cache_dir=str(tmp_path), fetcher=_fake_fundamentals
    )
    df = df.sort_values("period_end").reset_index(drop=True)

    assert math.isclose(df.loc[0, "net_margin"], 0.10, rel_tol=1e-9)
    assert math.isclose(df.loc[0, "debt_to_equity"], 0.5, rel_tol=1e-9)
    assert pd.isna(df.loc[0, "roe"])
    q4 = df[df["period_end"] == pd.Timestamp("2022-12-31")].iloc[0]
    assert math.isclose(q4["roe"], (10 + 10 + 10 + 10) / 200, rel_tol=1e-9)
    q1_23 = df[df["period_end"] == pd.Timestamp("2023-03-31")].iloc[0]
    assert math.isclose(q1_23["earnings_growth_yoy"], 1.0, rel_tol=1e-9)


def test_fundamentals_cache_roundtrip(tmp_path):
    calls = []

    def counting(ticker):
        calls.append(ticker)
        return _fake_fundamentals(ticker)

    first = U.load_fundamentals(["AAPL"], cache_dir=str(tmp_path), fetcher=counting)
    second = U.load_fundamentals(["AAPL"], cache_dir=str(tmp_path), fetcher=counting)
    assert calls == ["AAPL"]
    assert pd.api.types.is_datetime64_any_dtype(second["available_date"])
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True), check_like=True
    )


def test_safe_div_handles_zero_denominator():
    num = pd.Series([1.0, 2.0, 3.0])
    den = pd.Series([0.0, 2.0, None])
    result = U._safe_div(num, den)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 1.0
    assert pd.isna(result.iloc[2])



def _fake_mixed_fundamentals(ticker):
    """Recent quarterly statements plus older annual ones, tagged by ``freq``.

    Mirrors what the real fetcher returns: only a few recent quarters, but several
    prior fiscal years of annual statements to extend the usable history.
    """
    quarterly = pd.DataFrame(
        {
            "net_income": [10, 10, 10, 10, 10],
            "total_revenue": [100, 100, 100, 100, 100],
            "total_equity": [200, 200, 200, 200, 200],
            "total_debt": [100, 100, 100, 100, 100],
            "freq": "quarterly",
        },
        index=pd.date_range("2024-06-30", periods=5, freq="QE"),
    )
    annual = pd.DataFrame(
        {
            "net_income": [20, 40, 40, 40],
            "total_revenue": [400, 400, 400, 400],
            "total_equity": [200, 200, 200, 200],
            "total_debt": [100, 100, 100, 100],
            "freq": "annual",
        },
        index=pd.date_range("2021-12-31", periods=4, freq="YE"),
    )
    out = pd.concat([quarterly, annual])
    out["shares_outstanding"] = 1000.0
    out.index.name = "period_end"
    return out.sort_index()


def test_annual_statements_extend_history_and_ratios(tmp_path):
    df = U.load_fundamentals(["A"], cache_dir=str(tmp_path), fetcher=_fake_mixed_fundamentals)

    assert df["period_end"].min() == pd.Timestamp("2021-12-31")

    fy21 = df[df["period_end"] == pd.Timestamp("2021-12-31")].iloc[0]
    assert math.isclose(fy21["ttm_net_income"], 20.0)
    assert math.isclose(fy21["roe"], 20 / 200)
    assert math.isclose(fy21["net_margin"], 20 / 400)
    assert math.isclose(fy21["debt_to_equity"], 0.5)
    fy22 = df[df["period_end"] == pd.Timestamp("2022-12-31")].iloc[0]
    assert math.isclose(fy22["earnings_growth_yoy"], 1.0)

    q = df[df["period_end"] == pd.Timestamp("2025-06-30")].iloc[0]
    assert math.isclose(q["ttm_net_income"], 40.0)
    assert math.isclose(q["roe"], 0.20)

    assert (df["period_end"] == pd.Timestamp("2024-12-31")).sum() == 1


def test_missing_freq_column_is_treated_as_quarterly(tmp_path):
    df = U.load_fundamentals(["AAPL"], cache_dir=str(tmp_path), fetcher=_fake_fundamentals)
    q4 = df[df["period_end"] == pd.Timestamp("2022-12-31")].iloc[0]
    assert math.isclose(q4["ttm_net_income"], 40.0)
    assert math.isclose(q4["roe"], 0.20)



def _fake_prices_with_dividends(ticker, start, end):
    """Raw close flat at 100 while Adj Close accretes: a pure dividend effect."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0] * 4,
            "High": [110.0] * 4,
            "Low": [90.0] * 4,
            "Close": [100.0] * 4,
            "Adj Close": [95.0, 96.0, 97.0, 100.0],
            "Volume": [1_000_000] * 4,
        },
        index=idx,
    )


def test_prices_are_returned_on_a_total_return_basis(tmp_path):
    out = U.load_prices(
        ["AAPL"], cache_dir=str(tmp_path), fetcher=_fake_prices_with_dividends
    )
    frame = out["AAPL"]

    assert frame["Close"].tolist() == pytest.approx([95.0, 96.0, 97.0, 100.0])
    assert frame["High"].iloc[0] == pytest.approx(110.0 * 0.95)
    assert frame["Low"].iloc[0] == pytest.approx(90.0 * 0.95)
    assert frame["Volume"].iloc[0] == 1_000_000


def test_price_cache_stays_in_raw_form(tmp_path):
    U.load_prices(["AAPL"], cache_dir=str(tmp_path), fetcher=_fake_prices_with_dividends)
    cached = pd.read_csv(tmp_path / "AAPL.csv", index_col=0)
    assert cached["Close"].iloc[0] == pytest.approx(100.0)


def test_prices_without_adj_close_are_unchanged(tmp_path):
    out = U.load_prices(["AAPL"], cache_dir=str(tmp_path), fetcher=_fake_prices)
    assert out["AAPL"]["Close"].iloc[0] == pytest.approx(100.0)

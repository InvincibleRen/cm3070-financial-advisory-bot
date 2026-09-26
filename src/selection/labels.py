"""Forward-return labelling for cross-sectional selection (Direction 1 core, Phase 1).

Frames the task as *relative* selection rather than absolute forecasting: a
stock is a positive example if its forward return over the horizon beats the
universe median that period.

Design notes
------------
* **The label is a training target only.** The forward window ``t -> t+horizon``
  is, by definition, unknown at prediction time. It is therefore never exposed as
  a feature (see ``features.py``); it exists purely to supervise the ranker.
* **Relative, not absolute.** Labelling against the *cross-sectional median*
  makes the task market-neutral: in a bull month roughly half the universe is
  still labelled 0, so the model learns *selection* skill rather than simply
  learning "the market went up".
* **As-of pricing.** Rebalance dates need not be trading days, so entry and exit
  prices are read *as-of* (the last close on or before the date). A ticker is
  dropped for a given rebalance when either price is unavailable - in particular
  when ``t + horizon`` runs past the end of that ticker's history, which would
  otherwise silently truncate the forward window.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.common.timeindex import to_naive_index


def make_labels(
    prices: Dict[str, pd.DataFrame],
    rebalance_dates: List[pd.Timestamp],
    horizon_months: int = 1,
    execution_lag: int = 0,
    benchmark: str = "median",
    members: Optional[Dict[pd.Timestamp, Set[str]]] = None,
) -> pd.Series:
    """Return a binary label Series indexed by ``(rebalance_date, ticker)``.

    ``label = 1`` if the stock's forward return from ``t`` to ``t + horizon`` is
    strictly greater than the cross-sectional median forward return of the
    universe at ``t``; else ``0``.

    Parameters
    ----------
    prices:
        ``{ticker -> daily OHLCV DataFrame}`` (as produced by
        ``universe.load_prices``). Only the ``Close`` column is used.
    rebalance_dates:
        The dates at which the portfolio is re-selected.
    horizon_months:
        Holding period, in calendar months, used to measure the forward return.
    benchmark:
        The cross-sectional statistic a stock must beat to be labelled positive.
        ``"median"`` (the default) is the convention in the cross-sectional factor
        literature and splits each date exactly in half. ``"mean"`` instead sets the
        bar at the equal-weight portfolio return, which is what the evaluation
        benchmarks the strategy against. The two are not the same: monthly
        cross-sectional returns are right-skewed, so the mean sits above the median
        and fewer than half the stocks clear it. Training on the median therefore
        optimises a slightly easier target than the one the portfolio is judged on,
        and ``"mean"`` closes that gap at the cost of mildly imbalanced classes.

    members:
        Optional point-in-time constituents per rebalance date. When given, only
        the names that were members on that date are labelled and enter the median.

    Notes
    -----
    Tickers whose entry or exit price is unavailable at a given rebalance date are
    excluded from that date's cross-section (and therefore from the median).
    """
    if benchmark not in ("median", "mean"):
        raise ValueError(f"benchmark must be 'median' or 'mean', got {benchmark!r}")
    closes = _close_series_by_ticker(prices)

    labels: Dict[tuple, int] = {}
    for rebalance_date in pd.to_datetime(list(rebalance_dates)):
        forward_returns = _forward_returns_at(
            closes, rebalance_date, horizon_months, execution_lag
        )
        if members is not None:
            forward_returns = forward_returns[forward_returns.index.isin(members.get(rebalance_date, ()))]
        if forward_returns.empty:
            continue
        bar = forward_returns.median() if benchmark == "median" else forward_returns.mean()
        for ticker, fwd_ret in forward_returns.items():
            labels[(rebalance_date, ticker)] = int(fwd_ret > bar)

    return _as_labelled_series(labels)



BETA_WINDOW_DAYS = 252
VOL_WINDOW_DAYS = 63
MIN_OBS_DAYS = 126
TRADING_DAYS_PER_MONTH = 21
BETA_SHRINKAGE = 1.0 / 3.0
TARGET_COLUMNS = ["target", "excess_beta", "excess_spy", "fwd_return", "beta", "vol"]


def make_excess_return_targets(
    prices: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    horizon_months: int = 1,
    execution_lag: int = 0,
    beta_window: int = BETA_WINDOW_DAYS,
    vol_window: int = VOL_WINDOW_DAYS,
    min_obs: int = MIN_OBS_DAYS,
    beta_shrinkage: float = BETA_SHRINKAGE,
    winsorize: Optional[Tuple[float, float]] = (0.01, 0.99),
    members: Optional[Dict[pd.Timestamp, Set[str]]] = None,
) -> pd.DataFrame:
    """Continuous training target: risk-scaled excess return over beta x SPY.

    For each ``(t, ticker)``::

        target = (r_stock[t, t+h] - beta_t * r_spy[t, t+h]) / vol_t

    ``beta_t`` is the (Blume-shrunk) OLS beta of daily returns on SPY over the
    ``beta_window`` trading days up to and including ``t``; ``vol_t`` is the stock's
    daily return volatility over the last ``vol_window`` days, scaled to one month.
    Subtracting ``beta x SPY`` stops the model from "beating SPY" simply by holding
    high-beta names in a rising market, and dividing by ``vol`` expresses the payoff
    per unit of risk taken. The target is winsorised within each date's
    cross-section so a handful of extreme months do not dominate the squared loss.

    Leakage safety: ``beta_t`` and ``vol_t`` use returns on or before ``t`` only;
    the forward returns (``t -> t+h``) are the label and are never used as features.

    Returns a frame indexed by ``(rebalance_date, ticker)`` with ``TARGET_COLUMNS``:
    the winsorised ``target`` plus the raw components (``excess_beta``,
    ``excess_spy`` = stock minus SPY, ``fwd_return``, ``beta``, ``vol``) for the
    evaluation. Rows lacking a forward window, a SPY return, or enough history for
    the beta/vol estimate are dropped. With ``members`` (point-in-time constituents
    per date) only that date's members are kept, before the winsorisation.
    """
    closes = _close_series_by_ticker(prices)
    bench_close = _close_series_by_ticker({"_bench": benchmark}).get("_bench")
    if bench_close is None or bench_close.empty:
        raise ValueError("benchmark must be a price frame with a 'Close' column.")
    bench_ret = bench_close.pct_change()

    risk = {
        ticker: _trailing_beta_and_vol(
            series, bench_ret, beta_window, vol_window, min_obs, beta_shrinkage
        )
        for ticker, series in closes.items()
    }

    rows: Dict[tuple, Dict[str, float]] = {}
    for rebalance_date in pd.to_datetime(list(rebalance_dates)):
        bench_fwd = _forward_returns_at(
            {"_bench": bench_close}, rebalance_date, horizon_months, execution_lag
        )
        if bench_fwd.empty:
            continue
        spy_fwd = float(bench_fwd.iloc[0])
        stock_fwd = _forward_returns_at(closes, rebalance_date, horizon_months, execution_lag)
        if members is not None:
            stock_fwd = stock_fwd[stock_fwd.index.isin(members.get(rebalance_date, ()))]
        for ticker, fwd in stock_fwd.items():
            beta, vol = _risk_asof(risk[ticker], rebalance_date)
            if not (np.isfinite(beta) and np.isfinite(vol)) or vol <= 0:
                continue
            excess_beta = fwd - beta * spy_fwd
            rows[(rebalance_date, ticker)] = {
                "target": excess_beta / vol,
                "excess_beta": excess_beta,
                "excess_spy": fwd - spy_fwd,
                "fwd_return": fwd,
                "beta": beta,
                "vol": vol,
            }

    if not rows:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["rebalance_date", "ticker"],
        )
        return pd.DataFrame(columns=TARGET_COLUMNS, index=empty_index, dtype="float64")

    frame = pd.DataFrame.from_dict(rows, orient="index")[TARGET_COLUMNS]
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["rebalance_date", "ticker"])
    frame = frame.sort_index()
    if winsorize is not None:
        lo, hi = winsorize
        grouped = frame.groupby(level="rebalance_date")["target"]
        frame["target"] = frame["target"].clip(
            lower=grouped.transform(lambda s: s.quantile(lo)),
            upper=grouped.transform(lambda s: s.quantile(hi)),
        )
    return frame


def ex_ante_risk(
    prices: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    beta_window: int = BETA_WINDOW_DAYS,
    vol_window: int = VOL_WINDOW_DAYS,
    min_obs: int = MIN_OBS_DAYS,
    beta_shrinkage: float = BETA_SHRINKAGE,
    stale_days: int = 7,
) -> pd.DataFrame:
    """``beta`` and ``vol`` per ``(t, ticker)``, from data on or before ``t`` only.

    Unlike the target, this does not require a forward return, so a stock that is
    delisted during the next month is still present. That matters for anything
    that must be decided at ``t`` (risk-matched null draws, a "buy the most
    volatile names" baseline): filtering on a future price would leak which names
    survive. A ticker whose last price is more than ``stale_days`` before ``t`` is
    treated as no longer trading.
    """
    closes = _close_series_by_ticker(prices)
    bench_close = _close_series_by_ticker({"_bench": benchmark}).get("_bench")
    if bench_close is None or bench_close.empty:
        raise ValueError("benchmark must be a price frame with a 'Close' column.")
    bench_ret = bench_close.pct_change()

    rows: Dict[tuple, Dict[str, float]] = {}
    for ticker, series in closes.items():
        risk = _trailing_beta_and_vol(series, bench_ret, beta_window, vol_window, min_obs, beta_shrinkage)
        last_trade = series.index.max()
        for when in pd.to_datetime(list(rebalance_dates)):
            if when < series.index.min() or (when - last_trade).days > stale_days:
                continue
            beta, vol = _risk_asof(risk, when)
            if np.isfinite(beta) and np.isfinite(vol) and vol > 0:
                rows[(when, ticker)] = {"beta": beta, "vol": vol}

    if not rows:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["rebalance_date", "ticker"],
        )
        return pd.DataFrame(columns=["beta", "vol"], index=empty_index, dtype="float64")
    frame = pd.DataFrame.from_dict(rows, orient="index")[["beta", "vol"]]
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["rebalance_date", "ticker"])
    return frame.sort_index()


def _trailing_beta_and_vol(
    close: pd.Series,
    bench_ret: pd.Series,
    beta_window: int,
    vol_window: int,
    min_obs: int,
    beta_shrinkage: float,
) -> pd.DataFrame:
    """Daily rolling ``beta`` and monthly-scaled ``vol``, each using data up to that day."""
    ret = close.pct_change()
    aligned = pd.concat([ret, bench_ret], axis=1, keys=["s", "b"]).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["beta", "vol"], dtype="float64")
    cov = aligned["s"].rolling(beta_window, min_periods=min_obs).cov(aligned["b"])
    var = aligned["b"].rolling(beta_window, min_periods=min_obs).var()
    raw_beta = cov / var.where(var > 0)
    beta = (1.0 - beta_shrinkage) * raw_beta + beta_shrinkage * 1.0
    vol_min = min(vol_window, min_obs)
    vol = aligned["s"].rolling(vol_window, min_periods=vol_min).std() * np.sqrt(TRADING_DAYS_PER_MONTH)
    return pd.DataFrame({"beta": beta, "vol": vol})


def _risk_asof(risk: pd.DataFrame, when: pd.Timestamp) -> Tuple[float, float]:
    """``(beta, vol)`` on the last trading day on or before ``when`` (NaN if none)."""
    if risk.empty:
        return float("nan"), float("nan")
    pos = risk.index.searchsorted(when, side="right") - 1
    if pos < 0:
        return float("nan"), float("nan")
    row = risk.iloc[pos]
    return float(row["beta"]), float(row["vol"])



def _close_series_by_ticker(prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Extract a clean, sorted, tz-naive ``Close`` Series per ticker."""
    closes: Dict[str, pd.Series] = {}
    for ticker, frame in prices.items():
        if frame is None or "Close" not in frame.columns or frame.empty:
            continue
        series = frame["Close"].dropna()
        series.index = to_naive_index(series.index, normalize=True)
        closes[ticker] = series.sort_index()
    return closes


def _forward_returns_at(
    closes: Dict[str, pd.Series],
    rebalance_date: pd.Timestamp,
    horizon_months: int,
    execution_lag: int = 0,
) -> pd.Series:
    """Forward return per ticker over ``[t, t + horizon]`` at the execution price.

    ``execution_lag`` selects the convention. With the default of ``0`` the entry
    and exit are priced at the close as of each date, which is the convention used
    throughout the monthly cross-sectional factor literature. With ``1`` they are
    priced at the following trading day's close, which removes the simultaneity of
    transacting at the very close used to form the signal; that setting is exercised
    as a robustness check. An example is dropped whenever either leg has no
    usable price, which is what keeps an incomplete forward window out of the data.
    """
    exit_date = rebalance_date + pd.DateOffset(months=horizon_months)

    returns: Dict[str, float] = {}
    for ticker, series in closes.items():
        entry_price = _execution_price(series, rebalance_date, execution_lag)
        exit_price = _execution_price(series, exit_date, execution_lag)
        if entry_price is None or exit_price is None or entry_price == 0:
            continue
        returns[ticker] = (exit_price / entry_price) - 1.0

    return pd.Series(returns, dtype="float64")


def _execution_price(
    series: pd.Series, when: pd.Timestamp, execution_lag: int = 0
) -> float | None:
    """Price at which a position decided on ``when`` is assumed to transact."""
    if execution_lag <= 0:
        return _price_asof(series, when)
    return _price_next_trading_day(series, when)


def _price_asof(series: pd.Series, when: pd.Timestamp) -> float | None:
    """Last close on or before ``when``; ``None`` if ``when`` predates the data.

    Note the simultaneity this convention carries: the signal at ``t`` is formed
    from data up to and including ``t``'s close, and the position is then priced at
    that same close.
    """
    if when < series.index.min():
        return None
    if when > series.index.max():
        return None
    value = series.asof(when)
    return None if pd.isna(value) else float(value)


def _price_next_trading_day(series: pd.Series, when: pd.Timestamp) -> float | None:
    """Close on the first trading day **strictly after** ``when`` (T+1 execution).

    Returns ``None`` when ``when`` predates the ticker's history (the stock was not
    trading at the decision date) or when no trading day follows ``when`` (the
    forward window runs past the available data).
    """
    if when < series.index.min():
        return None
    pos = series.index.searchsorted(when, side="right")
    if pos >= len(series):
        return None
    value = series.iloc[pos]
    return None if pd.isna(value) else float(value)


def _as_labelled_series(labels: Dict[tuple, int]) -> pd.Series:
    """Build the MultiIndex ``(rebalance_date, ticker)`` label Series."""
    if not labels:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["rebalance_date", "ticker"],
        )
        return pd.Series([], index=empty_index, name="label", dtype="int64")

    index = pd.MultiIndex.from_tuples(labels.keys(), names=["rebalance_date", "ticker"])
    return pd.Series(list(labels.values()), index=index, name="label", dtype="int64").sort_index()

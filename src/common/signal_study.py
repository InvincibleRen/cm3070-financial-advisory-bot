"""Do the advisor's technical Buy / Sell signals predict what a stock does next?

The advisor scores each stock from moving averages, golden / death crosses, MACD
and ADX and says Buy (score >= 3), Sell (score <= -3) or Hold. This module replays
that exact rule (``backtester.calculate_backtest_signals``) day by day on every
stock that was in the S&P 500 *on that day*, and measures what followed.

Two kinds of event:

* ``new_buy`` / ``new_sell`` - the day the signal changes to Buy / Sell (what the
  advisor would announce as news);
* ``buy_state`` / ``sell_state`` - every day the signal *is* Buy / Sell (what the
  advisor shows whenever someone asks).

For each event the position is taken at that day's close and held ``h`` trading
days. Three yardsticks:

* **vs SPY** - the stock's return minus SPY's over the same days;
* **vs universe** - minus the average return of all S&P 500 members that same day
  over the same horizon. This removes both the market's direction and the gap
  between the average stock and SPY, so it isolates whether the signal *picks*
  better stocks;
* **beta-adjusted** - minus beta x SPY, beta estimated on data up to the event day.

Inference: events on one day share one market move, so each day's events are
averaged first; consecutive days' holding periods then overlap, so the t-stat is
Newey-West with ``h`` lags on that daily series rather than a naive one.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.common.backtester import calculate_backtest_signals
from src.common.timeindex import to_naive_index

HORIZONS = (5, 20, 60)
EVENTS = ("new_buy", "new_sell", "buy_state", "sell_state", "all_member_days")
EVENT_LABELS = {
    "new_buy": "New Buy signal (the day it appears)",
    "new_sell": "New Sell signal (the day it appears)",
    "buy_state": "Any day the signal is Buy",
    "sell_state": "Any day the signal is Sell",
    "all_member_days": "Every member stock-day (reference)",
}


BETA_WINDOW, BETA_MIN_OBS, BETA_SHRINKAGE = 252, 126, 1.0 / 3.0


def rolling_beta(close: pd.Series, spy_close: pd.Series) -> pd.Series:
    """Blume-shrunk beta of daily returns on SPY over the trailing year, as of each day."""
    r = close.pct_change()
    m = spy_close.pct_change()
    cov = r.rolling(BETA_WINDOW, min_periods=BETA_MIN_OBS).cov(m)
    var = m.rolling(BETA_WINDOW, min_periods=BETA_MIN_OBS).var()
    raw = cov / var.where(var > 0)
    return (1.0 - BETA_SHRINKAGE) * raw + BETA_SHRINKAGE


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = to_naive_index(out.index, normalize=True)
    return out[~out.index.duplicated(keep="last")].sort_index()


def membership_mask(dates: pd.DatetimeIndex, ticker: str, snapshots: pd.Series) -> np.ndarray:
    """True on the days ``ticker`` was an S&P 500 member (last snapshot on or before each day)."""
    in_snap = np.array([ticker in s for s in snapshots.to_numpy()], dtype=bool)
    pos = snapshots.index.searchsorted(dates, side="right") - 1
    mask = np.zeros(len(dates), dtype=bool)
    ok = pos >= 0
    mask[ok] = in_snap[pos[ok]]
    return mask


def stock_panel(
    ticker: str,
    frame: pd.DataFrame,
    spy_close: pd.Series,
    snapshots: Optional[pd.Series],
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """One row per member trading day: signal, event flags, forward stock / SPY returns, beta."""
    data = _clean(frame)
    signals = calculate_backtest_signals(data)
    close = data["Close"].astype("float64")
    idx = close.index
    spy = spy_close.reindex(idx, method="ffill")

    raw = signals["raw_signal"]
    prev = raw.shift(1)
    out = pd.DataFrame(index=idx)
    out["ticker"] = ticker
    out["score"] = signals["score"]
    out["new_buy"] = (raw == "Buy") & (prev != "Buy")
    out["new_sell"] = (raw == "Sell") & (prev != "Sell")
    out["buy_state"] = raw == "Buy"
    out["sell_state"] = raw == "Sell"
    out["all_member_days"] = True
    warm = data["Close"].rolling(200).count() >= 200
    out["beta"] = rolling_beta(close, spy)
    for h in horizons:
        out[f"r{h}"] = close.shift(-h) / close - 1.0
        out[f"spy{h}"] = spy.shift(-h) / spy - 1.0
    member = membership_mask(idx, ticker, snapshots) if snapshots is not None else np.ones(len(idx), bool)
    return out[member & warm.to_numpy()]


def build_panel(
    prices: Dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    snapshots: Optional[pd.Series],
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Stack every stock's member days, adding each day's universe-average forward return."""
    spy_close = _clean(spy)["Close"].astype("float64")
    parts = [stock_panel(t, f, spy_close, snapshots, horizons)
             for t, f in prices.items() if f is not None and not f.empty and "Close" in f]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel["ticker"] = panel["ticker"].astype("category")
    for h in horizons:
        uni = panel.groupby("date")[f"r{h}"].transform("mean")
        panel[f"x_spy{h}"] = panel[f"r{h}"] - panel[f"spy{h}"]
        panel[f"x_uni{h}"] = panel[f"r{h}"] - uni
        panel[f"x_beta{h}"] = panel[f"r{h}"] - panel["beta"] * panel[f"spy{h}"]
    return panel


def newey_west_t(series: pd.Series, lags: int) -> float:
    """t-stat of the mean of a time series with Newey-West (Bartlett) standard error."""
    x = series.dropna().to_numpy(dtype="float64")
    n = x.size
    if n < 3 or np.abs(x).max() < 1e-12:
        return float("nan")
    e = x - x.mean()
    var = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / math.sqrt(var / n))


def summarise(panel: pd.DataFrame, event: str, h: int) -> Dict[str, float]:
    """Hit rates and mean excess returns for one event type and horizon."""
    sub = panel[panel[event]].dropna(subset=[f"r{h}", f"spy{h}"])
    nan = float("nan")
    if sub.empty:
        return {k: nan for k in ("n", "n_days", "beat_spy", "x_spy", "t_spy", "beat_uni",
                                 "x_uni", "t_uni", "x_beta", "t_beta")}
    daily = sub.groupby("date")[[f"x_spy{h}", f"x_uni{h}", f"x_beta{h}"]].mean()
    return {
        "n": float(len(sub)),
        "n_days": float(len(daily)),
        "beat_spy": float((sub[f"x_spy{h}"] > 0).mean()),
        "x_spy": float(sub[f"x_spy{h}"].mean()),
        "t_spy": newey_west_t(daily[f"x_spy{h}"], h),
        "beat_uni": float((sub[f"x_uni{h}"] > 0).mean()),
        "x_uni": float(sub[f"x_uni{h}"].mean()),
        "t_uni": newey_west_t(daily[f"x_uni{h}"], h),
        "x_beta": float(sub[f"x_beta{h}"].mean()),
        "t_beta": newey_west_t(daily[f"x_beta{h}"], h),
    }


def results_table(panel: pd.DataFrame, events: Iterable[str] = EVENTS,
                  horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    rows = {(e, h): summarise(panel, e, h) for e in events for h in horizons}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["event", "horizon"])
    return frame


def split_by_period(panel: pd.DataFrame, cut: str) -> Dict[str, pd.DataFrame]:
    c = pd.Timestamp(cut)
    return {f"before {c:%Y}": panel[panel["date"] < c], f"{c:%Y} onwards": panel[panel["date"] >= c]}

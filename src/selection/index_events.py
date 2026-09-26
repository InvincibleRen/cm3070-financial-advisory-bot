"""Event study: what happens to a stock's price around S&P 500 additions and removals?

For every membership change in the point-in-time constituent file, day 0 is the
first trading day on or after the effective date (index funds trade at the
close of day -1). For each window of trading days ``[a, b]`` relative to day 0 the
stock's return ``close[b] / close[a-1] - 1`` is compared with SPY's over the same
dates:

* **market-adjusted** excess = r_stock - r_SPY (the standard measure in index
  inclusion studies, needing no estimated parameters);
* **beta-adjusted** excess = r_stock - beta * r_SPY, with beta estimated on trading
  days -310..-61 (before the pre-event run-up, Blume-shrunk), as a robustness check.

Events are grouped, because the reasons differ:

* ``added``          - an established stock (at least a year of trading before day 0)
  promoted into the index, typically after its market value has grown;
* ``added_new``      - a spin-off or recent listing admitted almost immediately, a
  mechanical inclusion rather than a reward for past performance;
* ``removed_trading``- still trading 60+ days after removal (typically demoted to a
  smaller index after shrinking);
* ``removed_stopped``- stopped trading at removal (acquired or delisted), so only
  the pre-event windows exist.

Only the effective date is available, not the announcement date, which is usually
about five trading days earlier; windows ending at day -1 therefore include the
announcement reaction.

Significance: events on the same day share the same market move, so the t-stat is
computed across event *dates* (each date's events averaged first), not across
individual events, which would overstate the evidence.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.timeindex import to_naive_index

WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("-60..-1", -60, -1),
    ("-20..-1", -20, -1),
    ("-5..-1", -5, -1),
    ("0..+4", 0, 4),
    ("0..+19", 0, 19),
    ("0..+59", 0, 59),
    ("0..+249", 0, 249),
)
PATH_DAYS = (-60, -40, -20, -10, -5, -1, 0, 5, 10, 20, 40, 60, 120, 250)
BETA_WINDOW = (-310, -61)
BETA_MIN_OBS = 126
BETA_SHRINKAGE = 1.0 / 3.0
STILL_TRADING_DAYS = 60
NEW_LISTING_DAYS = 250
STALE_DAYS = 10
GROUPS = ("added", "added_new", "removed_trading", "removed_stopped")
GROUP_LABELS = {
    "added": "Added (established stock)",
    "added_new": "Added (spin-off / recent listing)",
    "removed_trading": "Removed, still trading (demoted)",
    "removed_stopped": "Removed, stopped trading (acquired / delisted)",
}


def membership_changes(snapshots: pd.Series, start: Optional[str] = None) -> pd.DataFrame:
    """One row per addition or removal: ``date, ticker, event`` (``"add"`` / ``"remove"``)."""
    rows: List[tuple] = []
    prev = None
    for date, members in snapshots.sort_index().items():
        if prev is not None:
            rows += [(date, t, "add") for t in sorted(members - prev)]
            rows += [(date, t, "remove") for t in sorted(prev - members)]
        prev = members
    frame = pd.DataFrame(rows, columns=["date", "ticker", "event"])
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    return frame.reset_index(drop=True)


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    close = frame["Close"].dropna()
    close.index = to_naive_index(close.index, normalize=True)
    return close[~close.index.duplicated(keep="last")].sort_index()


def event_returns(
    events: pd.DataFrame,
    prices: Dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    data_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Per event: its group plus market- and beta-adjusted excess returns for every window.

    ``data_start`` is the first date of the price history that was downloaded; a
    stock whose data begins right there is treated as established even if the
    cache holds less than a year before day 0 (its history was cut, not short).
    Without it, only the length of the trading history decides. Windows the data
    does not cover are NaN.
    """
    spy_close = _clean_close(spy)
    closes = {t: _clean_close(f) for t, f in prices.items() if f is not None and not f.empty}

    out: List[Dict[str, object]] = []
    for date, ticker, event in events[["date", "ticker", "event"]].itertuples(index=False):
        row: Dict[str, object] = {"date": date, "ticker": ticker, "event": event}
        close = closes.get(ticker)
        if close is None or close.empty:
            row["group"] = "no_data"
            out.append(row)
            continue

        idx = close.index
        p0 = int(idx.searchsorted(date, side="left"))
        after = len(idx) - p0
        if event == "add":
            history = p0
            established = history >= NEW_LISTING_DAYS or (
                data_start is not None and idx.min() <= data_start + pd.Timedelta(days=7)
            )
            row["group"] = "added" if established else "added_new"
        else:
            if after >= STILL_TRADING_DAYS:
                row["group"] = "removed_trading"
            elif p0 > 0 and (date - idx[p0 - 1]).days <= STALE_DAYS:
                row["group"] = "removed_stopped"
            else:
                row["group"] = "no_data"
                out.append(row)
                continue

        beta = _beta(close, spy_close, p0)
        row["beta"] = beta
        for label, a, b in WINDOWS:
            r, m = _window_returns(close, spy_close, p0, a, b)
            row[f"mkt {label}"] = r - m if np.isfinite(r) and np.isfinite(m) else np.nan
            row[f"beta {label}"] = (r - beta * m) if np.isfinite(r) and np.isfinite(m) and np.isfinite(beta) else np.nan
        for day in PATH_DAYS:
            r, m = _window_returns(close, spy_close, p0, -60, day) if day >= -60 else (np.nan, np.nan)
            row[f"path {day}"] = r - m if np.isfinite(r) and np.isfinite(m) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def _window_returns(close: pd.Series, spy: pd.Series, p0: int, a: int, b: int) -> Tuple[float, float]:
    """Stock and SPY returns from the close of day ``a - 1`` to the close of day ``b``."""
    i0, i1 = p0 + a - 1, p0 + b
    if i0 < 0 or i1 >= len(close) or i1 <= i0:
        return float("nan"), float("nan")
    d0, d1 = close.index[i0], close.index[i1]
    s0, s1 = spy.asof(d0), spy.asof(d1)
    if pd.isna(s0) or pd.isna(s1) or s0 == 0:
        return float("nan"), float("nan")
    return float(close.iloc[i1] / close.iloc[i0] - 1.0), float(s1 / s0 - 1.0)


def _beta(close: pd.Series, spy: pd.Series, p0: int) -> float:
    lo, hi = p0 + BETA_WINDOW[0], p0 + BETA_WINDOW[1]
    if hi <= 0:
        return float("nan")
    seg = close.iloc[max(lo, 0): hi + 1]
    stock_ret = seg.pct_change()
    spy_ret = spy.reindex(seg.index, method="ffill").pct_change()
    both = pd.concat([stock_ret, spy_ret], axis=1).dropna()
    if len(both) < BETA_MIN_OBS or both.iloc[:, 1].var() == 0:
        return float("nan")
    raw = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
    return float((1.0 - BETA_SHRINKAGE) * raw + BETA_SHRINKAGE)


def summarise(values: pd.Series, dates: pd.Series) -> Dict[str, float]:
    """N, share positive, mean, median and a date-clustered t-stat of excess returns."""
    df = pd.DataFrame({"v": values.to_numpy(dtype="float64"), "d": dates.to_numpy()}).dropna()
    nan = float("nan")
    if df.empty:
        return {"n": 0.0, "n_dates": 0.0, "share_positive": nan, "mean": nan, "median": nan, "t": nan}
    by_date = df.groupby("d")["v"].mean()
    sd = float(by_date.std(ddof=1)) if len(by_date) > 1 else nan
    t = float(by_date.mean() / (sd / math.sqrt(len(by_date)))) if sd and sd > 0 else nan
    return {
        "n": float(len(df)),
        "n_dates": float(len(by_date)),
        "share_positive": float((df["v"] > 0).mean()),
        "mean": float(df["v"].mean()),
        "median": float(df["v"].median()),
        "t": t,
    }


def window_table(results: pd.DataFrame, group: str, measure: str = "mkt") -> pd.DataFrame:
    """Rows = windows, columns = the ``summarise`` statistics, for one event group."""
    sub = results[results["group"] == group]
    rows = {label: summarise(sub[f"{measure} {label}"], sub["date"]) for label, _, _ in WINDOWS}
    return pd.DataFrame.from_dict(rows, orient="index")


def average_path(results: pd.DataFrame, group: str) -> pd.Series:
    """Mean cumulative market-adjusted return from day -61, at each day in ``PATH_DAYS``."""
    sub = results[results["group"] == group]
    return pd.Series({day: sub[f"path {day}"].mean() for day in PATH_DAYS}, name=group)


def split_by_period(results: pd.DataFrame, cut: str) -> Dict[str, pd.DataFrame]:
    """The results before and on/after ``cut``, to see whether an effect has faded."""
    cut_ts = pd.Timestamp(cut)
    return {
        f"before {cut_ts:%Y}": results[results["date"] < cut_ts],
        f"{cut_ts:%Y} onwards": results[results["date"] >= cut_ts],
    }


def group_counts(results: pd.DataFrame, groups: Sequence[str] = GROUPS) -> Dict[str, int]:
    counts = results["group"].value_counts()
    out = {g: int(counts.get(g, 0)) for g in groups}
    out["no_data"] = int(counts.get("no_data", 0))
    return out


def drop_ticker_changes(
    events: pd.DataFrame, prices: Dict[str, pd.DataFrame], lookback: int = 20, tol: float = 0.01
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Remove add/remove pairs that are really one company changing ticker.

    ``constituents.TICKER_ALIASES`` handles the known renames. Any it misses show
    up as a removal and an addition on the same date whose price histories are the
    same series (yfinance files the full history under both symbols). Such a pair
    is detected when the two closes over the ``lookback`` days before the date
    differ by less than ``tol`` everywhere, and both rows are dropped.
    Returns the cleaned events and the ``(old, new)`` pairs removed.
    """
    closes = {t: _clean_close(f) for t, f in prices.items() if f is not None and not f.empty}
    drop: set = set()
    pairs: List[Tuple[str, str]] = []
    for date, day in events.groupby("date"):
        removed = [t for t in day.loc[day["event"] == "remove", "ticker"] if t in closes]
        added = [t for t in day.loc[day["event"] == "add", "ticker"] if t in closes]
        for old in removed:
            a = closes[old][closes[old].index < date].tail(lookback)
            for new in added:
                b = closes[new].reindex(a.index)
                if len(a) == lookback and b.notna().all() and ((b / a - 1).abs() < tol).all():
                    drop.update({(date, old), (date, new)})
                    pairs.append((old, new))
    keep = [(d, t) not in drop for d, t in zip(events["date"], events["ticker"])]
    return events[keep].reset_index(drop=True), pairs

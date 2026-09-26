"""Point-in-time S&P 500 membership, to remove survivorship bias from the universe.

``universe.DEFAULT_UNIVERSE`` is *today's* constituent list. Back-testing on it
quietly selects the winners: a small, volatile company that is in the index today
got there by rising, while the ones that fell out are absent from the history.
This module instead answers "which stocks were in the S&P 500 on date t?", so each
month's cross-section contains only names an investor could actually have held.

Data: ``data/constituents/sp500_historical_components.csv`` from the MIT-licensed
repository github.com/fja05680/sp500 (one row per membership change, with the full
constituent list after it).

Ticker changes
--------------
The file uses the ticker in force at each date, while yfinance files a renamed
company's whole history under its *current* ticker. ``TICKER_ALIASES`` maps each
old ticker to its successor so a company that merely changed ticker is not
dropped for the years before the change. Every pair was checked against the price
data: the successor's history covers the old ticker's membership period and the
two never appear in the same snapshot. Companies that were acquired or went
bankrupt are deliberately *not* aliased; their prices are simply missing from
yfinance, which ``membership_coverage`` reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

DEFAULT_PATH = (Path(__file__).resolve().parents[2]
                / "data" / "constituents" / "sp500_historical_components.csv")

TICKER_ALIASES: Dict[str, str] = {
    "ABC": "COR",
    "ANTM": "ELV",
    "BLL": "BALL",
    "BK": "BNY",
    "FI": "FISV",
    "FLT": "CPAY",
    "JEC": "J",
    "SYMC": "GEN",
    "NLOK": "GEN",
    "PKI": "RVTY",
    "RE": "EG",
    "WLTW": "WTW",
    "HRS": "LHX",
    "UTX": "RTX",
    "MYL": "VTRS",
    "DWDP": "DD",
    "TMK": "GL",
    "BHGE": "BKR",
    "ARNC": "HWM",
    "DISCA": "WBD",
    "MMC": "MRSH",
    "CBS": "PSKY",
    "VIAC": "PSKY",
    "GPS": "GAP",
    "CTL": "LUMN",
    "FBHS": "FBIN",
    "HFC": "DINO",
    "ADS": "BFH",
    "HCP": "DOC",
    "PEAK": "DOC",
}


def normalise_ticker(raw: str) -> str:
    """Source ticker -> the yfinance ticker the price cache uses."""
    ticker = raw.strip().upper().replace(".", "-")
    return TICKER_ALIASES.get(ticker, ticker)


def load_snapshots(path: Optional[Path] = None) -> pd.Series:
    """Membership snapshots: a Series ``date -> frozenset(tickers)``, sorted by date."""
    frame = pd.read_csv(path or DEFAULT_PATH, parse_dates=["date"])
    sets = frame["tickers"].fillna("").str.split(",").apply(
        lambda names: frozenset(normalise_ticker(t) for t in names if t.strip())
    )
    return pd.Series(sets.to_numpy(), index=pd.DatetimeIndex(frame["date"]), name="members").sort_index()


def members_at(snapshots: pd.Series, dates: Iterable[pd.Timestamp]) -> Dict[pd.Timestamp, Set[str]]:
    """Constituents in force on each date (the last snapshot on or before it).

    Dates before the first snapshot get an empty set, so they contribute nothing
    rather than silently borrowing a later list.
    """
    out: Dict[pd.Timestamp, Set[str]] = {}
    for when in pd.to_datetime(list(dates)):
        pos = snapshots.index.searchsorted(when, side="right") - 1
        out[when] = set(snapshots.iloc[pos]) if pos >= 0 else set()
    return out


def historical_universe(snapshots: pd.Series, dates: Iterable[pd.Timestamp]) -> List[str]:
    """Every ticker that was a member on at least one of ``dates``."""
    return sorted(set().union(*members_at(snapshots, dates).values()))


def filter_to_members(frame: pd.DataFrame, membership: Dict[pd.Timestamp, Set[str]]) -> pd.DataFrame:
    """Keep only the ``(date, ticker)`` rows where the ticker was a member on that date."""
    if frame.empty:
        return frame
    dates = frame.index.get_level_values(0)
    tickers = frame.index.get_level_values(1)
    keep = [t in membership.get(d, ()) for d, t in zip(dates, tickers)]
    return frame[keep]


def membership_coverage(
    membership: Dict[pd.Timestamp, Set[str]], present: Dict[pd.Timestamp, Set[str]]
) -> pd.DataFrame:
    """Per date: how many members there were, and how many made it into the data.

    ``present`` is the set of tickers actually in the cross-section on each date
    (e.g. the feature matrix after ``filter_to_members``).
    """
    rows = {
        d: {"members": len(m), "with_prices": len(m & present.get(d, set()))}
        for d, m in membership.items()
    }
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame["coverage"] = frame["with_prices"] / frame["members"].where(frame["members"] > 0)
    return frame

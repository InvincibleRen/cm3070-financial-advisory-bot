"""S&P 500 addition / removal event study: do stocks rise on entry and fall on exit?

Usage::

    python -m src.cli.index_events_cli                  # 2015 onwards, report to reports/
    python -m src.cli.index_events_cli --split 2020     # also compare before / after 2020
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.selection import constituents as K
from src.selection import index_events as E
from src.selection import universe as U

BENCHMARK = "SPY"


def _pct(v: float, digits: int = 2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{digits}f}%"


def _share(v: float) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"


def _num(v: float) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"


def _table(lines: List[str], table: pd.DataFrame) -> None:
    lines.append("| Window (trading days) | Events | Event dates | Beat SPY | Mean excess | Median excess | t (by date) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, r in table.iterrows():
        lines.append(f"| {label} | {int(r['n'])} | {int(r['n_dates'])} | {_share(r['share_positive'])} "
                     f"| {_pct(r['mean'])} | {_pct(r['median'])} | {_num(r['t'])} |")
    lines.append("")


def build_report(results: pd.DataFrame, pairs, n_raw: int, start: str, split: Optional[str]) -> str:
    L: List[str] = []
    add = L.append
    add("# S&P 500 Additions and Removals: Event Study")
    add("")
    add(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("## Setup")
    add("")
    add(f"- Events: every membership change since {start} in the point-in-time constituent "
        f"file ({n_raw} changes; {len(pairs)} add/remove pair(s) dropped as ticker changes: "
        + (", ".join(f"{a}->{b}" for a, b in pairs) if pairs else "none") + ").")
    add("- Day 0 = first trading day on or after the effective date. A window `a..b` runs from "
        "the close of day a-1 to the close of day b, so `-5..-1` ends at the close when index "
        "funds trade.")
    add("- **Excess return** = stock return minus SPY return over the same days (market-adjusted). "
        "Prices are dividend-adjusted. A beta-adjusted version (beta from days -310..-61) is "
        "given at the end as a robustness check.")
    add("- **Beat SPY** = share of events whose excess return was positive (the \"probability of "
        "rising relative to the market\").")
    add("- **t (by date)**: several changes often share one effective date and one market move, "
        "so each date's events are averaged first and the t-stat is taken across dates. "
        "|t| above about 2 is the usual bar.")
    add("- Only effective dates are available. The announcement is usually about five trading "
        "days earlier, so windows ending at day -1 already contain the announcement reaction; "
        "a trader could only have acted after the announcement.")
    add("")
    counts = E.group_counts(results)
    add("| Group | Events with price data |")
    add("|---|---:|")
    for g in E.GROUPS:
        add(f"| {E.GROUP_LABELS[g]} | {counts[g]} |")
    add(f"| No usable price data (mostly acquired / bankrupt, removed from yfinance) | {counts['no_data']} |")
    add("")

    for i, g in enumerate(E.GROUPS, start=1):
        add(f"## {i}. {E.GROUP_LABELS[g]}")
        add("")
        _table(L, E.window_table(results, g, "mkt"))

    add("## 5. Average path (cumulative excess return from day -61)")
    add("")
    paths = pd.concat([E.average_path(results, g) for g in E.GROUPS], axis=1)
    add("| Day | " + " | ".join(E.GROUP_LABELS[g] for g in E.GROUPS) + " |")
    add("|---:|" + "---:|" * len(E.GROUPS))
    for day, row in paths.iterrows():
        add(f"| {day} | " + " | ".join(_pct(row[g]) for g in E.GROUPS) + " |")
    add("")
    add("*Each cell averages whatever events cover that day, so later days can rest on fewer "
        "events (recent changes have no +250 yet).*")
    add("")

    if split:
        add(f"## 6. Has the effect faded? (split at {split})")
        add("")
        for g in ("added", "removed_trading"):
            add(f"**{E.GROUP_LABELS[g]}**")
            add("")
            add("| Window | Period | Events | Beat SPY | Mean excess | t (by date) |")
            add("|---|---|---:|---:|---:|---:|")
            for period, sub in E.split_by_period(results, split).items():
                table = E.window_table(sub, g, "mkt")
                for label, r in table.iterrows():
                    add(f"| {label} | {period} | {int(r['n'])} | {_share(r['share_positive'])} "
                        f"| {_pct(r['mean'])} | {_num(r['t'])} |")
            add("")

    add("## 7. Robustness: beta-adjusted excess returns")
    add("")
    add("| Group | Window | Events | Beat beta x SPY | Mean | t (by date) |")
    add("|---|---|---:|---:|---:|---:|")
    for g in ("added", "removed_trading"):
        for label, r in E.window_table(results, g, "beta").iterrows():
            add(f"| {E.GROUP_LABELS[g]} | {label} | {int(r['n'])} | {_share(r['share_positive'])} "
                f"| {_pct(r['mean'])} | {_num(r['t'])} |")
    add("")
    add("## Caveats")
    add("")
    add("- Removals that stopped trading (acquisitions, bankruptcies) mostly have no yfinance data "
        "at all, so the removal groups under-represent the worst and the acquired outcomes.")
    add("- Many windows are tested across several groups; a single |t| slightly above 2 is weak "
        "evidence on its own.")
    add("- Descriptive research on past data, not financial advice.")
    add("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2015-01-01", help="First effective date to include")
    parser.add_argument("--split", default="2020-01-01",
                        help="Also compare events before / after this date ('' to skip)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    events = E.membership_changes(K.load_snapshots(), start=args.start)
    tickers = sorted(set(events["ticker"]))
    cache = Path("data/prices")
    priced = [t for t in tickers if (cache / f"{t}.csv").exists()]
    print(f"{len(events)} membership changes since {args.start}; {len(priced)}/{len(tickers)} "
          "tickers have cached prices")
    prices = U.load_prices(priced, start="2015-01-01")
    spy = U.load_prices([BENCHMARK], start="2015-01-01")[BENCHMARK]

    clean, pairs = E.drop_ticker_changes(events, prices)
    if pairs:
        print(f"Dropped ticker changes: {pairs}")
    data_start = min(pd.to_datetime(f.index.min(), utc=True).tz_convert(None).normalize()
                     for f in prices.values() if not f.empty)
    results = E.event_returns(clean, prices, spy, data_start=data_start)

    for g in E.GROUPS:
        t = E.window_table(results, g, "mkt")
        print(f"\n{E.GROUP_LABELS[g]}")
        print(t[["n", "share_positive", "mean", "t"]].to_string(float_format=lambda v: f"{v:.3f}"))

    if not args.no_save:
        out = Path(__file__).resolve().parent.parent.parent / "reports" / \
            f"index_events_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(build_report(results, pairs, len(events), args.start, args.split or None))
        print(f"\nReport written to {out}")
    return results


if __name__ == "__main__":
    main()

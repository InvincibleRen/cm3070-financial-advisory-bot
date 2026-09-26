"""Do the advisor's technical Buy / Sell signals work? A survivorship-free test.

Replays the advisor's rule on every stock on the days it was an S&P 500 member
(2015 onwards) and reports what happened over the next 5 / 20 / 60 trading days.

Usage::

    python -m src.cli.signal_study_cli
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common import signal_study as Z
from src.selection import constituents as K
from src.selection import universe as U


def _pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v * 100:+.2f}%"


def _share(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v * 100:.1f}%"


def _num(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.2f}"


def _table(add, table: pd.DataFrame, events: Sequence[str]) -> None:
    add("| Signal | Hold (days) | Events | Days | Beat SPY | Mean vs SPY (t) | Beat same-day universe "
        "| Mean vs universe (t) | Mean vs beta x SPY (t) |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for event in events:
        for h in Z.HORIZONS:
            r = table.loc[(event, h)]
            add(f"| {Z.EVENT_LABELS[event]} | {h} | {int(r['n']):,} | {int(r['n_days']):,} "
                f"| {_share(r['beat_spy'])} | {_pct(r['x_spy'])} ({_num(r['t_spy'])}) "
                f"| {_share(r['beat_uni'])} | {_pct(r['x_uni'])} ({_num(r['t_uni'])}) "
                f"| {_pct(r['x_beta'])} ({_num(r['t_beta'])}) |")
    add("")


def build_report(table: pd.DataFrame, periods, meta: dict) -> str:
    L: List[str] = []
    add = L.append
    add("# Do the Advisor's Technical Signals Work?")
    add("")
    add(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("## Setup")
    add("")
    for k, v in meta.items():
        add(f"- {k}: {v}")
    add("- Rule: the advisor's score from price vs SMA20/50/200, golden/death crosses, MACD vs "
        "signal and the ADX amplifier (`backtester.calculate_backtest_signals`); Buy at score "
        ">= 3, Sell at <= -3. The live advisor scales the cross by its strength (+1 to +3) where "
        "this replay uses a fixed +2, and adds a small intraday nudge; otherwise the rules match.")
    add("- Each event is entered at that day's close and held 5, 20 or 60 trading days.")
    add("- **Beat same-day universe**: did the stock beat the average S&P 500 member over the same "
        "days? This removes the market's direction and is the cleanest test of whether the "
        "signal picks better stocks. A useful Buy signal should be well above 50% and its mean "
        "clearly positive; a useful Sell signal the opposite.")
    add("- t-stats: each day's events are averaged first, then a Newey-West t-stat with as many "
        "lags as the holding period (holding periods of consecutive days overlap). |t| above "
        "about 2 is the usual bar. The mean shown is pooled over events, while the t-stat tests "
        "the average *day*, so days with many events weigh less in the t-stat; when the effect "
        "is near zero the two can even differ in sign.")
    add("")
    add("## 1. Results, 2015 onwards")
    add("")
    _table(add, table, Z.EVENTS)
    add("## 2. Before and after 2020")
    add("")
    for label, t in periods.items():
        add(f"**{label}**")
        add("")
        _table(add, t, ("new_buy", "new_sell", "buy_state", "sell_state"))
    add("## Caveats")
    add("")
    add("- Members that were later acquired or went bankrupt are missing where yfinance has no "
        "prices, and a holding that stops trading inside the window is dropped.")
    add("- Transaction costs are not deducted; each new signal implies a trade.")
    add("- Research on past data; not financial advice.")
    add("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--split", default="2020-01-01")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    snapshots = K.load_snapshots()
    months = pd.date_range(args.start, pd.Timestamp.today(), freq="ME")
    tickers = K.historical_universe(snapshots, months)
    cache = Path("data/prices")
    priced = [t for t in tickers if (cache / f"{t}.csv").exists()]
    print(f"{len(priced)}/{len(tickers)} historical members have cached prices")
    prices = U.load_prices(priced, start=args.start)
    spy = U.load_prices(["SPY"], start=args.start)["SPY"]

    print("Replaying the advisor's rule on every member stock-day...")
    panel = Z.build_panel(prices, spy, snapshots)
    print(f"{len(panel):,} member stock-days, {panel['ticker'].nunique()} stocks")
    table = Z.results_table(panel)
    periods = {k: Z.results_table(v) for k, v in Z.split_by_period(panel, args.split).items()}
    print(table[["n", "beat_spy", "x_spy", "t_spy", "beat_uni", "x_uni", "t_uni"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    if not args.no_save:
        meta = {
            "Universe": "point-in-time S&P 500 (each day's actual members)",
            "Stocks": f"{panel['ticker'].nunique()} with prices out of {len(tickers)} members since {args.start}",
            "Member stock-days": f"{len(panel):,} ({panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d})",
        }
        out = (Path(__file__).resolve().parent.parent.parent / "reports"
               / f"signal_study_{datetime.now():%Y%m%d_%H%M%S}.md")
        out.write_text(build_report(table, periods, meta))
        print(f"Report written to {out}")
    return table


if __name__ == "__main__":
    main()

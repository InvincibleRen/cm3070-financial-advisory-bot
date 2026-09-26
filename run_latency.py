"""Measure single-stock recommendation latency, reproducibly, from the local cache.

The user-facing responsiveness of the app is dominated by *data retrieval* (the
network call to the provider). This benchmark isolates the part the system itself
controls -- the recommendation *compute*: turning an in-memory price history into
an indicator snapshot and a rule-based recommendation
(:func:`latest_indicator_snapshot` + :func:`generate_recommendation`), exactly the
path :func:`src.ui.data_access.analyse_single_stock` runs once the data is in hand.

Prices are read from the committed cache in ``data/prices/`` (no network), every
cached ticker with enough history is timed several times, and the median /
percentiles are written both to a timestamped report under ``reports/`` and into
``data/evidence/evidence.json`` under a ``latency`` key, so the latency figure
traces to a run rather than being hand-typed.

Usage::

    python run_latency.py [--repeats N] [--max-tickers M]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.data.yfinance_provider import Quote
from src.common.indicators import latest_indicator_snapshot
from src.explanation.advisor import generate_recommendation

ROOT = Path(__file__).resolve().parent
PRICES = ROOT / "data" / "prices"
REPORTS = ROOT / "reports"
EVIDENCE = ROOT / "data" / "evidence" / "evidence.json"

MIN_ROWS = 200


def _load_prices(path: Path) -> pd.DataFrame | None:
    """Load one cached price CSV into a Date-indexed OHLCV frame, or None."""
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if "Date" not in df.columns or "Close" not in df.columns or len(df) < MIN_ROWS:
        return None
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index = df.index.tz_localize(None)
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df if len(df) >= MIN_ROWS else None


def _quote(ticker: str, prices: pd.DataFrame) -> Quote:
    last = float(prices["Close"].iloc[-1])
    prev = float(prices["Close"].iloc[-2])
    return Quote(
        ticker=ticker, latest_price=last, previous_close=prev,
        change=last - prev, change_percent=(last / prev - 1.0) * 100.0,
        timestamp=str(prices.index[-1].date()), source="cache", note="latency-benchmark",
    )


def _recommend(prices: pd.DataFrame, quote: Quote) -> None:
    """The timed recommendation compute (data already retrieved)."""
    indicators = latest_indicator_snapshot(historical_data=prices, intraday_data=None)
    generate_recommendation(quote=quote, indicators=indicators)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats per ticker")
    ap.add_argument("--max-tickers", type=int, default=0, help="0 = all cached tickers")
    args = ap.parse_args(argv)

    paths = sorted(PRICES.glob("*.csv"))
    if args.max_tickers:
        paths = paths[: args.max_tickers]

    loaded: list[tuple[str, pd.DataFrame, Quote]] = []
    for p in paths:
        df = _load_prices(p)
        if df is not None:
            loaded.append((p.stem, df, _quote(p.stem, df)))
    if not loaded:
        raise SystemExit("no usable cached price files found under data/prices/")

    _recommend(loaded[0][1], loaded[0][2])

    samples_ms: list[float] = []
    for _ticker, df, q in loaded:
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            _recommend(df, q)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    n = len(samples_ms)
    quant = statistics.quantiles(samples_ms, n=100)
    result = {
        "metric": "single-stock recommendation compute (indicators + rule-based advisor), "
                  "from cached prices, excluding data retrieval",
        "median_ms": round(statistics.median(samples_ms), 2),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
        "p95_ms": round(quant[94], 2),
        "p99_ms": round(quant[98], 2),
        "min_ms": round(samples_ms[0], 2),
        "max_ms": round(samples_ms[-1], 2),
        "n_samples": n,
        "n_tickers": len(loaded),
        "repeats_per_ticker": args.repeats,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORTS.mkdir(exist_ok=True)
    md = [
        "# Recommendation latency", "",
        f"_{result['metric']}._", "",
        f"- Median: **{result['median_ms']} ms**",
        f"- Mean: {result['mean_ms']} ms  |  p95: {result['p95_ms']} ms  |  p99: {result['p99_ms']} ms",
        f"- Range: {result['min_ms']}-{result['max_ms']} ms",
        f"- Samples: {result['n_samples']} ({result['n_tickers']} tickers × {result['repeats_per_ticker']} repeats)",
        f"- Generated: {result['generated_at']}", "",
    ]
    report_path = REPORTS / f"latency_{ts}.md"
    report_path.write_text("\n".join(md))
    (REPORTS / f"latency_{ts}.json").write_text(json.dumps(result, indent=2))

    if EVIDENCE.exists():
        evidence = json.loads(EVIDENCE.read_text())
        evidence["latency"] = result
        EVIDENCE.write_text(json.dumps(evidence, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nwrote {report_path.relative_to(ROOT)} and updated {EVIDENCE.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()

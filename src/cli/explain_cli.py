"""Explain the walk-forward XGBoost return model: which features drive it, and do they work?

Usage::

    python -m src.cli.explain_cli --universe current --normalize rank     # the best-looking run
    python -m src.cli.explain_cli --universe historical --normalize rank  # the unbiased one
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.cli.hurdle_cli import load_inputs
from src.selection import explain as X
from src.selection.feature_labels import FEATURE_LABELS

DESCRIPTIONS = FEATURE_LABELS
PERIODS = (("2015-2017", "2015-01-01", "2017-12-31"), ("2018-2020", "2018-01-01", "2020-12-31"),
           ("2021-2023", "2021-01-01", "2023-12-31"), ("2024-2026", "2024-01-01", "2026-12-31"))
TOP_DEPENDENCE = 6


def _direction_text(d: float) -> str:
    if not np.isfinite(d):
        return "n/a"
    if d > 0.1:
        return "higher -> higher prediction"
    if d < -0.1:
        return "higher -> lower prediction"
    return "no clear direction"


def _support_text(direction: float, ic: float, t: float) -> str:
    """Does the feature's own out-of-sample IC back the direction the model learned?"""
    if not (np.isfinite(direction) and np.isfinite(ic) and np.isfinite(t)) or abs(direction) <= 0.1:
        return "n/a"
    if abs(t) < 2:
        return "no support (own IC not significant)"
    return "supported" if np.sign(direction) == np.sign(ic) else "contradicted (IC has the opposite sign)"


def build_report(table: pd.DataFrame, periods: pd.DataFrame, dep: pd.DataFrame, meta: dict) -> str:
    L: List[str] = []
    add = L.append
    add("# What the XGBoost Return Model Learned")
    add("")
    add(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("## Setup")
    add("")
    for k, v in meta.items():
        add(f"- {k}: {v}")
    add("")
    add("Every number below is out of sample: each month's model (trained only on the previous "
        "12 months) explains the predictions it made for that month's stocks. **Reliance** is the "
        "average absolute SHAP contribution, i.e. how far the feature moves a prediction; the "
        "shares add up to 100%. **Direction** is the rank correlation between a feature's value "
        "and its contribution. **Own IC** is how well the feature alone ranks the realised "
        "target each month (mean Spearman correlation, t-stat across months). A feature the model "
        "leans on without its own IC behind it is being used to fit noise.")
    add("")
    add("## 1. Reliance, direction and whether the data backs it")
    add("")
    add("| # | Feature | Meaning | Reliance | Direction learned | Own IC (t) | Backed by own IC? | Coverage |")
    add("|---:|---|---|---:|---|---:|---|---:|")
    for i, (f, r) in enumerate(table.iterrows(), start=1):
        add(f"| {i} | `{f}` | {DESCRIPTIONS.get(f, f)} | {r['share'] * 100:.1f}% "
            f"| {_direction_text(r['direction'])} ({r['direction']:+.2f}) "
            f"| {r['own_ic']:+.3f} ({r['own_ic_t']:+.2f}) "
            f"| {_support_text(r['direction'], r['own_ic'], r['own_ic_t'])} | {r['coverage'] * 100:.0f}% |")
    add("")
    add(f"## 2. Shape of the top {len(dep.columns)} features")
    add("")
    add("Average contribution to the predicted target by the feature's quintile within each "
        "month (Q1 = lowest fifth of stocks, Q5 = highest). Positive = pushes the prediction up.")
    add("")
    add("| Feature | " + " | ".join(dep.index) + " |")
    add("|---|" + "---:|" * len(dep.index))
    for f in dep.columns:
        add(f"| `{f}` | " + " | ".join(f"{v:+.4f}" for v in dep[f]) + " |")
    add("")
    add("## 3. Does the model lean on the same features over time?")
    add("")
    top = list(table.index[:8])
    add("| Feature | " + " | ".join(periods.columns) + " |")
    add("|---|" + "---:|" * len(periods.columns))
    for f in top:
        add(f"| `{f}` | " + " | ".join(f"{periods.loc[f, c] * 100:.1f}%" for c in periods.columns) + " |")
    add("")
    add("## Reading guide")
    add("")
    add("- Reliance says what the model *uses*, not whether it is right; the IC column says "
        "whether the data supports it. Only rows marked 'supported' reflect a relationship "
        "that held out of sample.")
    add("- Fundamental features are mostly missing (see coverage); the booster routes missing "
        "values down a default branch, so their reliance mostly reflects whether a value exists.")
    add("- Out-of-sample research only; not financial advice.")
    add("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--universe", choices=["historical", "current"], default="current")
    parser.add_argument("--normalize", choices=["rank", "zscore", "none"], default="rank")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-train", type=int, default=6)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    inp = load_inputs(args.tickers, args.universe, args.start, args.end, args.normalize,
                      1, args.min_train)
    print("Running the XGBoost walk-forward and collecting SHAP values...")
    contribs, preds = X.walk_forward_contributions(inp.feature_matrix, inp.target, inp.dates,
                                                   min_train_dates=args.min_train)
    predicted = pd.concat({p.date: p.scores for p in preds}, names=["rebalance_date", "ticker"])
    gap = float((contribs.sum(axis=1) - predicted.reindex(contribs.index)).abs().max())
    print(f"SHAP sanity check: max |sum of contributions - prediction| = {gap:.2e}")

    table = X.importance_table(contribs, inp.feature_matrix, inp.target)
    periods = X.importance_by_period(contribs, PERIODS)
    dep = pd.concat({f: X.dependence(contribs, inp.feature_matrix, f)
                     for f in table.index[:TOP_DEPENDENCE]}, axis=1)
    print(table[["share", "direction", "own_ic", "own_ic_t", "coverage"]]
          .to_string(float_format=lambda v: f"{v:+.3f}"))

    if not args.no_save:
        meta = {
            "Model": "XGBoost regressor on the risk-adjusted excess-return target",
            "Universe": {"current": "today's S&P 500 list (survivorship-biased)",
                         "historical": "point-in-time S&P 500",
                         "custom": "user-supplied tickers"}[inp.mode],
            "Features": f"{args.normalize} per month",
            "Months explained": f"{len(preds)} ({preds[0].date:%Y-%m} to {preds[-1].date:%Y-%m})",
            "Predictions explained": f"{len(contribs)}",
            "SHAP check": f"contributions sum to the predictions (max error {gap:.1e})",
        }
        out = (Path(__file__).resolve().parent.parent.parent / "reports"
               / f"explain_xgb_{inp.mode}_{args.normalize}_{datetime.now():%Y%m%d_%H%M%S}.md")
        out.write_text(build_report(table, periods, dep, meta))
        print(f"Report written to {out}")
    return table


if __name__ == "__main__":
    main()

"""Evaluate the selector as a machine-learning model, not as a portfolio.

Answers, on the regression task the model is trained on: does it beat a constant
predictor out of sample, does it fit its training window better than the month it
scores, does more training data help, is its ordering stable month to month, and
where do its errors sit.

Usage::

    python -m src.cli.ml_eval_cli --universe historical            # the unbiased run
    python -m src.cli.ml_eval_cli --universe historical --target excess_spy
    python -m src.cli.ml_eval_cli --universe current               # for the bias comparison
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.cli.hurdle_cli import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    RETURN_KINDS,
    MODEL_NAMES,
    TARGET_COLUMNS,
    load_inputs,
)
from src.selection import ml_eval as M

BASELINE_LABEL = "Training-window mean (constant)"


def _num(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.{digits}f}"


def _verdict(r2: float) -> str:
    if not np.isfinite(r2):
        return "undefined"
    if r2 > 0.005:
        return "beats the constant"
    if r2 > -0.005:
        return "indistinguishable from the constant"
    return "worse than the constant"


def model_table(inp, kinds: Sequence[str], min_train: int,
                train_window: Optional[int] = M.S.TRAIN_WINDOW_MONTHS) -> pd.DataFrame:
    """Pooled prediction metrics per model family, on identical folds."""
    rows: Dict[str, Dict[str, float]] = {}
    for kind in kinds:
        print(f"  scoring {MODEL_NAMES.get(kind, kind)}...")
        scores = M.fold_scores(inp.feature_matrix, inp.target, inp.dates, kind, min_train,
                               train_window)
        rows[MODEL_NAMES.get(kind, kind)] = M.pooled_metrics(scores)
    return pd.DataFrame(rows).T


def build_report(
    models: pd.DataFrame,
    curve: pd.DataFrame,
    stability: Dict[str, float],
    errors: Dict[str, pd.DataFrame],
    meta: Dict[str, str],
) -> str:
    L: List[str] = []
    add = L.append
    add("# Machine-Learning Evaluation of the Prediction Task")
    add("")
    add(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("## Setup")
    add("")
    for key, value in meta.items():
        add(f"- {key}: {value}")
    add("")
    add("Every figure is out of sample. Each month's model is trained only on earlier "
        "rebalances and scores that month's cross-section; the metrics pool the rows of "
        "all months, so a wide month counts for more than a thin one. The baseline "
        "throughout is the mean of the model's **own training window** - the only "
        "constant a forecaster could have used without seeing the future (Campbell and "
        "Thompson, 2008). R² above zero means the model predicts better than that "
        "constant; below zero means worse.")
    add("")

    add("## 1. Does the model beat a constant? (out-of-sample R²)")
    add("")
    add("| Model | R² out of sample | R² in sample | Gap | RMSE | Baseline RMSE | Mean IC (t) | Folds | Verdict |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, row in models.iterrows():
        add(f"| {name} | {_num(row['r2_oos'])} | {_num(row['r2_train'])} | {_num(row['gap'])} "
            f"| {row['rmse']:.4f} | {row['rmse_baseline']:.4f} "
            f"| {_num(row['ic'], 3)} ({_num(row['ic_t'], 2)}) | {int(row['folds'])} "
            f"| {_verdict(row['r2_oos'])} |")
    add("")
    add("**Gap** is in-sample R² minus out-of-sample R². A large positive gap means the "
        "model explains the window it was fitted on and not the month it then scores, "
        "which is memorisation rather than learning.")
    add("")

    add("## 2. Does more training data help? (learning curve)")
    add("")
    add("| Training window | R² out of sample | R² in sample | Gap | Mean IC (t) | Folds |")
    add("|---|---:|---:|---:|---:|---:|")
    for name, row in curve.iterrows():
        add(f"| {name} | {_num(row['r2_oos'])} | {_num(row['r2_train'])} | {_num(row['gap'])} "
            f"| {_num(row['ic'], 3)} ({_num(row['ic_t'], 2)}) | {int(row['folds'])} |")
    add("")
    add("A model learning a real relationship improves, or at least stops degrading, as "
        "the window lengthens. One fitting noise mostly trades one kind of variance for "
        "another. The pipeline's fixed twelve-month window is one row of this table, and "
        "this is the first test of whether it was the right choice.")
    add("")

    add("## 3. Is the ordering stable month to month?")
    add("")
    add(f"- Mean rank correlation between this month's and last month's predictions for "
        f"the same stocks: **{_num(stability['prediction_rho'], 3)}** "
        f"({int(stability['pairs'])} adjacent pairs)")
    add(f"- The same statistic for the realised target: "
        f"**{_num(stability['target_rho'], 3)}**")
    add("")
    add("The second line bounds the first: no honest model can be more persistent than "
        "the thing it predicts. A prediction correlation far above the target's means "
        "the model is repeating itself rather than tracking anything.")
    add("")

    add("## 4. Where do the errors sit?")
    add("")
    add("### By year")
    add("")
    add("| Year | R² out of sample | IC | Rows |")
    add("|---|---:|---:|---:|")
    for year, row in errors["by_year"].iterrows():
        add(f"| {year} | {_num(row['r2_oos'])} | {_num(row['ic'], 3)} | {int(row['n'])} |")
    add("")
    by_vol = errors.get("by_volatility")
    if by_vol is not None and not by_vol.empty:
        add("### By ex-ante volatility bucket (0 = calmest fifth, 4 = most volatile)")
        add("")
        add("| Bucket | R² out of sample | IC | Rows |")
        add("|---|---:|---:|---:|")
        for bucket, row in by_vol.iterrows():
            add(f"| {int(bucket)} | {_num(row['r2_oos'])} | {_num(row['ic'], 3)} | {int(row['n'])} |")
        add("")
        add("The top bucket matters most: those are the names a hurdle selector tends to "
            "buy, so an error concentrated there is an error in the part of the "
            "cross-section the product actually acts on.")
        add("")

    add("## Reading guide")
    add("")
    add("- These are model-quality results, not investment results. A model can predict "
        "poorly and still sit inside a portfolio that rises, because the market rises; "
        "the portfolio evidence is reported separately.")
    add("- Hyperparameters are fixed at the values in `model.ReturnModel` and were not "
        "tuned against any of these numbers. No search was run, so nothing here is a "
        "best-of-many result, and no held-out tuning split is needed.")
    add("- Out-of-sample research only; not financial advice.")
    add("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--universe", choices=["historical", "current"], default="historical")
    parser.add_argument("--normalize", choices=["rank", "zscore", "none"], default="rank")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--first-rebalance", default=None)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--target", choices=list(TARGET_COLUMNS), default=DEFAULT_TARGET)
    parser.add_argument("--models", default=",".join(RETURN_KINDS))
    parser.add_argument("--min-train", type=int, default=6)
    parser.add_argument("--train-window", type=int, default=None,
                        help="Rolling training window in rebalances (default 12); e.g. 3 or 6")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    kinds = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in RETURN_KINDS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(RETURN_KINDS)}.")

    window = args.train_window if args.train_window is not None else M.S.TRAIN_WINDOW_MONTHS
    inp = load_inputs(args.tickers, args.universe, args.start, args.end, args.normalize,
                      1, args.min_train, args.first_rebalance, False, args.features,
                      args.target)
    print(f"Scoring the prediction task over {len(inp.dates)} rebalances "
          f"on {inp.feature_matrix.shape[1]} features, {window}-rebalance training window...")
    models = model_table(inp, kinds, args.min_train, window)
    print(models[["r2_oos", "r2_train", "gap", "ic", "ic_t"]]
          .to_string(float_format=lambda v: f"{v:+.4f}"))

    print("Learning curve over training-window length...")
    curve = M.learning_curve(inp.feature_matrix, inp.target, inp.dates, kinds[0],
                             args.min_train)
    print("Prediction stability...")
    stability = M.prediction_stability(inp.feature_matrix, inp.target, inp.dates,
                                       kinds[0], args.min_train, window)
    print("Error analysis...")
    volatility = inp.risk["vol"] if "vol" in getattr(inp.risk, "columns", []) else None
    errors = M.error_analysis(inp.feature_matrix, inp.target, inp.dates, volatility,
                              kinds[0], args.min_train, window)

    if not args.no_save:
        meta = {
            "Task": f"regression on the {args.target} target",
            "Universe": {"current": "today's S&P 500 list (survivorship-biased)",
                         "historical": "point-in-time S&P 500",
                         "custom": "user-supplied tickers"}[inp.mode],
            "Features": f"{inp.feature_matrix.shape[1]} ({args.features}), {args.normalize} per month",
            "Rebalances": f"{len(inp.dates)} ({inp.dates[0]:%Y-%m} to {inp.dates[-1]:%Y-%m})",
            "Training window": f"{window} rebalances rolling",
            "First rebalance": args.first_rebalance or args.start,
            "Baseline": BASELINE_LABEL,
        }
        tags = "" if args.features == DEFAULT_FEATURES else f"_feat-{args.features}"
        if args.target != DEFAULT_TARGET:
            tags += f"_tgt-{args.target}"
        out = (Path(__file__).resolve().parent.parent.parent / "reports"
               / f"ml_evaluation_{inp.mode}_{args.normalize}{tags}"
                 f"_{datetime.now():%Y%m%d_%H%M%S}.md")
        out.write_text(build_report(models, curve, stability, errors, meta))
        print(f"Report written to {out}")
    return models


if __name__ == "__main__":
    main()

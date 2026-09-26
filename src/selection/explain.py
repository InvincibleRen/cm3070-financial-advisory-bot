"""What has the walk-forward XGBoost return model actually learned?

Two complementary views, both strictly out of sample:

* **What the model relies on** - TreeSHAP contributions of each feature to every
  prediction the walk-forward actually made (each month's model explaining that
  month's scored cross-section). ``mean |SHAP|`` measures how much a feature moves
  the predictions; the sign of the rank correlation between a feature's value and
  its contribution gives the direction the model learned ("higher momentum ->
  higher predicted return" or the reverse).
* **Whether that is justified** - each feature's own information coefficient: the
  per-month Spearman correlation between the feature and the realised target. A
  feature the model leans on heavily but whose own IC is ~0 (or of the opposite
  sign) is being used for noise.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.selection import hurdle as H
from src.selection.feature_labels import label


def walk_forward_contributions(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    min_train_dates: int = 6,
    horizon_months: int = 1,
) -> Tuple[pd.DataFrame, List[H.Prediction]]:
    """Run the XGBoost walk-forward and keep every fold's out-of-sample SHAP values.

    Returns the contributions (indexed ``(rebalance_date, ticker)``, one column per
    feature plus ``bias``) and the predictions, which the contributions sum to.
    """
    frames: Dict[pd.Timestamp, pd.DataFrame] = {}

    def keep(date, model, X_now):
        frames[date] = model.contributions(X_now)

    preds = H.walk_forward_predictions(
        feature_matrix, target, rebalance_dates, model_kind="xgb",
        min_train_dates=min_train_dates, horizon_months=horizon_months, on_fold=keep,
    )
    contribs = pd.concat(frames, names=["rebalance_date", "ticker"]) if frames else pd.DataFrame()
    return contribs, preds


def importance_table(
    contribs: pd.DataFrame, feature_matrix: pd.DataFrame, realised_target: pd.Series
) -> pd.DataFrame:
    """Per feature: reliance (mean |SHAP|, share), learned direction, coverage and own IC."""
    features = [c for c in contribs.columns if c != "bias"]
    X = feature_matrix.reindex(contribs.index)
    mean_abs = contribs[features].abs().mean()
    rows = {}
    for f in features:
        both = pd.concat([X[f], contribs[f]], axis=1, keys=["x", "s"]).dropna()
        direction = (both["x"].corr(both["s"], method="spearman")
                     if len(both) > 2 and both["x"].nunique() > 1 else np.nan)
        preds = [H.Prediction(d, g.droplevel(0)) for d, g in X[f].groupby(level=0)]
        ic = H.information_coefficient(preds, realised_target)
        rows[f] = {
            "mean_abs_shap": float(mean_abs[f]),
            "share": float(mean_abs[f] / mean_abs.sum()) if mean_abs.sum() > 0 else np.nan,
            "direction": float(direction),
            "coverage": float(X[f].notna().mean()),
            "own_ic": ic["mean_ic"],
            "own_ic_t": ic["ic_t_stat"],
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("mean_abs_shap", ascending=False)


def importance_by_period(
    contribs: pd.DataFrame, periods: Sequence[Tuple[str, str, str]]
) -> pd.DataFrame:
    """Share of total mean |SHAP| per feature within each ``(label, start, end)`` period."""
    features = [c for c in contribs.columns if c != "bias"]
    dates = contribs.index.get_level_values(0)
    out = {}
    for label, lo, hi in periods:
        part = contribs[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))][features]
        if part.empty:
            continue
        m = part.abs().mean()
        out[label] = m / m.sum() if m.sum() > 0 else m
    return pd.DataFrame(out)


def dependence(contribs: pd.DataFrame, feature_matrix: pd.DataFrame, feature: str,
               n_bins: int = 5) -> pd.Series:
    """Average contribution of ``feature`` by its within-month quintile (Q1 = lowest).

    Shows the *shape* the model learned: rising, falling, or U-shaped.
    """
    x = feature_matrix.reindex(contribs.index)[feature]
    pct = x.groupby(level=0).rank(pct=True)
    bins = np.ceil(pct * n_bins).clip(1, n_bins)
    frame = pd.DataFrame({"bin": bins, "s": contribs[feature]}).dropna()
    means = frame.groupby("bin")["s"].mean()
    means.index = [f"Q{int(b)}" for b in means.index]
    return means


def pick_reasons(
    contributions: pd.DataFrame,
    features: pd.DataFrame,
    picks: Sequence[str],
    top_n: int = 3,
) -> Dict[str, List[Dict[str, object]]]:
    """Why the model scored each pick highly: its largest SHAP contributions.

    ``contributions`` and ``features`` cover one rebalance date, indexed by ticker.
    For each pick the ``top_n`` features that moved its score most are returned with
    the direction they pushed it and where the stock sat in that month's
    cross-section for that feature (1.0 = highest of all stocks scored that month).
    """
    columns = [c for c in contributions.columns if c != "bias"]
    percentile = features[columns].rank(pct=True)
    out: Dict[str, List[Dict[str, object]]] = {}
    for ticker in picks:
        if ticker not in contributions.index:
            continue
        row = contributions.loc[ticker, columns]
        reasons = []
        for feature in row.abs().sort_values(ascending=False).head(top_n).index:
            value = float(row[feature])
            pct = percentile.loc[ticker, feature] if ticker in percentile.index else np.nan
            reasons.append({
                "feature": feature,
                "label": label(feature),
                "contribution": value,
                "direction": "raised" if value > 0 else "lowered",
                "percentile": None if pd.isna(pct) else float(pct),
            })
        out[ticker] = reasons
    return out

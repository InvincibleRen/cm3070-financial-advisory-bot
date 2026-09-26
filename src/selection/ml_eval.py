"""Machine-learning evaluation of the prediction task itself.

Everything else in ``src/selection`` judges the selector the way an investor would:
what a portfolio of its picks returned, how much market risk it carried, whether its
alpha was significant. Those are domain results. They answer "was this worth doing?"
but not "did the model learn anything?", and a model can fail the second question
long before the first one is asked.

This module asks the second question, on the regression task the model is actually
trained on:

* **Out-of-sample R²** against the only baseline a forecaster is allowed at the time
  of the forecast - the mean of its own training window (Campbell and Thompson,
  2008). A negative value means the fitted model predicts the target *worse* than a
  constant, which is the cleanest possible statement that there is nothing to learn.
* **Training loss against test loss**, fold by fold. A model that fits its training
  window far better than the month it then scores is memorising; this measures the
  gap directly rather than inferring it from attribution.
* **Learning curves** over the training-window length. The pipeline fixes the window
  at twelve months and has never justified it. If the model were learning a real
  relationship, a longer window should help; if it is fitting noise, window length
  mostly moves the variance around.
* **Prediction stability**, the rank correlation between what the model says this
  month and what it said last month about the same stocks. A model tracking a
  persistent signal is persistent itself; one that reshuffles its ordering every
  month is reading noise.
* **Error analysis** by volatility bucket and by year, so a single pooled number
  cannot hide a model that works in one regime and not another.

The metrics are pooled across folds rather than averaged per fold: each month
contributes its own rows, and months differ in size, so pooling weights a month by
how much of the cross-section it actually held.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.selection import hurdle as H
from src.selection import select as S

WINDOW_GRID: Sequence[Optional[int]] = (3, 6, 12, 24, 36, None)
VOL_BUCKETS = 5


@dataclass
class FoldScore:
    """Metrics for one walk-forward fold."""

    date: pd.Timestamp
    n_train: int
    n_test: int
    train_r2: float
    test_r2: float
    test_rmse: float
    baseline_rmse: float
    ic: float


def _r2(y: np.ndarray, pred: np.ndarray, baseline: float) -> float:
    """R² against a constant ``baseline``, the forecaster's own training mean.

    Returns NaN when the baseline already explains the sample exactly (a flat target),
    where the ratio is undefined rather than zero.
    """
    if y.size == 0:
        return float("nan")
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - baseline) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - sse / sst


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _clean(y: pd.Series, pred: pd.Series):
    """Aligned, finite ``(y, prediction)`` pairs as arrays."""
    frame = pd.DataFrame({"y": y, "pred": pred}).replace([np.inf, -np.inf], np.nan).dropna()
    return frame["y"].to_numpy(dtype="float64"), frame["pred"].to_numpy(dtype="float64")


def _target_at(target: pd.Series, date: pd.Timestamp, tickers: pd.Index) -> pd.Series:
    """That date's slice of the target, indexed by ticker to match a scored fold.

    :attr:`Fold.test` comes from ``feature_matrix.loc[date]``, which drops the date
    level, so the two-level target has to be sliced before it can be aligned.
    """
    if date not in target.index.get_level_values(0):
        return pd.Series(np.nan, index=tickers, dtype="float64")
    return target.xs(date, level=0).reindex(tickers)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def fold_scores(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    model_kind: str = "xgb",
    min_train_dates: int = 6,
    train_window: Optional[int] = S.TRAIN_WINDOW_MONTHS,
) -> List[FoldScore]:
    """Per-fold training and test metrics for one model and one training window.

    The baseline for both sides is the mean of the *training* target, the only
    constant a forecaster could have used at that point without seeing the future.
    """
    out: List[FoldScore] = []
    for fold in H.walk_forward_folds(feature_matrix, target, rebalance_dates,
                                     model_kind, min_train_dates, train_window):
        y_tr, p_tr = _clean(target.reindex(fold.train.index),
                            fold.model.predict_scores(fold.train))
        y_te, p_te = _clean(_target_at(target, fold.date, fold.test.index),
                            fold.model.predict_scores(fold.test))
        if y_tr.size == 0 or y_te.size == 0:
            continue
        baseline = float(np.mean(y_tr))
        out.append(FoldScore(
            date=fold.date,
            n_train=int(y_tr.size),
            n_test=int(y_te.size),
            train_r2=_r2(y_tr, p_tr, baseline),
            test_r2=_r2(y_te, p_te, baseline),
            test_rmse=_rmse(y_te, p_te),
            baseline_rmse=_rmse(y_te, np.full_like(y_te, baseline)),
            ic=_spearman(y_te, p_te),
        ))
    return out


def pooled_metrics(scores: Sequence[FoldScore]) -> Dict[str, float]:
    """Sample-weighted summary of per-fold scores, plus the train-test gap.

    ``r2_oos`` pools by test-set size, so a month holding 480 names counts for more
    than one holding 90, matching how the squared errors would pool if the folds were
    concatenated. ``ic_t`` is the t-statistic of the monthly information coefficients,
    the standard test of whether a ranking is better than chance.
    """
    if not scores:
        return {k: float("nan") for k in
                ("r2_oos", "r2_train", "gap", "rmse", "rmse_baseline", "ic", "ic_t", "folds")}
    weights = np.array([s.n_test for s in scores], dtype="float64")
    test_r2 = np.array([s.test_r2 for s in scores], dtype="float64")
    train_r2 = np.array([s.train_r2 for s in scores], dtype="float64")
    ic = np.array([s.ic for s in scores], dtype="float64")
    ic = ic[np.isfinite(ic)]

    def wmean(values: np.ndarray) -> float:
        keep = np.isfinite(values)
        if not keep.any():
            return float("nan")
        return float(np.average(values[keep], weights=weights[keep]))

    ic_t = float("nan")
    if ic.size > 1 and np.std(ic, ddof=1) > 0:
        ic_t = float(np.mean(ic) / (np.std(ic, ddof=1) / np.sqrt(ic.size)))

    return {
        "r2_oos": wmean(test_r2),
        "r2_train": wmean(train_r2),
        "gap": wmean(train_r2) - wmean(test_r2),
        "rmse": wmean(np.array([s.test_rmse for s in scores])),
        "rmse_baseline": wmean(np.array([s.baseline_rmse for s in scores])),
        "ic": float(np.mean(ic)) if ic.size else float("nan"),
        "ic_t": ic_t,
        "folds": float(len(scores)),
    }


def learning_curve(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    model_kind: str = "xgb",
    min_train_dates: int = 6,
    windows: Sequence[Optional[int]] = WINDOW_GRID,
) -> pd.DataFrame:
    """Pooled metrics at each training-window length, indexed by the window.

    Folds are re-run per window, so every row is a complete walk-forward rather than
    a subsample of one. Longer windows start later only through ``min_train_dates``,
    which is held fixed, so the rows cover the same test months.
    """
    rows = {}
    for window in windows:
        scores = fold_scores(feature_matrix, target, rebalance_dates, model_kind,
                             min_train_dates, window)
        rows["expanding" if window is None else f"{window}m"] = pooled_metrics(scores)
    return pd.DataFrame(rows).T


def prediction_stability(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    model_kind: str = "xgb",
    min_train_dates: int = 6,
    train_window: Optional[int] = S.TRAIN_WINDOW_MONTHS,
) -> Dict[str, float]:
    """How much of the model's ordering survives from one month to the next.

    For every adjacent pair of rebalances the predictions are restricted to the
    stocks scored in both months and rank-correlated. A model tracking a persistent
    signal repeats itself; one reading noise does not. The same statistic is computed
    for the realised target, which bounds how persistent any honest model could be.
    """
    preds = H.walk_forward_predictions(feature_matrix, target, rebalance_dates,
                                       model_kind, min_train_dates, train_window)
    pred_rho: List[float] = []
    target_rho: List[float] = []
    for previous, current in zip(preds, preds[1:]):
        shared = previous.scores.index.intersection(current.scores.index)
        if len(shared) < 3:
            continue
        pred_rho.append(_spearman(previous.scores.reindex(shared).to_numpy(),
                                  current.scores.reindex(shared).to_numpy()))
        y_prev = target.reindex(
            pd.MultiIndex.from_product([[previous.date], shared],
                                       names=feature_matrix.index.names)).to_numpy()
        y_curr = target.reindex(
            pd.MultiIndex.from_product([[current.date], shared],
                                       names=feature_matrix.index.names)).to_numpy()
        frame = pd.DataFrame({"a": y_prev, "b": y_curr}).dropna()
        if len(frame) >= 3:
            target_rho.append(_spearman(frame["a"].to_numpy(), frame["b"].to_numpy()))

    def summarise(values: List[float]) -> float:
        clean = [v for v in values if np.isfinite(v)]
        return float(np.mean(clean)) if clean else float("nan")

    return {
        "prediction_rho": summarise(pred_rho),
        "target_rho": summarise(target_rho),
        "pairs": float(len([v for v in pred_rho if np.isfinite(v)])),
    }


def error_analysis(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    volatility: Optional[pd.Series] = None,
    model_kind: str = "xgb",
    min_train_dates: int = 6,
    train_window: Optional[int] = S.TRAIN_WINDOW_MONTHS,
) -> Dict[str, pd.DataFrame]:
    """Out-of-sample R² and IC broken down by year and by volatility bucket.

    A pooled number can hide a model that works in one regime and not another, and
    the volatility split is the one that matters here: if the model's errors are
    concentrated in the names it actually buys, the pooled figure understates the
    problem. ``volatility`` is the ex-ante series used by the portfolio run; without
    it only the yearly table is returned.
    """
    frames: List[pd.DataFrame] = []
    for fold in H.walk_forward_folds(feature_matrix, target, rebalance_dates,
                                     model_kind, min_train_dates, train_window):
        y_tr = target.reindex(fold.train.index).dropna()
        if y_tr.empty:
            continue
        pred = fold.model.predict_scores(fold.test)
        frames.append(pd.DataFrame({
            "date": fold.date,
            "ticker": fold.test.index,
            "y": _target_at(target, fold.date, fold.test.index).to_numpy(),
            "pred": pred.to_numpy(),
            "baseline": float(y_tr.mean()),
        }))
    if not frames:
        empty = pd.DataFrame(columns=["r2_oos", "ic", "n"])
        return {"by_year": empty, "by_volatility": empty}

    panel = pd.concat(frames, ignore_index=True).dropna(subset=["y", "pred"])

    def group_metrics(frame: pd.DataFrame) -> pd.Series:
        y = frame["y"].to_numpy(dtype="float64")
        return pd.Series({
            "r2_oos": _r2(y, frame["pred"].to_numpy(dtype="float64"),
                          float(frame["baseline"].mean())),
            "ic": _spearman(y, frame["pred"].to_numpy(dtype="float64")),
            "n": float(len(frame)),
        })

    out = {"by_year": panel.groupby(panel["date"].dt.year).apply(
        group_metrics, include_groups=False)}

    if volatility is not None:
        index = pd.MultiIndex.from_arrays([panel["date"], panel["ticker"]],
                                          names=feature_matrix.index.names)
        panel = panel.assign(vol=volatility.reindex(index).to_numpy())
        ranked = panel.dropna(subset=["vol"])
        if not ranked.empty:
            bucket = ranked.groupby("date")["vol"].transform(
                lambda s: pd.qcut(s.rank(method="first"), min(VOL_BUCKETS, max(1, s.nunique())),
                                  labels=False, duplicates="drop"))
            ranked = ranked.assign(bucket=bucket).dropna(subset=["bucket"])
            out["by_volatility"] = ranked.groupby("bucket").apply(
                group_metrics, include_groups=False)
        else:
            out["by_volatility"] = pd.DataFrame(columns=["r2_oos", "ic", "n"])
    return out

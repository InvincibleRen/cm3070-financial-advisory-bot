"""Ranking model + baseline (Direction 1 core, Phase 2).

A gradient-boosting classifier predicts ``P(stock beats the universe median)``.
A logistic-regression baseline is kept for comparison so the evaluation can show
the ML model earns its complexity against a simple linear benchmark.

Design notes
------------
* **Model-agnostic surface.** The walk-forward selector in ``select.py`` only
  ever calls ``fit`` and ``predict_scores``; swapping ``kind`` between ``"gbm"``
  and ``"logistic"`` (or adding a new estimator) changes nothing downstream.
* **Ranking, not thresholding.** ``predict_scores`` returns a continuous
  probability, index-aligned to the input rows, so the selector can rank the
  cross-section and take the top-N. We never hard-classify.
* **Missing-value policy.** Features carry NaNs by design (a momentum window that
  has not filled yet, a stock with no filing so far). Gradient boosting handles
  NaNs natively; the logistic pipeline imputes (median) then standardises, since
  a linear model cannot consume NaNs and is scale-sensitive.
* **Optional dependency.** scikit-learn is imported lazily inside ``fit`` so the
  rest of the project still imports without it installed.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

_VALID_KINDS = ("gbm", "logistic", "rf", "xgb")

_GBM_EXCLUDE = ("volatility",)


@contextlib.contextmanager
def _silence_empty_feature_warning():
    """Suppress the one benign imputer warning about all-NaN feature columns.

    In early walk-forward folds an entire fundamental column can be all-NaN
    (yfinance has no filing that far back). scikit-learn's ``SimpleImputer`` emits
    "Skipping features without any observed values" during *both* fit and
    transform; over a multi-year monthly backtest that is hundreds of identical
    lines. It carries no information the run cares about, so we silence exactly
    that message (nothing else) around estimator calls.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        yield


@dataclass
class RankerModel:
    """Thin, model-agnostic wrapper around a scikit-learn classifier.

    Parameters
    ----------
    kind:
        ``"gbm"`` -> ``HistGradientBoostingClassifier`` (native NaN support);
        ``"logistic"`` -> median-impute + standardise + ``LogisticRegression``;
        ``"rf"`` -> median-impute + ``RandomForestClassifier`` (bagged trees);
        ``"xgb"`` -> ``XGBClassifier`` with row/column subsampling (native NaN).

        All four expose the same ``fit``/``predict_scores`` surface, so the
        walk-forward selector is indifferent to which is used. They are compared
        on identical folds, selecting on out-of-sample stability rather than on
        the best full-period return.
    random_state:
        Seed passed to estimators that accept one, for reproducible folds.
    """

    kind: str = "gbm"
    random_state: int = 0
    _estimator: Optional[object] = field(default=None, repr=False)
    _feature_names: Optional[List[str]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")


    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankerModel":
        """Fit the chosen estimator on ``(features, binary label)``.

        Rows whose label is missing are dropped. If, after cleaning, only a
        single class remains (common in the earliest, tiny walk-forward folds),
        no estimator is fitted - ``predict_scores`` then returns that class as a
        constant, which is the honest degenerate behaviour.
        """
        X, y = self._align_xy(X, y)
        if self.kind == "gbm":
            X = X.drop(columns=[c for c in _GBM_EXCLUDE if c in X.columns])
        self._feature_names = list(X.columns)

        classes = pd.unique(y)
        if len(classes) < 2:
            self._estimator = _ConstantClassifier(int(classes[0]) if len(classes) else 0)
            return self

        self._estimator = self._build_estimator(len(X))
        with _silence_empty_feature_warning():
            self._estimator.fit(X.to_numpy(dtype="float64"), y.to_numpy(dtype="int64"))
        return self

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """Return ``P(label = 1)`` per row, as a Series aligned to ``X.index``."""
        if self._estimator is None:
            raise RuntimeError("RankerModel.predict_scores called before fit().")

        X = X.reindex(columns=self._feature_names)
        with _silence_empty_feature_warning():
            proba = self._estimator.predict_proba(X.to_numpy(dtype="float64"))
        classes = list(self._estimator.classes_)
        if 1 in classes:
            scores = proba[:, classes.index(1)]
        else:
            scores = np.zeros(len(X), dtype="float64")
        return pd.Series(scores, index=X.index, name="score")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row, per-feature contributions to the score (TreeSHAP), plus ``bias``.

        Only the XGBoost ranker supports this; its booster computes exact tree SHAP
        values natively. Contributions are in log-odds, so they explain the *ranking*
        rather than adding up to the probability. A degenerate (single-class) fold
        contributes nothing.
        """
        if self._feature_names is None:
            raise RuntimeError("RankerModel.contributions called before fit().")
        if self.kind != "xgb":
            raise NotImplementedError("contributions are only available for kind='xgb'.")
        columns = list(self._feature_names) + ["bias"]
        if isinstance(self._estimator, _ConstantClassifier):
            return pd.DataFrame(0.0, index=X.index, columns=columns)
        import xgboost

        X = X.reindex(columns=self._feature_names)
        dmat = xgboost.DMatrix(X.to_numpy(dtype="float64"), missing=np.nan)
        contrib = self._estimator.get_booster().predict(dmat, pred_contribs=True)
        return pd.DataFrame(contrib, index=X.index, columns=columns)


    def _build_estimator(self, n_samples: int = 0):
        """Construct the underlying scikit-learn estimator (imported lazily)."""
        if self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier

            min_samples_leaf = min(60, max(20, n_samples // 25))

            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=3,
                max_iter=300,
                l2_regularization=1.0,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
            )

        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            return Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(
                        n_estimators=300, max_depth=6,
                        min_samples_leaf=min(40, max(5, n_samples // 25)),
                        n_jobs=1, random_state=self.random_state,
                    )),
                ]
            )

        if self.kind == "xgb":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=1, random_state=self.random_state,
                eval_metric="logloss", tree_method="hist",
            )

        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ]
        )


    @staticmethod
    def _align_xy(X: pd.DataFrame, y: pd.Series):
        """Align X and y on their shared index and drop rows with a missing label."""
        if X.empty:
            raise ValueError("Cannot fit RankerModel on an empty feature matrix.")
        y = y.reindex(X.index)
        keep = y.notna()
        return X.loc[keep], y.loc[keep].astype("int64")


class _ConstantClassifier:
    """Minimal stand-in used for single-class folds.

    Mimics the small slice of the scikit-learn API that ``predict_scores`` needs
    (``classes_`` and ``predict_proba``) so degenerate folds need no special-case
    branching upstream.
    """

    def __init__(self, constant_class: int) -> None:
        self.classes_ = np.array([constant_class])

    def predict_proba(self, X) -> np.ndarray:
        return np.ones((len(X), 1), dtype="float64")



_VALID_RETURN_KINDS = ("xgb", "rf", "ridge")


@dataclass
class ReturnModel:
    """Regressor for the continuous target of ``labels.make_excess_return_targets``.

    Where :class:`RankerModel` predicts ``P(beat the median)`` and so only knows
    *whether* a stock won, this model predicts *how much* risk-adjusted excess
    return it earns, which is what a hurdle ("only buy when the expected payoff is
    positive") needs.

    Parameters
    ----------
    kind:
        ``"xgb"`` -> ``XGBRegressor`` (primary; native NaN support);
        ``"rf"`` -> median-impute + ``RandomForestRegressor`` (primary);
        ``"ridge"`` -> median-impute + standardise + ``Ridge`` (auxiliary linear
        baseline, the regression counterpart of the logistic classifier).

        Tree settings mirror the classifier versions so the two targets are
        compared on the same model capacity.
    """

    kind: str = "xgb"
    random_state: int = 0
    _estimator: Optional[object] = field(default=None, repr=False)
    _feature_names: Optional[List[str]] = field(default=None, repr=False)
    _constant: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_RETURN_KINDS:
            raise ValueError(f"kind must be one of {_VALID_RETURN_KINDS}, got {self.kind!r}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReturnModel":
        """Fit on ``(features, continuous target)``; rows with a missing target are dropped.

        With fewer than two usable rows no estimator is fitted and the model
        predicts the training mean (0 if there is none).
        """
        if X.empty:
            raise ValueError("Cannot fit ReturnModel on an empty feature matrix.")
        y = y.reindex(X.index)
        keep = y.notna()
        X, y = X.loc[keep], y.loc[keep].astype("float64")
        self._feature_names = list(X.columns)
        if len(y) < 2:
            self._estimator = None
            self._constant = float(y.mean()) if len(y) else 0.0
            return self

        self._constant = None
        self._estimator = self._build_estimator(len(X))
        with _silence_empty_feature_warning():
            self._estimator.fit(X.to_numpy(dtype="float64"), y.to_numpy(dtype="float64"))
        return self

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """Predicted target per row, as a Series aligned to ``X.index``."""
        if self._feature_names is None:
            raise RuntimeError("ReturnModel.predict_scores called before fit().")
        if self._estimator is None:
            return pd.Series(self._constant, index=X.index, name="score", dtype="float64")
        X = X.reindex(columns=self._feature_names)
        with _silence_empty_feature_warning():
            pred = self._estimator.predict(X.to_numpy(dtype="float64"))
        return pd.Series(np.asarray(pred, dtype="float64"), index=X.index, name="score")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row, per-feature contributions to the prediction (TreeSHAP), plus ``bias``.

        Each row sums exactly to :meth:`predict_scores`. Only the XGBoost model
        supports this (its booster computes exact tree SHAP values natively). A
        degenerate fit contributes nothing and puts its constant in ``bias``.
        """
        if self._feature_names is None:
            raise RuntimeError("ReturnModel.contributions called before fit().")
        if self.kind != "xgb":
            raise NotImplementedError("contributions are only available for kind='xgb'.")
        columns = list(self._feature_names) + ["bias"]
        if self._estimator is None:
            out = pd.DataFrame(0.0, index=X.index, columns=columns)
            out["bias"] = self._constant
            return out
        import xgboost

        X = X.reindex(columns=self._feature_names)
        dmat = xgboost.DMatrix(X.to_numpy(dtype="float64"), missing=np.nan)
        contrib = self._estimator.get_booster().predict(dmat, pred_contribs=True)
        return pd.DataFrame(contrib, index=X.index, columns=columns)

    def _build_estimator(self, n_samples: int = 0):
        if self.kind == "xgb":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=1, random_state=self.random_state, tree_method="hist",
            )

        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        if self.kind == "rf":
            from sklearn.ensemble import RandomForestRegressor

            return Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("reg", RandomForestRegressor(
                    n_estimators=300, max_depth=6,
                    min_samples_leaf=min(40, max(5, n_samples // 25)),
                    n_jobs=1, random_state=self.random_state,
                )),
            ])

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        return Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])

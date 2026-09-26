"""Hurdle selection: buy only the stocks whose predicted payoff clears a bar.

The top-N selector in ``select.py`` always holds ``N`` names, so its hit rate is
diluted by however many it is forced to buy. This module instead asks the return
model (``model.ReturnModel``) for each stock's expected risk-adjusted excess return
and buys only the names above a fixed hurdle, capped at ``max_names`` and
equal-weighted among themselves. A month in which nothing clears the bar holds a
fallback asset instead:

* ``"spy"``   - SPY. Months without a pick then earn exactly zero active return, so
  every bit of out/under-performance is attributable to the stock picks.
* ``"bil"``   - short-term Treasury bills (BIL), the lowest-risk option.
* ``"trend"`` - SPY when SPY's month-end close is above its 10-month average,
  otherwise BIL. This is a separate market-timing rule, so it is reported as its
  own variant rather than mixed into the evaluation of the stock picks.

Leakage safety: the hurdle is fixed in advance (not tuned on the results), the
walk-forward trains only on dates whose target has resolved, and the trend rule
reads SPY closes on or before the rebalance date only.

The evaluation judges the picks the way a trader would: win rate against SPY,
payoff ratio (average win / average loss, the realised risk-reward), and
expectancy (mean excess return per pick), each set against the same-month
universe so a strong market month is not mistaken for skill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common import metrics
from src.selection import select as S
from src.selection.model import ReturnModel

PERIODS_PER_YEAR = S.PERIODS_PER_YEAR
BENCHMARK_TICKER = S.BENCHMARK_TICKER
RISK_FREE_TICKER = "BIL"
DEFAULT_HURDLE = 0.0
DEFAULT_MAX_NAMES = 5
FALLBACKS = ("spy", "bil", "trend")
TREND_MONTHS = 10
STRICTNESS_LEVELS = (1, 3, 5, 10, 20, 50)
MATCH_NEIGHBOURS = 20
NEWEY_WEST_LAGS = 3

Picker = Callable[[pd.Series], List[str]]



@dataclass
class Prediction:
    """Out-of-sample scores for one rebalance date's cross-section."""

    date: pd.Timestamp
    scores: pd.Series


@dataclass
class Fold:
    """One walk-forward fold: the fitted model and both sides of the split.

    ``train`` is every row the model saw, ``test`` the cross-section it then scored.
    Machine-learning diagnostics (training loss against test loss, learning curves)
    need the training side, which :class:`Prediction` alone does not carry.
    """

    date: pd.Timestamp
    model: ReturnModel
    train: pd.DataFrame
    test: pd.DataFrame


def walk_forward_folds(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    model_kind: str = "xgb",
    min_train_dates: int = 6,
    train_window: Optional[int] = S.TRAIN_WINDOW_MONTHS,
    horizon_months: int = 1,
):
    """Yield each :class:`Fold` of the walk-forward, in date order.

    The fold rules live here and :func:`walk_forward_predictions` consumes them, so
    the portfolio run and the machine-learning diagnostics can never drift apart.
    """
    dates = S._usable_dates(feature_matrix, rebalance_dates)
    date_level = feature_matrix.index.get_level_values("rebalance_date")
    for i, rebalance_date in enumerate(dates):
        if i < min_train_dates:
            continue
        start = 0 if train_window is None else max(0, i - train_window)
        window = dates[start:i]
        if horizon_months > 1:
            window = [d for d in window
                      if d + pd.DateOffset(months=horizon_months) <= rebalance_date]
        if not window:
            continue
        X_train = feature_matrix[date_level.isin(window)]
        if X_train.empty:
            continue
        model = ReturnModel(kind=model_kind).fit(X_train, target)
        yield Fold(rebalance_date, model, X_train, feature_matrix.loc[rebalance_date])


def walk_forward_predictions(
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    model_kind: str = "xgb",
    min_train_dates: int = 6,
    train_window: Optional[int] = S.TRAIN_WINDOW_MONTHS,
    horizon_months: int = 1,
    on_fold: Optional[Callable[[pd.Timestamp, ReturnModel, pd.DataFrame], None]] = None,
) -> List[Prediction]:
    """Train a :class:`ReturnModel` on strictly-past dates and score each cross-section.

    Mirrors the fold rules of ``select._walk_forward_records`` (rolling window,
    ``min_train_dates`` warm-up, embargo when the horizon exceeds one month) so the
    classifier and the return model are compared on identical folds. ``on_fold``,
    if given, is called with ``(date, fitted model, features scored that date)`` for
    each fold, e.g. to explain the out-of-sample predictions.
    """
    out: List[Prediction] = []
    for fold in walk_forward_folds(feature_matrix, target, rebalance_dates, model_kind,
                                   min_train_dates, train_window, horizon_months):
        out.append(Prediction(fold.date, fold.model.predict_scores(fold.test)))
        if on_fold is not None:
            on_fold(fold.date, fold.model, fold.test)
    return out



def hurdle_picker(hurdle: float = DEFAULT_HURDLE, max_names: int = DEFAULT_MAX_NAMES) -> Picker:
    """Names whose score is strictly above ``hurdle``, best first, at most ``max_names``."""
    def pick(scores: pd.Series) -> List[str]:
        s = scores.dropna()
        return list(s[s > hurdle].sort_values(ascending=False).head(max_names).index)
    return pick


def top_k_picker(k: int) -> Picker:
    """The ``k`` highest-scoring names, with no hurdle."""
    def pick(scores: pd.Series) -> List[str]:
        return list(scores.dropna().sort_values(ascending=False).head(k).index)
    return pick


def picks_by_date(predictions: Sequence[Prediction], picker: Picker) -> Dict[pd.Timestamp, List[str]]:
    return {p.date: picker(p.scores) for p in predictions}



def holding_periods(predictions: Sequence[Prediction]) -> List[tuple]:
    """``(entry, exit)`` pairs: each rebalance to the next. The last date has no exit."""
    dates = [p.date for p in predictions]
    return list(zip(dates[:-1], dates[1:]))


def asset_period_returns(
    close: Optional[pd.Series], periods: Sequence[tuple], execution_lag: int = 0
) -> pd.Series:
    """One asset's return over each holding period (NaN where unpriced)."""
    out: Dict[pd.Timestamp, float] = {}
    for entry, exit_ in periods:
        r = None if close is None else S._period_return(close, entry, exit_, execution_lag)
        out[entry] = float("nan") if r is None else r
    return pd.Series(out, dtype="float64")


def realised_excess_by_date(
    predictions: Sequence[Prediction],
    closes: Dict[str, pd.Series],
    execution_lag: int = 0,
    betas: Optional[pd.Series] = None,
) -> Dict[str, Dict[pd.Timestamp, pd.Series]]:
    """Per entry date, each scored ticker's holding-period excess return, two ways.

    * ``"raw"`` - return minus SPY's. This is what a trader sees, but a high-beta
      name "wins" it in any rising month without any skill.
    * ``"beta_adjusted"`` - return minus ``beta_t x`` SPY's, with ``beta_t`` the
      ex-ante beta at the entry date (``betas``, indexed ``(date, ticker)``). Names
      without a beta are left out of this view only. Without ``betas`` it equals raw.
    """
    if BENCHMARK_TICKER not in closes:
        raise ValueError(f"{BENCHMARK_TICKER} prices are required to measure excess returns.")
    periods = holding_periods(predictions)
    spy = asset_period_returns(closes[BENCHMARK_TICKER], periods, execution_lag)
    scored = {p.date: p.scores.index for p in predictions}
    beta_by_date = _by_date(betas)
    raw_out: Dict[pd.Timestamp, pd.Series] = {}
    adj_out: Dict[pd.Timestamp, pd.Series] = {}
    for entry, exit_ in periods:
        spy_ret = spy.get(entry)
        if spy_ret is None or not np.isfinite(spy_ret):
            continue
        returns: Dict[str, float] = {}
        for ticker in scored[entry]:
            if ticker not in closes:
                continue
            r = S._period_return(closes[ticker], entry, exit_, execution_lag)
            if r is not None:
                returns[ticker] = r
        if not returns:
            continue
        r = pd.Series(returns, dtype="float64")
        raw_out[entry] = r - spy_ret
        if betas is None:
            adj_out[entry] = raw_out[entry]
        else:
            b = beta_by_date.get(entry, pd.Series(dtype="float64")).reindex(r.index)
            adj = (r - b * spy_ret).dropna()
            if not adj.empty:
                adj_out[entry] = adj
    return {"raw": raw_out, "beta_adjusted": adj_out}


def volatility_ranks(
    predictions: Sequence[Prediction], vols: Optional[pd.Series]
) -> Optional[Dict[pd.Timestamp, pd.Series]]:
    """Per date, each scored ticker's ex-ante volatility percentile (0 = calmest, 1 = wildest).

    Drives the risk-matched null: a random stand-in for a pick is drawn from the
    names closest to it in volatility that same month, so a model that merely
    favours volatile names gains nothing over it. Names with no volatility
    estimate are NaN.
    """
    if vols is None:
        return None
    vol_by_date = _by_date(vols)
    return {
        p.date: vol_by_date.get(p.date, pd.Series(dtype="float64"))
        .reindex(p.scores.index).rank(pct=True)
        for p in predictions
    }


def _by_date(values: Optional[pd.Series]) -> Dict[pd.Timestamp, pd.Series]:
    if values is None or values.empty:
        return {}
    return {d: g.droplevel(0) for d, g in values.groupby(level=0)}



def trade_statistics(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Win rate, average win/loss, payoff ratio and expectancy of excess returns.

    A "win" is an excess return above zero. ``payoff_ratio`` is the average win
    divided by the average loss (the realised risk-reward), and
    ``breakeven_win_rate = 1 / (1 + payoff_ratio)`` is the win rate at which that
    payoff ratio would exactly break even. ``expectancy`` is the (weighted) mean
    excess return, i.e. ``win_rate * avg_win - (1 - win_rate) * avg_loss``.
    """
    v = np.asarray(values, dtype="float64")
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype="float64")
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    nan = float("nan")
    if v.size == 0:
        return {"n": 0.0, "win_rate": nan, "avg_win": nan, "avg_loss": nan,
                "payoff_ratio": nan, "breakeven_win_rate": nan, "expectancy": nan}
    win = v > 0
    total = w.sum()
    avg_win = float(np.average(v[win], weights=w[win])) if win.any() else nan
    avg_loss = float(np.average(-v[~win], weights=w[~win])) if (~win).any() else nan
    payoff = avg_win / avg_loss if (np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss > 0) else nan
    return {
        "n": float(v.size),
        "win_rate": float(w[win].sum() / total),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "breakeven_win_rate": 1.0 / (1.0 + payoff) if np.isfinite(payoff) else nan,
        "expectancy": float(np.average(v, weights=w)),
    }


def pick_statistics(
    picks: Dict[pd.Timestamp, List[str]],
    excess: Dict[pd.Timestamp, pd.Series],
    n_draws: int = 2000,
    seed: int = 0,
    vol_ranks: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    n_neighbours: int = MATCH_NEIGHBOURS,
) -> Dict[str, Dict[str, float]]:
    """Trade statistics of the picks, of the same-month universe, and permutation tests.

    The universe baseline weights each month by the number of picks made in it, so
    it answers "what would the same number of trades, spread over the whole
    cross-section of those same months, have produced?". A month with no picks
    contributes nothing to either side.

    Two nulls replace each month's picks with random names from that month:
    ``p_expectancy`` / ``p_win_rate`` draw from the whole cross-section;
    ``*_matched`` draw each pick's stand-in from the ``n_neighbours`` names nearest
    to it in volatility that month (``vol_ranks``), which removes the credit for
    simply holding volatile stocks. Each p-value is the one-sided share of draws
    that match or beat the picks. ``pick_vol_percentile`` is the picks' average
    volatility percentile (0.5 = no volatility tilt).
    """
    pick_vals: List[float] = []
    uni_vals: List[np.ndarray] = []
    uni_wts: List[np.ndarray] = []
    whole: List[tuple] = []
    matched: List[tuple] = []
    pick_pcts: List[float] = []
    for date, names in picks.items():
        month = excess.get(date)
        if month is None or not names:
            continue
        chosen = month.reindex(names).dropna()
        if chosen.empty:
            continue
        k = len(chosen)
        arr = month.to_numpy(dtype="float64")
        pick_vals.extend(chosen.to_numpy())
        uni_vals.append(arr)
        uni_wts.append(np.full(arr.size, k / arr.size))
        whole.append((arr, [(np.arange(arr.size), k)]))
        if vol_ranks is not None:
            ranks = vol_ranks.get(date, pd.Series(dtype="float64")).reindex(month.index).to_numpy()
            known = np.flatnonzero(np.isfinite(ranks))
            unknown = np.flatnonzero(~np.isfinite(ranks))
            groups = []
            for ticker in chosen.index:
                r = ranks[month.index.get_loc(ticker)]
                if np.isfinite(r) and known.size:
                    pick_pcts.append(float(r))
                    nearest = known[np.argsort(np.abs(ranks[known] - r), kind="stable")[:n_neighbours]]
                    groups.append((nearest, 1))
                else:
                    groups.append((unknown if unknown.size else np.arange(arr.size), 1))
            matched.append((arr, groups))

    picks_stats = trade_statistics(np.asarray(pick_vals))
    universe_stats = (
        trade_statistics(np.concatenate(uni_vals), np.concatenate(uni_wts))
        if uni_vals else trade_statistics(np.array([]))
    )
    null = _random_pick_null(whole, n_draws, seed)
    null_m = _random_pick_null(matched, n_draws, seed) if vol_ranks is not None else None
    nan = float("nan")
    test = {
        "p_expectancy": _upper_p(picks_stats["expectancy"], null["expectancy"]),
        "p_win_rate": _upper_p(picks_stats["win_rate"], null["win_rate"]),
        "p_expectancy_matched": (_upper_p(picks_stats["expectancy"], null_m["expectancy"])
                                 if null_m else nan),
        "p_win_rate_matched": (_upper_p(picks_stats["win_rate"], null_m["win_rate"])
                               if null_m else nan),
        "matched_null_expectancy": float(null_m["expectancy"].mean()) if null_m and null_m["expectancy"].size else nan,
        "pick_vol_percentile": float(np.mean(pick_pcts)) if pick_pcts else nan,
        "n_months": float(len(whole)),
        "n_draws": float(null["expectancy"].size),
    }
    return {"picks": picks_stats, "universe": universe_stats, "test": test}


def _random_pick_null(months: List[tuple], n_draws: int, seed: int) -> Dict[str, np.ndarray]:
    """Expectancy and win rate of ``n_draws`` random books with the same shape as the picks."""
    rng = np.random.default_rng(seed)
    total = sum(c for _, groups in months for _, c in groups)
    if not months or total == 0:
        return {"expectancy": np.array([]), "win_rate": np.array([])}
    exp = np.empty(n_draws)
    wins = np.empty(n_draws)
    for d in range(n_draws):
        s = 0.0
        w = 0
        for arr, groups in months:
            for pool, count in groups:
                idx = rng.choice(pool, size=count, replace=count > pool.size)
                draw = arr[idx]
                s += draw.sum()
                w += int((draw > 0).sum())
        exp[d] = s / total
        wins[d] = w / total
    return {"expectancy": exp, "win_rate": wins}


def _upper_p(real: float, null: np.ndarray) -> float:
    if null.size == 0 or not np.isfinite(real):
        return float("nan")
    return float(((null >= real).sum() + 1.0) / (null.size + 1.0))


def strictness_curve(
    predictions: Sequence[Prediction],
    excess: Dict[pd.Timestamp, pd.Series],
    levels: Sequence[int] = STRICTNESS_LEVELS,
    n_draws: int = 500,
    seed: int = 0,
    vol_ranks: Optional[Dict[pd.Timestamp, pd.Series]] = None,
) -> pd.DataFrame:
    """Pick statistics for the top-k names each month, k from strict to loose.

    A model with real skill shows win rate, payoff ratio and expectancy that rise
    as k shrinks; a flat curve means the top of the ranking is no better than the
    rest.
    """
    rows = {}
    for k in levels:
        st = pick_statistics(picks_by_date(predictions, top_k_picker(k)), excess, n_draws, seed, vol_ranks)
        rows[k] = _curve_row(st)
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "top_k"
    return frame


def _curve_row(st: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    p, u, t = st["picks"], st["universe"], st["test"]
    return {
        "n_picks": p["n"],
        "win_rate": p["win_rate"], "universe_win_rate": u["win_rate"],
        "payoff_ratio": p["payoff_ratio"], "universe_payoff_ratio": u["payoff_ratio"],
        "expectancy": p["expectancy"], "universe_expectancy": u["expectancy"],
        "p_expectancy": t["p_expectancy"],
        "p_expectancy_matched": t["p_expectancy_matched"],
        "pick_vol_percentile": t["pick_vol_percentile"],
    }


def information_coefficient(
    predictions: Sequence[Prediction], realised: pd.Series
) -> Dict[str, float]:
    """Mean per-date Spearman correlation of scores with the realised target, and its t-stat."""
    ics: List[float] = []
    for p in predictions:
        try:
            actual = realised.loc[p.date]
        except KeyError:
            continue
        aligned = pd.concat([p.scores, actual], axis=1, keys=["s", "r"]).dropna()
        if len(aligned) >= 3 and aligned["s"].nunique() > 1 and aligned["r"].nunique() > 1:
            ics.append(float(aligned["s"].corr(aligned["r"], method="spearman")))
    nan = float("nan")
    if len(ics) < 2:
        return {"mean_ic": ics[0] if ics else nan, "ic_t_stat": nan, "n_periods": float(len(ics))}
    arr = np.asarray(ics)
    sd = float(arr.std(ddof=1))
    return {
        "mean_ic": float(arr.mean()),
        "ic_t_stat": float(arr.mean() / sd * math.sqrt(arr.size)) if sd > 0 else nan,
        "n_periods": float(arr.size),
    }



def trend_risk_on(spy_close: pd.Series, when: pd.Timestamp, months: int = TREND_MONTHS) -> bool:
    """True when SPY's latest month-end close (on or before ``when``) is above its ``months``-month mean.

    Uses only closes on or before ``when``. With too little history it defaults to
    risk-on (hold SPY), the neutral choice for this project's benchmark.
    """
    history = spy_close[spy_close.index <= when]
    monthly = history.resample("ME").last().dropna()
    if len(monthly) < months:
        return True
    return bool(monthly.iloc[-1] > monthly.iloc[-months:].mean())


def fallback_asset(fallback: str, when: pd.Timestamp, closes: Dict[str, pd.Series]) -> str:
    if fallback == "spy":
        return BENCHMARK_TICKER
    if fallback == "bil":
        return RISK_FREE_TICKER
    if fallback == "trend":
        return BENCHMARK_TICKER if trend_risk_on(closes[BENCHMARK_TICKER], when) else RISK_FREE_TICKER
    raise ValueError(f"fallback must be one of {FALLBACKS}, got {fallback!r}")


def build_weights(
    picks: Dict[pd.Timestamp, List[str]],
    fallback: str,
    closes: Dict[str, pd.Series],
) -> pd.DataFrame:
    """``(date x ticker)`` weights: picks equal-weighted, or 100% in the fallback asset."""
    cells: Dict[tuple, float] = {}
    for date, names in picks.items():
        if names:
            for ticker in names:
                cells[(date, ticker)] = 1.0 / len(names)
        else:
            asset = fallback_asset(fallback, date, closes)
            if asset not in closes:
                raise ValueError(f"Fallback '{fallback}' needs {asset} prices, which were not loaded.")
            cells[(date, asset)] = 1.0
    return S._weights_frame(cells)


def portfolio_returns(
    weights: pd.DataFrame,
    closes: Dict[str, pd.Series],
    cost_per_turnover: float = S.DEFAULT_COST_PER_TURNOVER,
    execution_lag: int = 0,
) -> Dict[str, pd.Series]:
    """Net-of-cost holding-period returns and the turnover charged at each rebalance."""
    turnover = S._turnover_schedule(weights)
    dates = list(weights.index)
    net: Dict[pd.Timestamp, float] = {}
    charged: Dict[pd.Timestamp, float] = {}
    for entry, exit_ in zip(dates[:-1], dates[1:]):
        gross = S._weighted_period_return(weights.loc[entry], closes, entry, exit_, execution_lag)
        if gross is None:
            continue
        charged[entry] = float(turnover.loc[entry])
        net[entry] = gross - charged[entry] * cost_per_turnover
    return {
        "returns": pd.Series(net, dtype="float64").sort_index(),
        "turnover": pd.Series(charged, dtype="float64").sort_index(),
    }



def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int = NEWEY_WEST_LAGS):
    """OLS coefficients, Newey-West (Bartlett) standard errors and R^2. ``X`` includes the constant."""
    y = np.asarray(y, dtype="float64")
    X = np.asarray(X, dtype="float64")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    xe = X * resid[:, None]
    meat = xe.T @ xe
    for lag in range(1, min(lags, len(y) - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = xe[lag:].T @ xe[:-lag]
        meat += weight * (gamma + gamma.T)
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2


def return_decomposition(
    strategy: pd.Series,
    market: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Dict[str, float]:
    """Why a book whose picks average a positive excess can still compound below the index.

    The pick statistics report an *arithmetic* mean excess per pick. An investor earns the
    *geometric* (compounded) return, and the two differ by roughly half the variance: for
    small returns ``geometric ~= arithmetic - sigma^2 / 2``. A five-name book is far more
    volatile than the index, so it loses more to that term, and the gap grows with
    concentration rather than with anything the model did wrong.

    Returns, all annualised: the arithmetic and geometric excess over ``market``, the
    variance-drag difference that separates them, and the residual the approximation does
    not account for (skew, fat tails, and the error in the small-return expansion). A large
    residual means the decomposition should not be leaned on.
    """
    frame = pd.DataFrame({"s": strategy, "m": market}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return {k: float("nan") for k in
                ("arithmetic_excess", "geometric_excess", "variance_drag", "residual",
                 "vol_strategy", "vol_market", "periods")}
    s, m = frame["s"].to_numpy(), frame["m"].to_numpy()
    arithmetic = float(np.mean(s) - np.mean(m)) * periods_per_year
    geometric = float((np.prod(1.0 + s) ** (periods_per_year / len(s)))
                      - (np.prod(1.0 + m) ** (periods_per_year / len(m))))
    drag = float(np.var(s, ddof=1) - np.var(m, ddof=1)) / 2.0 * periods_per_year
    return {
        "arithmetic_excess": arithmetic,
        "geometric_excess": geometric,
        "variance_drag": drag,
        "residual": geometric - (arithmetic - drag),
        "vol_strategy": float(np.std(s, ddof=1)) * math.sqrt(periods_per_year),
        "vol_market": float(np.std(m, ddof=1)) * math.sqrt(periods_per_year),
        "periods": float(len(s)),
    }


def capm(
    strategy: pd.Series,
    market: pd.Series,
    risk_free: Optional[pd.Series] = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    lags: int = NEWEY_WEST_LAGS,
) -> Dict[str, float]:
    """``strategy - rf = alpha + beta * (market - rf)``, with Newey-West t-stats.

    ``alpha_ann`` is the monthly intercept times 12. ``beta_t_vs_1`` tests whether
    the beta differs from the market's. Also reports the alpha on each half of the
    sample, since a full-sample alpha can come from a single regime.
    """
    rf = risk_free if risk_free is not None else pd.Series(0.0, index=strategy.index)
    df = pd.concat([strategy, market, rf], axis=1, keys=["s", "m", "rf"]).dropna()
    nan = float("nan")
    out = {"n": float(len(df))}
    if len(df) < 6:
        return {**out, "alpha_ann": nan, "alpha_t": nan, "beta": nan, "beta_t_vs_1": nan, "r2": nan}
    y = (df["s"] - df["rf"]).to_numpy()
    x = (df["m"] - df["rf"]).to_numpy()
    if np.var(x) == 0:
        return {**out, "alpha_ann": nan, "alpha_t": nan, "beta": nan, "beta_t_vs_1": nan, "r2": nan}
    coef, se, r2 = ols_newey_west(y, np.column_stack([np.ones(len(x)), x]), lags)
    out.update({
        "alpha_ann": float(coef[0] * periods_per_year),
        "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else nan,
        "beta": float(coef[1]),
        "beta_t_vs_1": float((coef[1] - 1.0) / se[1]) if se[1] > 0 else nan,
        "r2": float(r2),
    })
    half = len(df) // 2
    for name, part in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
        yy, xx = y[part], x[part]
        if len(yy) < 6:
            out[f"alpha_ann_{name}"] = out[f"alpha_t_{name}"] = nan
            continue
        c, s, _ = ols_newey_west(yy, np.column_stack([np.ones(len(xx)), xx]), lags)
        out[f"alpha_ann_{name}"] = float(c[0] * periods_per_year)
        out[f"alpha_t_{name}"] = float(c[0] / s[0]) if s[0] > 0 else nan
    return out



@dataclass
class PortfolioResult:
    fallback: str
    returns: pd.Series = field(repr=False)
    metrics: Dict[str, float] = field(default_factory=dict)
    capm: Dict[str, float] = field(default_factory=dict)
    avg_turnover: float = 0.0
    yearly: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    decomposition: Dict[str, float] = field(default_factory=dict)


@dataclass
class HurdleEvaluation:
    """Everything the evaluation needs for one model."""

    label: str
    picks: Dict[pd.Timestamp, List[str]] = field(repr=False)
    pick_stats: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    curve: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    ic: Dict[str, float] = field(default_factory=dict)
    months_invested: float = 0.0
    avg_names_when_invested: float = 0.0
    portfolios: Dict[str, PortfolioResult] = field(default_factory=dict)
    latest_date: Optional[pd.Timestamp] = None
    latest_picks: List[str] = field(default_factory=list)


def evaluate(
    label: str,
    predictions: Sequence[Prediction],
    picker: Picker,
    closes: Dict[str, pd.Series],
    realised_target: pd.Series,
    risk: Optional[pd.DataFrame] = None,
    fallbacks: Sequence[str] = ("spy",),
    cost_per_turnover: float = S.DEFAULT_COST_PER_TURNOVER,
    execution_lag: int = 0,
    n_draws: int = 2000,
    curve_draws: int = 500,
    seed: int = 0,
) -> HurdleEvaluation:
    """Pick statistics, strictness curve, IC and one portfolio per fallback.

    ``risk`` is the ex-ante ``beta`` / ``vol`` frame (``labels.ex_ante_risk``). With
    it, the picks are also judged net of beta, against a volatility-matched null,
    and the strictness curve uses that stricter view. Without it both views
    coincide and the matched p-values are NaN.
    """
    predictions = sorted(predictions, key=lambda p: p.date)
    picks = picks_by_date(predictions, picker)
    betas = risk["beta"] if risk is not None else None
    excess = realised_excess_by_date(predictions, closes, execution_lag, betas)
    vol_ranks = volatility_ranks(predictions, risk["vol"] if risk is not None else None)
    periods = holding_periods(predictions)
    spy = asset_period_returns(closes.get(BENCHMARK_TICKER), periods, execution_lag)
    rf = (asset_period_returns(closes[RISK_FREE_TICKER], periods, execution_lag)
          if RISK_FREE_TICKER in closes else None)

    traded = [picks[d] for d, _ in periods]
    invested = [n for n in traded if n]
    result = HurdleEvaluation(
        label=label,
        picks=picks,
        pick_stats={view: pick_statistics(picks, excess[view], n_draws, seed, vol_ranks)
                    for view in ("raw", "beta_adjusted")},
        curve=strictness_curve(predictions, excess["beta_adjusted"], n_draws=curve_draws,
                               seed=seed, vol_ranks=vol_ranks),
        ic=information_coefficient(predictions, realised_target),
        months_invested=len(invested) / len(traded) if traded else float("nan"),
        avg_names_when_invested=float(np.mean([len(n) for n in invested])) if invested else 0.0,
    )
    if predictions:
        result.latest_date = predictions[-1].date
        result.latest_picks = picks[predictions[-1].date]

    for fb in fallbacks:
        weights = build_weights(picks, fb, closes)
        streams = portfolio_returns(weights, closes, cost_per_turnover, execution_lag)
        r = streams["returns"]
        result.portfolios[fb] = PortfolioResult(
            fallback=fb,
            returns=r,
            metrics=metrics.compute_metrics(r, 0.0, PERIODS_PER_YEAR),
            capm=capm(r, spy, rf),
            avg_turnover=float(streams["turnover"].mean()) if not streams["turnover"].empty else 0.0,
            yearly=S.yearly_breakdown(r, spy.dropna()),
            decomposition=return_decomposition(r, spy),
        )
    return result

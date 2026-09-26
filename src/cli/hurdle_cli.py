"""Hurdle-selection experiment: regress the excess-return target, buy only what clears the bar.

Pipeline: load prices (universe + SPY + BIL) -> leakage-safe features -> continuous
target ``(r - beta * r_SPY) / vol`` -> walk-forward ``ReturnModel`` per model ->
hold the names predicted above the hurdle (at most ``--max-names``, equal-weighted
among themselves) or the fallback asset when none qualify -> evaluate the picks
(win rate, payoff ratio, expectancy vs the same-month universe, permutation test),
the strictness curve, IC and a CAPM regression net of the risk-free rate.

Usage::

    python -m src.cli.hurdle_cli                       # xgb + rf + ridge, all fallbacks
    python -m src.cli.hurdle_cli --compare-classifier  # + the old median-label xgb, top-N
    python -m src.cli.hurdle_cli --model xgb --fallback spy --hurdle 0.1
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.cli.select_cli import (
    _coverage_lines,
    feature_coverage,
    fundamental_coverage_mean,
    FUND_COVERAGE_WARN,
    month_end_rebalances,
)
from src.common import metrics
from src.selection import constituents as K
from src.selection import hurdle as H
from src.selection import select as S
from src.selection import universe as U
from src.selection.features import (
    FUNDAMENTAL_FEATURES,
    TECHNICAL_FEATURES,
    build_feature_matrix,
    cross_sectional_normalize,
    drop_stale_rows,
)
from src.selection.labels import ex_ante_risk, make_excess_return_targets, make_labels

RETURN_KINDS = ("xgb", "rf", "ridge")
CLASSIFIER_LABEL = "xgb classifier (median label, top-N)"
VOL_BASELINE = "Rank on volatility alone (single factor)"
BETA_BASELINE = "Rank on beta alone (single factor)"
PRIMARY_KINDS = ("xgb", "rf")
MODEL_NAMES = {"xgb": "XGBoost", "rf": "Random forest", "ridge": "Ridge (auxiliary)"}
MIN_MEMBER_COVERAGE = 0.5
FEATURE_GROUPS = {
    "technical": list(TECHNICAL_FEATURES),
    "risk": ["beta", "volatility"],
    "fundamental": list(FUNDAMENTAL_FEATURES),
}
DEFAULT_FEATURES = "technical,fundamental"
TARGET_COLUMNS = {"risk_adjusted": "target", "excess_spy": "excess_spy"}
DEFAULT_TARGET = "risk_adjusted"
WINSORISE = (0.01, 0.99)
SURVIVORSHIP_NOTES = {
    "historical": (
        "The universe is the point-in-time S&P 500: each month only the stocks that "
        "were in the index on that date. Members whose prices yfinance no longer has "
        "(mostly companies later acquired or bankrupt) are missing; see the "
        "membership coverage above. Acquired names tend to have risen and bankrupt "
        "ones to have fallen, so the remaining bias is smaller but its sign is not "
        "certain. Out-of-sample research evaluation only; not financial advice."
    ),
    "current": (
        "The universe is today's S&P 500 constituent list, so names that were later "
        "dropped are missing from the history (survivorship bias); every return above "
        "is likely flattered by that. Out-of-sample research evaluation only; not "
        "financial advice."
    ),
    "custom": (
        "The universe is a user-supplied ticker list, held fixed over the whole "
        "history. Out-of-sample research evaluation only; not financial advice."
    ),
}



def _predict_return_model(kind, feature_matrix, target, dates, min_train, horizon,
                          train_window=S.TRAIN_WINDOW_MONTHS):
    return H.walk_forward_predictions(
        feature_matrix, target, dates, model_kind=kind,
        min_train_dates=min_train, train_window=train_window, horizon_months=horizon,
    )


def _predict_classifier(kind, feature_matrix, labels, dates, min_train, horizon, n):
    records = S._walk_forward_records(
        feature_matrix, labels, dates, n, kind, min_train, label_horizon_months=horizon,
    )
    return [H.Prediction(r.date, r.scores) for r in records]


def _baseline_predictions(
    predictions: Dict[str, List[H.Prediction]],
    feature_matrix: pd.DataFrame,
    risk: pd.DataFrame,
) -> Dict[str, List[H.Prediction]]:
    """No-model rules scored on the models' own dates and cross-sections.

    Each month rank the live names by their ex-ante volatility (or beta) and treat
    that as the score. Any model worth keeping has to beat these.
    """
    dates = sorted({p.date for preds in predictions.values() for p in preds})
    by_date = {d: g.droplevel(0) for d, g in risk.groupby(level=0)}
    out: Dict[str, List[H.Prediction]] = {VOL_BASELINE: [], BETA_BASELINE: []}
    for d in dates:
        live = feature_matrix.loc[d].index
        r = by_date.get(d, pd.DataFrame(columns=["beta", "vol"])).reindex(live)
        out[VOL_BASELINE].append(H.Prediction(d, r["vol"].astype("float64")))
        out[BETA_BASELINE].append(H.Prediction(d, r["beta"].astype("float64")))
    return out


def _run_jobs(jobs: Dict[str, tuple], n_workers: int) -> Dict[str, List[H.Prediction]]:
    if n_workers <= 1 or len(jobs) == 1:
        return {name: fn(*args) for name, (fn, args) in jobs.items()}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {name: pool.submit(fn, *args) for name, (fn, args) in jobs.items()}
        return {name: f.result() for name, f in futures.items()}



def _pct(v: float, digits: int = 2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.{digits}f}%"


def _num(v: float, digits: int = 2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _p(v: float) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"


def _benchmark_rows(evals: Dict[str, H.HurdleEvaluation], closes) -> Dict[str, Dict[str, float]]:
    """SPY / BIL buy-and-hold over the same holding periods the models traded."""
    any_eval = next(iter(evals.values()))
    dates = sorted(any_eval.picks)
    periods = list(zip(dates[:-1], dates[1:]))
    out = {}
    for ticker in (H.BENCHMARK_TICKER, H.RISK_FREE_TICKER):
        if ticker in closes:
            r = H.asset_period_returns(closes[ticker], periods).dropna()
            out[f"{ticker} (buy & hold)"] = metrics.compute_metrics(r, 0.0, H.PERIODS_PER_YEAR)
    return out


def _verdict(ev: H.HurdleEvaluation, vol_baseline: Optional[H.HurdleEvaluation]) -> str:
    """One plain sentence per model, on return.

    The primary question is the one a user asks: did these picks earn more than the
    same number of names drawn at random from the same months? That is
    ``p_expectancy``, which draws stand-ins from the whole cross-section. The
    volatility-matched variant is reported beside it, but it is deliberately
    constructed to strip out any return earned by holding volatile names, so it
    answers the narrower question of whether anything remains *beyond* beta and
    volatility. A rule that selects by taking risk is doing selection; it should not
    be failed for that here, so the matched test never decides this sentence.
    """
    st = ev.pick_stats["beta_adjusted"]
    p, u, t = st["picks"], st["universe"], st["test"]
    raw = ev.pick_stats["raw"]["picks"]
    p_random = t.get("p_expectancy", np.nan)
    head = (f"picks earn {_pct(raw['expectancy'])} per pick against {_pct(u['expectancy'])} for "
            f"the same-month universe, winning {_pct(p['win_rate'], 1)} of the time "
            f"vs {_pct(u['win_rate'], 1)}")
    if not np.isfinite(p_random):
        return f"{head}. The random-pick comparison could not be computed."
    if p_random < 0.05 and p["expectancy"] > 0:
        extra = ""
        matched = t.get("p_expectancy_matched", np.nan)
        if np.isfinite(matched):
            extra = (f" It also beats volatility-matched stand-ins (p = {matched:.3f}), so the "
                     "advantage is not only the risk it takes."
                     if matched < 0.05 else
                     f" Against volatility-matched stand-ins p = {matched:.3f}, so most of the "
                     "advantage comes through beta and volatility.")
        return (f"{head}, beating random picks from the same months at p = {p_random:.3f}."
                f"{extra}")
    return (f"{head}. Against random picks from the same months p = {p_random:.3f}, so this "
            "selector did not earn more than picking at random.")


def build_markdown_report(
    evals: Dict[str, H.HurdleEvaluation],
    closes: Dict[str, pd.Series],
    config: Dict[str, str],
    coverage: Dict[str, float],
    target_summary: Dict[str, float],
    survivorship_note: str = SURVIVORSHIP_NOTES["current"],
) -> str:
    L: List[str] = []
    add = L.append
    add("# Hurdle Selection on a Risk-Adjusted Excess-Return Target")
    add("")
    add(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("## Configuration")
    add("")
    for k, v in config.items():
        add(f"- {k}: {v}")
    add("- Feature coverage (non-NaN cells, before normalisation):")
    add("    -" + _coverage_lines(coverage)[0].replace("  technical:", " technical:"))
    add("    -" + _coverage_lines(coverage)[1].replace("  fundamental:", " fundamental:"))
    add(f"- Target rows: {int(target_summary['rows'])}, mean {target_summary['mean']:.3f}, "
        f"share above zero {_pct(target_summary['share_positive'], 1)}")
    add("")
    add("**Target.** `y = (r_stock - beta * r_SPY) / vol` over the next month, where beta "
        "(Blume-adjusted, 252 trading days) and vol (63 days, monthly scale) use only data "
        "up to the rebalance date, winsorised at the 1st/99th percentile of each month. "
        "Subtracting beta x SPY means a stock cannot score well just by being high-beta in "
        "a rising market; dividing by vol puts the payoff in units of risk taken.")
    add("")
    add("**Rule.** Buy the names whose predicted target is above the hurdle, best first, up "
        "to the cap, equal-weighted among themselves. If none qualify, hold the fallback. The "
        "hurdle is fixed before the run, not tuned on it.")
    add("")

    add("## 1. The picks: win rate, payoff ratio, expectancy")
    add("")
    add("Each pick is judged over the month it was held, two ways. **Raw** = its return minus "
        "SPY's, what a trader sees. **Beta-adjusted** = its return minus beta x SPY's (beta "
        "estimated before the pick), which a high-beta stock cannot win just because the "
        "market rose. The universe column is every stock in the same months, weighted by how "
        "many picks were made that month.")
    add("")
    add("Two random benchmarks replace each month's picks with the same number of names from "
        "that month. **`p (random)` is the primary test**: stand-ins are drawn from the whole "
        "cross-section, so it answers the question a user asks - did these picks earn more than "
        "picking at random? `p (vol-matched)` draws each stand-in from the 20 names nearest the "
        "pick in volatility, which by construction removes any return earned by holding volatile "
        "names; it therefore answers the narrower question of whether anything remains *beyond* "
        "beta and volatility, and a selector that works by taking risk is not failed for losing "
        "it. p is the share of 2,000 draws that did at least as well; below 0.05 is the usual "
        "bar. The two single-factor rows rank on one measure with nothing fitted.")
    add("")
    for view, title in (("raw", "Raw, minus SPY (primary)"), ("beta_adjusted", "Beta-adjusted")):
        add(f"**{title}**")
        add("")
        add("| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio "
            "| Universe | Expectancy / pick | Universe | Vol-matched random | p (random) "
            "| p (vol-matched) |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, ev in evals.items():
            st = ev.pick_stats[view]
            p, u, t = st["picks"], st["universe"], st["test"]
            add(f"| {name} | {int(p['n'])} | {_pct(t['pick_vol_percentile'], 0)} "
                f"| {_pct(p['win_rate'], 1)} | {_pct(u['win_rate'], 1)} "
                f"| {_num(p['payoff_ratio'])} | {_num(u['payoff_ratio'])} "
                f"| {_pct(p['expectancy'])} | {_pct(u['expectancy'])} "
                f"| {_pct(t['matched_null_expectancy'])} | {_p(t['p_expectancy'])} "
                f"| {_p(t['p_expectancy_matched'])} |")
        add("")
    add("*Payoff ratio = average winning excess return / average losing excess return (the "
        "realised risk-reward, no stop-loss or take-profit). Expectancy = mean excess return "
        "per pick. Vol percentile of picks = where the picks sat in that month's volatility "
        "ranking (50% = no tilt, 100% = always the most volatile). Vol-matched random = the "
        "average expectancy of random stand-ins drawn from the 20 names nearest each pick in "
        "volatility, i.e. what the picks' risk profile alone would have earned.*")
    add("")

    add("### 1b. Why per-pick excess and compounded return can disagree")
    add("")
    add("The expectancy above is an **arithmetic** mean per pick. An investor earns the "
        "**compounded** return, and for small returns the two differ by about half the "
        "variance, so a concentrated book gives more back to that term than the index does. "
        "The table separates the two; a small residual means the approximation accounts for "
        "the gap. All figures annualised, against SPY, on the `spy` fallback variant.")
    add("")
    add("| Model | Arithmetic excess | Variance drag | Residual | Compounded excess | Book vol | SPY vol |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for name, ev in evals.items():
        port = ev.portfolios.get("spy") or next(iter(ev.portfolios.values()), None)
        d = getattr(port, "decomposition", None) or {}
        if not d or not np.isfinite(d.get("arithmetic_excess", np.nan)):
            continue
        add(f"| {name} | {_pct(d['arithmetic_excess'])} | -{_pct(abs(d['variance_drag']))} "
            f"| {_pct(d['residual'])} | {_pct(d['geometric_excess'])} "
            f"| {_pct(d['vol_strategy'], 1)} | {_pct(d['vol_market'], 1)} |")
    add("")
    add("*Arithmetic excess - variance drag + residual = compounded excess. The drag is not a "
        "fault in the selector: it is the arithmetic of compounding a more volatile series, and "
        "it shrinks if the book is held wider.*")
    add("")

    add("## 2. Does stricter selection do better? (beta-adjusted)")
    add("")
    add("Top-k by score each month, ignoring the hurdle. If the score orders stocks usefully, "
        "the pick columns improve as k shrinks - the best-scored few should earn more than the "
        "best-scored many.")
    add("")
    for name, ev in evals.items():
        add(f"**{name}**")
        add("")
        add("| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy "
            "| Universe | p (random) | p (vol-matched) |")
        add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, r in ev.curve.iterrows():
            add(f"| {k} | {int(r['n_picks'])} | {_pct(r['pick_vol_percentile'], 0)} "
                f"| {_pct(r['win_rate'], 1)} "
                f"| {_pct(r['universe_win_rate'], 1)} | {_num(r['payoff_ratio'])} "
                f"| {_num(r['universe_payoff_ratio'])} | {_pct(r['expectancy'])} "
                f"| {_pct(r['universe_expectancy'])} | {_p(r['p_expectancy'])} "
                f"| {_p(r['p_expectancy_matched'])} |")
        add("")

    add("## 3. Ranking signal (information coefficient)")
    add("")
    add("| Model | Mean IC | IC t-stat | Months |")
    add("|---|---:|---:|---:|")
    for name, ev in evals.items():
        add(f"| {name} | {_num(ev.ic['mean_ic'], 3)} | {_num(ev.ic['ic_t_stat'])} "
            f"| {int(ev.ic['n_periods'])} |")
    add("")
    add("*Spearman correlation between the predicted and the realised target, per month.*")
    add("")

    add("## 4. Portfolio (monthly, net of costs)")
    add("")
    add("| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, ev in evals.items():
        for fb, pr in ev.portfolios.items():
            m = pr.metrics
            add(f"| {name} | {fb} | {_pct(m['annualised_return'])} | {_num(m['sharpe_ratio'])} "
                f"| {_pct(m['max_drawdown'])} | {_pct(ev.months_invested, 0)} "
                f"| {_num(ev.avg_names_when_invested, 1)} | {_num(pr.avg_turnover)} |")
    for label, m in _benchmark_rows(evals, closes).items():
        add(f"| {label} | - | {_pct(m['annualised_return'])} | {_num(m['sharpe_ratio'])} "
            f"| {_pct(m['max_drawdown'])} | - | - | - |")
    add("")
    add("*`spy`: hold SPY in months with no pick, so those months add zero active return and "
        "the result isolates the stock picks. `bil`: hold Treasury bills instead. `trend`: "
        "SPY if its month-end close is above its 10-month average, else bills; this is a "
        "separate timing rule. Sharpe uses a zero risk-free rate, as in the other reports.*")
    add("")

    add("## 5. Risk-adjusted: CAPM net of the risk-free rate")
    add("")
    add("`r - rf = alpha + beta * (r_SPY - rf)`, monthly, Newey-West t-stats (3 lags), "
        "rf = BIL. |t| above about 2 is the usual bar.")
    add("")
    add("| Model | Fallback | Alpha / yr | t | Beta | t (beta vs 1) | R² | Alpha 1st half (t) | Alpha 2nd half (t) |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, ev in evals.items():
        for fb, pr in ev.portfolios.items():
            c = pr.capm
            add(f"| {name} | {fb} | {_pct(c.get('alpha_ann'))} | {_num(c.get('alpha_t'))} "
                f"| {_num(c.get('beta'))} | {_num(c.get('beta_t_vs_1'))} | {_num(c.get('r2'))} "
                f"| {_pct(c.get('alpha_ann_first_half'))} ({_num(c.get('alpha_t_first_half'))}) "
                f"| {_pct(c.get('alpha_ann_second_half'))} ({_num(c.get('alpha_t_second_half'))}) |")
    add("")

    add("## 6. Year by year (fallback = spy)")
    add("")
    names = [n for n, ev in evals.items() if "spy" in ev.portfolios]
    if names:
        first = evals[names[0]].portfolios["spy"].yearly
        add("| Year | SPY | " + " | ".join(names) + " |")
        add("|---|---:|" + "---:|" * len(names))
        for year in first.index:
            cells = []
            for n in names:
                y = evals[n].portfolios["spy"].yearly
                cells.append(_pct(y.loc[year, "strategy_return"], 1) if year in y.index else "n/a")
            add(f"| {year} | {_pct(first.loc[year, 'benchmark_return'], 1)} | " + " | ".join(cells) + " |")
        add("")

    add("## 7. Latest decision")
    add("")
    for name, ev in evals.items():
        date = f"{ev.latest_date:%Y-%m-%d}" if ev.latest_date is not None else "n/a"
        picks = ", ".join(ev.latest_picks) if ev.latest_picks else "none qualify (hold the fallback)"
        add(f"- **{name}** ({date}): {picks}")
    add("")

    add("## Verdict")
    add("")
    vol_baseline = evals.get(VOL_BASELINE)
    for name, ev in evals.items():
        add(f"- **{name}**: {_verdict(ev, vol_baseline)}")
    add("")
    add(survivorship_note)
    add("")
    return "\n".join(L)



@dataclass
class Inputs:
    """Everything the walk-forward needs, built once from the prices on disk."""

    mode: str
    universe: List[str]
    never_priced: List[str]
    prices: Dict[str, pd.DataFrame]
    spy: Dict[str, pd.DataFrame]
    bil: Dict[str, pd.DataFrame]
    dates: List[pd.Timestamp]
    feature_matrix: pd.DataFrame
    membership: Optional[Dict[pd.Timestamp, set]]
    member_note: str
    coverage: Dict[str, float]
    targets: pd.DataFrame
    target: pd.Series
    target_summary: Dict[str, float]
    risk: pd.DataFrame


def load_inputs(
    tickers: Sequence[str] = (),
    universe_mode: str = "historical",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    normalize: str = "rank",
    horizon: int = 1,
    min_train: int = 6,
    first_rebalance: Optional[str] = None,
    drop_fundamentals: bool = False,
    features: str = DEFAULT_FEATURES,
    target_kind: str = DEFAULT_TARGET,
) -> Inputs:
    """Universe, prices, point-in-time membership, features and the excess-return target.

    Shared by this experiment and the model-explanation CLI so both see exactly the
    same cross-sections. ``first_rebalance`` starts the walk-forward (training and
    predictions) at that date while prices still load from ``start``, so trailing
    technical features are complete from the first month. ``drop_fundamentals``
    removes the six filing-based features, for a with/without comparison.
    ``features`` selects which groups of :data:`FEATURE_GROUPS` the model may see, so
    technical signals, a risk tilt and fundamentals can be compared inside one
    framework on identical folds.
    """
    if target_kind not in TARGET_COLUMNS:
        raise SystemExit(f"Unknown target {target_kind!r}; choose from {list(TARGET_COLUMNS)}.")
    groups = [g.strip() for g in features.split(",") if g.strip()]
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise SystemExit(f"Unknown feature group(s) {unknown}; choose from {list(FEATURE_GROUPS)}.")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    mode = "custom" if tickers else universe_mode
    snapshots = K.load_snapshots() if mode == "historical" else None
    if mode == "custom":
        universe = [t.upper().strip() for t in tickers]
    elif mode == "historical":
        months = pd.date_range(start, end or pd.Timestamp.today(), freq="ME")
        universe = K.historical_universe(snapshots, months)
    else:
        universe = list(U.DEFAULT_UNIVERSE)
    print(f"Loading prices + fundamentals for {len(universe)} tickers ({mode} universe)...")
    prices = U.load_prices(universe, start=start, end=end)
    prices = {t: f for t, f in prices.items() if f is not None and not f.empty}
    never_priced = sorted(set(universe) - set(prices))
    if never_priced:
        print(f"  {len(never_priced)} tickers have no price data (delisted/acquired): "
              f"{', '.join(never_priced[:12])}{' ...' if len(never_priced) > 12 else ''}")
    fundamentals = U.load_fundamentals(sorted(prices))
    spy = U.load_prices([H.BENCHMARK_TICKER], start=start, end=end)
    if H.BENCHMARK_TICKER not in spy:
        raise SystemExit("SPY prices could not be loaded; they define the target.")
    bil = U.load_prices([H.RISK_FREE_TICKER], start=start, end=end)

    dates = month_end_rebalances(prices)
    if len(dates) <= min_train + 1:
        raise SystemExit("Not enough history for a walk-forward run; widen --start.")

    feature_matrix = drop_stale_rows(build_feature_matrix(prices, fundamentals, dates), prices)
    membership = None
    member_note = "n/a (fixed universe)"
    if snapshots is not None:
        membership = K.members_at(snapshots, dates)
        feature_matrix = K.filter_to_members(feature_matrix, membership)
        present = {d: set(g.index.get_level_values(1)) for d, g in feature_matrix.groupby(level=0)}
        cov = K.membership_coverage(membership, present)
        thin = [d for d in dates if not cov.loc[d, "coverage"] >= MIN_MEMBER_COVERAGE]
        if thin:
            print(f"  Skipping {len(thin)} month(s) where under {MIN_MEMBER_COVERAGE:.0%} of members "
                  f"have prices: {', '.join(f'{d:%Y-%m}' for d in thin)}")
            dates = [d for d in dates if d not in thin]
            feature_matrix = feature_matrix[~feature_matrix.index.get_level_values(0).isin(thin)]
            cov = cov.drop(index=thin)
        member_note = (f"{len(universe)} tickers were members at some point; {len(never_priced)} "
                       f"have no price data. Share of each month's members with prices: median "
                       f"{cov['coverage'].median():.0%}, lowest {cov['coverage'].min():.0%} "
                       f"({cov['coverage'].idxmin():%Y-%m}), highest {cov['coverage'].max():.0%}"
                       + (f". Skipped months under {MIN_MEMBER_COVERAGE:.0%}: "
                          + ", ".join(f"{d:%Y-%m}" for d in thin) if thin else ""))
        print(f"  Membership coverage: median {cov['coverage'].median():.0%}, "
              f"lowest {cov['coverage'].min():.0%}")
    if first_rebalance is not None:
        cutoff = pd.Timestamp(first_rebalance)
        dates = [d for d in dates if d >= cutoff]
        feature_matrix = feature_matrix[feature_matrix.index.get_level_values(0) >= cutoff]
        if len(dates) <= min_train + 1:
            raise SystemExit("Too few rebalances after --first-rebalance for a walk-forward run.")
    risk = ex_ante_risk(prices, spy[H.BENCHMARK_TICKER], dates)
    if "risk" in groups:
        feature_matrix = feature_matrix.join(risk["beta"], how="left")
    if drop_fundamentals and "fundamental" in groups:
        groups = [g for g in groups if g != "fundamental"]
    keep = [c for g in groups for c in FEATURE_GROUPS[g]]
    keep = [c for c in dict.fromkeys(keep) if c in feature_matrix.columns]
    feature_matrix = feature_matrix[keep]
    print(f"Building features + targets over {len(dates)} monthly rebalances "
          f"on {len(keep)} features ({', '.join(groups)})...")
    coverage = feature_coverage(feature_matrix)
    for line in _coverage_lines(coverage):
        print(line)
    if not drop_fundamentals and fundamental_coverage_mean(coverage) < FUND_COVERAGE_WARN:
        print("WARNING: fundamentals are mostly missing; this is effectively a technical-factor model.")
    if normalize != "none":
        feature_matrix = cross_sectional_normalize(feature_matrix, method=normalize)

    targets = make_excess_return_targets(
        prices, spy[H.BENCHMARK_TICKER], dates, horizon_months=horizon, members=membership,
    )
    target = targets[TARGET_COLUMNS[target_kind]]
    if target_kind != "risk_adjusted":
        lo, hi = WINSORISE
        by_date = target.groupby(level="rebalance_date")
        target = target.clip(lower=by_date.transform(lambda s: s.quantile(lo)),
                             upper=by_date.transform(lambda s: s.quantile(hi)))
    target_summary = {"rows": float(len(target)), "mean": float(target.mean()),
                      "share_positive": float((target > 0).mean())}
    print(f"Target ({target_kind}) rows {len(target)}, mean {target.mean():.4f}, "
          f"share > 0 {(target > 0).mean() * 100:.1f}%")
    return Inputs(
        mode=mode, universe=universe, never_priced=never_priced, prices=prices, spy=spy,
        bil=bil, dates=dates, feature_matrix=feature_matrix, membership=membership,
        member_note=member_note, coverage=coverage, targets=targets, target=target,
        target_summary=target_summary, risk=risk,
    )


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, H.HurdleEvaluation]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*",
                        help="Fixed universe (default: the point-in-time S&P 500, see --universe)")
    parser.add_argument("--universe", choices=["historical", "current"], default="historical",
                        help="historical = each month's actual S&P 500 members (removes "
                             "survivorship bias); current = today's list for every month")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--model", choices=list(RETURN_KINDS) + ["primary", "all"], default="all",
                        help="all = xgb + rf (primary) + ridge (auxiliary); primary = xgb + rf")
    parser.add_argument("--hurdle", type=float, default=H.DEFAULT_HURDLE,
                        help="Buy only names whose predicted target is above this (default 0: "
                             "expected to beat beta x SPY)")
    parser.add_argument("--max-names", type=int, default=H.DEFAULT_MAX_NAMES)
    parser.add_argument("--fallback", choices=list(H.FALLBACKS) + ["all"], default="all")
    parser.add_argument("--horizon", type=int, default=1, help="Target horizon in months")
    parser.add_argument("--min-train", type=int, default=6)
    parser.add_argument("--train-window", type=int, default=None,
                        help="Rolling training window in rebalances (default: the pipeline's "
                             "12); use e.g. 3 or 6 for a short look-back")
    parser.add_argument("--cost", type=float, default=S.DEFAULT_COST_PER_TURNOVER)
    parser.add_argument("--normalize", choices=["rank", "zscore", "none"], default="rank")
    parser.add_argument("--compare-classifier", action="store_true",
                        help="Also run the previous xgb classifier (median label, fixed top-N) "
                             "and evaluate it on the same pick statistics")
    parser.add_argument("--draws", type=int, default=2000, help="Permutation draws")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel processes (0 = one per model)")
    parser.add_argument("--first-rebalance", default=None,
                        help="Start the walk-forward (training + predictions) here; prices still "
                             "load from --start so technical features are complete")
    parser.add_argument("--no-fundamentals", action="store_true",
                        help="Drop the six filing-based features (technical-only model)")
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=list(TARGET_COLUMNS),
                        help="What the model predicts: risk_adjusted = excess over beta x SPY "
                             "divided by volatility (skill beyond risk); excess_spy = plain "
                             "return minus SPY (did it beat the index)")
    parser.add_argument("--features", default=DEFAULT_FEATURES,
                        help="Comma-separated feature groups the model may see: "
                             f"{', '.join(FEATURE_GROUPS)} (default: {DEFAULT_FEATURES})")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    inp = load_inputs(args.tickers, args.universe, args.start, args.end, args.normalize,
                      args.horizon, args.min_train, args.first_rebalance, args.no_fundamentals,
                      args.features, args.target)
    mode, universe, prices, spy, bil = inp.mode, inp.universe, inp.prices, inp.spy, inp.bil
    dates, feature_matrix, membership = inp.dates, inp.feature_matrix, inp.membership
    member_note, coverage, target = inp.member_note, inp.coverage, inp.target
    target_summary, risk = inp.target_summary, inp.risk

    fallbacks = list(H.FALLBACKS) if args.fallback == "all" else [args.fallback]
    if H.RISK_FREE_TICKER not in bil:
        print(f"WARNING: {H.RISK_FREE_TICKER} unavailable; using fallback 'spy' only and rf = 0.")
        fallbacks = ["spy"]

    kinds = {"all": list(RETURN_KINDS), "primary": list(PRIMARY_KINDS)}.get(args.model, [args.model])
    train_window = args.train_window if args.train_window is not None else S.TRAIN_WINDOW_MONTHS
    jobs: Dict[str, tuple] = {
        MODEL_NAMES[k]: (_predict_return_model,
                         (k, feature_matrix, target, dates, args.min_train, args.horizon,
                          train_window))
        for k in kinds
    }
    if args.compare_classifier:
        labels = make_labels(prices, dates, horizon_months=args.horizon, members=membership)
        jobs[CLASSIFIER_LABEL] = (_predict_classifier,
                                  ("xgb", feature_matrix, labels, dates, args.min_train,
                                   args.horizon, args.max_names))
    workers = args.jobs or len(jobs)
    print(f"Running {len(jobs)} walk-forward model(s) on {workers} process(es): {', '.join(jobs)}")
    predictions = _run_jobs(jobs, workers)
    predictions.update(_baseline_predictions(predictions, feature_matrix, risk))

    closes = S._close_series_by_ticker({**prices, **spy, **bil})
    evals: Dict[str, H.HurdleEvaluation] = {}
    for name, preds in predictions.items():
        picker = (H.hurdle_picker(args.hurdle, args.max_names) if name in MODEL_NAMES.values()
                  else H.top_k_picker(args.max_names))
        print(f"Evaluating {name}...")
        evals[name] = H.evaluate(
            name, preds, picker, closes, target, risk=risk, fallbacks=fallbacks,
            cost_per_turnover=args.cost, n_draws=args.draws,
        )

    print("Beta-adjusted pick statistics (p vs volatility-matched random picks):")
    for name, ev in evals.items():
        st = ev.pick_stats["beta_adjusted"]
        p, u, t = st["picks"], st["universe"], st["test"]
        print(f"  {name:42s} win {p['win_rate'] * 100:5.1f}% (univ {u['win_rate'] * 100:5.1f}%)  "
              f"payoff {p['payoff_ratio']:.2f}  exp {p['expectancy'] * 100:+.2f}% "
              f"(vol-matched {t['matched_null_expectancy'] * 100:+.2f}%)  "
              f"p={t['p_expectancy_matched']:.3f}  vol pct {t['pick_vol_percentile'] * 100:.0f}%  "
              f"IC t={ev.ic['ic_t_stat']:.2f}  "
              f"invested {ev.months_invested * 100:.0f}%")

    if not args.no_save:
        config = {
            "Universe": {"historical": "point-in-time S&P 500 (each month's actual members)",
                         "current": f"today's S&P 500 list ({len(universe)} tickers) for every month",
                         "custom": f"{len(universe)} user-supplied tickers"}[mode],
            "Membership coverage": member_note,
            "History": f"{args.start} to {args.end or 'latest'}, {len(dates)} monthly rebalances",
            "Training window": f"{train_window} rebalances rolling",
            "Models": ", ".join(jobs),
            "Hurdle": f"predicted target > {args.hurdle}",
            "Max names": f"{args.max_names} (equal weight among those that qualify)",
            "Fallbacks": ", ".join(fallbacks),
            "Transaction cost": f"{args.cost * 1e4:.0f} bps per unit turnover (fallback switches included)",
            "Feature normalisation": args.normalize,
            "Features": f"{args.features} ({len(inp.feature_matrix.columns)} columns: "
                        f"{', '.join(inp.feature_matrix.columns)})",
            "Training target": ("(r - beta x r_SPY) / vol, winsorised per month"
                                if args.target == "risk_adjusted"
                                else "r - r_SPY, winsorised per month"),
            "Walk-forward starts": (f"{args.first_rebalance} (prices from {args.start})"
                                    if args.first_rebalance else args.start),
        }
        report = build_markdown_report(evals, closes, config, coverage, target_summary,
                                       SURVIVORSHIP_NOTES[mode])
        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        feature_tag = ("" if args.features == DEFAULT_FEATURES
                       else "_feat-" + args.features.replace(",", "+"))
        target_tag = "" if args.target == DEFAULT_TARGET else f"_tgt-{args.target}"
        tag = "".join([f"_from{args.first_rebalance[:4]}" if args.first_rebalance else "",
                       "_nofund" if args.no_fundamentals else "", feature_tag, target_tag])
        out = reports_dir / f"hurdle_{mode}_{args.normalize}{tag}_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(report)
        print(f"Report written to {out}")
    return evals


if __name__ == "__main__":
    main()

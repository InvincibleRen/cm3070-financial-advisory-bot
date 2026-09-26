"""Data/logic layer for the web UI.

Deliberately **UI-framework-free**: every function here is pure application logic
that takes an injectable data provider, so it can be unit-tested offline without
any web server. The FastAPI layer (``src/api/main.py``) serialises these results
for the React front end. This keeps the UI thin and the logic testable (Rule 1).

Each function wires together modules that already exist and are tested elsewhere:
the rule-based advisor (core), the FinBERT sentiment scorer (Extension A), and the
cross-sectional selector (Direction-1 core).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.common.data.yfinance_provider import Quote, YFinanceProvider
from src.common.indicators import add_moving_averages, latest_indicator_snapshot
from src.explanation.advisor import Recommendation, generate_recommendation

DEFAULT_HISTORY = "2y"


SINGLE_STOCK_PRESETS: Dict[str, dict] = {
    "1 day":    {"period": "1d",  "interval": "5m",  "trim": None},
    "1 week":   {"period": "5d",  "interval": "30m", "trim": None},
    "1 month":  {"period": "1mo", "interval": "1d",  "trim": None},
    "3 months": {"period": "3mo", "interval": "1d",  "trim": None},
    "6 months": {"period": "6mo", "interval": "1d",  "trim": None},
    "1 year":   {"period": "1y",  "interval": "1d",  "trim": None},
    "3 years":  {"period": "5y",  "interval": "1d",  "trim": 756},
    "5 years":  {"period": "5y",  "interval": "1d",  "trim": None},
}

BACKTEST_PRESETS: Dict[str, dict] = {
    label: {"period": p["period"], "interval": p["interval"], "trim": p["trim"]}
    for label, p in SINGLE_STOCK_PRESETS.items()
    if label not in ("1 day", "1 week")
}


def default_universe() -> List[str]:
    """The built-in large-cap universe, offered as suggestions in the UI."""
    from src.selection import universe as U

    return list(U.DEFAULT_UNIVERSE)


def _fetch_prices(provider, ticker: str, period: str, interval: str, trim: Optional[int]):
    """Fetch OHLC and optionally keep only the last ``trim`` rows (for '3 years')."""
    data = provider.get_historical_data(ticker, period=period, interval=interval)
    if trim:
        data = data.tail(int(trim))
    return data



@dataclass
class SingleStockAnalysis:
    ticker: str
    quote: Quote
    indicators: Dict
    recommendation: Recommendation
    price_chart: pd.DataFrame


def analyse_single_stock(
    ticker: str,
    period: str = DEFAULT_HISTORY,
    interval: str = "1d",
    trim_rows: Optional[int] = None,
    provider: Optional[object] = None,
) -> SingleStockAnalysis:
    """Full single-stock view: quote, indicators, recommendation, price chart.

    ``period``/``interval``/``trim_rows`` come from a ``SINGLE_STOCK_PRESETS`` entry
    (the view resolves the friendly label). Moving-average columns that never fill
    on a short/intraday range are dropped from the chart rather than shown empty.
    """
    provider = provider or YFinanceProvider()
    ticker = ticker.upper().strip()

    historical = _fetch_prices(provider, ticker, period, interval, trim_rows)
    try:
        intraday = provider.get_intraday_data(ticker, period="5d", interval="5m")
    except Exception:  # noqa: BLE001 - intraday is optional context
        intraday = None

    quote = provider.get_latest_quote(ticker)
    indicators = latest_indicator_snapshot(historical_data=historical, intraday_data=intraday)
    recommendation = generate_recommendation(quote=quote, indicators=indicators)

    sma = add_moving_averages(historical[["Close"]])
    price_chart = sma[["Close", "SMA20", "SMA50", "SMA200"]].dropna(axis=1, how="all").copy()

    return SingleStockAnalysis(
        ticker=ticker, quote=quote, indicators=indicators, recommendation=recommendation,
        price_chart=price_chart,
    )



@dataclass
class TickerEvaluation:
    ticker: str
    walkforward: object


def evaluate_ticker(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    train: int = 252,
    test: int = 63,
    cost: float = 0.001,
    trim_rows: Optional[int] = None,
    provider: Optional[object] = None,
) -> TickerEvaluation:
    """Walk-forward rule-based backtest of the advisor's signals.

    Raises a clear ``ValueError`` when the chosen period is too short to form a
    single train+test walk-forward window, so the UI can show a friendly hint.
    """
    from src.common.walkforward import run_walk_forward

    provider = provider or YFinanceProvider()
    ticker = ticker.upper().strip()
    prices = _fetch_prices(provider, ticker, period, interval, trim_rows)

    if len(prices) < train + test + 5:
        raise ValueError(
            f"Only {len(prices)} bars for this period: a walk-forward backtest needs "
            f"more than train+test ({train}+{test}). Pick a longer period or a smaller "
            "train window."
        )

    walkforward = run_walk_forward(
        prices, ticker=ticker, train_size=train, test_size=test, cost_per_turnover=cost,
    )
    return TickerEvaluation(ticker=ticker, walkforward=walkforward)



@dataclass
class SentimentResult:
    ticker: str
    aggregate: float
    headlines: List[Tuple[pd.Timestamp, str, float, str]] = field(default_factory=list)
    n_total: int = 0
    error: Optional[str] = None


def sentiment_interpretation(score: float) -> Tuple[str, str]:
    """Turn a signed sentiment score in [-1, 1] into (label, plain-English meaning)."""
    if score is None or score != score:
        return "Unknown", "No sentiment score could be computed."
    if score >= 0.5:
        return "Strongly positive", "Recent coverage is clearly upbeat about this stock."
    if score >= 0.15:
        return "Positive", "Recent coverage leans favourable on balance."
    if score > -0.15:
        return "Neutral", "Coverage is mixed or neutral: no clear positive or negative tilt."
    if score > -0.5:
        return "Negative", "Recent coverage leans unfavourable on balance."
    return "Strongly negative", "Recent coverage is clearly downbeat about this stock."


def score_ticker_sentiment(
    ticker: str,
    lookback_days: int = 30,
    backend: str = "onnx-local",
    max_headlines: int = 10,
    asof: Optional[pd.Timestamp] = None,
    source: Optional[object] = None,
) -> SentimentResult:
    """Fetch recent headlines and score with FinBERT.

    The aggregate is the mean signed score over **all** headlines in the window;
    the returned ``headlines`` list is the most recent ``max_headlines`` (each with
    its own score and article URL) for display.
    """
    ticker = ticker.upper().strip()
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()

    try:
        if source is None:
            from src.sentiment.news_source import YFinanceNewsSource

            source = YFinanceNewsSource()
        headlines = source.fetch_headlines(ticker, asof, lookback_days)
    except Exception as exc:  # noqa: BLE001 - a news failure must not crash the UI
        return SentimentResult(ticker, float("nan"), [], 0, f"News fetch failed: {exc}")

    if not headlines:
        return SentimentResult(ticker, 0.0, [], 0, None)

    try:
        from src.sentiment.finbert import FinBERTScorer

        scorer = FinBERTScorer(backend=backend)
        aggregate = scorer.score_texts([h.text for h in headlines])
        recent = sorted(headlines, key=lambda h: h.published_at, reverse=True)[:max_headlines]
        rows = [(h.published_at, h.text, scorer.score_texts([h.text]), h.url) for h in recent]
    except Exception as exc:  # noqa: BLE001 - missing deps / model download issues
        return SentimentResult(ticker, float("nan"), [], len(headlines), f"Scoring failed: {exc}")

    return SentimentResult(ticker, float(aggregate), rows, len(headlines), None)



@dataclass
class SelectionResult:
    result: object
    latest_date: Optional[pd.Timestamp]
    latest_picks: List[str]
    universe: List[str]
    n_rebalances: int
    next_rebalance: Optional[pd.Timestamp] = None
    data_asof: Optional[pd.Timestamp] = None
    pick_reasons: Dict[str, List[dict]] = field(default_factory=dict)
    previous_issue: Optional[dict] = None


def run_universe_selection(
    tickers: Optional[List[str]] = None,
    start: str = "2018-01-01",
    top_n: int = 3,
    model_kind: str = "xgb",
    min_train: int = 6,
    cost: float = 0.001,
    normalize: str = "rank",
) -> SelectionResult:
    """Run the walk-forward selector over a universe and return the latest top-N.

    ``normalize`` applies leakage-safe cross-sectional feature standardisation per
    rebalance date ("rank" by default: the setting that improves ranking AUC /
    precision@N; pass "none" for the raw-level behaviour).
    """
    from src.cli.select_cli import BENCHMARK_TICKER, month_end_rebalances
    from src.selection import universe as U
    from src.selection.features import (
        build_feature_matrix, cross_sectional_normalize, drop_stale_rows,
    )
    from src.selection.labels import make_labels
    from src.selection.select import run_selection_backtest

    universe = [t.upper().strip() for t in tickers] if tickers else list(U.DEFAULT_UNIVERSE)
    prices = U.load_prices(universe, start=start, end=None)
    fundamentals = U.load_fundamentals(universe)
    spy = U.load_prices([BENCHMARK_TICKER], start=start, end=None)

    rebalance_dates = month_end_rebalances(prices)
    if len(rebalance_dates) <= min_train + 1:
        raise ValueError("Not enough history for a walk-forward run; use an earlier start date.")

    feature_matrix = drop_stale_rows(build_feature_matrix(prices, fundamentals, rebalance_dates), prices)
    if normalize != "none":
        feature_matrix = cross_sectional_normalize(feature_matrix, method=normalize)
    labels = make_labels(prices, rebalance_dates, horizon_months=1)
    prices_eval = dict(prices)
    prices_eval.update(spy)

    last_fold: dict = {}

    def remember(date, model, scored):
        last_fold.update(date=date, model=model, scored=scored)

    result = run_selection_backtest(
        feature_matrix, labels, prices_eval, rebalance_dates,
        n=top_n, model_kind=model_kind, min_train_dates=min_train, cost_per_turnover=cost,
        on_fold=remember,
    )

    latest_date = result.weights.index.max() if not result.weights.empty else None
    latest_picks: List[str] = []
    if latest_date is not None:
        row = result.weights.loc[latest_date]
        latest_picks = list(row[row > 0].index)

    stamps = [pd.to_datetime(df.index.max(), errors="coerce") for df in prices.values() if not df.empty]
    stamps = [s for s in stamps if pd.notna(s)]
    data_asof = max(stamps) if stamps else None
    anchor = pd.to_datetime(latest_date, errors="coerce")
    next_rebalance = (anchor + pd.offsets.MonthEnd(1)) if pd.notna(anchor) else None

    return SelectionResult(
        result=result, latest_date=latest_date, latest_picks=latest_picks,
        universe=universe, n_rebalances=len(rebalance_dates),
        next_rebalance=next_rebalance, data_asof=data_asof,
        pick_reasons=_explain_picks(last_fold, latest_date, latest_picks),
        previous_issue=_previous_issue(result, spy.get(BENCHMARK_TICKER)),
    )


def _previous_issue(result, spy_prices) -> Optional[dict]:
    """Last month's picks and how they did, against SPY over the same weeks.

    The walk-forward already holds every past rebalance, so the most recent completed
    holding period can be reported without storing anything between runs: its return
    is the one the backtest realised for that month, net of costs.
    """
    from src.cli.select_cli import BENCHMARK_TICKER
    from src.selection.select import _close_series_by_ticker, _period_return

    dates = list(result.weights.index)
    if len(dates) < 2:
        return None
    entry, exit_ = dates[-2], dates[-1]
    row = result.weights.loc[entry]
    picks = list(row[row > 0].index)
    strategy = result.strategy_returns.get(entry)
    spy_return = None
    if spy_prices is not None and not spy_prices.empty:
        closes = _close_series_by_ticker({BENCHMARK_TICKER: spy_prices})
        spy_return = _period_return(closes[BENCHMARK_TICKER], entry, exit_)
    if strategy is None or not picks:
        return None
    return {
        "entry": entry, "exit": exit_, "picks": picks,
        "strategy_return": float(strategy),
        "spy_return": None if spy_return is None else float(spy_return),
        "excess": None if spy_return is None else float(strategy) - float(spy_return),
    }


def _explain_picks(last_fold: dict, latest_date, picks: List[str]) -> Dict[str, List[dict]]:
    """SHAP reasons for the latest picks, or ``{}`` when the model cannot explain itself."""
    from src.selection.explain import pick_reasons

    if not last_fold or not picks or last_fold.get("date") != latest_date:
        return {}
    try:
        contributions = last_fold["model"].contributions(last_fold["scored"])
    except (NotImplementedError, ImportError, RuntimeError):
        return {}
    return pick_reasons(contributions, last_fold["scored"], picks)

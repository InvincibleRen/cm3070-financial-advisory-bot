"""FastAPI backend for the React frontend.

Thin REST wrapper around ``src.ui.data_access``: the same UI-framework-free
logic layer, serialising the dataclass results into JSON that the React
client expects.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.common.data.yfinance_provider import YFinanceProvider
from src.direction.evaluate import DirectionEvaluation, evaluate_direction
from src.direction.forecaster import DirectionForecast, DirectionForecaster
from src.ui import evidence as ev
from src.ui.profile import profile_guidance
from src.ui.data_access import (
    BACKTEST_PRESETS,
    SINGLE_STOCK_PRESETS,
    SelectionResult,
    SingleStockAnalysis,
    SentimentResult,
    TickerEvaluation,
    analyse_single_stock,
    default_universe,
    evaluate_ticker,
    run_universe_selection,
    score_ticker_sentiment,
    sentiment_interpretation,
)

app = FastAPI(
    title="Financial Advisor Bot API",
    version="1.0.0",
    description="REST API for the explainable stock-selection advisor.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)



def _safe(v: Any) -> Any:
    """Convert NaN / Inf / Timestamp to JSON-safe values."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _quote_dict(q) -> Dict[str, Any]:
    return {
        "ticker": q.ticker,
        "latest_price": _safe(q.latest_price),
        "previous_close": _safe(q.previous_close),
        "change": _safe(q.change),
        "change_percent": _safe(q.change_percent),
        "timestamp": q.timestamp,
    }


def _recommendation_dict(r) -> Dict[str, Any]:
    return {
        "signal": r.signal,
        "risk_level": r.risk_level,
        "score": r.score,
        "reasons": r.reasons,
        "warnings": r.warnings,
        "disclaimer": r.disclaimer,
        "trend_strength": r.trend_strength,
    }


def _chart_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a price-chart DataFrame to a list of {date, Close, SMA20, ...}."""
    records = []
    for idx, row in df.iterrows():
        entry: Dict[str, Any] = {"date": str(idx)[:10]}
        for col in df.columns:
            val = row[col]
            entry[col] = round(float(val), 2) if pd.notna(val) else None
        records.append(entry)
    return records



@app.get("/api/universe")
def get_universe():
    """Return the default stock universe."""
    return {"tickers": default_universe()}


@app.get("/api/presets")
def get_presets():
    """Return period-preset labels for the single-stock and backtest views."""
    return {"single_stock": list(SINGLE_STOCK_PRESETS.keys()),
            "backtest": list(BACKTEST_PRESETS.keys())}


@app.get("/api/selection")
def get_selection(
    tickers: Optional[str] = Query(None, description="Comma-separated ticker list"),
    start: str = Query("2018-01-01"),
    top_n: int = Query(5, ge=1, le=20),
    model: str = Query("xgb"),
):
    """Run the walk-forward cross-sectional selector and return the latest picks."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    try:
        result: SelectionResult = run_universe_selection(
            tickers=ticker_list, start=start, top_n=top_n, model_kind=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    picks = []
    for ticker in result.latest_picks:
        try:
            analysis = analyse_single_stock(ticker, period="1y")
            picks.append({
                "ticker": ticker,
                "quote": _quote_dict(analysis.quote),
                "recommendation": _recommendation_dict(analysis.recommendation),
                "model_reasons": _model_reasons(result, ticker),
            })
        except Exception as exc:  # noqa: BLE001
            picks.append({
                "ticker": ticker,
                "quote": None,
                "recommendation": None,
                "error": str(exc),
                "model_reasons": _model_reasons(result, ticker),
            })

    overall = {}
    if hasattr(result.result, "overall_strategy"):
        overall = {k: _safe(v) for k, v in result.result.overall_strategy.items()}

    return {
        "latest_date": _safe(result.latest_date),
        "next_rebalance": _safe(result.next_rebalance),
        "data_asof": _safe(result.data_asof),
        "n_rebalances": result.n_rebalances,
        "universe": result.universe,
        "picks": picks,
        "previous_issue": _previous_issue_dict(result.previous_issue),
        "overall_metrics": overall,
    }


EVIDENCE_MODEL = "xgb classifier (median label, top-N)"


@app.get("/api/evidence")
def get_evidence(signal: Optional[str] = Query(None, description="Buy / Sell / Hold")):
    """How this kind of signal, and the selector, did in the survivorship-free tests."""
    evidence = ev.load_evidence()
    if evidence is None:
        return {"available": False, "signal_lines": [], "model_lines": [], "index_reference": ""}
    return {
        "available": True,
        "generated_at": evidence.get("generated_at"),
        "signal_lines": ev.signal_evidence(signal, evidence) if signal else [],
        "model_lines": ev.model_evidence_lines(evidence, EVIDENCE_MODEL, "this app's stock selector"),
        "index_reference": ev.index_reference(evidence, EVIDENCE_MODEL),
    }


def _previous_issue_dict(issue: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Last month's advice and what it earned, for the 'how did we do' panel."""
    if not issue:
        return None
    return {**issue, "entry": _safe(issue["entry"]), "exit": _safe(issue["exit"])}


def _model_reasons(result: SelectionResult, ticker: str) -> List[Dict[str, Any]]:
    """Why the ranker scored this pick highly, as sentences plus the raw numbers."""
    return [{**r, "text": ev.pick_reason_text(r)} for r in result.pick_reasons.get(ticker, [])]


@app.get("/api/profile")
def get_profile(
    risk: str = Query("medium", description="low / medium / high"),
    horizon: str = Query("1_to_5y", description="under_1y / 1_to_5y / over_5y"),
    single_stocks: bool = Query(True, description="Willing to hold individual shares"),
):
    """How three answers change the layout and the number of names shown."""
    return profile_guidance(risk=risk, horizon=horizon, single_stocks=single_stocks)


@app.get("/api/research")
def get_research():
    """The survivorship-free study tables behind the app's Evidence page."""
    evidence = ev.load_evidence()
    if evidence is None or "research" not in evidence:
        return {"available": False}
    return {"available": True, "generated_at": evidence.get("generated_at"), **evidence["research"]}


@app.get("/api/stock/{ticker}")
def get_stock(
    ticker: str,
    period: str = Query("1 year"),
):
    """Single-stock analysis: quote, indicators, recommendation and price chart."""
    preset = SINGLE_STOCK_PRESETS.get(period)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown period '{period}'. Use one of: {list(SINGLE_STOCK_PRESETS.keys())}")

    try:
        analysis: SingleStockAnalysis = analyse_single_stock(
            ticker,
            period=preset["period"],
            interval=preset["interval"],
            trim_rows=preset["trim"],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

    indicators = {}
    for k, v in analysis.indicators.items():
        indicators[k] = _safe(v)

    return {
        "ticker": analysis.ticker,
        "quote": _quote_dict(analysis.quote),
        "indicators": indicators,
        "recommendation": _recommendation_dict(analysis.recommendation),
        "price_chart": _chart_records(analysis.price_chart),
    }


@app.get("/api/sentiment/{ticker}")
def get_sentiment(ticker: str):
    """FinBERT news sentiment for a single stock."""
    try:
        result: SentimentResult = score_ticker_sentiment(ticker)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    label, meaning = sentiment_interpretation(result.aggregate)

    headlines = []
    for row in result.headlines:
        date, text, score, url = row
        tone_label, _ = sentiment_interpretation(score)
        headlines.append({
            "date": _safe(date),
            "text": text,
            "score": _safe(score),
            "tone": tone_label,
            "url": url,
        })

    return {
        "ticker": result.ticker,
        "aggregate": _safe(result.aggregate),
        "label": label,
        "meaning": meaning,
        "n_total": result.n_total,
        "error": result.error,
        "headlines": headlines,
    }


@app.get("/api/evaluate/{ticker}")
def get_evaluate(
    ticker: str,
    period: str = Query("5 years"),
    train: int = Query(252, ge=50),
):
    """Walk-forward backtest of the advisor's rule-based signals for one stock."""
    preset = BACKTEST_PRESETS.get(period)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown period '{period}'. Use one of: {list(BACKTEST_PRESETS.keys())}")

    try:
        result: TickerEvaluation = evaluate_ticker(
            ticker,
            period=preset["period"],
            interval=preset["interval"],
            train=train,
            trim_rows=preset["trim"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    wf = result.walkforward
    strategy = {k: _safe(v) for k, v in wf.overall_strategy.items()}
    benchmark = {k: _safe(v) for k, v in wf.overall_benchmark.items()}

    equity_curve = []
    if hasattr(wf, "stitched_returns") and not wf.stitched_returns.empty:
        cumulative = (1 + wf.stitched_returns).cumprod()
        for idx, val in cumulative.items():
            equity_curve.append({"date": str(idx)[:10], "value": round(float(val), 4)})

    return {
        "ticker": ticker.upper(),
        "strategy": strategy,
        "benchmark": benchmark,
        "equity_curve": equity_curve,
        "train_size": wf.train_size,
        "test_size": wf.test_size,
        "n_folds": len(wf.folds),
    }


DIRECTION_KIND = "gbm"
DIRECTION_HORIZON = 5

_DIRECTION_CACHE: Dict[tuple, Dict[str, Any]] = {}


@app.get("/api/direction/{ticker}")
def get_direction(
    ticker: str,
    horizon: int = Query(DIRECTION_HORIZON, ge=1, le=30),
    period: str = Query("5y"),
):
    """Per-stock ML direction forecast, returned together with its walk-forward reliability.

    The current calibrated up-probability and the out-of-sample evaluation (accuracy vs
    the naive base rate) are delivered in one payload on purpose: the probability must
    never be shown without the evidence that this kind of forecast is close to a coin flip.
    """
    ticker = ticker.upper()
    cache_key = (ticker, horizon, period)
    if cache_key in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[cache_key]

    provider = YFinanceProvider()
    try:
        prices = provider.get_historical_data(ticker, period=period, interval="1d")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not load prices for {ticker}: {exc}")
    if prices is None or len(prices) == 0:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    try:
        forecaster = DirectionForecaster(kind=DIRECTION_KIND, horizon_days=horizon).fit(prices, ticker)
        call: DirectionForecast = forecaster.predict_direction(prices, ticker)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Direction forecast failed for {ticker}: {exc}")

    try:
        ev_res: Optional[DirectionEvaluation] = evaluate_direction(
            prices, ticker, kind=DIRECTION_KIND, horizon_days=horizon)
    except Exception:  # noqa: BLE001
        ev_res = None

    forecast = {
        "prob_up": _safe(call.prob_up),
        "prob_up_pct": round(call.prob_up * 100, 1) if isinstance(call.prob_up, float) else None,
        "direction": call.direction,
        "horizon_days": call.horizon_days,
        "as_of": _safe(call.as_of),
    }

    evaluation = None
    verdict = None
    if ev_res is not None:
        edge = ev_res.edge
        beats = isinstance(edge, float) and edge > 0.0
        auc = ev_res.auc
        reliable = bool(beats and auc is not None and auc > 0.55)
        if ev_res.n_predictions == 0:
            verdict = "Not enough history to test this forecast on this stock."
        elif reliable:
            verdict = (
                f"Out of sample this forecast beat the naive base rate by {edge * 100:.1f} "
                "points: a small edge, not a reliable one."
            )
        else:
            verdict = (
                f"Out of sample this forecast was no better than the base rate: "
                f"{ev_res.accuracy * 100:.1f}% accurate versus {ev_res.base_rate * 100:.1f}% "
                "from always guessing the majority direction. Treat the probability above as "
                "roughly a coin flip."
            )
        evaluation = {
            "n_predictions": ev_res.n_predictions,
            "accuracy": _safe(ev_res.accuracy),
            "base_rate": _safe(ev_res.base_rate),
            "edge": _safe(ev_res.edge),
            "auc": _safe(ev_res.auc),
            "beats_base_rate": bool(beats),
            "reliable": reliable,
        }

    payload = {
        "ticker": ticker,
        "kind": DIRECTION_KIND,
        "horizon_days": horizon,
        "forecast": forecast,
        "evaluation": evaluation,
        "verdict": verdict,
        "caveat": (
            "This probability is shown for transparency. The evaluation beside it shows this "
            "kind of short-term forecast is not reliably better than chance."
        ),
    }
    _DIRECTION_CACHE[cache_key] = payload
    return payload


TRADING_DAYS = 252
_COMPARE_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _perf_metrics(close: pd.Series) -> Dict[str, Any]:
    """Buy-and-hold performance of one (dividend-adjusted) close series."""
    close = close.astype("float64").dropna()
    if len(close) < 30:
        return {}
    ret = close.pct_change().dropna()
    n = len(ret)
    total = float(close.iloc[-1] / close.iloc[0]) - 1.0
    years = n / TRADING_DAYS
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and (1.0 + total) > 0 else float("nan")
    sd = float(ret.std())
    vol = sd * (TRADING_DAYS ** 0.5)
    sharpe = (float(ret.mean()) / sd) * (TRADING_DAYS ** 0.5) if sd > 0 else float("nan")
    cum = (1.0 + ret).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "ann_return": cagr,
        "total_return": total,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_days": n,
    }


def _beta_to(stock_ret: pd.Series, spy_ret: pd.Series) -> Optional[float]:
    df = pd.concat([stock_ret, spy_ret], axis=1, join="inner").dropna()
    if len(df) < 30 or float(df.iloc[:, 1].var()) == 0.0:
        return None
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())


@app.get("/api/compare/{ticker}")
def get_compare(ticker: str, period: str = Query("5y")):
    """Neutral, educational comparison of holding this stock vs holding SPY.

    Returns objective buy-and-hold metrics for both over the same window. No
    recommendation is made (the difference is framed as risk, not skill).
    """
    ticker = ticker.upper()
    cache_key = (ticker, period)
    if cache_key in _COMPARE_CACHE:
        return _COMPARE_CACHE[cache_key]

    provider = YFinanceProvider()
    try:
        stock = provider.get_historical_data(ticker, period=period, interval="1d")
        spy = provider.get_historical_data("SPY", period=period, interval="1d")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not load prices: {exc}")
    if stock is None or len(stock) == 0:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    stock_m = _perf_metrics(stock["Close"])
    spy_m = _perf_metrics(spy["Close"])
    stock_ret = stock["Close"].astype("float64").pct_change().dropna()
    spy_ret = spy["Close"].astype("float64").pct_change().dropna()
    stock_m["beta"] = _beta_to(stock_ret, spy_ret)
    spy_m["beta"] = 1.0

    payload = {
        "ticker": ticker,
        "period": period,
        "stock": {k: _safe(v) for k, v in stock_m.items()},
        "spy": {k: _safe(v) for k, v in spy_m.items()},
        "note": (
            "Objective comparison over the same window. The difference comes mainly from "
            "risk: a higher return (if any) is accompanied by higher volatility and beta: "
            "the extra risk you would carry. SPY gives the market's average return with lower "
            "volatility and no stock-picking. This is educational information, not personal "
            "financial advice; whether that risk is worth it is your decision."
        ),
    }
    _COMPARE_CACHE[cache_key] = payload
    return payload

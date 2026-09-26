"""Plain-English names for the ranker's features, shared by the app and the report generators.

The feature columns are terse identifiers (``mom_12m``); anything a user or a report
reader sees should say what they mean, and both must say the same thing.
"""
from __future__ import annotations

from typing import Dict

FEATURE_LABELS: Dict[str, str] = {
    "mom_1m": "1-month return",
    "mom_3m": "3-month return",
    "mom_6m": "6-month return",
    "mom_12m": "12-1 month return (skips the last month)",
    "reversal_5d": "last 5 days' return",
    "mom_risk_adj": "12-1 momentum / volatility",
    "high_52w": "price / 52-week high",
    "volatility": "21-day volatility",
    "macd": "MACD",
    "adx": "ADX trend strength",
    "pe": "price / earnings",
    "pb": "price / book",
    "roe": "return on equity",
    "earnings_growth": "earnings growth (YoY)",
    "net_margin": "net margin",
    "debt_to_equity": "debt / equity",
    "sentiment": "news sentiment",
}


def label(feature: str) -> str:
    """Readable name for a feature column, falling back to the column itself."""
    return FEATURE_LABELS.get(feature, feature)

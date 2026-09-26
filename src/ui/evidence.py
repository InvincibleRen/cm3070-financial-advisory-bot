"""Evidence shown next to every signal and pick: how did this kind of advice actually do?

The advisor's job is not only to say Buy or Sell but to be honest about how much
that is worth. This module reads ``data/evidence/evidence.json`` (written by
``python -m src.cli.evidence_cli`` from the survivorship-free studies) and turns it
into plain sentences. It is UI-framework-free, so the web API and the tests use
the same wording.

The text is general, historical information about the tool's own signals, not
personal investment advice.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "evidence" / "evidence.json"

NOT_ADVICE = ("This is general, historical information about this tool, not personal "
              "financial advice.")

_SIGNAL_EVENT = {
    "watch / buy": "new_buy",
    "watch with caution": "new_buy",
    "buy": "new_buy",
    "sell / avoid": "new_sell",
    "sell": "new_sell",
}


def signal_event(signal: str) -> Optional[str]:
    """Study row for an advisor label: ``new_buy``, ``new_sell`` or ``None`` (Hold)."""
    return _SIGNAL_EVENT.get(signal.strip().lower())


def load_evidence(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The evidence file, or ``None`` if it has not been generated yet."""
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _pct(v: Optional[float], signed: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:+.2f}%" if signed else f"{v * 100:.1f}%"


def signal_evidence(signal: str, evidence: Optional[Dict[str, Any]]) -> List[str]:
    """Plain sentences on how the advisor's current kind of signal has performed.

    ``signal`` is the advisor's label (Buy / Sell / Hold, any case). Returns an
    empty list when no evidence file is available, so callers can skip the section.
    """
    if not evidence or "signals" not in evidence:
        return []
    sig = evidence["signals"]
    base = sig.get("all_member_days", {})
    event = signal_event(signal)
    scope = (f"Tested on every stock on the days it was in the S&P 500, "
             f"{sig.get('period', '')}.").strip()

    if event is None:
        b20 = base.get("20", {})
        lines = [
            f"A Hold signal makes no directional call. {scope}",
            f"For comparison, a typical S&P 500 stock beat the average member over the next "
            f"20 trading days {_pct(b20.get('beat_uni'), signed=False)} of the time.",
        ]
        if sig.get("verdict"):
            lines.append(sig["verdict"])
        return lines

    row = sig.get(event, {})
    r20, r60 = row.get("20", {}), row.get("60", {})
    b20 = base.get("20", {})
    kind = "Buy" if event == "new_buy" else "Sell"
    lines = [
        f"{scope} There were {int(r20.get('n', 0)):,} new {kind} signals.",
        f"Over the next 20 trading days, stocks with a new {kind} signal beat the average "
        f"S&P 500 member {_pct(r20.get('beat_uni'), signed=False)} of the time "
        f"(a typical stock: {_pct(b20.get('beat_uni'), signed=False)}), with an average "
        f"difference of {_pct(r20.get('x_uni'))}.",
        f"Over 60 trading days they did {_pct(r60.get('x_spy'))} versus SPY on average "
        f"(beat SPY {_pct(r60.get('beat_spy'), signed=False)} of the time).",
    ]
    verdict = sig.get("verdict")
    if verdict:
        lines.append(verdict)
    return lines


def index_reference(evidence: Optional[Dict[str, Any]], model: str = "XGBoost") -> str:
    """The index-fund comparison, worded from the evidence rather than assumed.

    Says the index did better only when the survivorship-free backtest shows the
    selector behind SPY on return or on risk-adjusted return (Sharpe).
    """
    m = model_evidence(evidence) or {}
    rows = m.get("unbiased", {}).get("rows", {})
    row, spy = rows.get(model), rows.get("SPY")
    if row and spy and (row.get("cagr", 0) < spy.get("cagr", 0)
                        or row.get("sharpe", 0) < spy.get("sharpe", 0)):
        return ("For reference: in the same survivorship-free test, simply holding a low-cost "
                "S&P 500 index fund (SPY) did better than the stock-selection model once risk "
                "is taken into account. That matches the broad research finding that, for most "
                "long-term investors, low-cost diversified index investing is hard to beat. "
                + NOT_ADVICE)
    return ("For reference: compare any signal with simply holding a low-cost S&P 500 index "
            "fund (SPY), the benchmark every result here is measured against. " + NOT_ADVICE)


def model_evidence(evidence: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Survivorship-free (and, for contrast, survivorship-biased) backtest of the selector."""
    if not evidence or "model" not in evidence:
        return None
    return evidence["model"]


def model_evidence_lines(
    evidence: Optional[Dict[str, Any]], model: str = "XGBoost", display_name: Optional[str] = None
) -> List[str]:
    """Sentences comparing the selector's honest backtest with SPY.

    ``model`` is the internal row label; ``display_name`` is what the user reads.
    """
    name = display_name or model
    m = model_evidence(evidence)
    if not m:
        return []
    unbiased = m.get("unbiased", {}).get("rows", {})
    biased = m.get("biased", {}).get("rows", {})
    row, spy = unbiased.get(model), unbiased.get("SPY")
    if not row or not spy:
        return []
    lines = [
        f"In a survivorship-free backtest ({m['unbiased'].get('period', '')}: each month only "
        f"the stocks actually in the S&P 500 then), {name} returned "
        f"{_pct(row.get('cagr'))} a year with a Sharpe ratio of {row.get('sharpe', 0):.2f}, "
        f"against {_pct(spy.get('cagr'))} and {spy.get('sharpe', 0):.2f} for SPY. Its "
        f"risk-adjusted alpha was {_pct(row.get('alpha'))} a year (t = {row.get('alpha_t', 0):.2f}; "
        "about 2 is needed to call it more than luck).",
    ]
    brow = biased.get(model)
    if brow:
        lines.append(
            f"Tested on today's S&P 500 list instead, the same selector shows {_pct(brow.get('cagr'))} "
            "a year. That figure is inflated: today's list only contains the companies that "
            "succeeded, which an investor could not have known in advance."
        )
    return lines


def pick_reason_text(reason: Dict[str, Any]) -> str:
    """One SHAP reason as a sentence, e.g. "6-month return (top 7% this month) raised its score"."""
    pct = reason.get("percentile")
    where = ""
    if pct is not None:
        if pct >= 0.995:
            where = " (the highest this month)"
        elif pct <= 0.005:
            where = " (the lowest this month)"
        elif pct >= 0.5:
            where = f" (top {max(1, round((1 - pct) * 100))}% this month)"
        else:
            where = f" (bottom {max(1, round(pct * 100))}% this month)"
    return f"{reason['label']}{where} {reason['direction']} its score"

"""Three questions that change what the advice page shows, and nothing it claims.

The system holds no information about a user, so the same five names are issued to
everyone. That is honest but blunt: a five-stock book is a different proposition for
someone saving for next year than for someone investing for a decade. This module
turns three answers into two presentation decisions - how many names to hold, and
whether the picks or the index-fund comparison leads the page - and a sentence
explaining why.

Every rule here follows from a measurement in this project rather than from a claim
about the user's suitability:

* **Diversification.** The sensitivity grid in the evaluation found the widest book
  (ten names) earned the highest Sharpe ratio of the three tested, so a user who
  says they want less risk is shown more names, not fewer.
* **Leading with the index.** The survivorship-free test found the selector behind a
  low-cost index fund on both return and Sharpe, so for a short horizon or a user
  uncomfortable with single stocks that comparison is put first.

What it deliberately does not do is claim that a particular stock suits a particular
person, or that the selector becomes reliable for anyone. The guidance is general
information about how to read this tool, not personal financial advice.
"""
from __future__ import annotations

from typing import Dict, List

RISK_LEVELS = ("low", "medium", "high")
HORIZONS = ("under_1y", "1_to_5y", "over_5y")
NAMES_BY_RISK = {"low": 10, "medium": 5, "high": 3}
DEFAULT_TOP_N = 5


def profile_guidance(
    risk: str = "medium", horizon: str = "1_to_5y", single_stocks: bool = True
) -> Dict[str, object]:
    """Presentation settings and the reasons for them, from three answers.

    ``risk`` is one of :data:`RISK_LEVELS`, ``horizon`` one of :data:`HORIZONS`, and
    ``single_stocks`` records whether the user is willing to hold individual shares at
    all. Unknown values fall back to the middle setting rather than raising, so a stale
    saved profile can never break the page.
    """
    risk = risk if risk in RISK_LEVELS else "medium"
    horizon = horizon if horizon in HORIZONS else "1_to_5y"

    top_n = NAMES_BY_RISK[risk]
    reasons: List[str] = []
    lead_with_index = False

    if not single_stocks:
        lead_with_index = True
        top_n = NAMES_BY_RISK["low"]
        reasons.append(
            "You said you would rather not hold individual shares, so the index-fund "
            "comparison is shown first and the stock picks are kept below it."
        )
    if horizon == "under_1y":
        lead_with_index = True
        reasons.append(
            "Over a horizon under a year, the month-to-month swings in this book are "
            "large relative to anything the model adds, so the index comparison leads."
        )
    if risk == "low":
        reasons.append(
            "Holding ten names rather than five is the more diversified setting, and it "
            "earned the best risk-adjusted result of the three tested in the evaluation."
        )
    elif risk == "high":
        reasons.append(
            "Holding three names concentrates the book. The evaluation found that the "
            "concentrated books carried the deepest drawdowns without ranking any better."
        )
    if not reasons:
        reasons.append(
            "These are the default settings: five names, picks first, with the track "
            "record beside them."
        )

    return {
        "top_n": top_n,
        "lead_with_index": lead_with_index,
        "reasons": reasons,
        "disclaimer": (
            "These answers change how this page is laid out and how many names are "
            "shown. They do not make the selector more reliable for you, and nothing "
            "here is personal financial advice."
        ),
    }

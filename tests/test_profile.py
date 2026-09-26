"""Tests for the three-question profile that tailors the advice page."""
import pytest

from src.ui.profile import DEFAULT_TOP_N, profile_guidance


def test_defaults_show_five_names_with_the_picks_first():
    g = profile_guidance()
    assert g["top_n"] == DEFAULT_TOP_N
    assert g["lead_with_index"] is False
    assert "default settings" in g["reasons"][0]


def test_lower_risk_is_answered_with_more_names_not_fewer():
    """The evaluation found the widest book had the best risk-adjusted result."""
    assert profile_guidance(risk="low")["top_n"] == 10
    assert profile_guidance(risk="medium")["top_n"] == 5
    assert profile_guidance(risk="high")["top_n"] == 3


def test_a_short_horizon_puts_the_index_comparison_first():
    g = profile_guidance(horizon="under_1y")
    assert g["lead_with_index"] is True
    assert any("under a year" in r for r in g["reasons"])


def test_declining_single_stocks_leads_with_the_index_and_widens_the_book():
    g = profile_guidance(risk="high", single_stocks=False)
    assert g["lead_with_index"] is True
    assert g["top_n"] == 10
    assert any("rather not hold individual shares" in r for r in g["reasons"])


def test_unknown_answers_fall_back_to_the_middle_setting():
    g = profile_guidance(risk="aggressive", horizon="forever")
    assert g["top_n"] == DEFAULT_TOP_N and g["lead_with_index"] is False


def test_every_profile_carries_the_not_advice_disclaimer():
    for risk in ("low", "medium", "high"):
        for horizon in ("under_1y", "1_to_5y", "over_5y"):
            g = profile_guidance(risk=risk, horizon=horizon)
            assert "personal financial advice" in g["disclaimer"]
            assert g["reasons"], "every profile explains why it looks the way it does"


def test_api_profile_endpoint():
    from fastapi.testclient import TestClient
    from src.api import main

    body = TestClient(main.app).get("/api/profile", params={"risk": "low", "horizon": "under_1y",
                                                           "single_stocks": "false"}).json()
    assert body["top_n"] == 10 and body["lead_with_index"] is True

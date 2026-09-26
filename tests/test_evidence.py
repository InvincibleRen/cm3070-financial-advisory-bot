"""Offline tests for the user-facing evidence layer (`ui/evidence.py`, `cli/evidence_cli.py`)."""
import json

import pytest

from src.cli import evidence_cli as C
from src.ui import evidence as E

MODEL = "xgb classifier (median label, top-N)"


def _stats(n, beat_uni, x_uni, beat_spy, x_spy):
    return {"n": n, "beat_uni": beat_uni, "x_uni": x_uni, "beat_spy": beat_spy, "x_spy": x_spy}


def _evidence(model_cagr=0.10, model_sharpe=0.50):
    return {
        "signals": {
            "period": "2015 to 2026",
            "new_buy": {"20": _stats(1000, 0.498, -0.0007, 0.48, -0.002),
                        "60": _stats(990, 0.481, -0.002, 0.461, -0.0082)},
            "new_sell": {"20": _stats(800, 0.494, -0.0001, 0.48, -0.002),
                         "60": _stats(790, 0.486, -0.0009, 0.47, -0.0047)},
            "all_member_days": {"20": _stats(10**6, 0.494, 0.0, 0.48, -0.001)},
            "verdict": "In this test, neither Buy nor Sell signals did meaningfully better.",
        },
        "model": {
            "unbiased": {"period": "2015 to latest", "rows": {
                MODEL: {"cagr": model_cagr, "sharpe": model_sharpe, "alpha": -0.019, "alpha_t": -0.36},
                "SPY": {"cagr": 0.142, "sharpe": 0.95}}},
            "biased": {"rows": {MODEL: {"cagr": 0.259, "sharpe": 0.93}}},
        },
    }


def test_buy_signal_evidence_states_the_numbers_and_the_verdict():
    lines = E.signal_evidence("Buy", _evidence())
    text = " ".join(lines)
    assert "1,000 new Buy signals" in text
    assert "49.8%" in text and "49.4%" in text
    assert "-0.82% versus SPY" in text
    assert "neither Buy nor Sell" in text


def test_hold_and_missing_evidence():
    hold = E.signal_evidence("hold", _evidence())
    assert "no directional call" in hold[0]
    assert hold[-1] == _evidence()["signals"]["verdict"]
    assert E.signal_evidence("Buy", None) == []
    assert E.model_evidence_lines(None) == []


def test_index_reference_only_claims_the_index_won_when_the_data_says_so():
    lost = E.index_reference(_evidence(model_cagr=0.10, model_sharpe=0.5), MODEL)
    assert "did better than the stock-selection model" in lost
    won = E.index_reference(_evidence(model_cagr=0.30, model_sharpe=1.5), MODEL)
    assert "did better" not in won
    assert "not personal financial advice" in lost and "not personal financial advice" in won


def test_model_lines_contrast_the_honest_and_the_inflated_backtest():
    lines = E.model_evidence_lines(_evidence(), MODEL, "this app's stock selector")
    assert "this app's stock selector returned +10.00% a year" in lines[0]
    assert "+14.20%" in lines[0] and "median label" not in lines[0]
    assert "+25.90%" in lines[1] and "inflated" in lines[1]


def test_load_evidence_round_trip(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_evidence()))
    assert E.load_evidence(path)["signals"]["period"] == "2015 to 2026"
    assert E.load_evidence(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert E.load_evidence(tmp_path / "bad.json") is None


_REPORT = """# Hurdle Selection
- History: 2015-01-01 to latest, 140 monthly rebalances

## 4. Portfolio (monthly, net of costs)

| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 8.86% | 0.48 | -41.86% | 99% | 5.0 | 1.84 |
| XGBoost | bil | 9.00% | 0.50 | -41.00% | 99% | 5.0 | 1.84 |
| SPY (buy & hold) | - | 14.19% | 0.95 | -23.93% | - | - | - |

## 5. Risk-adjusted: CAPM net of the risk-free rate

| Model | Fallback | Alpha / yr | t | Beta | t (beta vs 1) | R² | Alpha 1st half (t) | Alpha 2nd half (t) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | -3.42% | -0.59 | 1.04 | 0.39 | 0.48 | -12.48% (-2.11) | 5.36% (0.62) |

## 6. Year by year
"""


def test_parse_hurdle_report_reads_the_spy_fallback_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    path = tmp_path / "hurdle_historical_rank_20260101_000000.md"
    path.write_text(_REPORT)
    parsed = C.parse_hurdle_report(path)
    xgb, spy = parsed["rows"]["XGBoost"], parsed["rows"]["SPY"]
    assert xgb["cagr"] == pytest.approx(0.0886) and xgb["sharpe"] == pytest.approx(0.48)
    assert xgb["alpha"] == pytest.approx(-0.0342) and xgb["alpha_t"] == pytest.approx(-0.59)
    assert spy["cagr"] == pytest.approx(0.1419)
    assert parsed["period"].startswith("2015-01-01")


def test_latest_report_skips_tagged_partial_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPORTS", tmp_path)
    for name in ("hurdle_historical_rank_20260101_000000.md",
                 "hurdle_historical_rank_from2023_20260201_000000.md",
                 "hurdle_historical_rank_from2023_nofund_20260301_000000.md"):
        (tmp_path / name).write_text("x")
    assert C.latest_report("historical").name == "hurdle_historical_rank_20260101_000000.md"


def test_api_evidence_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from src.api import main

    monkeypatch.setattr(main.ev, "load_evidence", lambda path=None: _evidence())
    body = TestClient(main.app).get("/api/evidence", params={"signal": "Sell"}).json()
    assert body["available"] is True
    assert any("new Sell signals" in line for line in body["signal_lines"])
    assert body["model_lines"] and "not personal financial advice" in body["index_reference"]

    monkeypatch.setattr(main.ev, "load_evidence", lambda path=None: None)
    assert TestClient(main.app).get("/api/evidence").json()["available"] is False


@pytest.mark.parametrize("label,event", [
    ("Watch / Buy", "new_buy"), ("Watch with caution", "new_buy"),
    ("Sell / Avoid", "new_sell"), ("Hold / Watch", None), ("Hold", None),
])
def test_every_advisor_label_maps_to_the_study_row_for_its_score_band(label, event):
    assert E.signal_event(label) == event


def test_labels_the_advisor_actually_emits_are_all_covered():
    """Every label generate_recommendation can return must map deliberately."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "explanation" / "advisor.py").read_text()
    labels = set(re.findall(r'signal = "([^"]+)"', source))
    assert labels == {"Watch / Buy", "Hold / Watch", "Sell / Avoid", "Hold", "Watch with caution"}
    buys = {l for l in labels if E.signal_event(l) == "new_buy"}
    assert buys == {"Watch / Buy", "Watch with caution"}
    assert {l for l in labels if E.signal_event(l) == "new_sell"} == {"Sell / Avoid"}


def test_markdown_table_reader_handles_repeated_headers_and_numbers():
    body = """
text before
| Model | Win rate | Universe | Payoff | Universe |
|---|---:|---:|---:|---:|
| A | 49.8% | 49.4% | 1.20 | 1.08 |

**second**

| x | y |
|---|---|
| 1,234 | -0.07% (-1.22) |
"""
    first, second = C.tables(body)
    assert first[0]["Universe"] == "49.4%" and first[0]["Universe#2"] == "1.08"
    assert C.num(first[0]["Win rate"]) == pytest.approx(0.498)
    assert C.num(second[0]["x"]) == 1234
    assert C.num(second[0]["y"]) == pytest.approx(-0.0007)
    assert C.t_stat(second[0]["y"]) == pytest.approx(-1.22)
    assert C.num("n/a") is None and C.t_stat("+0.00% (n/a)") is None


_SIGNALS = """# Do the Advisor's Technical Signals Work?
- Member stock-days: 1,249,730 (2015-10-16 to 2026-09-21)

## 1. Results, 2015 onwards

| Signal | Hold (days) | Events | Days | Beat SPY | Mean vs SPY (t) | Beat same-day universe | Mean vs universe (t) | Mean vs beta x SPY (t) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| New Buy signal (the day it appears) | 20 | 32,813 | 2,634 | 48.5% | -0.24% (-2.08) | 49.8% | -0.07% (-1.22) | -0.22% (-1.83) |
| New Sell signal (the day it appears) | 20 | 24,748 | 2,592 | 48.2% | -0.21% (-0.48) | 49.4% | -0.01% (0.42) | -0.14% (-0.08) |
| Every member stock-day (reference) | 20 | 1,239,688 | 2,727 | 48.3% | -0.14% (-1.01) | 49.4% | +0.00% (n/a) | -0.13% (-1.03) |

## 2. Before and after 2020
"""


def test_signal_study_report_is_read_into_the_shape_the_app_uses(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    path = tmp_path / "signal_study_20260101_000000.md"
    path.write_text(_SIGNALS)
    sig = C.parse_signal_study(path)
    assert sig["period"] == "2015 to 2026"
    assert sig["new_buy"]["20"]["n"] == 32813
    assert sig["new_buy"]["20"]["t_uni"] == pytest.approx(-1.22)
    assert sig["all_member_days"]["20"]["t_uni"] is None
    assert "neither Buy nor Sell" in sig["verdict"]
    lines = E.signal_evidence("Watch / Buy", {"signals": sig})
    assert any("32,813 new Buy signals" in line for line in lines)


def test_api_research_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from src.api import main

    research = {"selectors": {"rows": [], "spy": {"cagr": 0.14}}, "signals": {"rows": []}}
    monkeypatch.setattr(main.ev, "load_evidence",
                        lambda path=None: {"generated_at": "2026-09-22 13:00", "research": research})
    body = TestClient(main.app).get("/api/research").json()
    assert body["available"] is True and body["selectors"]["spy"]["cagr"] == 0.14

    monkeypatch.setattr(main.ev, "load_evidence", lambda path=None: {"signals": {}})
    assert TestClient(main.app).get("/api/research").json() == {"available": False}

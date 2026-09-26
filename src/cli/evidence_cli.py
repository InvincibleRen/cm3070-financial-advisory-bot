"""Build ``data/evidence/evidence.json``: the honest track record the app shows users.

Every number is read from the generated reports in ``reports/`` (the same files
``reports/FINDINGS.md`` cites), never recomputed here, so the app always agrees with
the write-up even after the price cache is refreshed. Sections:

* ``signals`` - the advisor's Buy / Sell signals (``signal_study_*.md``);
* ``model``   - the app's selector, survivorship-free vs today's list (``hurdle_*.md``);
* ``research``- the tables behind the app's Evidence page: survivorship bias, the
  survivorship-free selectors, SHAP, the signal study, index events and the
  2023+ fundamentals comparison.

Usage::

    python -m src.cli.evidence_cli
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.common.signal_study import EVENT_LABELS
from src.ui.evidence import DEFAULT_PATH

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
_FULL_RUN = re.compile(r"^hurdle_(historical|current)_(rank|none|zscore)_\d{8}_\d{6}\.md$")
_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")
DISPLAY_NAMES = {
    "XGBoost": "XGBoost regressor",
    "Random forest": "Random forest regressor",
    "Ridge (auxiliary)": "Ridge regressor (baseline)",
    "xgb classifier (median label, top-N)": "XGBoost classifier (the app's selector)",
    "Rank on volatility alone (single factor)": "Single factor: 5 most volatile",
    "Rank on beta alone (single factor)": "Single factor: 5 highest-beta",
}



def section(text: str, heading: str) -> str:
    """Body of the ``## heading...`` section, up to the next ``## `` heading."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## ") and l[3:].startswith(heading)), None)
    if start is None:
        raise ValueError(f"section '{heading}' not found")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1:end])


def tables(body: str) -> List[List[Dict[str, str]]]:
    """Every pipe table in ``body`` as a list of row dicts (repeated headers get ``#2``, ``#3``)."""
    out: List[List[Dict[str, str]]] = []
    block: List[str] = []
    for line in body.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line.strip())
            continue
        if len(block) >= 2:
            header = _cells(block[0])
            keys, seen = [], {}
            for h in header:
                seen[h] = seen.get(h, 0) + 1
                keys.append(h if seen[h] == 1 else f"{h}#{seen[h]}")
            out.append([dict(zip(keys, _cells(r))) for r in block[2:]])
        block = []
    return out


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def num(cell: Optional[str]) -> Optional[float]:
    """First number in a cell; percentages become fractions. ``None`` if there is none."""
    if cell is None:
        return None
    m = _NUMBER.search(cell.replace("−", "-"))
    if not m:
        return None
    value = float(m.group(0).replace(",", ""))
    return round(value / 100, 8) if cell[m.end():m.end() + 1] == "%" else value


def t_stat(cell: Optional[str]) -> Optional[float]:
    """The number in parentheses, e.g. the t-stat in ``-0.07% (-1.22)``."""
    if cell is None:
        return None
    m = re.search(r"\(([^)]*)\)", cell)
    return num(m.group(1)) if m else None


def _optional_tables(text: str, heading: str) -> List[List[Dict[str, str]]]:
    try:
        return tables(section(text, heading))
    except ValueError:
        return []


def latest(pattern: str) -> Optional[Path]:
    runs = sorted(REPORTS.glob(pattern))
    return runs[-1] if runs else None


def latest_report(mode: str, normalize: str = "rank") -> Optional[Path]:
    runs = sorted(p for p in REPORTS.glob(f"hurdle_{mode}_{normalize}_*.md") if _FULL_RUN.match(p.name))
    return runs[-1] if runs else None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)



def parse_hurdle_report(path: Path) -> Dict[str, object]:
    """Per model: portfolio (fallback = spy), CAPM, beta-adjusted pick stats and IC; plus SPY."""
    text = path.read_text()
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    for r in tables(section(text, "4."))[0]:
        if r["Fallback"] == "spy" or r["Model"] == "SPY (buy & hold)":
            key = "SPY" if r["Model"] == "SPY (buy & hold)" else r["Model"]
            rows[key] = {"cagr": num(r["CAGR"]), "sharpe": num(r["Sharpe"]), "max_dd": num(r["Max DD"])}
    for r in tables(section(text, "5."))[0]:
        if r["Fallback"] == "spy" and r["Model"] in rows:
            rows[r["Model"]].update({"alpha": num(r["Alpha / yr"]), "alpha_t": num(r["t"]),
                                     "beta": num(r["Beta"])})
    picks = _optional_tables(text, "1.")
    if picks:
        for r in picks[0]:
            if r["Model"] in rows:
                rows[r["Model"]].update({
                    "expectancy": num(r.get("Expectancy / pick")),
                    "matched_random": num(r.get("Vol-matched random")),
                    "p_random": num(r.get("p (random)")),
                    "p_matched": num(r.get("p (vol-matched)")),
                    "vol_percentile": num(r.get("Vol percentile of picks")),
                    "win_rate": num(r.get("Win rate")),
                })
    for r in (_optional_tables(text, "3.")[:1] or [[]])[0]:
        if r["Model"] in rows:
            rows[r["Model"]].update({"ic": num(r["Mean IC"]), "ic_t": num(r["IC t-stat"])})
    period = re.search(r"- History: (.*)", text)
    start = re.search(r"- Walk-forward starts: (.*)", text)
    return {"source": _rel(path), "period": period.group(1) if period else "",
            "walk_forward_start": start.group(1) if start else "", "rows": rows}


def parse_signal_study(path: Path) -> Dict[str, object]:
    """``{event: {horizon: stats}}`` in the shape ``ui/evidence.py`` reads, plus a verdict."""
    text = path.read_text()
    by_label = {v: k for k, v in EVENT_LABELS.items()}
    out: Dict[str, object] = {"source": _rel(path)}
    for r in tables(section(text, "1."))[0]:
        event = by_label.get(r["Signal"])
        if event is None:
            continue
        out.setdefault(event, {})[str(int(num(r["Hold (days)"])))] = {
            "n": num(r["Events"]), "beat_spy": num(r["Beat SPY"]),
            "x_spy": num(r["Mean vs SPY (t)"]), "t_spy": t_stat(r["Mean vs SPY (t)"]),
            "beat_uni": num(r["Beat same-day universe"]),
            "x_uni": num(r["Mean vs universe (t)"]), "t_uni": t_stat(r["Mean vs universe (t)"]),
        }
    days = re.search(r"Member stock-days: [\d,]+ \((\d{4})-\d\d-\d\d to (\d{4})", text)
    out["period"] = f"{days.group(1)} to {days.group(2)}" if days else ""
    buy, sell = out["new_buy"]["20"], out["new_sell"]["20"]
    buy_works = (buy["x_uni"] or 0) > 0 and (buy["t_uni"] or 0) > 2
    sell_works = (sell["x_uni"] or 0) < 0 and (sell["t_uni"] or 0) < -2
    if buy_works or sell_works:
        which = " and ".join(n for n, ok in (("Buy", buy_works), ("Sell", sell_works)) if ok)
        out["verdict"] = f"In this test, {which} signals showed a statistically meaningful edge."
    else:
        out["verdict"] = ("In this test, neither Buy nor Sell signals did meaningfully better than "
                          "a typical stock: they carried no reliable information about what a "
                          "stock would do next.")
    return out


def parse_explain_report(path: Path) -> Dict[str, object]:
    text = path.read_text()
    rows = [{
        "feature": r["Feature"].strip("`"), "meaning": r["Meaning"],
        "reliance": num(r["Reliance"]), "direction": r["Direction learned"],
        "own_ic": num(r["Own IC (t)"]), "own_ic_t": t_stat(r["Own IC (t)"]),
        "backed": r["Backed by own IC?"], "coverage": num(r["Coverage"]),
    } for r in tables(section(text, "1."))[0]]
    return {"source": _rel(path), "rows": rows}


def parse_index_events(path: Path) -> Dict[str, object]:
    text = path.read_text()
    out: Dict[str, object] = {"source": _rel(path)}
    for key, heading in (("added", "1."), ("demoted", "3.")):
        out[key] = [{
            "window": r["Window (trading days)"], "n": num(r["Events"]),
            "beat_spy": num(r["Beat SPY"]), "mean": num(r["Mean excess"]),
            "median": num(r["Median excess"]), "t": num(r["t (by date)"]),
        } for r in tables(section(text, heading))[0]]
    return out


def parse_ml_eval(path: Path) -> Dict[str, object]:
    """The machine-learning evaluation report: out-of-sample R², learning curve,
    prediction stability and error breakdowns. All numbers, never recomputed."""
    text = path.read_text()
    out: Dict[str, object] = {"source": _rel(path)}
    out["r2"] = [{
        "model": r["Model"],
        "r2_oos": num(r["R² out of sample"]),
        "r2_is": num(r["R² in sample"]),
        "gap": num(r["Gap"]),
        "rmse": num(r["RMSE"]),
        "baseline_rmse": num(r["Baseline RMSE"]),
        "ic": num(r["Mean IC (t)"]),
        "ic_t": t_stat(r["Mean IC (t)"]),
        "folds": num(r["Folds"]),
        "verdict": r.get("Verdict"),
    } for r in tables(section(text, "1."))[0]]
    out["learning_curve"] = [{
        "window": r["Training window"],
        "r2_oos": num(r["R² out of sample"]),
        "r2_is": num(r["R² in sample"]),
        "gap": num(r["Gap"]),
        "ic": num(r["Mean IC (t)"]),
        "ic_t": t_stat(r["Mean IC (t)"]),
    } for r in tables(section(text, "2."))[0]]
    vals = re.findall(r"\*\*([+-]?\d*\.?\d+)\*\*", section(text, "3."))
    out["stability"] = {
        "pred_autocorr": float(vals[0]) if len(vals) > 0 else None,
        "target_autocorr": float(vals[1]) if len(vals) > 1 else None,
    }
    err = tables(section(text, "4."))
    if len(err) >= 1:
        out["by_year"] = [{"year": r["Year"], "r2_oos": num(r["R² out of sample"]),
                           "ic": num(r["IC"]), "rows": num(r["Rows"])} for r in err[0]]
    if len(err) >= 2:
        out["by_vol"] = [{"bucket": r["Bucket"], "r2_oos": num(r["R² out of sample"]),
                          "ic": num(r["IC"]), "rows": num(r["Rows"])} for r in err[1]]
    return out


def _display(rows: Dict[str, Dict], keep: Sequence[str]) -> List[Dict[str, object]]:
    return [{"model": DISPLAY_NAMES.get(m, m), **rows[m]} for m in keep if m in rows]


def research_section(hist: Path, cur: Optional[Path], explain: Optional[Path],
                     signals: Dict[str, object], events: Optional[Path],
                     fund: Optional[Path], nofund: Optional[Path],
                     ml_eval: Optional[Path] = None) -> Dict[str, object]:
    models = ["XGBoost", "Random forest", "Ridge (auxiliary)", "xgb classifier (median label, top-N)",
              "Rank on volatility alone (single factor)", "Rank on beta alone (single factor)"]
    h = parse_hurdle_report(hist)
    out: Dict[str, object] = {
        "selectors": {"source": h["source"], "period": h["period"],
                      "rows": _display(h["rows"], models), "spy": h["rows"].get("SPY")},
        "signals": {"source": signals["source"], "period": signals["period"],
                    "rows": [{"signal": EVENT_LABELS[e], "horizon": int(hz), **signals[e][hz]}
                             for e in EVENT_LABELS if e in signals for hz in sorted(signals[e], key=int)]},
    }
    if cur:
        c = parse_hurdle_report(cur)
        out["survivorship"] = {
            "sources": [c["source"], h["source"]],
            "rows": [{"model": DISPLAY_NAMES.get(m, m),
                      "current": {k: c["rows"][m].get(k) for k in ("cagr", "alpha", "alpha_t")},
                      "historical": {k: h["rows"][m].get(k) for k in ("cagr", "alpha", "alpha_t")}}
                     for m in models if m in c["rows"] and m in h["rows"]],
            "spy": h["rows"].get("SPY"),
        }
    if ml_eval:
        out["model_eval"] = parse_ml_eval(ml_eval)
    if explain:
        out["shap"] = parse_explain_report(explain)
    if events:
        out["index_events"] = parse_index_events(events)
    if fund and nofund:
        f, n = parse_hurdle_report(fund), parse_hurdle_report(nofund)
        out["fundamentals_2023"] = {
            "sources": [f["source"], n["source"]], "walk_forward_start": f["walk_forward_start"],
            "rows": [{"variant": variant, **row}
                     for variant, rep in (("With fundamentals", f), ("Technical only", n))
                     for row in _display(rep["rows"], ["XGBoost", "Random forest"])],
            "spy": f["rows"].get("SPY"),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    hist, cur = latest_report("historical"), latest_report("current")
    signal_report = latest("signal_study_*.md")
    if hist is None or signal_report is None:
        raise SystemExit("Need a survivorship-free hurdle report and a signal study report; "
                         "run hurdle_cli and signal_study_cli first.")
    signals = parse_signal_study(signal_report)
    lat = latest("latency_*.json")
    evidence = {
        "generated_at": f"{datetime.now():%Y-%m-%d %H:%M}",
        "signals": signals,
        **({"latency": json.loads(lat.read_text())} if lat else {}),
        "model": {"unbiased": parse_hurdle_report(hist),
                  **({"biased": parse_hurdle_report(cur)} if cur else {})},
        "research": research_section(
            hist, cur, latest("explain_xgb_historical_rank_*.md"), signals,
            latest("index_events_*.md"),
            latest("hurdle_historical_rank_from2023_2*.md"),
            latest("hurdle_historical_rank_from2023_nofund_*.md"),
            latest("ml_evaluation_historical_rank_*.md"),
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2))
    used = [hist, cur, signal_report]
    print("Evidence built from: " + ", ".join(p.name for p in used if p))
    print(f"Evidence written to {args.out}")
    return evidence


if __name__ == "__main__":
    main()

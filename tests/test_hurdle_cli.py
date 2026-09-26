"""Offline integration test for the hurdle-selection CLI.

The universe loaders are monkeypatched with synthetic in-memory data (noisy paths,
so beta and volatility are defined), and the whole chain runs without network:
load -> features -> excess-return target -> walk-forward regressors (+ the
classifier comparison) -> pick statistics -> portfolios -> report.
"""
import numpy as np
import pandas as pd
import pytest

from src.cli import hurdle_cli as C
from src.selection.features import FUNDAMENTAL_FEATURES, TECHNICAL_FEATURES
from src.selection import universe as U

_TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _synthetic_prices(tickers, start=None, end=None, **_):
    idx = pd.date_range("2015-01-01", periods=900, freq="B")
    rng = np.random.default_rng(0)
    market = rng.normal(0.0004, 0.01, len(idx))
    out = {}
    for i, t in enumerate(tickers):
        if t == "BIL":
            closes = 100.0 * np.cumprod(np.full(len(idx), 1.0001))
        elif t == "SPY":
            closes = 100.0 * np.cumprod(1.0 + market)
        else:
            beta = 0.6 + 0.2 * (i % 5)
            closes = 100.0 * np.cumprod(1.0 + beta * market + rng.normal(0, 0.01, len(idx)))
        out[t] = pd.DataFrame({"Close": closes}, index=idx)
    return out


def _synthetic_fundamentals(tickers=None, **_):
    frames = []
    for t in tickers or _TICKERS:
        periods = pd.date_range("2014-03-31", periods=16, freq="QE")
        frames.append(pd.DataFrame({
            "ticker": t, "period_end": periods, "available_date": periods + pd.Timedelta(days=60),
            "net_income": 10.0, "total_revenue": 100.0, "total_equity": 200.0,
            "total_debt": 100.0, "shares_outstanding": 1000.0,
            "roe": 0.2, "net_margin": 0.1, "debt_to_equity": 0.5, "earnings_growth_yoy": 0.05,
        }))
    return pd.concat(frames, ignore_index=True)


def test_hurdle_cli_runs_end_to_end_offline(monkeypatch):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)

    evals = C.main(_TICKERS + ["--no-save", "--jobs", "1", "--draws", "50",
                               "--max-names", "2", "--compare-classifier"])

    assert set(evals) == {"XGBoost", "Random forest", "Ridge (auxiliary)", C.CLASSIFIER_LABEL,
                          C.VOL_BASELINE, C.BETA_BASELINE}
    for ev in evals.values():
        assert set(ev.portfolios) == {"spy", "bil", "trend"}
        assert ev.pick_stats["beta_adjusted"]["picks"]["n"] > 0
        assert np.isfinite(ev.pick_stats["beta_adjusted"]["test"]["p_expectancy_matched"])
        assert 0.0 <= ev.months_invested <= 1.0
    assert evals[C.CLASSIFIER_LABEL].months_invested == 1.0

    closes = C.S._close_series_by_ticker(_synthetic_prices(_TICKERS + ["SPY", "BIL"]))
    report = C.build_markdown_report(
        evals, closes, {"Universe": "6 tickers"},
        {c: 1.0 for c in TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES},
        {"rows": 10.0, "mean": 0.0, "share_positive": 0.5},
    )
    assert "vol-matched" in report and C.VOL_BASELINE in report
    for section in ("The picks", "stricter selection", "information coefficient",
                    "Portfolio", "CAPM", "Year by year", "Latest decision", "Verdict"):
        assert section in report


def test_historical_universe_mode_only_trades_that_months_members(monkeypatch):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    snaps = pd.Series(
        [frozenset(_TICKERS[:5]), frozenset(_TICKERS)],
        index=pd.DatetimeIndex(["2014-01-01", "2017-01-01"]),
    )
    monkeypatch.setattr(C.K, "load_snapshots", lambda path=None: snaps)

    evals = C.main(["--no-save", "--jobs", "1", "--draws", "20", "--max-names", "2",
                    "--model", "ridge", "--fallback", "spy"])

    ev = evals["Ridge (auxiliary)"]
    early = [d for d, picks in ev.picks.items() if d < pd.Timestamp("2017-01-01")]
    assert early, "expected rebalances before FFF joined"
    assert all("FFF" not in ev.picks[d] for d in early)
    base = evals[C.VOL_BASELINE]
    assert all("FFF" not in base.picks[d] for d in early)


def test_first_rebalance_and_no_fundamentals_options(monkeypatch):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    inp = C.load_inputs(_TICKERS, "current", first_rebalance="2016-01-01", drop_fundamentals=True)
    assert min(inp.dates) >= pd.Timestamp("2016-01-01")
    assert inp.feature_matrix.index.get_level_values(0).min() >= pd.Timestamp("2016-01-01")
    assert not set(FUNDAMENTAL_FEATURES) & set(inp.feature_matrix.columns)
    first = inp.feature_matrix.xs(min(inp.dates), level=0)
    assert first["mom_6m"].notna().all()


@pytest.mark.parametrize("features,expected", [
    ("technical", set(TECHNICAL_FEATURES)),
    ("risk", {"beta", "volatility"}),
    ("technical,risk", set(TECHNICAL_FEATURES) | {"beta"}),
])
def test_feature_groups_select_what_the_model_may_see(monkeypatch, features, expected):
    """Each group can be run through the identical walk-forward, so they are comparable."""
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    inp = C.load_inputs(_TICKERS, "current", features=features)
    assert set(inp.feature_matrix.columns) == expected


def test_beta_feature_is_the_ex_ante_estimate(monkeypatch):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    inp = C.load_inputs(_TICKERS, "current", normalize="none", features="risk")
    pd.testing.assert_series_equal(
        inp.feature_matrix["beta"].dropna(),
        inp.risk["beta"].reindex(inp.feature_matrix.index).dropna(),
    )


def test_unknown_feature_group_is_refused(monkeypatch):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    with pytest.raises(SystemExit):
        C.load_inputs(_TICKERS, "current", features="technical,astrology")


def test_target_choice_switches_what_the_model_predicts(monkeypatch):
    """The default target removes market exposure; excess_spy leaves it in."""
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    risk_adj = C.load_inputs(_TICKERS, "current", features="risk")
    plain = C.load_inputs(_TICKERS, "current", features="risk", target_kind="excess_spy")

    assert risk_adj.target.name == "target" and plain.target.name == "excess_spy"
    assert not risk_adj.target.equals(plain.target)
    for t in (risk_adj.target, plain.target):
        by_date = t.groupby(level="rebalance_date")
        assert (t <= by_date.transform("max")).all()
    pd.testing.assert_series_equal(
        plain.target, plain.targets["excess_spy"].clip(
            lower=plain.targets["excess_spy"].groupby(level="rebalance_date").transform(
                lambda s: s.quantile(0.01)),
            upper=plain.targets["excess_spy"].groupby(level="rebalance_date").transform(
                lambda s: s.quantile(0.99))))


def test_unknown_target_is_refused(monkeypatch):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)
    with pytest.raises(SystemExit):
        C.load_inputs(_TICKERS, "current", target_kind="sharpe")

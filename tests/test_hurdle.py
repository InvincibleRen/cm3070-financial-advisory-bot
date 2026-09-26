"""Offline tests for hurdle selection (`selection/hurdle.py`).

Covers the picking rule (hurdle + cap, equal weight among the qualifiers), the
fallback asset when nothing qualifies, the leakage-safe trend rule, the trade
statistics (win rate / payoff ratio / expectancy), the CAPM regression, and an
end-to-end walk-forward on an engineered universe where one feature drives returns.
"""
import numpy as np
import pandas as pd
import pytest

from src.selection import hurdle as H

_D = list(pd.date_range("2020-01-31", periods=4, freq="ME"))


def _scores(**kw):
    return pd.Series(kw, dtype="float64")



def test_hurdle_picker_keeps_only_names_above_the_bar_best_first():
    pick = H.hurdle_picker(hurdle=0.0, max_names=5)
    assert pick(_scores(A=0.3, B=-0.1, C=0.5, D=0.0)) == ["C", "A"]


def test_hurdle_picker_caps_the_number_of_names():
    pick = H.hurdle_picker(hurdle=0.0, max_names=2)
    assert pick(_scores(A=0.3, B=0.1, C=0.5, D=0.2)) == ["C", "A"]


def test_top_k_picker_ignores_the_hurdle():
    assert H.top_k_picker(2)(_scores(A=-3.0, B=-1.0, C=-2.0)) == ["B", "C"]



def _closes(n_days=500, spy_drift=0.001):
    idx = pd.date_range("2019-01-01", periods=n_days, freq="D")
    return {
        "SPY": pd.Series(100.0 * np.exp(spy_drift * np.arange(n_days)), index=idx),
        "BIL": pd.Series(100.0 + 0.001 * np.arange(n_days), index=idx),
        "A": pd.Series(100.0 * np.exp(0.002 * np.arange(n_days)), index=idx),
        "B": pd.Series(np.full(n_days, 100.0), index=idx),
    }


def test_partial_qualification_puts_all_capital_in_the_qualifiers():
    w = H.build_weights({_D[0]: ["A", "B"]}, "spy", _closes())
    assert w.loc[_D[0]].to_dict() == {"A": 0.5, "B": 0.5}


@pytest.mark.parametrize("fallback,asset", [("spy", "SPY"), ("bil", "BIL")])
def test_no_qualifier_holds_the_fallback(fallback, asset):
    w = H.build_weights({_D[0]: []}, fallback, _closes())
    assert w.loc[_D[0]].to_dict() == {asset: 1.0}


def test_trend_fallback_follows_spy_versus_its_moving_average():
    rising = _closes(spy_drift=0.001)
    falling = _closes(spy_drift=-0.001)
    when = pd.Timestamp("2020-01-31")
    assert H.fallback_asset("trend", when, rising) == "SPY"
    assert H.fallback_asset("trend", when, falling) == "BIL"


def test_trend_rule_ignores_prices_after_the_decision_date():
    closes = _closes(spy_drift=0.001)
    when = pd.Timestamp("2019-12-31")
    before = H.trend_risk_on(closes["SPY"], when)
    crashed = closes["SPY"].copy()
    crashed[crashed.index > when] *= 0.1
    assert H.trend_risk_on(crashed, when) == before


def test_missing_fallback_prices_raise():
    closes = _closes()
    del closes["BIL"]
    with pytest.raises(ValueError):
        H.build_weights({_D[0]: []}, "bil", closes)


def test_portfolio_returns_charge_turnover():
    closes = _closes()
    w = H.build_weights({_D[0]: ["A"], _D[1]: ["A"], _D[2]: []}, "spy", closes)
    out = H.portfolio_returns(w, closes, cost_per_turnover=0.01)
    gross_first = closes["A"].asof(_D[1]) / closes["A"].asof(_D[0]) - 1
    assert out["returns"].loc[_D[0]] == pytest.approx(gross_first - 0.01)
    assert out["turnover"].loc[_D[1]] == pytest.approx(0.0)
    assert len(out["returns"]) == 2



def test_trade_statistics_by_hand():
    st = H.trade_statistics(np.array([0.04, 0.02, -0.01, -0.03]))
    assert st["win_rate"] == pytest.approx(0.5)
    assert st["avg_win"] == pytest.approx(0.03)
    assert st["avg_loss"] == pytest.approx(0.02)
    assert st["payoff_ratio"] == pytest.approx(1.5)
    assert st["breakeven_win_rate"] == pytest.approx(0.4)
    assert st["expectancy"] == pytest.approx(0.005)


def test_universe_baseline_is_weighted_by_picks_per_month():
    excess = {
        _D[0]: pd.Series({"A": 0.10, "B": -0.10}),
        _D[1]: pd.Series({"A": 0.02, "B": 0.02, "C": 0.02, "D": 0.02}),
    }
    picks = {_D[0]: ["A"], _D[1]: ["A", "B", "C"], _D[2]: []}
    st = H.pick_statistics(picks, excess, n_draws=50)
    assert st["universe"]["expectancy"] == pytest.approx((0.0 * 1 + 0.02 * 3) / 4)
    assert st["picks"]["n"] == 4
    assert st["test"]["n_months"] == 2


def test_permutation_test_flags_a_perfect_picker():
    rng = np.random.default_rng(0)
    dates = list(pd.date_range("2018-01-31", periods=40, freq="ME"))
    excess = {d: pd.Series(rng.normal(0, 0.05, 50), index=[f"T{i}" for i in range(50)])
              for d in dates}
    best = {d: list(s.sort_values(ascending=False).index[:3]) for d, s in excess.items()}
    worst = {d: list(s.sort_values().index[:3]) for d, s in excess.items()}
    assert H.pick_statistics(best, excess, n_draws=200)["test"]["p_expectancy"] < 0.01
    assert H.pick_statistics(worst, excess, n_draws=200)["test"]["p_expectancy"] > 0.99



def test_capm_recovers_alpha_and_beta_net_of_the_risk_free_rate():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    m = pd.Series(rng.normal(0.008, 0.04, 120), index=idx)
    rf = pd.Series(0.002, index=idx)
    s = rf + 0.005 + 1.5 * (m - rf) + pd.Series(rng.normal(0, 0.001, 120), index=idx)
    out = H.capm(s, m, rf)
    assert out["alpha_ann"] == pytest.approx(0.06, abs=0.005)
    assert out["beta"] == pytest.approx(1.5, abs=0.02)
    assert out["alpha_t"] > 5 and out["beta_t_vs_1"] > 5
    assert "alpha_t_second_half" in out


def test_capm_with_a_flat_market_is_undefined_not_an_error():
    idx = pd.date_range("2015-01-31", periods=24, freq="ME")
    out = H.capm(pd.Series(0.01, index=idx), pd.Series(0.0, index=idx))
    assert np.isnan(out["beta"])



def test_decomposition_identity_holds_exactly():
    """Arithmetic excess - variance drag + residual must reconstruct compounded excess."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-31", periods=140, freq="ME")
    market = pd.Series(rng.normal(0.010, 0.042, 140), index=idx)
    book = market * 1.1 + pd.Series(rng.normal(0.004, 0.075, 140), index=idx)
    d = H.return_decomposition(book, market, periods_per_year=12)
    assert (d["arithmetic_excess"] - d["variance_drag"] + d["residual"]
            == pytest.approx(d["geometric_excess"]))


def test_a_more_volatile_book_gives_more_back_to_variance_drag():
    """A positive average monthly excess can still compound below the index."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-31", periods=240, freq="ME")
    market = pd.Series(rng.normal(0.008, 0.030, 240), index=idx)
    book = market + pd.Series(rng.normal(0.002, 0.090, 240), index=idx)
    d = H.return_decomposition(book, market, periods_per_year=12)
    assert d["vol_strategy"] > d["vol_market"]
    assert d["variance_drag"] > 0
    assert d["arithmetic_excess"] > d["geometric_excess"]


def test_decomposition_of_a_book_identical_to_the_market_is_all_zero():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    market = pd.Series(np.random.default_rng(2).normal(0.01, 0.04, 60), index=idx)
    d = H.return_decomposition(market, market, periods_per_year=12)
    for key in ("arithmetic_excess", "geometric_excess", "variance_drag", "residual"):
        assert d[key] == pytest.approx(0.0, abs=1e-12)


def test_decomposition_needs_two_overlapping_points():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    d = H.return_decomposition(pd.Series([0.01], index=idx), pd.Series([0.02], index=idx))
    assert np.isnan(d["arithmetic_excess"]) and np.isnan(d["geometric_excess"])



def _engineered_universe(n_tickers=30, n_months=30, seed=0):
    """Each month one feature, ``signal``, is exactly each stock's next-month excess return."""
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2018-01-31", periods=n_months, freq="ME"))
    days = pd.date_range(dates[0], dates[-1], freq="D")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    closes = {"SPY": pd.Series(100.0, index=days), "BIL": pd.Series(100.0, index=days)}
    monthly = pd.DataFrame(rng.normal(0, 0.05, (n_months, n_tickers)), index=dates, columns=tickers)
    level = (1 + monthly).cumprod().shift(1).fillna(1.0) * 100
    for t in tickers:
        closes[t] = level[t].reindex(days, method="ffill")
    rows = {(d, t): {"signal": monthly.loc[d, t], "noise": rng.normal()}
            for d in dates for t in tickers}
    fm = pd.DataFrame.from_dict(rows, orient="index")
    fm.index = pd.MultiIndex.from_tuples(fm.index, names=["rebalance_date", "ticker"])
    target = fm["signal"].rename("target")
    return fm.sort_index(), target.sort_index(), dates, closes


def test_end_to_end_finds_the_engineered_signal():
    fm, target, dates, closes = _engineered_universe()
    preds = H.walk_forward_predictions(fm, target, dates, model_kind="ridge", min_train_dates=6)
    ev = H.evaluate("ridge", preds, H.hurdle_picker(0.0, 5), closes, target,
                    fallbacks=("spy", "bil", "trend"), cost_per_turnover=0.0, n_draws=200,
                    curve_draws=50)
    st = ev.pick_stats["raw"]
    p, u = st["picks"], st["universe"]
    assert p["win_rate"] > u["win_rate"] + 0.3
    assert p["expectancy"] > u["expectancy"]
    assert st["test"]["p_expectancy"] < 0.01
    assert ev.pick_stats["beta_adjusted"]["picks"]["expectancy"] == p["expectancy"]
    assert np.isnan(st["test"]["p_expectancy_matched"])
    assert ev.ic["mean_ic"] > 0.9
    assert ev.curve.loc[1, "expectancy"] >= ev.curve.loc[20, "expectancy"]
    assert set(ev.portfolios) == {"spy", "bil", "trend"}
    assert ev.portfolios["spy"].metrics["annualised_return"] > 0


def test_walk_forward_never_trains_on_the_scoring_date():
    fm, target, dates, _ = _engineered_universe(n_months=10)
    poisoned = target.copy()
    last = dates[-1]
    poisoned.loc[last] = 1e6
    clean = H.walk_forward_predictions(fm, target, dates, model_kind="ridge", min_train_dates=6)
    dirty = H.walk_forward_predictions(fm, poisoned, dates, model_kind="ridge", min_train_dates=6)
    pd.testing.assert_series_equal(clean[-1].scores, dirty[-1].scores)



def _vol_driven_month(rng, n=200):
    """Excess return rises with volatility, buried in realistic monthly noise (8%)."""
    tickers = [f"T{i:03d}" for i in range(n)]
    vol = pd.Series(np.linspace(0.02, 0.20, n), index=tickers)
    excess = pd.Series(0.3 * vol.to_numpy() + rng.normal(0, 0.08, n), index=tickers)
    return vol, excess


def _vol_world(score_by, seed, months=60):
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2018-01-31", periods=months, freq="ME"))
    preds, excess, vols = [], {}, {}
    for d in dates:
        vol, ex = _vol_driven_month(rng)
        preds.append(H.Prediction(d, vol if score_by == "vol" else ex))
        excess[d], vols[d] = ex, vol
    ranks = H.volatility_ranks(preds, pd.concat(vols, names=["rebalance_date", "ticker"]))
    return H.pick_statistics(H.picks_by_date(preds, H.top_k_picker(5)), excess, 300, 0, ranks)


def test_matched_null_removes_the_credit_for_holding_volatile_stocks():
    st = _vol_world("vol", seed=0)
    assert st["test"]["p_expectancy"] < 0.01
    assert st["test"]["p_expectancy_matched"] > 0.05
    assert st["test"]["pick_vol_percentile"] > 0.95


def test_matched_null_still_credits_real_skill():
    st = _vol_world("excess", seed=1)
    assert st["test"]["p_expectancy_matched"] < 0.01


def test_volatility_ranks_are_percentiles_per_date():
    d = pd.Timestamp("2020-01-31")
    scores = pd.Series(0.0, index=[f"T{i}" for i in range(10)] + ["NOVOL"])
    vols = pd.Series(np.arange(10, dtype=float), index=[f"T{i}" for i in range(10)])
    vols.index = pd.MultiIndex.from_product([[d], vols.index])
    r = H.volatility_ranks([H.Prediction(d, scores)], vols)[d]
    assert r["T0"] == pytest.approx(0.1) and r["T9"] == pytest.approx(1.0)
    assert np.isnan(r["NOVOL"])


def test_beta_adjusted_excess_subtracts_ex_ante_beta_times_spy():
    closes = _closes()
    d0, d1 = _D[0], _D[1]
    preds = [H.Prediction(d0, _scores(A=1.0)), H.Prediction(d1, _scores(A=1.0))]
    betas = pd.Series([2.0], index=pd.MultiIndex.from_tuples([(d0, "A")]))
    out = H.realised_excess_by_date(preds, closes, betas=betas)
    r_a = closes["A"].asof(d1) / closes["A"].asof(d0) - 1
    r_s = closes["SPY"].asof(d1) / closes["SPY"].asof(d0) - 1
    assert out["raw"][d0]["A"] == pytest.approx(r_a - r_s)
    assert out["beta_adjusted"][d0]["A"] == pytest.approx(r_a - 2.0 * r_s)

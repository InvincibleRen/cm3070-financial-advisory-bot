"""Offline tests for point-in-time S&P 500 membership (`selection/constituents.py`)."""
import numpy as np
import pandas as pd
import pytest

from src.selection import constituents as K
from src.selection import labels as L


def _snapshots(tmp_path):
    path = tmp_path / "members.csv"
    pd.DataFrame({
        "date": ["2015-01-02", "2016-06-15", "2020-03-01"],
        "tickers": ["AAA,BBB,BF.B,ANTM", "AAA,CCC,BF.B,ANTM", "AAA,CCC,BF.B,ELV"],
    }).to_csv(path, index=False)
    return K.load_snapshots(path)


def test_tickers_are_normalised_and_renames_aliased(tmp_path):
    snap = _snapshots(tmp_path)
    first = snap.iloc[0]
    assert "BF-B" in first and "BF.B" not in first
    assert "ELV" in first and "ANTM" not in first


def test_members_use_the_last_snapshot_on_or_before_each_date(tmp_path):
    snap = _snapshots(tmp_path)
    m = K.members_at(snap, [pd.Timestamp("2014-12-31"), pd.Timestamp("2016-06-14"),
                            pd.Timestamp("2016-06-15"), pd.Timestamp("2025-01-31")])
    assert m[pd.Timestamp("2014-12-31")] == set()
    assert "BBB" in m[pd.Timestamp("2016-06-14")]
    assert "BBB" not in m[pd.Timestamp("2016-06-15")] and "CCC" in m[pd.Timestamp("2016-06-15")]
    assert m[pd.Timestamp("2025-01-31")] == {"AAA", "CCC", "BF-B", "ELV"}


def test_historical_universe_is_the_union_over_the_dates(tmp_path):
    snap = _snapshots(tmp_path)
    dates = pd.date_range("2015-01-31", "2017-01-31", freq="ME")
    assert K.historical_universe(snap, dates) == ["AAA", "BBB", "BF-B", "CCC", "ELV"]


def test_filter_to_members_drops_non_members_per_date():
    d1, d2 = pd.Timestamp("2016-01-31"), pd.Timestamp("2017-01-31")
    idx = pd.MultiIndex.from_tuples([(d1, "AAA"), (d1, "CCC"), (d2, "AAA"), (d2, "CCC")])
    frame = pd.DataFrame({"x": [1, 2, 3, 4]}, index=idx)
    out = K.filter_to_members(frame, {d1: {"AAA"}, d2: {"AAA", "CCC"}})
    assert list(out.index) == [(d1, "AAA"), (d2, "AAA"), (d2, "CCC")]


def test_membership_coverage_counts_members_present_in_the_data():
    d = pd.Timestamp("2016-01-31")
    cov = K.membership_coverage({d: {"A", "B", "C", "D"}}, {d: {"A", "B", "C", "Z"}})
    assert cov.loc[d, "coverage"] == pytest.approx(0.75)


def test_every_alias_points_at_a_different_ticker():
    assert all(old != new for old, new in K.TICKER_ALIASES.items())
    assert not set(K.TICKER_ALIASES.values()) & set(K.TICKER_ALIASES)


def test_labels_and_targets_only_use_that_dates_members():
    idx = pd.date_range("2019-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    spy = pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 500))}, index=idx)
    prices = {t: pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, 500))},
                              index=idx) for t in ("IN1", "IN2", "IN3", "OUT")}
    t = pd.Timestamp("2020-06-30")
    members = {t: {"IN1", "IN2", "IN3"}}
    labels = L.make_labels(prices, [t], members=members)
    targets = L.make_excess_return_targets(prices, spy, [t], members=members)
    assert set(labels.index.get_level_values(1)) == {"IN1", "IN2", "IN3"}
    assert set(targets.index.get_level_values(1)) == {"IN1", "IN2", "IN3"}

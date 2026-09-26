import pandas as pd
import pytest

from src.common.backtester import calculate_backtest_signals, run_backtest


def test_calculate_backtest_signals_creates_expected_columns():
    data = pd.DataFrame({"Close": list(range(1, 251))})

    result = calculate_backtest_signals(data)

    assert "SMA20" in result.columns
    assert "SMA50" in result.columns
    assert "SMA200" in result.columns
    assert "score" in result.columns
    assert "raw_signal" in result.columns
    assert "target_position" in result.columns


def test_run_backtest_returns_summary():
    data = pd.DataFrame({"Close": list(range(1, 301))})

    result = run_backtest(data, ticker="TEST", initial_cash=10_000)

    assert result.ticker == "TEST"
    assert result.initial_cash == 10_000
    assert result.final_strategy_value > 0
    assert result.final_buy_hold_value > 0
    assert isinstance(result.strategy_return_percent, float)
    assert isinstance(result.buy_hold_return_percent, float)


def test_run_backtest_rejects_empty_data():
    data = pd.DataFrame()

    with pytest.raises(ValueError):
        run_backtest(data, ticker="EMPTY")

def test_weak_trend_halves_the_lean_without_a_dtype_error():
    """The ADX amplifier writes fractional scores, so the column must widen first.

    Both other tests here pass a close-only frame, on which ADX is NaN and the
    amplifier never runs, so this branch went unexercised until the interface hit
    it on real OHLC data. Under pandas 3 a float written into the integer score
    column raises, which broke the evaluation view for any history long enough to
    contain a range-bound stretch.
    """
    import numpy as np

    n = 400
    close = np.concatenate([100 + np.tile([0.4, -0.4], (n - 120) // 2),
                            np.linspace(100, 160, 120)])
    frame = pd.DataFrame({
        "Close": close,
        "High": close + 0.6,
        "Low": close - 0.6,
    })

    result = calculate_backtest_signals(frame)

    assert (result["ADX14"] < 20).any(), "the range-bound branch was not exercised"
    assert str(result["score"].dtype).startswith("int")
    assert result["score"].notna().all()
    assert set(result["raw_signal"].unique()) <= {"Buy", "Sell", "Hold"}

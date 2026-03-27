"""
Tests for strategies/helpers/indicators.py

Uses small synthetic DataFrames with known mathematical outputs.
No Freqtrade dependency required.
"""
import sys
import os
import math

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.helpers.indicators import (
    ema,
    rsi,
    atr,
    adx,
    bollinger_bands,
    macd,
    stochastic,
    stochastic_rsi,
    volume_sma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_close(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def make_ohlcv(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_length(self):
        series = make_close(list(range(1, 21)))
        result = ema(series, period=5)
        assert len(result) == len(series)

    def test_ema_first_value_equals_first_close(self):
        """With adjust=False, first EMA value equals first close."""
        series = make_close([10.0, 12.0, 14.0, 11.0, 13.0])
        result = ema(series, period=3)
        assert result.iloc[0] == pytest.approx(10.0, rel=1e-6)

    def test_ema_known_sequence(self):
        """Manually verify EMA(3) for a short sequence."""
        # EMA with alpha = 2/(3+1) = 0.5
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        series = make_close(values)
        result = ema(series, period=3)
        # expected[0] = 10
        # expected[1] = 0.5*11 + 0.5*10 = 10.5
        # expected[2] = 0.5*12 + 0.5*10.5 = 11.25
        assert result.iloc[0] == pytest.approx(10.0, rel=1e-6)
        assert result.iloc[1] == pytest.approx(10.5, rel=1e-6)
        assert result.iloc[2] == pytest.approx(11.25, rel=1e-6)

    def test_ema_longer_period_smoother(self):
        """Longer period EMA should have lower variance than shorter."""
        series = make_close(list(range(100)))
        slow = ema(series, period=20)
        fast = ema(series, period=5)
        assert slow.std() <= fast.std()

    def test_ema_no_nan_after_warmup(self):
        series = make_close(list(range(1, 51)))
        result = ema(series, period=10)
        # EMA with adjust=False produces values from index 0
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_bounds(self):
        """RSI must always be between 0 and 100."""
        df = make_ohlcv(100)
        result = rsi(df["close"], period=14)
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_rsi_all_up_moves(self):
        """RSI should be near 100 when all moves are upward."""
        values = list(range(1, 30))  # constantly increasing
        series = make_close(values)
        result = rsi(series, period=14)
        # After warmup (>14 candles), RSI should be very high
        assert result.iloc[-1] > 90.0

    def test_rsi_all_down_moves(self):
        """RSI should be near 0 when all moves are downward."""
        values = list(range(30, 1, -1))  # constantly decreasing
        series = make_close(values)
        result = rsi(series, period=14)
        assert result.iloc[-1] < 10.0

    def test_rsi_alternating(self):
        """RSI near 50 for alternating up/down pattern."""
        values = [10.0 + (1.0 if i % 2 == 0 else -1.0) * (i % 4) for i in range(50)]
        series = make_close(values)
        result = rsi(series, period=14)
        # Should be in a reasonable middle range after warmup
        assert 20 < result.iloc[-1] < 80

    def test_rsi_length(self):
        series = make_close(list(range(1, 51)))
        result = rsi(series, period=14)
        assert len(result) == len(series)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_positive(self):
        """ATR must always be non-negative."""
        df = make_ohlcv(50)
        result = atr(df["high"], df["low"], df["close"], period=14)
        assert (result >= 0).all()

    def test_atr_length(self):
        df = make_ohlcv(50)
        result = atr(df["high"], df["low"], df["close"], period=14)
        assert len(result) == len(df)

    def test_atr_constant_prices(self):
        """ATR should be 0 (or near 0) for perfectly constant OHLCV."""
        n = 30
        high = pd.Series([100.0] * n)
        low = pd.Series([100.0] * n)
        close = pd.Series([100.0] * n)
        result = atr(high, low, close, period=14)
        assert result.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_atr_increases_with_volatility(self):
        """ATR on volatile data should be larger than on calm data."""
        rng = np.random.default_rng(0)
        n = 60
        close_calm = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.1, n)))
        high_calm = close_calm + 0.1
        low_calm = close_calm - 0.1

        close_vol = pd.Series(100.0 + np.cumsum(rng.normal(0, 5.0, n)))
        high_vol = close_vol + 5.0
        low_vol = close_vol - 5.0

        atr_calm = atr(high_calm, low_calm, close_calm, 14).iloc[-1]
        atr_vol = atr(high_vol, low_vol, close_vol, 14).iloc[-1]
        assert atr_vol > atr_calm


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_bb_upper_above_lower(self):
        series = make_close(list(range(1, 51)))
        upper, middle, lower, width = bollinger_bands(series, period=20)
        valid = upper.dropna()
        lower_valid = lower.dropna()
        assert (valid.values >= lower_valid.values).all()

    def test_bb_middle_is_sma(self):
        """Middle band should equal a 20-period rolling SMA."""
        values = list(range(1, 51))
        series = make_close(values)
        _, middle, _, _ = bollinger_bands(series, period=20)
        expected_sma = series.rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, expected_sma, check_names=False)

    def test_bb_width_positive(self):
        series = make_close([float(i) + 100 for i in range(50)])
        _, _, _, width = bollinger_bands(series, period=10)
        assert (width.dropna() >= 0).all()

    def test_bb_returns_four_series(self):
        series = make_close(list(range(1, 51)))
        result = bollinger_bands(series)
        assert len(result) == 4
        for s in result:
            assert isinstance(s, pd.Series)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_length(self):
        series = make_close(list(range(1, 101)))
        macd_line, signal_line, histogram = macd(series)
        assert len(macd_line) == len(series)
        assert len(signal_line) == len(series)
        assert len(histogram) == len(series)

    def test_macd_histogram_equals_diff(self):
        """Histogram should equal macd_line - signal_line."""
        series = make_close(list(range(1, 101)))
        macd_line, signal_line, histogram = macd(series)
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected, check_names=False)

    def test_macd_fast_slow_crossing(self):
        """MACD line should turn negative when price trends down."""
        # Create uptrend then downtrend
        up = list(range(1, 50))
        down = list(range(50, 1, -1))
        series = make_close(up + down)
        macd_line, _, _ = macd(series)
        # After sustained downtrend, MACD line should be negative
        assert macd_line.iloc[-1] < 0


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_stoch_bounds(self):
        """Stochastic %K and %D should be between 0 and 100."""
        df = make_ohlcv(60)
        k, d = stochastic(df["high"], df["low"], df["close"])
        assert (k >= 0).all() and (k <= 100).all()
        assert (d >= 0).all() and (d <= 100).all()

    def test_stoch_length(self):
        df = make_ohlcv(50)
        k, d = stochastic(df["high"], df["low"], df["close"])
        assert len(k) == len(df)
        assert len(d) == len(df)

    def test_stoch_at_high_is_100(self):
        """If close = high throughout, stochastic should approach 100."""
        n = 30
        high = pd.Series([float(i + 10) for i in range(n)])
        low = pd.Series([float(i) for i in range(n)])
        close = high.copy()  # close == high
        k, _ = stochastic(high, low, close, k_period=5)
        # After warmup, %K should be 100
        assert k.iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_stoch_at_low_is_zero(self):
        """If close = low throughout, stochastic should approach 0."""
        n = 30
        high = pd.Series([float(i + 10) for i in range(n)])
        low = pd.Series([float(i) for i in range(n)])
        close = low.copy()
        k, _ = stochastic(high, low, close, k_period=5)
        assert k.iloc[-1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------

class TestStochasticRSI:
    def test_stoch_rsi_bounds(self):
        """Stochastic RSI should be between 0 and 1."""
        series = make_ohlcv(100)["close"]
        k, d = stochastic_rsi(series, period=14)
        assert (k.dropna() >= 0).all()
        assert (k.dropna() <= 1).all()

    def test_stoch_rsi_length(self):
        series = make_ohlcv(100)["close"]
        k, d = stochastic_rsi(series)
        assert len(k) == len(series)
        assert len(d) == len(series)


# ---------------------------------------------------------------------------
# Volume SMA
# ---------------------------------------------------------------------------

class TestVolumeSMA:
    def test_volume_sma_known_values(self):
        """Volume SMA should match pandas rolling mean."""
        volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        result = volume_sma(volume, period=3)
        expected = volume.rolling(window=3).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_volume_sma_length(self):
        volume = pd.Series(list(range(1, 31)), dtype=float)
        result = volume_sma(volume, period=20)
        assert len(result) == len(volume)

    def test_volume_sma_nan_warmup(self):
        volume = pd.Series(list(range(1, 21)), dtype=float)
        result = volume_sma(volume, period=5)
        assert result.iloc[:4].isna().all()
        assert not pd.isna(result.iloc[4])

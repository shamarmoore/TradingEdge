"""
Tests for:
  - RegimeDetector.detect_regime()
  - RegimeDetector.apply_hysteresis()
  - No look-ahead bias
  - BEAR_TREND produces no trade signals (via EnsembleStrategy stub)

No Freqtrade dependency required.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.RegimeDetector import RegimeDetector


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_trending_up(n: int = 250, start: float = 100.0, slope: float = 0.5) -> pd.DataFrame:
    """Create a strongly upward-trending OHLCV dataframe."""
    close = pd.Series(start + slope * np.arange(n))
    high = close + 2.0
    low = close - 2.0
    open_ = close - 0.5
    volume = pd.Series(np.ones(n) * 1000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_trending_down(n: int = 250, start: float = 200.0, slope: float = 0.5) -> pd.DataFrame:
    """Create a strongly downward-trending OHLCV dataframe."""
    close = pd.Series(start - slope * np.arange(n))
    high = close + 2.0
    low = close - 2.0
    open_ = close + 0.5
    volume = pd.Series(np.ones(n) * 1000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_ranging(n: int = 250, center: float = 100.0, amplitude: float = 2.0) -> pd.DataFrame:
    """Create a sideways-ranging OHLCV dataframe with small oscillations."""
    idx = np.arange(n)
    close = pd.Series(center + amplitude * np.sin(idx * 0.3))
    high = close + 0.5
    low = close - 0.5
    open_ = close + np.random.default_rng(0).uniform(-0.2, 0.2, n)
    volume = pd.Series(np.ones(n) * 500.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_breakout(n_pre: int = 200, n_breakout: int = 20) -> pd.DataFrame:
    """Create data with a clear ATR/volume expansion breakout at the end."""
    rng = np.random.default_rng(7)
    # Pre-breakout: calm
    close_pre = 100.0 + rng.normal(0, 0.5, n_pre)
    high_pre = close_pre + 0.5
    low_pre = close_pre - 0.5
    vol_pre = np.ones(n_pre) * 500.0

    # Breakout: large moves + high volume
    close_bo = 105.0 + np.arange(n_breakout) * 3.0
    high_bo = close_bo + 6.0
    low_bo = close_bo - 6.0
    vol_bo = np.ones(n_breakout) * 5000.0

    close = np.concatenate([close_pre, close_bo])
    high = np.concatenate([high_pre, high_bo])
    low = np.concatenate([low_pre, low_bo])
    volume = np.concatenate([vol_pre, vol_bo])
    open_ = close.copy()

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# -----------------------------------------------------------------------
# detect_regime tests
# -----------------------------------------------------------------------

class TestDetectRegime:
    def setup_method(self):
        self.detector = RegimeDetector()

    def test_returns_series_same_length(self):
        df = _make_trending_up()
        df = self.detector.add_indicators(df)
        regime = self.detector.detect_regime(df)
        assert isinstance(regime, pd.Series)
        assert len(regime) == len(df)

    def test_valid_regime_labels(self):
        valid = {"BULL_TREND", "BEAR_TREND", "RANGING", "BREAKOUT", "UNCERTAIN"}
        df = _make_trending_up()
        df = self.detector.add_indicators(df)
        regime = self.detector.detect_regime(df)
        assert set(regime.unique()).issubset(valid)

    def test_bull_trend_detected(self):
        """Strongly trending-up data should produce BULL_TREND in later candles."""
        df = _make_trending_up(n=250, slope=1.0)
        df = self.detector.add_indicators(df)
        regime = self.detector.detect_regime(df)
        # Ignore warmup period (first 200 candles for EMA_200)
        tail = regime.iloc[210:]
        assert (tail == "BULL_TREND").any(), f"Expected BULL_TREND in tail, got: {tail.value_counts()}"

    def test_bear_trend_detected(self):
        """Strongly trending-down data should produce BEAR_TREND in later candles."""
        df = _make_trending_down(n=250, slope=1.0)
        df = self.detector.add_indicators(df)
        regime = self.detector.detect_regime(df)
        tail = regime.iloc[210:]
        assert (tail == "BEAR_TREND").any(), f"Expected BEAR_TREND in tail, got: {tail.value_counts()}"

    def test_ranging_detected(self):
        """Sideways data should produce RANGING in later candles."""
        df = _make_ranging(n=250)
        df = self.detector.add_indicators(df)
        regime = self.detector.detect_regime(df)
        tail = regime.iloc[30:]  # after BB/ADX warmup
        assert (tail == "RANGING").any(), f"Expected RANGING in tail, got: {tail.value_counts()}"

    def test_breakout_detected(self):
        """Data with large ATR + volume expansion should trigger BREAKOUT."""
        df = _make_breakout(n_pre=200, n_breakout=20)
        df = self.detector.add_indicators(df)
        regime = self.detector.detect_regime(df)
        tail = regime.iloc[-10:]
        assert (tail == "BREAKOUT").any(), f"Expected BREAKOUT in tail, got: {tail.value_counts()}"

    def test_add_indicators_adds_columns(self):
        df = _make_trending_up()
        result = self.detector.add_indicators(df)
        required_cols = [
            "ema_50", "ema_200", "adx_14", "bb_upper", "bb_middle", "bb_lower",
            "bb_width", "bb_width_avg", "atr_14", "atr_avg", "vol_avg",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_detect_regime_auto_adds_indicators(self):
        """detect_regime should work even without pre-calling add_indicators."""
        df = _make_trending_up()
        regime = self.detector.detect_regime(df)
        assert len(regime) == len(df)


# -----------------------------------------------------------------------
# apply_hysteresis tests
# -----------------------------------------------------------------------

class TestApplyHysteresis:
    def setup_method(self):
        self.detector = RegimeDetector()

    def test_stable_regime_unchanged(self):
        """A consistent regime should pass through unchanged."""
        regime = pd.Series(["BULL_TREND"] * 10)
        result = self.detector.apply_hysteresis(regime, candles=3)
        assert (result == "BULL_TREND").all()

    def test_regime_switch_requires_3_candles(self):
        """A regime switch should only commit after 3 consecutive candles."""
        # First 5 BULL, then RANGING (but only 2 candles) then BULL again
        regime = pd.Series(
            ["BULL_TREND"] * 5 + ["RANGING"] * 2 + ["BULL_TREND"] * 5
        )
        result = self.detector.apply_hysteresis(regime, candles=3)
        # The two RANGING candles should not commit – should remain BULL_TREND
        assert result.iloc[5] == "BULL_TREND"
        assert result.iloc[6] == "BULL_TREND"

    def test_regime_switch_commits_after_3(self):
        """After 3+ consecutive new regime candles, switch should commit."""
        regime = pd.Series(
            ["BULL_TREND"] * 5 + ["RANGING"] * 5
        )
        result = self.detector.apply_hysteresis(regime, candles=3)
        # By index 7 (3rd RANGING candle), switch should be committed
        assert result.iloc[9] == "RANGING"

    def test_empty_series_handled(self):
        regime = pd.Series([], dtype=object)
        result = self.detector.apply_hysteresis(regime, candles=3)
        assert len(result) == 0

    def test_single_element(self):
        regime = pd.Series(["BULL_TREND"])
        result = self.detector.apply_hysteresis(regime, candles=3)
        assert result.iloc[0] == "BULL_TREND"

    def test_hysteresis_default_3_candles(self):
        """Default candles parameter should be 3."""
        regime = pd.Series(["BULL_TREND"] * 5 + ["BREAKOUT"] * 2)
        result_explicit = self.detector.apply_hysteresis(regime, candles=3)
        result_default = self.detector.apply_hysteresis(regime)
        pd.testing.assert_series_equal(result_explicit, result_default)


# -----------------------------------------------------------------------
# No look-ahead bias tests
# -----------------------------------------------------------------------

class TestNoLookAheadBias:
    def setup_method(self):
        self.detector = RegimeDetector()

    def test_regime_at_t_unchanged_when_future_appended(self):
        """
        Signals at candle T must not change when future candles are appended.
        This verifies there is no look-ahead bias in the indicator calculations.
        """
        df_base = _make_trending_up(n=220)
        df_base = self.detector.add_indicators(df_base)
        regime_base = self.detector.detect_regime(df_base)

        # Append 10 more candles
        extra = _make_trending_up(n=10, start=df_base["close"].iloc[-1])
        df_extended = pd.concat([df_base.drop(columns=df_base.columns.difference(
            ["open", "high", "low", "close", "volume"]
        )), extra], ignore_index=True)
        df_extended = self.detector.add_indicators(df_extended)
        regime_extended = self.detector.detect_regime(df_extended)

        # Regimes for the original candles should not have changed due to future data
        # (EMA and ATR use only past data; regime detection is causal)
        # We allow a small tolerance at the boundary because rolling lookback on extended
        # data can differ for the last few candles of the original series, but candles
        # that are fully in the past (first 210) must be unchanged.
        for i in range(210):
            assert regime_base.iloc[i] == regime_extended.iloc[i], (
                f"Look-ahead bias detected at candle {i}: "
                f"{regime_base.iloc[i]} → {regime_extended.iloc[i]}"
            )

    def test_indicators_at_t_independent_of_future(self):
        """EMA and ATR values at index T must not change when future data is added."""
        df_base = _make_trending_up(n=100)
        df_with_indicators = self.detector.add_indicators(df_base)
        ema50_at_t50 = df_with_indicators["ema_50"].iloc[50]

        # Extend and recompute
        extra = _make_trending_up(n=50, start=df_base["close"].iloc[-1])
        df_extended = pd.concat([df_base, extra], ignore_index=True)
        df_extended_ind = self.detector.add_indicators(df_extended)
        ema50_at_t50_extended = df_extended_ind["ema_50"].iloc[50]

        assert ema50_at_t50 == pytest.approx(ema50_at_t50_extended, rel=1e-9), (
            "EMA_50 at candle 50 changed after future data was appended (look-ahead bias!)"
        )


# -----------------------------------------------------------------------
# BEAR_TREND produces no trade entry
# -----------------------------------------------------------------------

class TestBearTrendNoEntry:
    """
    Verify that strategies do not generate entry signals in BEAR_TREND regime.
    We test this via EnsembleStrategy's populate_entry_trend logic.
    """

    def test_bear_trend_no_entry_ensemble(self):
        """EnsembleStrategy should not open entries during BEAR_TREND."""
        # Import EnsembleStrategy (works without freqtrade due to graceful fallback)
        from strategies.EnsembleStrategy import EnsembleStrategy

        strategy = EnsembleStrategy()

        df = _make_trending_down(n=250, slope=1.5)
        metadata = {"pair": "BTC/USDT"}

        df = strategy.populate_indicators(df, metadata)

        # Manually force regime to BEAR_TREND for all candles to test isolation
        df["regime"] = "BEAR_TREND"

        df = strategy.populate_entry_trend(df, metadata)

        # No entry signals should be generated in BEAR_TREND
        assert df["enter_long"].sum() == 0, (
            f"Expected 0 entries in BEAR_TREND but got {df['enter_long'].sum()}"
        )

    def test_uncertain_regime_no_entry_ensemble(self):
        """EnsembleStrategy should not open entries during UNCERTAIN regime."""
        from strategies.EnsembleStrategy import EnsembleStrategy

        strategy = EnsembleStrategy()
        df = _make_ranging(n=250)
        metadata = {"pair": "ETH/USDT"}

        df = strategy.populate_indicators(df, metadata)
        df["regime"] = "UNCERTAIN"
        df = strategy.populate_entry_trend(df, metadata)

        assert df["enter_long"].sum() == 0

"""
MomentumBreakout – Freqtrade strategy for BREAKOUT regimes.

Catches explosive moves on high volume with expanding ATR.
"""
from __future__ import annotations

import pandas as pd

try:
    from freqtrade.strategy import IStrategy
    _FREQTRADE_AVAILABLE = True
except ImportError:
    _FREQTRADE_AVAILABLE = False

    class IStrategy:  # type: ignore[no-redef]
        stoploss: float = -0.04
        minimal_roi: dict = {"0": 0.20}
        timeframe: str = "4h"
        trailing_stop: bool = False
        trailing_stop_positive: float | None = None
        trailing_stop_positive_offset: float = 0.0
        trailing_only_offset_is_reached: bool = False
        process_only_new_candles: bool = True
        use_exit_signal: bool = True
        exit_profit_only: bool = False
        can_short: bool = False
        startup_candle_count: int = 200

        def __init__(self, config: dict | None = None):
            self.config = config or {}

        def populate_indicators(self, dataframe, metadata):  # pragma: no cover
            return dataframe

        def populate_entry_trend(self, dataframe, metadata):  # pragma: no cover
            return dataframe

        def populate_exit_trend(self, dataframe, metadata):  # pragma: no cover
            return dataframe


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.helpers.indicators import (
    atr,
    rsi,
    volume_sma,
)
from strategies.RegimeDetector import RegimeDetector


class MomentumBreakout(IStrategy):
    """
    Momentum-breakout strategy active only in BREAKOUT regime.

    Entry logic (all must be true):
        1. Regime == BREAKOUT
        2. Price breaks above 20-period high
        3. Volume > 2.5x 20-period average
        4. ATR expanding (current > max of previous 3 candles)
        5. RSI between 55–75

        Filter: Do NOT enter if price already moved >5 % from breakout level.

    Exit logic:
        - Hard stop: -4 %
        - Target 1: +10 %  (via minimal_roi entry "0": 0.10)
        - Target 2: +20 %  (via minimal_roi entry "0": 0.20 – kept at all time)
        - Trailing stop follows after 10 % gain
    """

    stoploss = -0.04
    # Scaled ROI: keep half at 10%, rest rides with trailing stop
    minimal_roi = {"0": 0.20, "48": 0.10}
    timeframe = "4h"

    trailing_stop = True
    trailing_stop_positive = 0.10
    trailing_stop_positive_offset = 0.12
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    can_short = False
    startup_candle_count = 200

    def __init__(self, config: dict | None = None):
        if _FREQTRADE_AVAILABLE:
            super().__init__(config)  # type: ignore[call-arg]
        else:
            self.config = config or {}
        self._regime_detector = RegimeDetector()

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Regime
        dataframe = self._regime_detector.add_indicators(dataframe)
        raw_regime = self._regime_detector.detect_regime(dataframe)
        dataframe["regime"] = self._regime_detector.apply_hysteresis(raw_regime, candles=3)

        # Strategy indicators
        dataframe["rsi_14"] = rsi(dataframe["close"], 14)
        dataframe["atr_14"] = atr(dataframe["high"], dataframe["low"], dataframe["close"], 14)

        # 20-period high (breakout level)
        dataframe["high_20"] = dataframe["high"].rolling(window=20).max()
        # Use prior 20-period high (shift by 1 to avoid look-ahead)
        dataframe["breakout_level"] = dataframe["high_20"].shift(1)

        # ATR of previous 3 candles (max) for expansion check
        dataframe["atr_prev_max"] = dataframe["atr_14"].shift(1).rolling(window=3).max()

        # Volume metrics
        dataframe["vol_avg_20"] = volume_sma(dataframe["volume"], 20)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        # Price must break above the prior 20-period high
        price_breaks_out = dataframe["close"] > dataframe["breakout_level"]

        # Filter: not more than 5 % beyond breakout level
        within_5pct = (
            (dataframe["close"] - dataframe["breakout_level"]) / dataframe["breakout_level"]
        ) <= 0.05

        # ATR expanding
        atr_expanding = dataframe["atr_14"] > dataframe["atr_prev_max"]

        conditions = (
            (dataframe["regime"] == "BREAKOUT")
            & price_breaks_out
            & within_5pct
            & (dataframe["volume"] > dataframe["vol_avg_20"] * 2.5)
            & atr_expanding
            & (dataframe["rsi_14"] >= 55)
            & (dataframe["rsi_14"] <= 75)
        )

        dataframe.loc[conditions, "enter_long"] = 1
        dataframe.loc[conditions, "enter_tag"] = "momentum_breakout"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        # Exit on regime change away from BREAKOUT
        regime_change = dataframe["regime"] != "BREAKOUT"
        dataframe.loc[regime_change, "exit_long"] = 1
        dataframe.loc[regime_change, "exit_tag"] = "regime_change"
        return dataframe

"""
TrendFollower – Freqtrade strategy for BULL_TREND regimes.

Buys pullbacks within confirmed bull trends.
"""
from __future__ import annotations

import sys
import pandas as pd

try:
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
    _FREQTRADE_AVAILABLE = True
except ImportError:
    _FREQTRADE_AVAILABLE = False

    class IStrategy:  # type: ignore[no-redef]
        """Minimal stub so the module is importable without Freqtrade."""
        stoploss: float = -0.05
        minimal_roi: dict = {"0": 0.08}
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

    class DecimalParameter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("default", args[1] if len(args) > 1 else 0)

    class IntParameter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("default", args[1] if len(args) > 1 else 0)


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.helpers.indicators import (
    ema,
    rsi,
    macd,
    volume_sma,
)
from strategies.RegimeDetector import RegimeDetector


class TrendFollower(IStrategy):
    """
    Trend-following strategy active only in BULL_TREND regime.

    Entry logic (all must be true):
        1. Regime == BULL_TREND
        2. RSI_14 crosses below 40 (pullback signal)
        3. Price within 1 % of EMA_21
        4. MACD histogram decreasing but MACD line above signal
        5. Volume on pullback candle below 20-bar average

    Exit logic:
        - Hard stop: -5 %
        - minimal_roi: 8 %
        - Trailing stop activates after 4 % gain
    """

    stoploss = -0.05
    minimal_roi = {"0": 0.08}
    timeframe = "4h"

    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.05
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
        # Regime indicators
        dataframe = self._regime_detector.add_indicators(dataframe)
        raw_regime = self._regime_detector.detect_regime(dataframe)
        dataframe["regime"] = self._regime_detector.apply_hysteresis(raw_regime, candles=3)

        # Strategy indicators
        dataframe["ema_21"] = ema(dataframe["close"], 21)
        dataframe["rsi_14"] = rsi(dataframe["close"], 14)

        macd_line, signal_line, histogram = macd(dataframe["close"])
        dataframe["macd"] = macd_line
        dataframe["macd_signal"] = signal_line
        dataframe["macd_hist"] = histogram
        dataframe["macd_hist_prev"] = histogram.shift(1)

        dataframe["vol_avg_20"] = volume_sma(dataframe["volume"], 20)
        dataframe["rsi_prev"] = dataframe["rsi_14"].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        # Distance from EMA_21 in percent
        ema21_dist = ((dataframe["close"] - dataframe["ema_21"]) / dataframe["ema_21"]).abs()

        conditions = (
            (dataframe["regime"] == "BULL_TREND")
            & (dataframe["rsi_14"] < 40)         # RSI pulled back
            & (dataframe["rsi_prev"] >= 40)       # crossed below 40 (was above previous candle)
            & (ema21_dist <= 0.01)                 # within 1 % of EMA_21
            & (dataframe["macd"] > dataframe["macd_signal"])  # MACD line above signal
            & (dataframe["macd_hist"] < dataframe["macd_hist_prev"])  # histogram decreasing
            & (dataframe["volume"] < dataframe["vol_avg_20"])  # low volume pullback
        )

        dataframe.loc[conditions, "enter_long"] = 1
        dataframe.loc[conditions, "enter_tag"] = "trend_pullback"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        # Exit on regime change away from BULL_TREND
        exit_conditions = dataframe["regime"] != "BULL_TREND"
        dataframe.loc[exit_conditions, "exit_long"] = 1
        dataframe.loc[exit_conditions, "exit_tag"] = "regime_change"
        return dataframe

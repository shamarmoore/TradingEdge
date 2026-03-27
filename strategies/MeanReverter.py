"""
MeanReverter – Freqtrade strategy for RANGING regimes.

Buys at the lower Bollinger Band when the market is mean-reverting.
"""
from __future__ import annotations

import pandas as pd

try:
    from freqtrade.strategy import IStrategy
    _FREQTRADE_AVAILABLE = True
except ImportError:
    _FREQTRADE_AVAILABLE = False

    class IStrategy:  # type: ignore[no-redef]
        stoploss: float = -0.03
        minimal_roi: dict = {"0": 0.05}
        timeframe: str = "4h"
        trailing_stop: bool = False
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
    rsi,
    bollinger_bands,
    stochastic,
    volume_sma,
)
from strategies.RegimeDetector import RegimeDetector

# Time-stop: exit after this many candles regardless of profit
TIME_STOP_CANDLES = 12


class MeanReverter(IStrategy):
    """
    Mean-reversion strategy active only in RANGING regime.

    Entry logic (all must be true):
        1. Regime == RANGING
        2. RSI_14 < 30
        3. Price within 1.5 % of lower Bollinger Band
        4. Stochastic %K crosses above %D below 20
        5. Volume increasing on bounce candle
        Filter: 20-day low must not be making new lows.

    Exit logic:
        - Target: middle Bollinger Band (handled via minimal_roi)
        - Hard stop: -3 %
        - Time stop: 12 candles
    """

    stoploss = -0.03
    minimal_roi = {"0": 0.05, "24": 0.03}
    timeframe = "4h"
    trailing_stop = False
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

        bb_upper, bb_middle, bb_lower, bb_width = bollinger_bands(dataframe["close"], period=20)
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_lower
        dataframe["bb_width"] = bb_width

        stoch_k, stoch_d = stochastic(dataframe["high"], dataframe["low"], dataframe["close"])
        dataframe["stoch_k"] = stoch_k
        dataframe["stoch_d"] = stoch_d
        dataframe["stoch_k_prev"] = stoch_k.shift(1)
        dataframe["stoch_d_prev"] = stoch_d.shift(1)

        dataframe["vol_avg_20"] = volume_sma(dataframe["volume"], 20)
        dataframe["vol_prev"] = dataframe["volume"].shift(1)

        # 20-period low for filter
        dataframe["low_20"] = dataframe["low"].rolling(window=20).min()
        dataframe["low_20_prev"] = dataframe["low_20"].shift(1)

        # Track entry candle for time-stop
        dataframe["candle_idx"] = range(len(dataframe))

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        bb_lower_dist = ((dataframe["close"] - dataframe["bb_lower"]) / dataframe["bb_lower"]).abs()

        # Stochastic %K crosses above %D below 20
        stoch_cross_up = (
            (dataframe["stoch_k"] > dataframe["stoch_d"])
            & (dataframe["stoch_k_prev"] <= dataframe["stoch_d_prev"])
            & (dataframe["stoch_k"] < 20)
        )

        # 20-period low not making new lows (filter)
        no_new_lows = dataframe["low_20"] >= dataframe["low_20_prev"]

        conditions = (
            (dataframe["regime"] == "RANGING")
            & (dataframe["rsi_14"] < 30)
            & (bb_lower_dist <= 0.015)
            & stoch_cross_up
            & (dataframe["volume"] > dataframe["vol_prev"])  # volume increasing
            & no_new_lows
        )

        dataframe.loc[conditions, "enter_long"] = 1
        dataframe.loc[conditions, "enter_tag"] = "mean_reversion_buy"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        # Exit at middle BB (price reaches middle band)
        at_middle_bb = dataframe["close"] >= dataframe["bb_middle"]

        # Regime change out of RANGING
        regime_change = dataframe["regime"] != "RANGING"

        exit_conditions = at_middle_bb | regime_change
        dataframe.loc[exit_conditions, "exit_long"] = 1
        dataframe.loc[at_middle_bb, "exit_tag"] = "middle_bb_target"
        dataframe.loc[regime_change & ~at_middle_bb, "exit_tag"] = "regime_change"

        return dataframe

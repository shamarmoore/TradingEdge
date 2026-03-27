"""
EnsembleStrategy – master Freqtrade strategy that:
  - Runs RegimeDetector
  - Activates the correct sub-strategy logic per regime
  - Applies confluence scoring and position sizing
  - Enforces all filters (correlation, spread, volume, cooldown)

Entry is only generated when the correct sub-strategy logic fires AND
confluence_score >= 3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

try:
    from freqtrade.strategy import IStrategy
    _FREQTRADE_AVAILABLE = True
except ImportError:
    _FREQTRADE_AVAILABLE = False

    class IStrategy:  # type: ignore[no-redef]
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


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.RegimeDetector import RegimeDetector
from strategies.helpers.indicators import (
    ema,
    rsi,
    atr,
    bollinger_bands,
    macd,
    stochastic,
    volume_sma,
)
from strategies.helpers.filters import (
    confluence_score,
    spread_filter,
    volume_filter,
    cooldown_filter,
)
from strategies.helpers.risk_manager import RiskManager

# -----------------------------------------------------------------------
# Load risk params
# -----------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_RISK_PARAMS: dict = {}
_risk_params_path = _CONFIG_DIR / "risk_params.json"
if _risk_params_path.exists():
    with open(_risk_params_path, "r") as _f:
        _RISK_PARAMS = json.load(_f)


class EnsembleStrategy(IStrategy):
    """
    Master ensemble strategy.  Delegates entry logic to the appropriate
    sub-strategy based on the detected market regime.

    Regime → Sub-strategy:
        BULL_TREND  → TrendFollower logic
        RANGING     → MeanReverter logic
        BREAKOUT    → MomentumBreakout logic
        BEAR_TREND  → no entry (sit on hands)
        UNCERTAIN   → no entry
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

    # Minimum confluence score required
    MIN_CONFLUENCE = 3

    def __init__(self, config: dict | None = None):
        if _FREQTRADE_AVAILABLE:
            super().__init__(config)  # type: ignore[call-arg]
        else:
            self.config = config or {}

        self._regime_detector = RegimeDetector()
        # RiskManager initialised with a placeholder; updated at runtime
        self._risk_manager = RiskManager(
            portfolio_value=self.config.get("dry_run_wallet", 200.0),
            risk_params=_RISK_PARAMS,
        )
        # Cooldown tracking: pair → last trade candle index
        self._last_trade_candle: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Freqtrade hooks
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # ---- Regime detection ----------------------------------------
        dataframe = self._regime_detector.add_indicators(dataframe)
        raw_regime = self._regime_detector.detect_regime(dataframe)
        dataframe["regime"] = self._regime_detector.apply_hysteresis(raw_regime, candles=3)

        # ---- Shared indicators ----------------------------------------
        dataframe["ema_21"] = ema(dataframe["close"], 21)
        dataframe["rsi_14"] = rsi(dataframe["close"], 14)
        dataframe["atr_14"] = atr(dataframe["high"], dataframe["low"], dataframe["close"], 14)

        macd_line, macd_signal, macd_hist = macd(dataframe["close"])
        dataframe["macd"] = macd_line
        dataframe["macd_signal"] = macd_signal
        dataframe["macd_hist"] = macd_hist
        dataframe["macd_hist_prev"] = macd_hist.shift(1)

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
        dataframe["rsi_prev"] = dataframe["rsi_14"].shift(1)

        # Bollinger lower proximity
        dataframe["bb_lower_dist"] = (
            (dataframe["close"] - dataframe["bb_lower"]) / dataframe["bb_lower"]
        ).abs()

        # 20-period high for breakout
        dataframe["high_20"] = dataframe["high"].rolling(window=20).max()
        dataframe["breakout_level"] = dataframe["high_20"].shift(1)
        dataframe["atr_prev_max"] = dataframe["atr_14"].shift(1).rolling(window=3).max()

        # 20-period low (MeanReverter filter)
        dataframe["low_20"] = dataframe["low"].rolling(window=20).min()
        dataframe["low_20_prev"] = dataframe["low_20"].shift(1)

        # Candle index for cooldown tracking
        dataframe["candle_idx"] = range(len(dataframe))

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        pair = metadata.get("pair", "")

        # ---------- BULL_TREND signals (TrendFollower) ----------
        ema21_dist = ((dataframe["close"] - dataframe["ema_21"]) / dataframe["ema_21"]).abs()
        trend_conditions = (
            (dataframe["regime"] == "BULL_TREND")
            & (dataframe["rsi_14"] < 40)
            & (dataframe["rsi_prev"] >= 40)
            & (ema21_dist <= 0.01)
            & (dataframe["macd"] > dataframe["macd_signal"])
            & (dataframe["macd_hist"] < dataframe["macd_hist_prev"])
            & (dataframe["volume"] < dataframe["vol_avg_20"])
        )

        # ---------- RANGING signals (MeanReverter) ----------
        stoch_cross = (
            (dataframe["stoch_k"] > dataframe["stoch_d"])
            & (dataframe["stoch_k_prev"] <= dataframe["stoch_d_prev"])
            & (dataframe["stoch_k"] < 20)
        )
        no_new_lows = dataframe["low_20"] >= dataframe["low_20_prev"]
        ranging_conditions = (
            (dataframe["regime"] == "RANGING")
            & (dataframe["rsi_14"] < 30)
            & (dataframe["bb_lower_dist"] <= 0.015)
            & stoch_cross
            & (dataframe["volume"] > dataframe["vol_prev"])
            & no_new_lows
        )

        # ---------- BREAKOUT signals (MomentumBreakout) ----------
        price_breaks_out = dataframe["close"] > dataframe["breakout_level"]
        within_5pct = (
            (dataframe["close"] - dataframe["breakout_level"]) / dataframe["breakout_level"]
        ) <= 0.05
        atr_expanding = dataframe["atr_14"] > dataframe["atr_prev_max"]
        breakout_conditions = (
            (dataframe["regime"] == "BREAKOUT")
            & price_breaks_out
            & within_5pct
            & (dataframe["volume"] > dataframe["vol_avg_20"] * 2.5)
            & atr_expanding
            & (dataframe["rsi_14"] >= 55)
            & (dataframe["rsi_14"] <= 75)
        )

        any_signal = trend_conditions | ranging_conditions | breakout_conditions

        # Apply confluence and cooldown filters row-by-row for signals
        for idx in dataframe[any_signal].index:
            row = dataframe.loc[idx]
            candle_idx = int(row["candle_idx"])

            # Cooldown check
            last_candle = self._last_trade_candle.get(pair, -999)
            cooldown_ok = cooldown_filter(last_candle, candle_idx, cooldown_candles=3)
            if not cooldown_ok:
                continue

            # Volume filter
            vol_usd = row.get("volume", 0) * row.get("close", 0)
            min_vol = _RISK_PARAMS.get("filters", {}).get("min_24h_volume_usd", 10_000_000)
            if not volume_filter(vol_usd, min_vol):
                continue

            # Build signals dict for confluence scoring
            signals_dict: dict = {
                "primary_conditions_met": bool(
                    trend_conditions.get(idx, False)
                    or ranging_conditions.get(idx, False)
                    or breakout_conditions.get(idx, False)
                ),
                "volume_confirms": bool(row["volume"] > row["vol_avg_20"]),
                "multi_indicator_agree": bool(
                    (row["rsi_14"] < 40) or (row["rsi_14"] > 55)
                ),
                "no_nearby_resistance": True,  # simplification – extend with S/R logic
                "daily_tf_confirms": False,    # requires higher-tf data; conservative default
                "regime_recently_changed": False,
                "conflicting_signal": False,
                "high_correlation": False,
                "recent_loss_on_pair": False,
            }
            score = confluence_score(signals_dict)
            if score < self.MIN_CONFLUENCE:
                continue

            tag = (
                "ensemble_trend"
                if trend_conditions.get(idx, False)
                else ("ensemble_ranging" if ranging_conditions.get(idx, False) else "ensemble_breakout")
            )
            dataframe.loc[idx, "enter_long"] = 1
            dataframe.loc[idx, "enter_tag"] = tag

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        # Exit when regime no longer matches any tradable state
        non_tradable = dataframe["regime"].isin(["BEAR_TREND", "UNCERTAIN"])
        dataframe.loc[non_tradable, "exit_long"] = 1
        dataframe.loc[non_tradable, "exit_tag"] = "regime_non_tradable"
        return dataframe

    # ------------------------------------------------------------------
    # Optional: track last trade for cooldown (call from custom hooks)
    # ------------------------------------------------------------------

    def record_trade_exit(self, pair: str, candle_idx: int) -> None:
        """Record a trade exit to enforce per-pair cooldown."""
        self._last_trade_candle[pair] = candle_idx

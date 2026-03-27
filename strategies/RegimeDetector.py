"""
RegimeDetector – standalone utility class (NOT a Freqtrade strategy).

Classifies each candle into one of five market regimes:
    BULL_TREND, BEAR_TREND, RANGING, BREAKOUT, UNCERTAIN
"""
from __future__ import annotations

import pandas as pd

# Local helpers – no Freqtrade dependency
from strategies.helpers.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    volume_sma,
)


class RegimeDetector:
    """
    Detect market regime from OHLCV data.

    Parameters
    ----------
    adx_trend_threshold : float
        ADX level above which a trend is considered strong (default 25).
    adx_ranging_threshold : float
        ADX level below which the market is considered ranging (default 20).
    breakout_atr_multiplier : float
        ATR must exceed ATR_avg * this factor for a breakout (default 1.5).
    breakout_volume_multiplier : float
        Volume must exceed VOL_avg * this factor for a breakout (default 2.0).
    hysteresis_candles : int
        Candles a new regime must persist before being confirmed (default 3).
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        breakout_atr_multiplier: float = 1.5,
        breakout_volume_multiplier: float = 2.0,
        hysteresis_candles: int = 3,
    ) -> None:
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_ranging_threshold = adx_ranging_threshold
        self.breakout_atr_multiplier = breakout_atr_multiplier
        self.breakout_volume_multiplier = breakout_volume_multiplier
        self.hysteresis_candles = hysteresis_candles

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def add_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add all regime indicator columns to *dataframe* (in-place copy).

        Required input columns: open, high, low, close, volume.

        Added columns:
            ema_50, ema_200, adx_14, bb_upper, bb_middle, bb_lower,
            bb_width, bb_width_avg, atr_14, atr_avg, vol_avg
        """
        df = dataframe.copy()

        df["ema_50"] = ema(df["close"], 50)
        df["ema_200"] = ema(df["close"], 200)
        df["adx_14"] = adx(df["high"], df["low"], df["close"], period=14)

        bb_upper, bb_middle, bb_lower, bb_width = bollinger_bands(df["close"], period=20, std_dev=2.0)
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["bb_width"] = bb_width
        df["bb_width_avg"] = bb_width.rolling(window=20).mean()

        df["atr_14"] = atr(df["high"], df["low"], df["close"], period=14)
        df["atr_avg"] = volume_sma(df["atr_14"], period=20)  # 20-period SMA of ATR
        df["vol_avg"] = volume_sma(df["volume"], period=20)

        return df

    def detect_regime(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Classify each candle into a market regime.

        The dataframe must already contain the indicator columns added by
        :meth:`add_indicators`.  If they are missing, they will be added
        automatically.

        Returns
        -------
        pd.Series of str
            One of: "BULL_TREND", "BEAR_TREND", "RANGING",
                    "BREAKOUT", "UNCERTAIN"
        """
        required = {"ema_50", "ema_200", "adx_14", "bb_width", "bb_width_avg", "atr_14", "atr_avg", "vol_avg"}
        if not required.issubset(dataframe.columns):
            dataframe = self.add_indicators(dataframe)

        df = dataframe

        regimes = pd.Series("UNCERTAIN", index=df.index, dtype=object)

        # BULL_TREND: close > EMA_50 > EMA_200 and ADX > threshold
        bull = (
            (df["close"] > df["ema_50"])
            & (df["ema_50"] > df["ema_200"])
            & (df["adx_14"] > self.adx_trend_threshold)
        )

        # BEAR_TREND: close < EMA_50 < EMA_200 and ADX > threshold
        bear = (
            (df["close"] < df["ema_50"])
            & (df["ema_50"] < df["ema_200"])
            & (df["adx_14"] > self.adx_trend_threshold)
        )

        # RANGING: ADX < ranging_threshold and BB_width < BB_width_avg
        ranging = (df["adx_14"] < self.adx_ranging_threshold) & (df["bb_width"] < df["bb_width_avg"])

        # BREAKOUT: ATR > ATR_avg * multiplier and volume > VOL_avg * multiplier
        breakout = (df["atr_14"] > df["atr_avg"] * self.breakout_atr_multiplier) & (
            df["volume"] > df["vol_avg"] * self.breakout_volume_multiplier
        )

        # Apply in priority order (breakout > trend > ranging > uncertain)
        regimes[ranging] = "RANGING"
        regimes[bull] = "BULL_TREND"
        regimes[bear] = "BEAR_TREND"
        regimes[breakout] = "BREAKOUT"

        return regimes

    def apply_hysteresis(self, regime_series: pd.Series, candles: int = 3) -> pd.Series:
        """
        Require a regime to persist for *candles* consecutive bars before
        switching.  Until confirmed, the previous regime is kept.

        Parameters
        ----------
        regime_series : pd.Series
            Raw regime classification from :meth:`detect_regime`.
        candles : int
            Number of consecutive candles required to confirm a regime change.

        Returns
        -------
        pd.Series
            Smoothed regime series.
        """
        if len(regime_series) == 0:
            return regime_series.copy()

        smoothed = regime_series.copy()
        confirmed_regime = regime_series.iloc[0]
        candidate_regime = confirmed_regime
        candidate_count = 1

        for i in range(1, len(regime_series)):
            current = regime_series.iloc[i]
            if current == candidate_regime:
                candidate_count += 1
            else:
                candidate_regime = current
                candidate_count = 1

            if candidate_count >= candles:
                confirmed_regime = candidate_regime

            smoothed.iloc[i] = confirmed_regime

        return smoothed

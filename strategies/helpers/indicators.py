"""
Technical indicators implemented using only numpy and pandas.
No TA-Lib dependency.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing method)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=high.index)
    minus_dm_series = pd.Series(minus_dm, index=high.index)

    # Smoothed using Wilder's method
    atr_smooth = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm_series.ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_series.ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth.replace(0, np.nan)

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / dx_denom
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx_series.fillna(0.0)


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns:
        (upper, middle, lower, width)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.

    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.

    Returns:
        (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100.0 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50.0), d.fillna(50.0)


def stochastic_rsi(
    series: pd.Series,
    period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI.

    Returns:
        (stoch_rsi_k, stoch_rsi_d)
    """
    rsi_values = rsi(series, period)
    lowest_rsi = rsi_values.rolling(window=period).min()
    highest_rsi = rsi_values.rolling(window=period).max()
    denom = (highest_rsi - lowest_rsi).replace(0, np.nan)
    stoch_rsi = (rsi_values - lowest_rsi) / denom
    stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
    return stoch_rsi_k.fillna(0.5), stoch_rsi_d.fillna(0.5)


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average of volume."""
    return volume.rolling(window=period).mean()

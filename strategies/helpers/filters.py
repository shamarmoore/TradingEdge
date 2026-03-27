"""
Signal confluence and market filters for the CryptoEdge trading bot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confluence_score(signals: dict) -> int:
    """
    Calculate confluence score from a signals dictionary.

    Expected keys (all optional, default to False/0):
        primary_conditions_met  (bool)  – core strategy conditions satisfied
        daily_tf_confirms       (bool)  – 1D timeframe alignment
        volume_confirms         (bool)  – volume confirms the move
        multi_indicator_agree   (bool)  – multiple indicators agree
        no_nearby_resistance    (bool)  – no major resistance nearby
        regime_recently_changed (bool)  – regime switched recently (penalty)
        conflicting_signal      (bool)  – another indicator disagrees (penalty)
        high_correlation        (bool)  – highly correlated open position (penalty)
        recent_loss_on_pair     (bool)  – recent losing trade on this pair (penalty)

    Scoring:
        +1 per positive condition (up to +5)
        -1 per negative condition (up to -4)
    Returns integer score (can be negative).
    """
    score = 0

    # Positive contributors
    if signals.get("primary_conditions_met", False):
        score += 1
    if signals.get("daily_tf_confirms", False):
        score += 1
    if signals.get("volume_confirms", False):
        score += 1
    if signals.get("multi_indicator_agree", False):
        score += 1
    if signals.get("no_nearby_resistance", False):
        score += 1

    # Negative contributors
    if signals.get("regime_recently_changed", False):
        score -= 1
    if signals.get("conflicting_signal", False):
        score -= 1
    if signals.get("high_correlation", False):
        score -= 1
    if signals.get("recent_loss_on_pair", False):
        score -= 1

    return score


def pair_correlation_filter(
    returns_dict: dict, max_correlation: float = 0.7
) -> bool:
    """
    Check whether adding a new pair would exceed the maximum allowed
    pairwise correlation among open positions.

    Parameters
    ----------
    returns_dict : dict
        Keys are pair symbols, values are pd.Series of returns.
        The *last* key is treated as the candidate new pair.
    max_correlation : float
        Maximum acceptable pairwise Pearson correlation (absolute value).

    Returns
    -------
    bool
        True  – correlation is acceptable (trade allowed).
        False – at least one existing pair is too correlated.
    """
    if len(returns_dict) < 2:
        return True  # Nothing to correlate against

    pairs = list(returns_dict.keys())
    candidate = pairs[-1]
    candidate_returns = returns_dict[candidate]

    for pair in pairs[:-1]:
        existing_returns = returns_dict[pair]
        # Align series
        combined = pd.concat(
            [existing_returns, candidate_returns], axis=1
        ).dropna()
        if len(combined) < 5:
            continue
        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        if abs(corr) > max_correlation:
            return False

    return True


def spread_filter(bid: float, ask: float, max_spread_pct: float = 0.1) -> bool:
    """
    Check whether the bid-ask spread is within acceptable limits.

    Parameters
    ----------
    bid : float
        Best bid price.
    ask : float
        Best ask price.
    max_spread_pct : float
        Maximum allowed spread as a percentage of mid-price.

    Returns
    -------
    bool
        True if spread is acceptable, False otherwise.
    """
    if bid <= 0 or ask <= 0:
        return False
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0
    return spread_pct <= max_spread_pct


def volume_filter(
    current_volume_usd: float, min_volume_usd: float = 10_000_000
) -> bool:
    """
    Check whether 24h volume meets the minimum threshold.

    Returns
    -------
    bool
        True if volume is adequate, False otherwise.
    """
    return current_volume_usd >= min_volume_usd


def cooldown_filter(
    last_trade_candle: int,
    current_candle: int,
    cooldown_candles: int = 3,
) -> bool:
    """
    Enforce a cooldown period after the previous trade on a pair.

    Parameters
    ----------
    last_trade_candle : int
        Candle index (or timestamp integer) of the last trade exit.
    current_candle : int
        Current candle index.
    cooldown_candles : int
        Number of candles that must pass before re-entering.

    Returns
    -------
    bool
        True if cooldown has passed (trade allowed), False otherwise.
    """
    return (current_candle - last_trade_candle) >= cooldown_candles

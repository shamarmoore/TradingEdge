"""
telegram_bot.py – Telegram alert functions for CryptoEdge monitoring.

All functions are stateless – they accept token/chat_id explicitly so
they can be called from any context without global state.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send_alert(token: str, chat_id: str, message: str, parse_mode: str = "HTML") -> bool:
    """
    Send a plain-text or HTML message to a Telegram chat.

    Parameters
    ----------
    token    : Telegram Bot API token.
    chat_id  : Target chat / channel ID.
    message  : Message text (HTML tags allowed when parse_mode='HTML').
    parse_mode : 'HTML' or 'Markdown'.

    Returns
    -------
    bool  True if sent successfully, False otherwise.
    """
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id is empty – skipping alert.")
        return False

    url = TELEGRAM_API_BASE.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
        return False


def send_trade_alert(
    token: str,
    chat_id: str,
    trade_info: dict,
) -> bool:
    """
    Format and send a trade notification.

    Expected keys in trade_info:
        pair, direction, entry_price, amount, stop_loss,
        take_profit, strategy, confluence_score, regime
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    direction = trade_info.get("direction", "LONG").upper()
    pair = trade_info.get("pair", "UNKNOWN")
    entry = trade_info.get("entry_price", 0)
    amount = trade_info.get("amount", 0)
    sl = trade_info.get("stop_loss", 0)
    tp = trade_info.get("take_profit", 0)
    strategy = trade_info.get("strategy", "EnsembleStrategy")
    score = trade_info.get("confluence_score", "?")
    regime = trade_info.get("regime", "UNKNOWN")

    icon = "🟢" if direction == "LONG" else "🔴"
    message = (
        f"{icon} <b>TRADE OPENED</b> – {ts}\n"
        f"Pair:       <code>{pair}</code>\n"
        f"Direction:  {direction}\n"
        f"Entry:      <code>{entry}</code>\n"
        f"Amount:     <code>{amount}</code>\n"
        f"Stop Loss:  <code>{sl}</code>\n"
        f"Take Profit:<code>{tp}</code>\n"
        f"Strategy:   {strategy}\n"
        f"Regime:     {regime}\n"
        f"Confluence: {score}"
    )
    return send_alert(token, chat_id, message)


def send_circuit_breaker_alert(
    token: str,
    chat_id: str,
    level: str,
    drawdown: float,
) -> bool:
    """
    Send a circuit-breaker notification.

    Parameters
    ----------
    level    : "yellow", "orange", or "red"
    drawdown : Current drawdown as a positive percentage (e.g. 5.2 for 5.2 %)
    """
    icons = {
        "yellow": "⚠️",
        "orange": "🟠",
        "red": "🛑",
    }
    actions = {
        "yellow": "Reduce position sizes to 50 %.",
        "orange": "Stop new entries. Manage existing trades only.",
        "red": "HALT all trading. Switch to paper mode. Review strategy.",
    }
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    icon = icons.get(level.lower(), "⚠️")
    action = actions.get(level.lower(), "Review trading activity.")

    message = (
        f"{icon} <b>CIRCUIT BREAKER – {level.upper()}</b> – {ts}\n"
        f"Current Drawdown: <code>{drawdown:.2f}%</code>\n"
        f"Action Required: {action}"
    )
    return send_alert(token, chat_id, message)


def send_daily_report(
    token: str,
    chat_id: str,
    report: dict,
) -> bool:
    """
    Send the daily performance report.

    Expected keys in report:
        date, total_trades, wins, losses, profit_pct,
        portfolio_value, daily_pnl_usd, win_rate, best_trade, worst_trade
    """
    date = report.get("date", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))
    total = report.get("total_trades", 0)
    wins = report.get("wins", 0)
    losses = report.get("losses", 0)
    profit_pct = report.get("profit_pct", 0.0)
    portfolio = report.get("portfolio_value", 0.0)
    daily_pnl = report.get("daily_pnl_usd", 0.0)
    win_rate = report.get("win_rate", 0.0)
    best = report.get("best_trade", "N/A")
    worst = report.get("worst_trade", "N/A")

    pnl_icon = "📈" if daily_pnl >= 0 else "📉"
    message = (
        f"📊 <b>Daily Report – {date}</b>\n"
        f"Trades:     {total}  (W:{wins} / L:{losses})\n"
        f"Win Rate:   {win_rate*100:.1f}%\n"
        f"{pnl_icon} Daily P&L: <code>{daily_pnl:+.2f} USDT  ({profit_pct:+.2f}%)</code>\n"
        f"Portfolio:  <code>{portfolio:.2f} USDT</code>\n"
        f"Best Trade: {best}\n"
        f"Worst Trade:{worst}"
    )
    return send_alert(token, chat_id, message)

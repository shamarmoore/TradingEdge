"""
daily_report.py – Generate and persist daily P&L reports for CryptoEdge.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_daily_report(
    trades: list[dict],
    portfolio_value: float,
    start_value: float,
) -> dict:
    """
    Generate a daily P&L report from a list of closed trades.

    Parameters
    ----------
    trades : list of dicts
        Each trade dict must contain at least:
            profit_ratio  (float) – trade return as a ratio (e.g. 0.02 = +2 %)
            profit_abs    (float) – absolute P&L in stake currency
            pair          (str)   – trading pair symbol
        Optional keys: open_date, close_date, strategy
    portfolio_value : float
        Current portfolio value after today's trades.
    start_value : float
        Portfolio value at the start of the day.

    Returns
    -------
    dict with comprehensive daily stats.
    """
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    if not trades:
        return {
            "date": date_str,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": 0.0,
            "profit_pct": 0.0,
            "daily_pnl_usd": 0.0,
            "portfolio_value": portfolio_value,
            "start_value": start_value,
            "best_trade": None,
            "worst_trade": None,
            "avg_profit_pct": 0.0,
            "profit_factor": 0.0,
            "trades": [],
        }

    profits = [t.get("profit_ratio", t.get("profit_percent", 0.0)) for t in trades]
    profits_abs = [t.get("profit_abs", 0.0) for t in trades]

    wins = [p for p in profits if p > 0]
    losses_list = [p for p in profits if p < 0]
    breakeven = len(profits) - len(wins) - len(losses_list)

    gross_profit = sum(p for p in profits_abs if p > 0)
    gross_loss = abs(sum(p for p in profits_abs if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_pnl_usd = portfolio_value - start_value
    profit_pct = (daily_pnl_usd / start_value * 100.0) if start_value > 0 else 0.0

    # Best and worst individual trade
    if profits:
        best_idx = profits.index(max(profits))
        worst_idx = profits.index(min(profits))
        best_trade = {
            "pair": trades[best_idx].get("pair", "UNKNOWN"),
            "profit_pct": round(profits[best_idx] * 100, 4),
        }
        worst_trade = {
            "pair": trades[worst_idx].get("pair", "UNKNOWN"),
            "profit_pct": round(profits[worst_idx] * 100, 4),
        }
    else:
        best_trade = worst_trade = None

    return {
        "date": date_str,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses_list),
        "breakeven": breakeven,
        "win_rate": round(len(wins) / len(trades), 4),
        "profit_pct": round(profit_pct, 4),
        "daily_pnl_usd": round(daily_pnl_usd, 4),
        "portfolio_value": round(portfolio_value, 4),
        "start_value": round(start_value, 4),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_profit_pct": round(sum(profits) / len(profits) * 100, 4),
        "profit_factor": round(profit_factor, 4) if not isinstance(profit_factor, float) or not profit_factor == float("inf") else "inf",
        "trades": trades,
    }


def save_report(report: dict, filepath: str) -> None:
    """
    Save a daily report as both JSON and CSV.

    Parameters
    ----------
    report   : dict returned by generate_daily_report()
    filepath : Base file path without extension (extensions are added automatically).
               E.g. "data/validation/report_2024-01-15"
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = filepath if filepath.endswith(".json") else filepath + ".json"
    with open(json_path, "w") as f:
        # Exclude full trade list from JSON summary
        summary = {k: v for k, v in report.items() if k != "trades"}
        json.dump(summary, f, indent=2, default=str)

    # Save CSV summary row
    csv_path = (filepath.rstrip(".json") if filepath.endswith(".json") else filepath) + ".csv"
    fieldnames = [
        "date", "total_trades", "wins", "losses", "win_rate",
        "profit_pct", "daily_pnl_usd", "portfolio_value", "profit_factor",
    ]
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {k: report.get(k, "") for k in fieldnames}
        writer.writerow(row)

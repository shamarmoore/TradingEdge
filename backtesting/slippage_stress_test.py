#!/usr/bin/env python3
"""
slippage_stress_test.py – Test strategy robustness across slippage levels.

Tests at 0 %, 0.05 %, 0.10 %, and 0.20 % slippage.

HARD GATE: strategy must be profitable at 0.10 % slippage.

Also calculates the Slippage Degradation Ratio (SDR):
    SDR = (Profit at 0% slippage - Profit at 0.20% slippage) /
          (Profit at 0% slippage)

Usage:
    python backtesting/slippage_stress_test.py \
        --trades-file backtesting/reports/trades.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SLIPPAGE_LEVELS = [0.0, 0.0005, 0.001, 0.002]  # 0%, 0.05%, 0.10%, 0.20%
HARD_GATE_SLIPPAGE = 0.001  # 0.10%


def load_returns(filepath: str) -> list[float]:
    """Load trade profit_ratio list from Freqtrade JSON or CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(filepath)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        trades = data if isinstance(data, list) else data.get("trades", data.get("results", []))
        df = pd.DataFrame(trades)
    else:
        df = pd.read_csv(path)

    df.columns = [c.lower().strip() for c in df.columns]
    profit_col = next(
        (c for c in ("profit_ratio", "profit_percent", "profit", "profit_abs") if c in df.columns),
        None,
    )
    if profit_col is None:
        raise ValueError("Trades file must contain a profit column.")
    return list(pd.to_numeric(df[profit_col], errors="coerce").dropna())


def apply_slippage(returns: list[float], slippage_pct: float) -> list[float]:
    """Apply per-trade round-trip slippage cost to every trade."""
    return [r - slippage_pct for r in returns]  # entry + exit both slip


def calc_metrics(returns: list[float]) -> dict:
    """Calculate performance metrics from a list of returns."""
    arr = np.array(returns)
    winners = arr[arr > 0]
    losers = arr[arr < 0]

    gross_profit = float(winners.sum())
    gross_loss = float(abs(losers.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_return = float((np.cumprod(1 + arr))[-1] - 1) if len(arr) > 0 else 0.0

    # Max drawdown
    equity = np.cumprod(1 + arr)
    equity = np.insert(equity, 0, 1.0)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    return {
        "total_trades": len(returns),
        "win_rate": round(len(winners) / max(len(arr), 1), 4),
        "profit_factor": round(profit_factor, 4),
        "total_return_pct": round(total_return * 100, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "profitable": total_return > 0,
    }


def run_stress_test(returns: list[float]) -> list[dict]:
    """Run slippage stress test across all levels."""
    rows: list[dict] = []
    for slip in SLIPPAGE_LEVELS:
        adjusted = apply_slippage(returns, slip)
        metrics = calc_metrics(adjusted)
        metrics["slippage_pct"] = round(slip * 100, 4)
        rows.append(metrics)
    return rows


def calculate_sdr(rows: list[dict]) -> float:
    """
    Slippage Degradation Ratio.
        SDR = (Profit@0% - Profit@0.20%) / Profit@0%
    """
    base = next((r for r in rows if r["slippage_pct"] == 0.0), None)
    high = next((r for r in rows if r["slippage_pct"] == 0.2), None)
    if base is None or high is None:
        return float("nan")
    base_profit = base["total_return_pct"]
    high_profit = high["total_return_pct"]
    if base_profit == 0:
        return float("nan")
    return round((base_profit - high_profit) / base_profit, 4)


def hard_gate_check(rows: list[dict]) -> bool:
    """HARD GATE: strategy must be profitable at 0.10 % slippage."""
    gate_row = next((r for r in rows if r["slippage_pct"] == 0.1), None)
    if gate_row is None:
        return False
    return gate_row["profitable"]


def print_results_table(rows: list[dict], sdr: float, gate_pass: bool) -> None:
    print("\n" + "=" * 75)
    print("Slippage Stress Test")
    print("=" * 75)
    header = f"{'Slippage':>10} {'Trades':>8} {'Win%':>8} {'PF':>8} {'TotalRet%':>12} {'MaxDD%':>10} {'Result':>8}"
    print(header)
    print("-" * 75)
    for r in rows:
        result = "PASS" if r["profitable"] else "FAIL"
        print(
            f"{r['slippage_pct']:>10.2f}% "
            f"{r['total_trades']:>8} "
            f"{r['win_rate']*100:>7.1f}% "
            f"{r['profit_factor']:>8.3f} "
            f"{r['total_return_pct']:>11.2f}% "
            f"{r['max_drawdown_pct']:>9.2f}% "
            f"{result:>8}"
        )
    print("=" * 75)
    print(f"Slippage Degradation Ratio (SDR): {sdr:.4f}")
    gate_str = "PASS" if gate_pass else "FAIL ← HARD GATE BREACHED"
    print(f"Hard Gate (profitable at 0.10%): {gate_str}")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Slippage stress test.")
    parser.add_argument("--trades-file", help="Freqtrade trades JSON or CSV.")
    parser.add_argument(
        "--returns",
        nargs="+",
        type=float,
        help="Space-separated trade profit ratios.",
    )
    parser.add_argument(
        "--output",
        default="backtesting/reports/slippage_stress_test.json",
        help="Output path for results.",
    )
    args = parser.parse_args()

    if args.trades_file:
        returns = load_returns(args.trades_file)
    elif args.returns:
        returns = args.returns
    else:
        parser.error("Provide either --trades-file or --returns.")

    rows = run_stress_test(returns)
    sdr = calculate_sdr(rows)
    gate_pass = hard_gate_check(rows)

    print_results_table(rows, sdr, gate_pass)

    # Save results
    output = {
        "slippage_levels": rows,
        "sdr": sdr,
        "hard_gate_pass": gate_pass,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    if not gate_pass:
        print("\nFAIL: Strategy is not profitable at 0.10% slippage!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

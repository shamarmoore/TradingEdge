#!/usr/bin/env python3
"""
monte_carlo.py – Monte Carlo simulation for trade return sequences.

Methodology:
  - 1 000 iterations
  - Per iteration: shuffle trade order, remove 10 % of winners randomly,
    add random 0.05–0.15 % slippage per trade
  - Reports:
      • 95th-percentile maximum drawdown
      • 5th-percentile final equity
      • Probability of ruin (equity < 50 % of starting equity)

Usage:
    python backtesting/monte_carlo.py --trades-file backtesting/reports/trades.json
    python backtesting/monte_carlo.py --returns 0.02 -0.01 0.03 -0.015 0.04
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


N_ITERATIONS = 1_000
SLIPPAGE_MIN = 0.0005   # 0.05 %
SLIPPAGE_MAX = 0.0015   # 0.15 %
WINNER_REMOVAL_PCT = 0.10
RUIN_THRESHOLD = 0.50   # equity below 50 % of start = ruin


def load_returns_from_file(filepath: str) -> list[float]:
    """Load trade profit_ratio values from a Freqtrade JSON export or CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(filepath)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            trades = data
        elif isinstance(data, dict):
            trades = data.get("trades", data.get("results", []))
        else:
            trades = []
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


def run_equity_curve(returns: np.ndarray, starting_equity: float = 1.0) -> np.ndarray:
    """Compound returns into an equity curve."""
    equity = np.empty(len(returns) + 1)
    equity[0] = starting_equity
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def max_drawdown(equity: np.ndarray) -> float:
    """Calculate maximum percentage drawdown from peak."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())  # negative value


def simulate(
    original_returns: list[float],
    n_iterations: int = N_ITERATIONS,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run Monte Carlo simulation.

    Returns dict with:
        iterations              – number of iterations
        p95_max_drawdown        – 95th-percentile worst drawdown (negative)
        p5_final_equity         – 5th-percentile final equity (starting=1.0)
        probability_of_ruin     – fraction of iterations where equity < 50%
        all_final_equities      – list of final equity values
        all_max_drawdowns       – list of max drawdown values
    """
    if rng is None:
        rng = np.random.default_rng()

    returns_arr = np.array(original_returns, dtype=float)
    winners_mask = returns_arr > 0

    final_equities: list[float] = []
    max_drawdowns: list[float] = []

    for _ in range(n_iterations):
        sim_returns = returns_arr.copy()

        # Remove 10 % of winners randomly
        winner_indices = np.where(winners_mask)[0]
        n_remove = max(1, int(len(winner_indices) * WINNER_REMOVAL_PCT))
        remove_idx = rng.choice(winner_indices, size=n_remove, replace=False)
        sim_returns = np.delete(sim_returns, remove_idx)

        # Shuffle trade order
        rng.shuffle(sim_returns)

        # Add random slippage (always negative cost)
        slippage = rng.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX, size=len(sim_returns))
        sim_returns = sim_returns - slippage

        equity = run_equity_curve(sim_returns)
        final_equities.append(float(equity[-1]))
        max_drawdowns.append(max_drawdown(equity))

    fe = np.array(final_equities)
    md = np.array(max_drawdowns)
    ruin_count = int(np.sum(fe < RUIN_THRESHOLD))

    return {
        "iterations": n_iterations,
        "p95_max_drawdown": round(float(np.percentile(md, 5)), 6),   # 5th pct of DD (worst)
        "p5_final_equity": round(float(np.percentile(fe, 5)), 6),
        "probability_of_ruin": round(float(ruin_count / n_iterations), 6),
        "median_final_equity": round(float(np.median(fe)), 6),
        "mean_final_equity": round(float(np.mean(fe)), 6),
        "all_final_equities": [round(v, 6) for v in final_equities],
        "all_max_drawdowns": [round(v, 6) for v in max_drawdowns.tolist()],
    }


def print_report(result: dict) -> None:
    print("\n" + "=" * 50)
    print(f"Monte Carlo Simulation  ({result['iterations']} iterations)")
    print("=" * 50)
    print(f"  95th-pct max drawdown : {result['p95_max_drawdown']*100:.2f}%")
    print(f"  5th-pct final equity  : {result['p5_final_equity']:.4f}x")
    print(f"  Median final equity   : {result['median_final_equity']:.4f}x")
    print(f"  Probability of ruin   : {result['probability_of_ruin']*100:.2f}%")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation for trade returns.")
    parser.add_argument("--trades-file", help="Path to Freqtrade trades JSON or CSV.")
    parser.add_argument(
        "--returns",
        nargs="+",
        type=float,
        help="Space-separated trade returns (e.g. 0.02 -0.01 0.03).",
    )
    parser.add_argument(
        "--output",
        default="backtesting/reports/monte_carlo_results.json",
        help="Output path for results JSON.",
    )
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    args = parser.parse_args()

    if args.trades_file:
        returns = load_returns_from_file(args.trades_file)
    elif args.returns:
        returns = args.returns
    else:
        parser.error("Provide either --trades-file or --returns.")

    if len(returns) < 10:
        print("WARNING: Very few trades – simulation may not be meaningful.", file=sys.stderr)

    print(f"Running simulation on {len(returns)} trades...")
    result = simulate(returns, n_iterations=args.iterations)
    print_report(result)

    # Save (strip large lists for report)
    save_data = {k: v for k, v in result.items() if k not in ("all_final_equities", "all_max_drawdowns")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

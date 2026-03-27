#!/usr/bin/env python3
"""
run_backtest.py – Launch Freqtrade backtesting via subprocess and
                  evaluate results against MUST-PASS performance metrics.

Usage:
    python backtesting/run_backtest.py \
        --strategy EnsembleStrategy \
        --timerange 20210101-20231231 \
        --config config/config.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------
# MUST-PASS performance thresholds
# -----------------------------------------------------------------------
MUST_PASS = {
    "win_rate_min": 0.45,         # >= 45 %
    "profit_factor_min": 1.3,     # gross profit / gross loss >= 1.3
    "max_drawdown_max": 0.25,     # max drawdown <= 25 %
    "sharpe_min": 0.5,            # annualised Sharpe >= 0.5
    "total_trades_min": 30,       # at least 30 trades
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Freqtrade backtest and evaluate results.")
    parser.add_argument("--strategy", required=True, help="Strategy class name")
    parser.add_argument("--timerange", default="", help="Date range, e.g. 20210101-20231231")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON")
    parser.add_argument("--results-dir", default="backtesting/reports", help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return parser.parse_args()


def build_freqtrade_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "freqtrade", "backtesting",
        "--strategy", args.strategy,
        "--config", args.config,
        "--export", "trades",
        "--export-filename", f"{args.results_dir}/{args.strategy}_backtest.json",
    ]
    if args.timerange:
        cmd += ["--timerange", args.timerange]
    return cmd


def run_backtest(cmd: list[str]) -> tuple[int, str, str]:
    """Run the backtest command, return (returncode, stdout, stderr)."""
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def parse_backtest_results(results_path: str) -> dict:
    """Parse Freqtrade JSON export and extract key metrics."""
    path = Path(results_path)
    if not path.exists():
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    # Freqtrade export structure: results keyed by strategy name
    if not isinstance(data, dict):
        return {}

    # Try to get the first strategy's results
    strategy_results = {}
    if "strategy" in data:
        strategy_results = data.get("strategy", {})
    elif "results" in data:
        strategy_results = data.get("results", {})

    # Flatten one level if needed
    if strategy_results and isinstance(list(strategy_results.values())[0], dict):
        strategy_results = list(strategy_results.values())[0]

    trades = data.get("trades", [])
    if not trades and isinstance(strategy_results, dict):
        trades = strategy_results.get("trades", [])

    return {
        "total_trades": strategy_results.get("total_trades", len(trades)),
        "win_rate": strategy_results.get("wins", 0) / max(strategy_results.get("total_trades", 1), 1),
        "profit_factor": strategy_results.get("profit_factor", 0.0),
        "max_drawdown": abs(strategy_results.get("max_drawdown", 0.0)),
        "sharpe": strategy_results.get("sharpe_ratio", strategy_results.get("sharpe", 0.0)),
        "total_profit_pct": strategy_results.get("profit_total_abs", 0.0),
    }


def evaluate_results(metrics: dict) -> tuple[bool, list[str]]:
    """Compare metrics against MUST-PASS thresholds."""
    failures: list[str] = []

    checks = [
        ("win_rate", metrics.get("win_rate", 0), ">=", MUST_PASS["win_rate_min"]),
        ("profit_factor", metrics.get("profit_factor", 0), ">=", MUST_PASS["profit_factor_min"]),
        ("max_drawdown", metrics.get("max_drawdown", 1), "<=", MUST_PASS["max_drawdown_max"]),
        ("sharpe", metrics.get("sharpe", 0), ">=", MUST_PASS["sharpe_min"]),
        ("total_trades", metrics.get("total_trades", 0), ">=", MUST_PASS["total_trades_min"]),
    ]

    for name, value, op, threshold in checks:
        passed = (value >= threshold) if op == ">=" else (value <= threshold)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value:.4f} {op} {threshold}")
        if not passed:
            failures.append(f"{name}: {value:.4f} (required {op} {threshold})")

    return len(failures) == 0, failures


def save_results_summary(
    args: argparse.Namespace,
    metrics: dict,
    passed: bool,
    failures: list[str],
) -> None:
    """Save a summary JSON alongside the backtest results."""
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy": args.strategy,
        "timerange": args.timerange,
        "run_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "must_pass_result": "PASS" if passed else "FAIL",
        "failures": failures,
    }
    out_path = f"{args.results_dir}/{args.strategy}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


def main() -> None:
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    cmd = build_freqtrade_command(args)

    if args.dry_run:
        print("Dry run – command that would be executed:")
        print(" ".join(cmd))
        sys.exit(0)

    returncode, stdout, stderr = run_backtest(cmd)

    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    if returncode != 0:
        print(f"\nERROR: freqtrade exited with code {returncode}", file=sys.stderr)
        sys.exit(returncode)

    # Parse and evaluate
    results_file = f"{args.results_dir}/{args.strategy}_backtest.json"
    metrics = parse_backtest_results(results_file)

    if not metrics:
        print("\nWARN: Could not parse backtest results file. Check freqtrade output.")
        sys.exit(0)

    print("\n--- MUST-PASS Evaluation ---")
    passed, failures = evaluate_results(metrics)

    save_results_summary(args, metrics, passed, failures)

    if not passed:
        print(f"\nRESULT: FAIL ({len(failures)} criteria not met)")
        sys.exit(1)
    else:
        print("\nRESULT: PASS – all criteria met")


if __name__ == "__main__":
    main()

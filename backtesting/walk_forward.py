#!/usr/bin/env python3
"""
walk_forward.py – Walk-forward optimisation analysis.

Splits a trade history into rolling in-sample / out-of-sample windows and
reports whether the strategy overfits (out-of-sample < 50 % of in-sample).

Window scheme:
    Training window : 6 months
    Test window     : 2 months
    Step            : 2 months (slide forward)

Usage:
    python backtesting/walk_forward.py --trades-file backtesting/reports/trades.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


TRAIN_MONTHS = 6
TEST_MONTHS = 2
STEP_MONTHS = 2
OVERFIT_THRESHOLD = 0.5  # OOS < 50 % of IS → overfitting flag


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime (approximate: 30 days per month)."""
    return dt + timedelta(days=months * 30)


def load_trades(filepath: str) -> pd.DataFrame:
    """Load trades from a JSON file exported by Freqtrade or a CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {filepath}")

    if path.suffix == ".json":
        with open(path, "r") as f:
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

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse close_date / close_time / exit_time
    for col in ("close_date", "close_time", "exit_time", "open_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    date_col = next((c for c in ("close_date", "close_time", "exit_time") if c in df.columns), None)
    if date_col:
        df["_close_dt"] = df[date_col]
    else:
        raise ValueError("Trades file must contain a close/exit date column.")

    profit_col = next(
        (c for c in ("profit_ratio", "profit_percent", "profit_abs", "profit") if c in df.columns),
        None,
    )
    if profit_col is None:
        raise ValueError("Trades file must contain a profit column.")
    df["_profit"] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)

    return df.dropna(subset=["_close_dt"]).sort_values("_close_dt").reset_index(drop=True)


def window_iterator(
    start: datetime, end: datetime
) -> Iterator[tuple[datetime, datetime, datetime, datetime]]:
    """Yield (train_start, train_end, test_start, test_end) windows."""
    train_start = start
    while True:
        train_end = add_months(train_start, TRAIN_MONTHS)
        test_start = train_end
        test_end = add_months(test_start, TEST_MONTHS)
        if test_end > end:
            break
        yield train_start, train_end, test_start, test_end
        train_start = add_months(train_start, STEP_MONTHS)


def calc_performance(trades: pd.DataFrame) -> dict:
    """Calculate performance metrics for a slice of trades."""
    if trades.empty:
        return {"total_trades": 0, "win_rate": 0.0, "profit_sum": 0.0, "profit_factor": 0.0}

    profits = trades["_profit"]
    winners = profits[profits > 0]
    losers = profits[profits < 0]

    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_trades": len(trades),
        "win_rate": len(winners) / len(trades),
        "profit_sum": round(float(profits.sum()), 6),
        "profit_factor": round(float(profit_factor), 4),
    }


def run_walk_forward(trades_df: pd.DataFrame) -> list[dict]:
    """Run all walk-forward windows and collect results."""
    start = trades_df["_close_dt"].min().to_pydatetime().replace(tzinfo=None)
    end = trades_df["_close_dt"].max().to_pydatetime().replace(tzinfo=None)

    results: list[dict] = []

    for i, (train_s, train_e, test_s, test_e) in enumerate(window_iterator(start, end), 1):
        ts = pd.Timestamp
        in_sample = trades_df[
            (trades_df["_close_dt"] >= ts(train_s, tz="UTC"))
            & (trades_df["_close_dt"] < ts(train_e, tz="UTC"))
        ]
        out_of_sample = trades_df[
            (trades_df["_close_dt"] >= ts(test_s, tz="UTC"))
            & (trades_df["_close_dt"] < ts(test_e, tz="UTC"))
        ]

        is_perf = calc_performance(in_sample)
        oos_perf = calc_performance(out_of_sample)

        # Overfitting check: OOS profit_factor < 50 % of IS profit_factor
        if is_perf["profit_factor"] > 0 and not np.isinf(is_perf["profit_factor"]):
            oos_ratio = oos_perf["profit_factor"] / is_perf["profit_factor"]
            overfit = oos_ratio < OVERFIT_THRESHOLD
        else:
            oos_ratio = None
            overfit = False

        result = {
            "window": i,
            "train_start": train_s.isoformat(),
            "train_end": train_e.isoformat(),
            "test_start": test_s.isoformat(),
            "test_end": test_e.isoformat(),
            "in_sample": is_perf,
            "out_of_sample": oos_perf,
            "oos_to_is_ratio": round(float(oos_ratio), 4) if oos_ratio is not None else None,
            "overfit_flag": overfit,
        }
        results.append(result)

    return results


def print_report(results: list[dict]) -> None:
    """Print walk-forward results to stdout."""
    overfit_count = sum(1 for r in results if r["overfit_flag"])
    print(f"\n{'='*70}")
    print(f"Walk-Forward Analysis  ({len(results)} windows)")
    print(f"{'='*70}")
    header = f"{'Win':<5} {'Train':<22} {'Test':<22} {'IS PF':>8} {'OOS PF':>8} {'Ratio':>7} {'Flag'}"
    print(header)
    print("-" * 80)
    for r in results:
        flag = "OVERFIT" if r["overfit_flag"] else "OK"
        ratio_str = f"{r['oos_to_is_ratio']:.3f}" if r["oos_to_is_ratio"] is not None else "N/A"
        print(
            f"{r['window']:<5} {r['train_start'][:10]}-{r['train_end'][:10]}  "
            f"{r['test_start'][:10]}-{r['test_end'][:10]}  "
            f"{r['in_sample']['profit_factor']:>8.3f} "
            f"{r['out_of_sample']['profit_factor']:>8.3f} "
            f"{ratio_str:>7}  {flag}"
        )
    print(f"\nOverfitting windows: {overfit_count}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward optimisation analysis.")
    parser.add_argument(
        "--trades-file",
        required=True,
        help="Path to trades JSON/CSV exported from Freqtrade.",
    )
    parser.add_argument(
        "--output",
        default="backtesting/reports/walk_forward_results.json",
        help="Output path for results JSON.",
    )
    args = parser.parse_args()

    trades_df = load_trades(args.trades_file)
    print(f"Loaded {len(trades_df)} trades from {args.trades_file}")

    results = run_walk_forward(trades_df)
    print_report(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

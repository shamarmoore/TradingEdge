#!/usr/bin/env python3
"""
validate_data.py – Validate OHLCV historical data quality.

Usage:
    python data/validate_data.py --datadir data/historical/

Validation checks:
  - Gap detection (4h candles should be 4h apart; flag >12h, exclude >24h)
  - Outlier wick detection (wick_ratio > 10 without next-candle confirmation)
  - Volume anomalies (zero-volume; volume > 20x 50-period average)
  - Timestamp UTC alignment
  - Completeness (3yr of 4h data ≈ 6570 candles; warn if < 95 %)

Results are saved to data/validation/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CANDLE_SECONDS = 4 * 60 * 60  # 4-hour candles
EXPECTED_CANDLES_3YR = 6570    # 3 years × 365.25 d × 6 candles/day ≈ 6570
COMPLETENESS_THRESHOLD = 0.95


def load_csv(filepath: str) -> pd.DataFrame:
    """Load OHLCV CSV with columns: timestamp, open, high, low, close, volume."""
    df = pd.read_csv(filepath)
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse timestamp – accept Unix ms, Unix s, or ISO string
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        # Detect ms vs s
        if df["timestamp"].iloc[0] > 1e12:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def check_gaps(df: pd.DataFrame) -> dict:
    """Detect timestamp gaps in 4h candle data."""
    issues: list[dict] = []
    excluded_rows: list[int] = []

    diffs = df["datetime"].diff().dt.total_seconds().dropna()

    for i, diff in diffs.items():
        if diff > CANDLE_SECONDS * 1.05:  # allow 5 % tolerance
            gap_hours = diff / 3600
            issue = {
                "row": i,
                "datetime": str(df["datetime"].iloc[i]),
                "gap_hours": round(gap_hours, 2),
            }
            if gap_hours > 24:
                issue["severity"] = "EXCLUDE"
                excluded_rows.append(i)
            elif gap_hours > 12:
                issue["severity"] = "FLAG"
            else:
                continue  # small gap – ignore
            issues.append(issue)

    return {"issues": issues, "excluded_rows": excluded_rows}


def check_outlier_wicks(df: pd.DataFrame) -> list[dict]:
    """Flag candles with extreme wick ratios not confirmed by the next candle."""
    flagged: list[dict] = []
    body = (df["close"] - df["open"]).abs().replace(0, np.nan)
    wick = df["high"] - df["low"]
    wick_ratio = wick / body

    for i in range(len(df) - 1):  # -1 to check next candle
        ratio = wick_ratio.iloc[i]
        if pd.isna(ratio) or ratio <= 10:
            continue
        # Check if next candle confirms (moves in same direction as candle body)
        body_dir = df["close"].iloc[i] - df["open"].iloc[i]
        next_body_dir = df["close"].iloc[i + 1] - df["open"].iloc[i + 1]
        if (body_dir * next_body_dir) > 0:
            continue  # confirmed
        flagged.append({
            "row": i,
            "datetime": str(df["datetime"].iloc[i]),
            "wick_ratio": round(float(ratio), 2),
        })

    return flagged


def check_volume_anomalies(df: pd.DataFrame) -> list[dict]:
    """Flag zero-volume candles and extreme volume spikes."""
    flagged: list[dict] = []
    vol_avg = df["volume"].rolling(window=50, min_periods=10).mean()

    for i, row in df.iterrows():
        vol = row["volume"]
        avg = vol_avg.iloc[i] if not pd.isna(vol_avg.iloc[i]) else None

        if vol == 0:
            flagged.append({
                "row": i,
                "datetime": str(row["datetime"]),
                "issue": "zero_volume",
            })
        elif avg is not None and vol > avg * 20:
            flagged.append({
                "row": i,
                "datetime": str(row["datetime"]),
                "issue": "volume_spike",
                "volume": float(vol),
                "avg_volume": round(float(avg), 2),
                "ratio": round(float(vol / avg), 2),
            })

    return flagged


def check_completeness(df: pd.DataFrame) -> dict:
    """Check whether enough candles exist for a reliable backtest."""
    count = len(df)
    ratio = count / EXPECTED_CANDLES_3YR
    return {
        "candle_count": count,
        "expected_3yr": EXPECTED_CANDLES_3YR,
        "completeness_pct": round(ratio * 100, 2),
        "pass": ratio >= COMPLETENESS_THRESHOLD,
    }


def check_utc_alignment(df: pd.DataFrame) -> bool:
    """Verify all timestamps are UTC-aware."""
    return all(
        (getattr(ts, "tzinfo", None) is not None) for ts in df["datetime"]
    )


def validate_file(filepath: str) -> dict:
    """Run all validation checks on a single OHLCV CSV file."""
    print(f"Validating: {filepath}")
    df = load_csv(filepath)

    gap_result = check_gaps(df)
    # Remove excluded rows before further checks
    excluded = set(gap_result["excluded_rows"])
    df_clean = df.drop(index=list(excluded)).reset_index(drop=True)

    report = {
        "file": filepath,
        "total_candles_raw": len(df),
        "excluded_candles": len(excluded),
        "total_candles_clean": len(df_clean),
        "utc_aligned": check_utc_alignment(df),
        "completeness": check_completeness(df_clean),
        "gap_issues": gap_result["issues"],
        "wick_outliers": check_outlier_wicks(df_clean),
        "volume_anomalies": check_volume_anomalies(df_clean),
    }
    return report


def save_validation_report(report: dict, output_dir: str) -> str:
    """Save validation report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    base = Path(report["file"]).stem
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"validation_{base}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {out_path}")
    return out_path


def print_summary(report: dict) -> None:
    """Print a human-readable summary of the validation report."""
    comp = report["completeness"]
    status = "PASS" if comp["pass"] else "WARN"
    print(f"  Completeness: {comp['completeness_pct']}% [{status}]")
    print(f"  Candles (raw/clean): {report['total_candles_raw']} / {report['total_candles_clean']}")
    print(f"  UTC aligned: {report['utc_aligned']}")
    print(f"  Gap issues: {len(report['gap_issues'])}")
    print(f"  Wick outliers: {len(report['wick_outliers'])}")
    print(f"  Volume anomalies: {len(report['volume_anomalies'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OHLCV historical data.")
    parser.add_argument(
        "--datadir",
        required=True,
        help="Directory containing OHLCV CSV files to validate.",
    )
    parser.add_argument(
        "--output",
        default="data/validation",
        help="Output directory for validation reports (default: data/validation/).",
    )
    args = parser.parse_args()

    data_path = Path(args.datadir)
    if not data_path.exists():
        print(f"ERROR: datadir '{args.datadir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{args.datadir}'.")
        sys.exit(0)

    for csv_file in csv_files:
        report = validate_file(str(csv_file))
        print_summary(report)
        save_validation_report(report, args.output)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()

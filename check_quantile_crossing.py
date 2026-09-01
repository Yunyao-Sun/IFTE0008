"""
Quantile crossing robustness check.

Theoretically P10 <= P50 <= P90 should always hold, but quantile neural
networks (including TFT's pinball-loss output heads) don't guarantee this
per-sample -- crossing can happen when the model's three quantile heads
disagree slightly. This script reports how often it happens.

Usage:
    python check_quantile_crossing.py --dir inputs/pointwise
Scans every {turbine}_{freq}_{experiment}_pointwise.csv in that folder.

Output: quantile_crossing_report.csv with one row per file:
    turbine, granularity, experiment, n_points, crossing_rate,
    n_p10_gt_p50, n_p50_gt_p90
"""

import argparse
import glob
import os
import re

import pandas as pd

FILENAME_PATTERN = re.compile(
    r"^(?P<turbine>[^_]+)_(?P<freq>\d+min)_(?P<experiment>E\d_[a-zA-Z0-9_]+)_pointwise\.csv$"
)


def check_one_file(path: str) -> dict:
    df = pd.read_csv(path)
    if not {"pred_p10", "pred_p50", "pred_p90"}.issubset(df.columns):
        return None

    p10_gt_p50 = (df["pred_p10"] > df["pred_p50"])
    p50_gt_p90 = (df["pred_p50"] > df["pred_p90"])
    any_crossing = p10_gt_p50 | p50_gt_p90

    n = len(df)
    return {
        "n_points": n,
        "n_p10_gt_p50": int(p10_gt_p50.sum()),
        "n_p50_gt_p90": int(p50_gt_p90.sum()),
        "n_any_crossing": int(any_crossing.sum()),
        "crossing_rate": float(any_crossing.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="inputs/pointwise")
    parser.add_argument("--out", default="quantile_crossing_report.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_pointwise.csv")))
    if not files:
        print(f"No pointwise files found in {args.dir}")
        return

    rows = []
    for path in files:
        name = os.path.basename(path)
        m = FILENAME_PATTERN.match(name)
        if not m:
            print(f"[skip] filename doesn't match pattern: {name}")
            continue

        result = check_one_file(path)
        if result is None:
            print(f"[skip] {name}: missing pred_p10/p50/p90 columns")
            continue

        row = {
            "turbine": m.group("turbine"),
            "granularity": m.group("freq"),
            "experiment": m.group("experiment"),
            **result,
        }
        rows.append(row)
        print(f"{name}: crossing_rate={result['crossing_rate']*100:.3f}%  "
              f"({result['n_any_crossing']}/{result['n_points']} points)")

    if not rows:
        print("No valid files processed.")
        return

    report_df = pd.DataFrame(rows)
    report_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {args.out}")

    print("\n--- Average crossing rate by experiment ---")
    print(report_df.groupby("experiment")["crossing_rate"].mean().apply(lambda x: f"{x*100:.3f}%"))


if __name__ == "__main__":
    main()

"""Audit the hand-landmarks dataset used by the detection pipeline.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --dataset data/hand_landmarks_dataset.csv
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

DEFAULT_DATASET = "data/hand_landmarks_dataset.csv"
NEAR_DUP_L2 = 1e-3
FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]


def bar(count, scale):
    width = max(1, int(round(count / scale)))
    return "#" * width


def count_near_duplicates(df_class):
    if len(df_class) < 2:
        return 0
    feats = df_class[FEATURE_COLS].to_numpy(dtype=np.float32)
    diffs = np.linalg.norm(feats[1:] - feats[:-1], axis=1)
    return int((diffs < NEAR_DUP_L2).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"path to landmarks CSV (default: {DEFAULT_DATASET})")
    args = parser.parse_args()

    path = args.dataset
    if not os.path.exists(path):
        print(f"Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(path)

    print("=" * 64)
    print(f"File        : {path}")
    print(f"Size        : {size_mb:.2f} MB")
    print(f"Modified    : {mtime}")
    print(f"Total rows  : {len(df)}")
    print(f"Columns     : {len(df.columns)} (expected 43: 42 features + label)")
    print("=" * 64)

    if "label" not in df.columns:
        print("ERROR: no 'label' column. Is this the raw landmarks CSV?", file=sys.stderr)
        sys.exit(2)

    counts = df["label"].value_counts().sort_values(ascending=False)
    max_count = int(counts.max())
    min_count = int(counts.min())
    scale = max(1, max_count // 30)

    print(f"\nPer-class counts  ({len(counts)} classes):")
    for label, count in counts.items():
        print(f"  {label:<14} {count:>5}  {bar(count, scale)}")

    ratio = max_count / min_count if min_count else float("inf")
    print(f"\nImbalance ratio : {ratio:.2f}  (max {max_count} / min {min_count})")
    if ratio > 2:
        print("  WARNING: ratio > 2. Minority classes will under-perform in training.")

    print("\nNear-duplicate estimate (consecutive L2 < {:g}):".format(NEAR_DUP_L2))
    total_dups = 0
    for label in counts.index:
        class_df = df[df["label"] == label]
        dups = count_near_duplicates(class_df)
        total_dups += dups
        pct = 100.0 * dups / len(class_df) if len(class_df) else 0.0
        print(f"  {label:<14} {dups:>5} / {len(class_df):<5}  ({pct:5.1f}%)")
    overall_pct = 100.0 * total_dups / len(df) if len(df) else 0.0
    print(f"  {'TOTAL':<14} {total_dups:>5} / {len(df):<5}  ({overall_pct:5.1f}%)")

    print("\nRecommendations:")
    target = max_count
    under = counts[counts < target * 0.75]
    if under.empty:
        print("  Classes are reasonably balanced. No action needed.")
    else:
        for label, count in under.tail(3).items():
            needed = target - count
            print(f"  Collect ~{needed} more samples of '{label}' to match '{counts.index[0]}' ({target}).")
    print("=" * 64)


if __name__ == "__main__":
    main()

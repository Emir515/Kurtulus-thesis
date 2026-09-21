"""
Create unsupervised sequence datasets following Stefan Helm's methodology.

Replicates the exact LSTM-AE data pipeline from Helm (2026):
  - Non-overlapping windows of size Wseq (350s, 701s, 1402s)
  - Raw multivariate sequences (all 96 sensor channels at 1 Hz)
  - Z-score standardization using TRAIN NORMAL statistics only
  - Training set: normal-only windows from first 80% of dev period
  - Validation set: normal-only windows from last 20% of dev period
  - Dev eval set: all windows from last 20% of dev period (for threshold tuning)
  - Test set: all windows from test period
  - Labels: shifted binary + current failure flag

Outputs 9 configs (3 windows x 3 horizons) to outputs/unsupervised_datasets/
"""

import glob
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\Emırhan\Desktop\MASTER THESIS\Kurtulus-thesis")
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
OUT_DIR = ROOT / "outputs" / "unsupervised_datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_CSV = ROOT / "data" / "asset_data" / "failure.csv"
REVISION_CSV = ROOT / "data" / "asset_data" / "revision.csv"

# ── Time boundaries ──────────────────────────────────────────────────────────
DEV_START = pd.Timestamp("2024-06-01")
DEV_END = pd.Timestamp("2025-02-11")
TEST_END = pd.Timestamp("2025-07-01")

# ── Window configs ───────────────────────────────────────────────────────────
WINDOW_CONFIGS = [
    (350, "6m"),
    (701, "11m"),
    (1402, "23m"),
]

LABEL_HORIZONS = [
    (1, "no_shift"),
    (11163, "11163_shift"),
    (22326, "22326_shift"),
]

# ── Sensor classification ───────────────────────────────────────────────────
BINARY_SUBSTRINGS = (
    "_ACTIVE", "_RUNNING", "_AVAILABLE",
    "MANUAL_MODE", "AUTOMATIC_MODE", "EMERGENCY_MODE",
    "BRAKE_SIGNAL",
)

META_COLS = [
    "TRAIN_IS_IN_FAILURE", "TRAIN_FAILURE_TYPE",
    "TRAIN_IS_IN_MAINTENANCE", "TRAIN_MAINTENANCE_TYPE",
    "TIMESTAMP", "year", "month", "day",
    "TRAIN_LINE", "TRAIN_CURRENT_SECTION", "TRAIN_IS_SPECIAL_SECTION",
]


def is_binary(col):
    return any(s in col for s in BINARY_SUBSTRINGS)


def parse_failure_intervals(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    intervals = []
    for _, row in df.iterrows():
        try:
            start_date, start_time = row.iloc[0], row.iloc[1]
            end_date, end_time = row.iloc[2], row.iloc[3]
            if pd.isna(start_time) or pd.isna(end_time):
                continue
            start = pd.to_datetime(f"{start_date} {start_time}", dayfirst=True)
            end = pd.to_datetime(f"{end_date} {end_time}", dayfirst=True)
            intervals.append((start, end))
        except Exception:
            continue
    return sorted(intervals, key=lambda x: x[0])


def parse_revision_intervals(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    intervals = []
    for _, row in df.iterrows():
        try:
            start = pd.Timestamp(row["start"])
            end = pd.Timestamp(row["end"])
            intervals.append((start, end))
        except Exception:
            continue
    return sorted(intervals, key=lambda x: x[0])


def is_in_failure_at(ts, failure_intervals):
    for start, end in failure_intervals:
        if start <= ts <= end:
            return True
        if start > ts:
            break
    return False


def failure_in_horizon(ts, horizon_s, failure_intervals):
    horizon_end = ts + timedelta(seconds=horizon_s)
    for start, end in failure_intervals:
        if start < horizon_end and end > ts:
            return True
        if start >= horizon_end:
            break
    return False


def get_sorted_parquet_files(directory):
    return sorted(glob.glob(str(directory / "**" / "*.parquet"), recursive=True))


def build_raw_windows(files, window_s, sensor_cols, binary_cols):
    """Build raw sequence windows from parquet files.

    Returns list of dicts with 'window_end', 'sequences' (dict of arrays),
    'is_in_maintenance' flag.
    """
    all_windows = []

    for fpath in tqdm(files, desc=f"W={window_s}s", leave=False):
        df = pd.read_parquet(fpath)
        if "TIMESTAMP" not in df.columns:
            continue

        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df = df.sort_values("TIMESTAMP").reset_index(drop=True)

        # Filter out maintenance periods
        df = df[df["TRAIN_IS_IN_MAINTENANCE"] == False].reset_index(drop=True)
        if len(df) < window_s * 0.8:
            continue

        ts = df["TIMESTAMP"]
        t0 = ts.iloc[0]
        offsets = (ts - t0).dt.total_seconds().values
        window_idx = (offsets // window_s).astype(int)
        df["_window_idx"] = window_idx

        for widx, group in df.groupby("_window_idx"):
            n_obs = len(group)
            if n_obs < window_s * 0.8:
                continue

            window_end = group["TIMESTAMP"].iloc[-1]

            # Build interpolated sequences of exact length window_s
            group_ts = group["TIMESTAMP"]
            t_start = group_ts.iloc[0]
            obs_offsets = (group_ts - t_start).dt.total_seconds().values
            grid = np.arange(window_s, dtype=float)

            sequences = {}
            for col in sensor_cols:
                vals = group[col].values.astype(np.float64)

                if col in binary_cols:
                    # Forward-fill interpolation for binary
                    if len(vals) <= 2:
                        seq = np.full(window_s, vals[0], dtype=np.float64)
                    else:
                        seq = np.empty(window_s, dtype=np.float64)
                        seq[:] = vals[0]
                        obs = dict(zip(obs_offsets.astype(int), vals))
                        last = vals[0]
                        for t in range(window_s):
                            if t in obs:
                                last = obs[t]
                            seq[t] = last
                else:
                    # Linear interpolation for analog
                    if len(vals) == 1:
                        seq = np.full(window_s, vals[0], dtype=np.float64)
                    else:
                        seq = np.interp(grid, obs_offsets, vals)

                sequences[col] = seq

            all_windows.append({
                "window_end": window_end,
                "sequences": sequences,
            })

    return all_windows


def compute_channel_stats(windows, sensor_cols):
    """Compute per-channel mean/std from a list of windows."""
    stats = {}
    for col in sensor_cols:
        all_vals = np.concatenate([w["sequences"][col] for w in windows])
        mu = float(np.nanmean(all_vals))
        sd = float(np.nanstd(all_vals))
        if sd == 0.0:
            sd = 1.0
        stats[col] = {"mean": mu, "std": sd}
    return stats


def standardize_windows(windows, stats, sensor_cols):
    """Apply z-score standardization in-place."""
    for w in windows:
        for col in sensor_cols:
            mu = stats[col]["mean"]
            sd = stats[col]["std"]
            w["sequences"][col] = (w["sequences"][col] - mu) / sd


def save_windows(windows, sensor_cols, out_dir, split_name):
    """Save windows as numpy array + metadata CSV.

    Creates:
      {split_name}_X.npy  — shape (n_windows, seq_len, n_channels), float32
      {split_name}_meta.csv — window_end, label, label_old, label_shift
    """
    n = len(windows)
    if n == 0:
        return
    seq_len = len(windows[0]["sequences"][sensor_cols[0]])
    n_ch = len(sensor_cols)

    X = np.empty((n, seq_len, n_ch), dtype=np.float32)
    meta_rows = []

    for i, w in enumerate(windows):
        for j, col in enumerate(sensor_cols):
            X[i, :, j] = w["sequences"][col]
        row = {"window_end": w["window_end"]}
        for key in ("label", "label_old", "label_shift"):
            if key in w:
                row[key] = w[key]
        meta_rows.append(row)

    np.save(out_dir / f"{split_name}_X.npy", X)
    pd.DataFrame(meta_rows).to_csv(out_dir / f"{split_name}_meta.csv", index=False)
    size_mb = X.nbytes / 1024 / 1024
    print(f"      {split_name}_X.npy: {X.shape} ({size_mb:.0f} MB)")


def main():
    print("=" * 60)
    print("Creating Unsupervised Sequence Datasets (Helm methodology)")
    print("=" * 60)

    # Parse intervals
    failure_intervals = parse_failure_intervals(FAILURE_CSV)
    revision_intervals = parse_revision_intervals(REVISION_CSV)
    print(f"Failure intervals: {len(failure_intervals)}")
    print(f"Revision intervals: {len(revision_intervals)}")

    # Get files
    train_files = get_sorted_parquet_files(TRAIN_DIR)
    test_files = get_sorted_parquet_files(TEST_DIR)
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

    # Determine sensor columns
    sample = pd.read_parquet(train_files[0])
    sensor_cols = [c for c in sample.columns if c not in META_COLS]
    binary_cols = [c for c in sensor_cols if is_binary(c)]
    print(f"Total channels: {len(sensor_cols)} ({len(binary_cols)} binary)")

    for window_s, window_tag in WINDOW_CONFIGS:
        print(f"\n{'-' * 40}")
        print(f"Sequence window: {window_tag} ({window_s}s)")
        print(f"{'-' * 40}")

        # Build raw windows
        print("  Building train windows...")
        dev_windows = build_raw_windows(
            train_files, window_s, sensor_cols, binary_cols
        )
        print(f"  Dev windows: {len(dev_windows)}")

        print("  Building test windows...")
        test_windows = build_raw_windows(
            test_files, window_s, sensor_cols, binary_cols
        )
        print(f"  Test windows: {len(test_windows)}")

        # For each label horizon
        for horizon_s, shift_tag in LABEL_HORIZONS:
            print(f"\n  Label horizon: {shift_tag} ({horizon_s}s)")

            # Add labels to all windows
            for w in dev_windows + test_windows:
                ts = w["window_end"]
                w["label_old"] = 1 if is_in_failure_at(ts, failure_intervals) else 0
                w["label_shift"] = (
                    1 if failure_in_horizon(ts, horizon_s, failure_intervals) else 0
                )

            # Split dev into train pool (80%) and dev eval (20%) by time
            dev_times = sorted([w["window_end"] for w in dev_windows])
            if len(dev_times) == 0:
                print("    [WARN] No dev windows, skipping")
                continue
            cutoff_idx = int(len(dev_times) * 0.8)
            eval_cutoff = dev_times[min(cutoff_idx, len(dev_times) - 1)]

            train_pool = [w for w in dev_windows if w["window_end"] < eval_cutoff]
            dev_eval = [w for w in dev_windows if w["window_end"] >= eval_cutoff]

            # Normal-only windows for AE training
            train_normal = [
                w for w in train_pool
                if w["label_old"] == 0 and w["label_shift"] == 0
            ]

            # Split normal into train (80%) and val (20%)
            train_normal.sort(key=lambda w: w["window_end"])
            if len(train_normal) == 0:
                print("    [WARN] No normal training windows, skipping")
                continue
            norm_cut_idx = int(len(train_normal) * 0.8)
            norm_cutoff = train_normal[min(norm_cut_idx, len(train_normal) - 1)]["window_end"]

            train_final = [w for w in train_normal if w["window_end"] < norm_cutoff]
            val_final = [w for w in train_normal if w["window_end"] >= norm_cutoff]

            # Compute standardization stats from train_final only
            print("    Computing channel statistics from normal train windows...")
            stats = compute_channel_stats(train_final, sensor_cols)

            # Standardize all window sets
            standardize_windows(train_final, stats, sensor_cols)
            standardize_windows(val_final, stats, sensor_cols)
            standardize_windows(dev_eval, stats, sensor_cols)
            standardize_windows(test_windows, stats, sensor_cols)

            # Add label field for convenience
            for w in dev_eval:
                w["label"] = w["label_shift"]
            for w in test_windows:
                w["label"] = w["label_shift"]

            # Save
            out_path = OUT_DIR / f"{window_tag}_{shift_tag}"
            out_path.mkdir(parents=True, exist_ok=True)

            print(f"    Saving train ({len(train_final)} windows)...")
            save_windows(train_final, sensor_cols, out_path, "train")

            print(f"    Saving val ({len(val_final)} windows)...")
            save_windows(val_final, sensor_cols, out_path, "val")

            print(f"    Saving dev_eval ({len(dev_eval)} windows)...")
            save_windows(dev_eval, sensor_cols, out_path, "dev_eval")

            print(f"    Saving test ({len(test_windows)} windows)...")
            save_windows(test_windows, sensor_cols, out_path, "test")

            # Save metadata
            meta = {
                "channels": sensor_cols,
                "seq_len_s": window_s,
                "stride_s": window_s,
                "standardization": stats,
                "dev_range": [str(DEV_START), str(DEV_END)],
                "test_range": [str(DEV_END), str(TEST_END)],
                "n_train": len(train_final),
                "n_val": len(val_final),
                "n_dev_eval": len(dev_eval),
                "n_test": len(test_windows),
            }
            with open(out_path / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    Saved to {out_path}")

            # Un-standardize test_windows for reuse in next horizon
            # (they'll be re-standardized with potentially different stats)
            for w in test_windows:
                for col in sensor_cols:
                    mu = stats[col]["mean"]
                    sd = stats[col]["std"]
                    w["sequences"][col] = w["sequences"][col] * sd + mu

    print("\n" + "=" * 60)
    print("Done! All unsupervised datasets created.")
    print("=" * 60)


if __name__ == "__main__":
    main()

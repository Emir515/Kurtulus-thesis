"""
Create supervised feature datasets following Stefan Helm's methodology.

Replicates the exact pipeline from Helm (2026) / Steiner et al. (RAMS 2026):
  - Non-overlapping windows of size Wfeat (350s, 701s, 1402s)
  - Analog sensors: min, max, mean, sum per window
  - Binary sensors: active_s (seconds active), flips (state transitions)
  - Asset features: days_since_last_failure, days_since_last_revision
  - Label: shifted binary (will failure occur within Wlabel seconds?)
  - Filtering: keep window if shifted_label==1 OR (shifted_label==0 AND current_label==0)
  - Revision/maintenance periods excluded

Outputs 9 configs (3 windows x 3 horizons) to outputs/supervised_datasets/
"""

import glob
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
OUT_DIR = ROOT / "outputs" / "supervised_datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_CSV = ROOT / "data" / "asset_data" / "failure.csv"
REVISION_CSV = ROOT / "data" / "asset_data" / "revision.csv"

# ── Time boundaries ──────────────────────────────────────────────────────────
DEV_START = pd.Timestamp("2024-06-01")
DEV_END = pd.Timestamp("2025-02-11")  # exclusive — test starts here
TEST_END = pd.Timestamp("2025-07-01")  # exclusive

# ── Window configs ───────────────────────────────────────────────────────────
WINDOW_CONFIGS = [
    (350, "6m"),    # ~6 min = 0.5 * tau_corr
    (701, "11m"),   # ~11.5 min = tau_corr
    (1402, "23m"),  # ~23 min = 2 * tau_corr
]

LABEL_HORIZONS = [
    (1, "no_shift"),        # no shift (current failure only)
    (11163, "11163_shift"),  # ~3.1h = 0.5 * tau_lead
    (22326, "22326_shift"),  # ~6.2h = tau_lead
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
    """Parse failure CSV into list of (start, end) datetime tuples."""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    intervals = []
    for _, row in df.iterrows():
        try:
            start_date = row.iloc[0]  # Störungsbeginn
            start_time = row.iloc[1]  # Störungsbeginn Zeit
            end_date = row.iloc[2]    # Störungsende
            end_time = row.iloc[3]    # Störungsende Zeit
            if pd.isna(start_time) or pd.isna(end_time):
                continue
            start = pd.to_datetime(f"{start_date} {start_time}", dayfirst=True)
            end = pd.to_datetime(f"{end_date} {end_time}", dayfirst=True)
            intervals.append((start, end))
        except Exception:
            continue
    return sorted(intervals, key=lambda x: x[0])


def parse_revision_intervals(csv_path):
    """Parse revision CSV into list of (start, end) datetime tuples."""
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


def days_since_last_event(ts, intervals):
    """Compute days since the end of the most recent completed event before ts."""
    best = None
    for start, end in intervals:
        if end <= ts:
            best = end
        elif start > ts:
            break
    if best is None:
        return np.nan
    return (ts - best).total_seconds() / 86400.0


def is_in_failure_at(ts, failure_intervals):
    """Check if timestamp ts falls within any failure interval."""
    for start, end in failure_intervals:
        if start <= ts <= end:
            return True
        if start > ts:
            break
    return False


def failure_in_horizon(ts, horizon_s, failure_intervals):
    """Check if any failure starts or is active within [ts, ts + horizon_s)."""
    horizon_end = ts + timedelta(seconds=horizon_s)
    for start, end in failure_intervals:
        # Failure overlaps with [ts, horizon_end) if start < horizon_end AND end > ts
        if start < horizon_end and end > ts:
            return True
        if start >= horizon_end:
            break
    return False


def is_in_revision_at(ts, revision_intervals):
    """Check if timestamp ts falls within any revision interval."""
    for start, end in revision_intervals:
        if start <= ts <= end:
            return True
        if start > ts:
            break
    return False


def get_sorted_parquet_files(directory):
    """Get all parquet files sorted by date."""
    files = sorted(glob.glob(str(directory / "**" / "*.parquet"), recursive=True))
    return files


def extract_window_features(window_df, analog_cols, binary_cols):
    """Extract Stefan's exact features from a window DataFrame."""
    feats = {}

    for col in analog_cols:
        if col not in window_df.columns:
            continue
        v = window_df[col].values.astype(np.float64)
        feats[f"{col}_min"] = np.nanmin(v)
        feats[f"{col}_max"] = np.nanmax(v)
        feats[f"{col}_mean"] = np.nanmean(v)
        feats[f"{col}_sum"] = np.nansum(v)

    for col in binary_cols:
        if col not in window_df.columns:
            continue
        v = window_df[col].values.astype(np.float64)
        feats[f"{col}_active_s"] = np.nansum(v)
        feats[f"{col}_flips"] = float(np.sum(np.abs(np.diff(np.nan_to_num(v))) > 0.5))

    return feats


def process_files(files, window_s, failure_intervals, revision_intervals,
                  analog_cols, binary_cols):
    """Process a list of parquet files into windowed features."""
    all_rows = []

    for fpath in tqdm(files, desc=f"W={window_s}s", leave=False):
        df = pd.read_parquet(fpath)
        if "TIMESTAMP" not in df.columns:
            continue

        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df = df.sort_values("TIMESTAMP").reset_index(drop=True)

        # Skip if too few rows
        if len(df) < window_s * 0.8:
            continue

        # Filter out revision/maintenance periods
        df = df[df["TRAIN_IS_IN_MAINTENANCE"] == False].reset_index(drop=True)
        if len(df) < window_s * 0.8:
            continue

        # Create non-overlapping windows aligned to first timestamp
        ts = df["TIMESTAMP"]
        t0 = ts.iloc[0]

        # Compute window indices
        offsets = (ts - t0).dt.total_seconds().values
        window_idx = (offsets // window_s).astype(int)
        df["_window_idx"] = window_idx

        for widx, group in df.groupby("_window_idx"):
            if len(group) < window_s * 0.8:  # min 80% coverage
                continue

            window_end = group["TIMESTAMP"].iloc[-1]

            # Extract features
            feats = extract_window_features(group, analog_cols, binary_cols)
            feats["window_end"] = window_end

            # Asset features: use window_end timestamp
            feats["days_since_last_failure"] = days_since_last_event(
                window_end, failure_intervals
            )
            feats["days_since_last_revision"] = days_since_last_event(
                window_end, revision_intervals
            )

            all_rows.append(feats)

    return pd.DataFrame(all_rows)


def add_labels_and_filter(features_df, horizon_s, failure_intervals):
    """Add shifted labels and apply Stefan's filtering rule."""
    labels = []
    old_labels = []

    for _, row in features_df.iterrows():
        ts = row["window_end"]
        # Shifted label: will failure occur within horizon?
        labels.append(
            1 if failure_in_horizon(ts, horizon_s, failure_intervals) else 0
        )
        # Old label: is currently in failure?
        old_labels.append(
            1 if is_in_failure_at(ts, failure_intervals) else 0
        )

    features_df = features_df.copy()
    features_df["label"] = labels
    features_df["old_label"] = old_labels

    # Stefan's filtering rule:
    # Keep if label==1 OR (label==0 AND old_label==0)
    # This discards windows that are in an ongoing failure but don't have
    # a failure within the shifted horizon (avoids trivial negatives)
    mask = (features_df["label"] == 1) | (
        (features_df["label"] == 0) & (features_df["old_label"] == 0)
    )
    features_df = features_df[mask].drop(columns=["old_label"]).reset_index(drop=True)

    return features_df


def main():
    print("=" * 60)
    print("Creating Supervised Feature Datasets (Helm methodology)")
    print("=" * 60)

    # Parse failure and revision intervals
    print("\nParsing failure intervals...")
    failure_intervals = parse_failure_intervals(FAILURE_CSV)
    print(f"  Found {len(failure_intervals)} failure intervals")

    print("Parsing revision intervals...")
    revision_intervals = parse_revision_intervals(REVISION_CSV)
    print(f"  Found {len(revision_intervals)} revision intervals")

    # Get file lists
    train_files = get_sorted_parquet_files(TRAIN_DIR)
    test_files = get_sorted_parquet_files(TEST_DIR)
    print(f"\nTrain files: {len(train_files)}, Test files: {len(test_files)}")

    # Determine sensor columns from first file
    sample = pd.read_parquet(train_files[0])
    all_cols = [c for c in sample.columns if c not in META_COLS]
    analog_cols = [c for c in all_cols if not is_binary(c)]
    binary_cols = [c for c in all_cols if is_binary(c)]
    print(f"Analog sensors: {len(analog_cols)}, Binary sensors: {len(binary_cols)}")
    n_features = len(analog_cols) * 4 + len(binary_cols) * 2 + 2  # +2 asset
    print(f"Features per window: {n_features}")

    # Build datasets for each window size
    for window_s, window_tag in WINDOW_CONFIGS:
        print(f"\n{'-' * 40}")
        print(f"Window: {window_tag} ({window_s}s)")
        print(f"{'-' * 40}")

        # Extract features for train and test
        print("  Processing train files...")
        train_feats = process_files(
            train_files, window_s, failure_intervals, revision_intervals,
            analog_cols, binary_cols
        )
        print(f"  Train windows: {len(train_feats)}")

        print("  Processing test files...")
        test_feats = process_files(
            test_files, window_s, failure_intervals, revision_intervals,
            analog_cols, binary_cols
        )
        print(f"  Test windows: {len(test_feats)}")

        # For each label horizon, create labeled dataset
        for horizon_s, shift_tag in LABEL_HORIZONS:
            print(f"\n  Label horizon: {shift_tag} ({horizon_s}s)")

            # Add labels and filter
            train_labeled = add_labels_and_filter(
                train_feats, horizon_s, failure_intervals
            )
            test_labeled = add_labels_and_filter(
                test_feats, horizon_s, failure_intervals
            )

            # Stats
            train_pos = train_labeled["label"].sum()
            test_pos = test_labeled["label"].sum()
            print(f"    Train: {len(train_labeled)} windows, "
                  f"{train_pos} positive ({100*train_pos/max(len(train_labeled),1):.1f}%)")
            print(f"    Test:  {len(test_labeled)} windows, "
                  f"{test_pos} positive ({100*test_pos/max(len(test_labeled),1):.1f}%)")

            # Save
            out_path = OUT_DIR / f"{window_tag}_{shift_tag}"
            out_path.mkdir(parents=True, exist_ok=True)

            train_labeled.to_parquet(out_path / "train.parquet", index=False)
            test_labeled.to_parquet(out_path / "test.parquet", index=False)

            print(f"    Saved to {out_path}")

    print("\n" + "=" * 60)
    print("Done! All supervised datasets created.")
    print("=" * 60)


if __name__ == "__main__":
    main()

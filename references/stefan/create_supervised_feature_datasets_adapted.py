"""
Adapted from Stefan Helm's create_supervised_feature_datasets.py.
Changes:
  - Polars replaced with pandas + pyarrow (Polars binary broken on this machine)
  - PARQ points to MetroAT partitioned parquet data (English column names)
  - TS_COL = "TIMESTAMP" (capital, matches MetroAT parquet schema)
  - TRAIN_IS_IN_REVISION -> TRAIN_IS_IN_MAINTENANCE
  - Sensor lists replaced with English names from outputs/windows/sensor_list.json
  - Same 9 configs, same output format: OUT_ROOT/{window}_{shift}/features.parquet + test.parquet
"""
from pathlib import Path
import glob, os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
TRAIN_DATA = os.path.expanduser("~/Desktop/MASTER THESIS/Stefan Work/Handover-Stefan/train")
TEST_DATA  = os.path.expanduser("~/Desktop/MASTER THESIS/Stefan Work/Handover-Stefan/test")
OUT_ROOT   = os.path.expanduser("~/Desktop/MASTER THESIS/Stefan Work/supervised_features")

SENSOR_LIST_JSON = os.path.expanduser(
    "~/Desktop/MASTER THESIS/Emirhan-thesis/outputs/windows/sensor_list.json"
)

# -----------------------------------------------------------
# Split boundaries (identical to Stefan)
# -----------------------------------------------------------
DEV_START = pd.Timestamp("2024-06-01")
DEV_END   = pd.Timestamp("2025-02-11")
TEST_END  = pd.Timestamp("2025-06-01")

TS_COL         = "TIMESTAMP"
MAINT_COL      = "TRAIN_IS_IN_MAINTENANCE"
FAILURE_COL    = "TRAIN_IS_IN_FAILURE"

# -----------------------------------------------------------
# Sensor lists (88 sensors, English names)
# -----------------------------------------------------------
with open(SENSOR_LIST_JSON) as f:
    ALL_SENSORS = json.load(f)

# Classify binary vs analog from parquet schema (double=binary, float=analog)
_sample_files = sorted(glob.glob(os.path.join(TRAIN_DATA, "**", "*.parquet"), recursive=True))
_schema = pq.ParquetFile(_sample_files[0]).read().schema
_dtype_map = {field.name: str(field.type) for field in _schema}

BINARY_SENSORS = [s for s in ALL_SENSORS if _dtype_map.get(s, "float") == "double"]
ANALOG_SENSORS = [s for s in ALL_SENSORS if s not in BINARY_SENSORS]
print(f"Sensors: {len(ALL_SENSORS)} total | {len(ANALOG_SENSORS)} analog | {len(BINARY_SENSORS)} binary")

LOAD_COLS = [TS_COL, MAINT_COL, FAILURE_COL] + ALL_SENSORS

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def _flips(series):
    s = series.values
    if len(s) < 2:
        return 0
    return int(np.sum(np.abs(np.diff((s > 0.5).astype(np.int8)))))


def scan_files(root, date_start, date_end):
    """Return sorted list of (date, path) for parquet files within [date_start, date_end)."""
    all_files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    result = []
    for f in all_files:
        parts = Path(f).parts
        try:
            year  = int([p for p in parts if p.startswith("year=")][0].split("=")[1])
            month = int([p for p in parts if p.startswith("month=")][0].split("=")[1])
            day   = int([p for p in parts if p.startswith("day=")][0].split("=")[1])
        except (IndexError, ValueError):
            continue
        d = pd.Timestamp(year, month, day)
        if date_start <= d < date_end:
            result.append((d, f))
    return result


# -----------------------------------------------------------
# Feature engineering (replicates Stefan's create_supervised_feature_datasets)
# -----------------------------------------------------------
def create_supervised_feature_datasets(file_list, window_s):
    """
    Aggregate sensor readings into non-overlapping epoch-aligned windows.
    Maintenance rows are filtered out first (same as Stefan's TRAIN_IS_IN_REVISION==0).
    Returns DataFrame indexed by window_end.
    """
    all_windows = []

    for _, fpath in tqdm(file_list, desc=f"Features {window_s}s"):
        try:
            df = pq.ParquetFile(fpath).read(columns=LOAD_COLS).to_pandas()
        except Exception as e:
            print(f"  skip {fpath}: {e}")
            continue

        df[TS_COL] = pd.to_datetime(df[TS_COL])
        df = df.sort_values(TS_COL)
        df = df[df[MAINT_COL] != True].copy()
        if df.empty:
            continue

        # Use seconds since epoch — works for timestamp[ms] and timestamp[ns]
        ts_s = df[TS_COL].values.astype("datetime64[s]").astype(np.int64)
        df["_wid"] = ts_s // window_s

        agg = {}
        for c in ANALOG_SENSORS:
            if c in df.columns:
                agg[f"{c}_min"]  = pd.NamedAgg(column=c, aggfunc="min")
                agg[f"{c}_max"]  = pd.NamedAgg(column=c, aggfunc="max")
                agg[f"{c}_mean"] = pd.NamedAgg(column=c, aggfunc="mean")
                agg[f"{c}_sum"]  = pd.NamedAgg(column=c, aggfunc="sum")
        for c in BINARY_SENSORS:
            if c in df.columns:
                agg[f"{c}_active_s"] = pd.NamedAgg(column=c, aggfunc="sum")
                agg[f"{c}_flips"]    = pd.NamedAgg(column=c, aggfunc=_flips)
        agg["_in_failure"] = pd.NamedAgg(column=FAILURE_COL, aggfunc="max")

        win_df = df.groupby("_wid").agg(**agg)
        win_df.index = pd.to_datetime((win_df.index + 1) * window_s, unit="s")
        win_df.index.name = "window_end"
        all_windows.append(win_df)

    if not all_windows:
        return pd.DataFrame()
    result = pd.concat(all_windows)
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


# -----------------------------------------------------------
# Label engineering (replicates Stefan's create_shifted_labels)
# -----------------------------------------------------------
def extract_failure_ts(file_list):
    """Return sorted int64 array of failure timestamps in SECONDS since epoch."""
    frames = []
    for _, fpath in tqdm(file_list, desc="Failure schedule"):
        try:
            df = pq.ParquetFile(fpath).read(columns=[TS_COL, FAILURE_COL]).to_pandas()
        except Exception:
            continue
        df[TS_COL] = pd.to_datetime(df[TS_COL])
        frames.append(df[df[FAILURE_COL] == True][TS_COL])
    if not frames:
        return np.array([], dtype=np.int64)
    all_ts = pd.concat(frames).drop_duplicates().sort_values()
    return all_ts.values.astype("datetime64[s]").astype(np.int64)


def create_shifted_labels(window_end_index, failure_ts_s, horizon_s):
    """
    For each window_end T: label=1 if any failure in [T, T+horizon_s).
    All timestamps and horizon are in seconds since epoch.
    """
    if len(failure_ts_s) == 0:
        return np.zeros(len(window_end_index), dtype=np.int8)
    t_arr = window_end_index.values.astype("datetime64[s]").astype(np.int64)
    idx = np.searchsorted(failure_ts_s, t_arr, side="left")
    labels = np.zeros(len(t_arr), dtype=np.int8)
    valid = idx < len(failure_ts_s)
    within = np.zeros(len(t_arr), dtype=bool)
    within[valid] = failure_ts_s[idx[valid]] < t_arr[valid] + horizon_s
    labels[within] = 1
    return labels


# -----------------------------------------------------------
# Main (replicates Stefan's __main__ block)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Scanning files...")
    dev_files  = scan_files(TRAIN_DATA, DEV_START, DEV_END)
    test_files = scan_files(TEST_DATA,  DEV_END,   TEST_END)
    print(f"DEV files: {len(dev_files)}  |  TEST files: {len(test_files)}")

    # Window sizes (seconds): 6min, ~11.5min, ~23min
    WINDOW_SIZES = {"6m": 350, "11m": 701, "23m": 1402}

    # Label horizons (seconds): no shift (current), 3.1h, 6.2h
    HORIZONS = {"no_shift": 1, "11163_shift": 11163, "22326_shift": 22326}

    configs = [
        ("6m",  "no_shift"),
        ("11m", "no_shift"),
        ("23m", "no_shift"),
        ("6m",  "11163_shift"),
        ("11m", "11163_shift"),
        ("23m", "11163_shift"),
        ("6m",  "22326_shift"),
        ("11m", "22326_shift"),
        ("23m", "22326_shift"),
    ]

    # ---- Pre-compute features for each window size ----
    print("\n=== Feature Engineering ===")
    dev_feats  = {}
    test_feats = {}
    for name, ws in WINDOW_SIZES.items():
        print(f"\n--- DEV window={name} ({ws}s) ---")
        dev_feats[name]  = create_supervised_feature_datasets(dev_files,  ws)
        print(f"\n--- TEST window={name} ({ws}s) ---")
        test_feats[name] = create_supervised_feature_datasets(test_files, ws)

    # ---- Pre-compute failure schedules ----
    print("\n=== Failure Schedules ===")
    dev_fail_ts  = extract_failure_ts(dev_files)
    test_fail_ts = extract_failure_ts(test_files)
    print(f"DEV failure seconds : {len(dev_fail_ts):,}")
    print(f"TEST failure seconds: {len(test_fail_ts):,}")

    # ---- Build 9 datasets and save ----
    print("\n=== Building Datasets ===")
    for window_name, shift_name in tqdm(configs, desc="Configs"):
        ws = WINDOW_SIZES[window_name]
        hs = HORIZONS[shift_name]

        feats_dev  = dev_feats[window_name].copy()
        feats_test = test_feats[window_name].copy()

        # Shifted labels
        feats_dev["label"]  = create_shifted_labels(feats_dev.index,  dev_fail_ts,  hs)
        feats_test["label"] = create_shifted_labels(feats_test.index, test_fail_ts, hs)

        # Current labels (horizon=1s) — for filter rule
        feats_dev["_old"]  = create_shifted_labels(feats_dev.index,  dev_fail_ts,  1)
        feats_test["_old"] = create_shifted_labels(feats_test.index, test_fail_ts, 1)

        # Filter rule: keep label==1 OR (label==0 AND old_label==0)
        mask_dev  = (feats_dev["label"]  == 1) | ((feats_dev["label"]  == 0) & (feats_dev["_old"]  == 0))
        mask_test = (feats_test["label"] == 1) | ((feats_test["label"] == 0) & (feats_test["_old"] == 0))

        combined_dev  = feats_dev[mask_dev].drop(columns=["_in_failure", "_old"]).reset_index()
        combined_test = feats_test[mask_test].drop(columns=["_in_failure", "_old"]).reset_index()

        out_dir = Path(OUT_ROOT) / f"{window_name}_{shift_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        combined_dev.to_parquet(out_dir / "features.parquet", index=False, compression="zstd")
        combined_test.to_parquet(out_dir / "test.parquet",    index=False, compression="zstd")

        pos_dev  = combined_dev["label"].sum()
        pos_test = combined_test["label"].sum()
        print(f"  {window_name}_{shift_name}: DEV {len(combined_dev):,} rows ({pos_dev} pos) | TEST {len(combined_test):,} rows ({pos_test} pos)")

    print(f"\nDone. Output: {OUT_ROOT}")

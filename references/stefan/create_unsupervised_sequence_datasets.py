# build_lstm_ae_datasets.py
import json
import os
from typing import Dict, List

import numpy as np
import polars as pl

# ========= CONFIG =========
OUT_ROOT = "V6_unsupervised_sequences"  # output base directory
PARQ = r"C:\Users\neosw\Desktop\WL\V6_base_dataset"

TS_COL = "timestamp"
REV_COL = "TRAIN_IS_IN_REVISION"
FAIL_COL = "TRAIN_IS_IN_FAILURE"

BINARY_SENSORS = [
    "SW2_P_BREMSE_WIRKSAM",
    "MW1_C_DRUCK_DG1_VORHANDEN",
    "MW4_C_DRUCK_DG1_VORHANDEN",
    "MW4_C_DRUCK_DG2_VORHANDEN",
    "SW1_C_DRUCK_DG1_VORHANDEN",
    "MW3_C_DRUCK_DG1_VORHANDEN",
    "SW1_C_DRUCK_DG2_VORHANDEN",
    "SW1_FSP_ANGELEGT_DG1",
    "MW2_C_DRUCK_DG1_VORHANDEN",
    "SW1_FSP_ANGELEGT_DG2",
    "SW2_C_DRUCK_DG2_VORHANDEN",
    "SW2_FSP_ANGELEGT_DG2",
    "MW1_C_DRUCK_DG2_VORHANDEN",
    "SW1_P_BREMSE_WIRKSAM",
    "SW2_KOMPRESSOR_LAEUFT",
    "SW1_KOMPRESSOR_LAEUFT",
    "SW2_C_DRUCK_DG1_VORHANDEN",
    "MW3_C_DRUCK_DG2_VORHANDEN",
    "MW2_C_DRUCK_DG2_VORHANDEN",
    "SW2_FSP_ANGELEGT_DG1",
    "TRAIN_BREMSBEFEHL",
]

CHANNELS = [
    "SW2_P_BREMSE_WIRKSAM",
    "MW1_C_DRUCK_DG1_VORHANDEN",
    "MW4_C_DRUCK_DG1_VORHANDEN",
    "MW4_C_DRUCK_DG2_VORHANDEN",
    "SW1_C_DRUCK_DG1_VORHANDEN",
    "MW3_C_DRUCK_DG1_VORHANDEN",
    "SW1_C_DRUCK_DG2_VORHANDEN",
    "SW1_FSP_ANGELEGT_DG1",
    "MW2_C_DRUCK_DG1_VORHANDEN",
    "SW1_FSP_ANGELEGT_DG2",
    "SW2_C_DRUCK_DG2_VORHANDEN",
    "SW2_FSP_ANGELEGT_DG2",
    "MW1_C_DRUCK_DG2_VORHANDEN",
    "SW1_P_BREMSE_WIRKSAM",
    "SW2_KOMPRESSOR_LAEUFT",
    "SW1_KOMPRESSOR_LAEUFT",
    "SW2_C_DRUCK_DG1_VORHANDEN",
    "MW3_C_DRUCK_DG2_VORHANDEN",
    "MW2_C_DRUCK_DG2_VORHANDEN",
    "SW2_FSP_ANGELEGT_DG1",
    "TRAIN_BREMSBEFEHL",
    "SW1_T_DRUCK_DG1",
    "MW4_ENERGIE_BREMSWIDERSTAND",
    "MW4_P_BREMSKRAFT_DG1",
    "MW2_C_DRUCK_DG1",
    "MW3_T_DRUCK_DG1",
    "MW1_LASTSIGNAL_M",
    "MW2_T_DRUCK_DG2",
    "MW4_FSP_DRUCK_DG2",
    "MW1_P_BREMSKRAFT_DG2",
    "MW3_P_BREMSKRAFT_DG2",
    "MW3_C_DRUCK_DG2",
    "SW1_P_BREMSKRAFT_DG1",
    "SW2_FSP_DRUCK_DG1",
    "SW1_FSP_DRUCK_DG1",
    "SW1_LASTSIGNAL_S",
    "MW1_ENERGIE_BREMSWIDERSTAND",
    "MW1_CV_DRUCK_DG2",
    "SW2_LASTSIGNAL_S",
    "SW2_HBL_DRUCK",
    "MW2_P_BREMSKRAFT_DG1",
    "MW1_CV_DRUCK_DG1",
    "SW1_FSP_DRUCK_DG2",
    "SW2_CV_DRUCK_DG2",
    "MW4_T_DRUCK_DG2",
    "SW2_T_DRUCK_DG2",
    "MW2_FSP_DRUCK_DG1",
    "MW2_ENERGIE_BREMSWIDERSTAND",
    "MW1_P_BREMSKRAFT_DG1",
    "MW3_LASTSIGNAL_M",
    "SW1_CV_DRUCK_DG2",
    "MW3_C_DRUCK_DG1",
    "MW1_C_DRUCK_DG2",
    "SW1_C_DRUCK_DG1",
    "MW1_T_DRUCK_DG2",
    "MW3_ENERGIE_BREMSWIDERSTAND",
    "SW2_C_DRUCK_DG2",
    "SW2_P_BREMSKRAFT_DG2",
    "MW2_CV_DRUCK_DG1",
    "MW3_FSP_DRUCK_DG1",
    "MW4_T_DRUCK_DG1",
    "MW3_FSP_DRUCK_DG2",
    "SW2_T_DRUCK_DG1",
    "MW1_FSP_DRUCK_DG2",
    "MW1_C_DRUCK_DG1",
    "MW2_P_BREMSKRAFT_DG2",
    "MW4_C_DRUCK_DG2",
    "MW4_C_DRUCK_DG1",
    "SW1_T_DRUCK_DG2",
    "MW4_FSP_DRUCK_DG1",
    "MW3_T_DRUCK_DG2",
    "MW4_P_BREMSKRAFT_DG2",
    "MW4_LASTSIGNAL_M",
    "MW4_CV_DRUCK_DG2",
    "SW1_HBL_DRUCK",
    "MW2_FSP_DRUCK_DG2",
    "SW1_P_BREMSKRAFT_DG2",
    "MW3_CV_DRUCK_DG1",
    "MW3_CV_DRUCK_DG2",
    "MW3_P_BREMSKRAFT_DG1",
    "SW1_CV_DRUCK_DG1",
    "MW2_T_DRUCK_DG1",
    "MW2_LASTSIGNAL_M",
    "SW2_FSP_DRUCK_DG2",
    "SW2_P_BREMSKRAFT_DG1",
    "SW2_CV_DRUCK_DG1",
    "MW2_CV_DRUCK_DG2",
    "MW1_FSP_DRUCK_DG1",
    "MW4_CV_DRUCK_DG1",
    "SW1_C_DRUCK_DG2",
    "MW2_C_DRUCK_DG2",
    "MW1_T_DRUCK_DG1",
    "SW2_C_DRUCK_DG1",
    "TRAIN_AUSSENTEMPERATUR",
    "TRAIN_V_IST",
]

# Sequence windows (seconds): 6m, 11m, 23m
SEQ_WINDOWS = [(350, "6m"), (701, "11m"), (1402, "23m")]

# Early-warning horizons (seconds): no shift, 3.1h, 6.3h
LABEL_HORIZONS = [(1, "no_shift"), (11163, "11163_shift"), (22326, "22326_shift")]

# Time boundaries (inclusive start, exclusive end)
DEV_START = pl.datetime(2024, 6, 1)
DEV_END = pl.datetime(2025, 2, 11)
TEST_START = DEV_END
TEST_END = pl.datetime(2025, 6, 1)

# ========= CORE HELPERS =========


def filter_revisions(lf: pl.LazyFrame) -> pl.LazyFrame:
    if REV_COL in lf.collect_schema().names():
        return lf.filter(pl.col(REV_COL) == 0)
    return lf


def create_shifted_labels(lf: pl.LazyFrame, horizon_s: int) -> pl.LazyFrame:
    """
    Label at time t is 1 iff a failure occurs in [t, t + horizon_s).
    """
    return (
        lf.sort(TS_COL)
        .group_by_dynamic(
            index_column=TS_COL,
            every="1s",
            period=f"{horizon_s + 1}s",  # +1s to include right edge at 1 Hz
            closed="left",
        )
        .agg(pl.col(FAIL_COL).max().alias("label"))
        .select([pl.col(TS_COL).alias("window_end"), pl.col("label").cast(pl.Int8)])
    )


def compute_channel_stats(lf: pl.LazyFrame, channels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute per-channel mean/std over the provided LazyFrame (raw 1Hz rows).
    """
    agg_exprs = []
    for c in channels:
        agg_exprs += [pl.col(c).mean().alias(f"{c}__mean"), pl.col(c).std().alias(f"{c}__std")]
    stats = lf.select(agg_exprs).collect().to_dicts()[0]
    out = {}
    for c in channels:
        mu = float(stats.get(f"{c}__mean", 0.0) or 0.0)
        sd = float(stats.get(f"{c}__std", 0.0) or 0.0)
        if sd == 0.0 or sd is None:
            sd = 1.0  # avoid division by zero (e.g., binary channels)
        out[c] = {"mean": mu, "std": sd}
    return out


def standardize(lf: pl.LazyFrame, stats: Dict[str, Dict[str, float]], channels: List[str]) -> pl.LazyFrame:
    exprs = []
    for c in channels:
        mu = stats[c]["mean"]
        sd = stats[c]["std"]
        exprs.append(((pl.col(c) - mu) / sd).alias(c))
    return lf.with_columns(exprs)


def make_sequences(lf_std: pl.LazyFrame, window_s: int) -> pl.LazyFrame:
    """
    Build non-overlapping sequences as list columns per channel.
    Keeps only full windows (length == window_s).
    """

    min_coverage = 0.8
    # 1) Collect timestamps and raw values per window
    rows = (
        lf_std.sort(TS_COL)
        .group_by_dynamic(
            index_column=TS_COL,
            every=f"{window_s}s",
            period=f"{window_s}s",
            include_boundaries=True,
            closed="left",
        )
        .agg([pl.col(TS_COL).implode().alias("ts_list")] + [pl.col(c).implode().alias(f"raw__{c}") for c in CHANNELS])
        .with_columns([pl.col("_upper_boundary").alias("window_end"), pl.col("ts_list").list.len().alias("n_obs")])
        .filter(pl.col("n_obs") >= (pl.lit(window_s) * min_coverage))
    )

    def _interp_row(row: dict) -> dict:
        start = row["_lower_boundary"]
        ts_list = row["ts_list"]
        # offsets (seconds) of observed points within the window
        offs = np.array([int(round((t - start).total_seconds())) for t in ts_list], dtype=int)
        grid = np.arange(window_s, dtype=float)

        out = {}
        for c in CHANNELS:
            vals = np.asarray(row[f"raw__{c}"], dtype=float)

            if vals.size == 0:
                # shouldn't happen due to coverage filter, but keep it safe
                seq = np.full(window_s, np.nan, dtype=float)

            elif c in BINARY_SENSORS:
                if vals.size <= 2:
                    # your rule: only 1–2 observed samples -> fill whole window with the first value
                    seq = np.full(window_s, vals[0], dtype=float)
                else:
                    # stepwise forward-fill on the 1 Hz grid (no fractional values)
                    seq = np.empty(window_s, dtype=float)
                    seq[:] = vals[0]  # left-edge hold
                    obs = dict(zip(offs, vals, strict=False))  # positions -> values
                    last = vals[0]
                    for t in range(window_s):
                        if t in obs:
                            last = obs[t]
                        seq[t] = last

            else:
                # analog channel
                if vals.size == 1:
                    seq = np.full(window_s, vals[0], dtype=float)
                else:
                    # linear interpolation, clamped to edge values outside observed range
                    seq = np.interp(grid, offs.astype(float), vals)

            out[f"seq__{c}"] = seq.tolist()

        return out

    seq = (
        rows.with_columns(
            pl.struct(["_lower_boundary", "ts_list"] + [f"raw__{c}" for c in CHANNELS])
            .map_elements(_interp_row)
            .alias("_seqs")
        )
        .unnest("_seqs")
        .drop(["ts_list", "n_obs"] + [f"raw__{c}" for c in CHANNELS] + ["_lower_boundary", "_upper_boundary"])
        .sort("window_end")
    )

    # ensure all seq columns hat length window_s
    for c in CHANNELS:
        seq = seq.filter(pl.col(f"seq__{c}").list.len() == window_s)

    return seq


def write_parquet(df: pl.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path, compression="zstd")


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ========= MAIN BUILD =========
def build_all_datasets(raw_lf: pl.LazyFrame):
    # mask out revisions globally
    lf_nr = filter_revisions(raw_lf)

    # Precompute "no-shift" labels once (for exclusion)
    labels_no = create_shifted_labels(lf_nr, horizon_s=1).rename({"label": "label_old"})

    for window_s, window_tag in SEQ_WINDOWS:
        # Build sequences after standardization: but we need stats from NORMAL TRAIN in DEV range.
        # We’ll loop horizons to compute the (old==0 & new==0) mask for stats.
        for horizon_s, shift_tag in LABEL_HORIZONS:
            print(f"== Building {window_tag} / {shift_tag} ==")

            labels_new = create_shifted_labels(lf_nr, horizon_s=horizon_s).rename({"label": "label_new"})

            # rename timestamp column to window_end for joining
            lf_nr = lf_nr.with_columns(pl.col(TS_COL).alias("window_end"))

            # Join labels onto raw (per second) rows to form masks by timestamp
            base = (
                lf_nr.join(labels_no, on=["window_end"], how="left")
                .join(labels_new, on=["window_end"], how="left")
                .with_columns(
                    [
                        pl.col("label_old").fill_null(0).cast(pl.Int8),
                        pl.col("label_new").fill_null(0).cast(pl.Int8),
                    ]
                )
            )

            # DEV and TEST row-level masks (per-second)
            dev_rows = base.filter((pl.col(TS_COL) >= DEV_START) & (pl.col(TS_COL) < DEV_END))
            test_rows = base.filter((pl.col(TS_COL) >= TEST_START) & (pl.col(TS_COL) < TEST_END))

            # TRAIN NORMAL rows = DEV rows where old==0 AND new==0
            train_norm_rows = dev_rows.filter((pl.col("label_old") == 0) & (pl.col("label_new") == 0))

            # Compute channel stats from TRAIN NORMAL rows only
            stats = compute_channel_stats(train_norm_rows, CHANNELS)

            # Standardize DEV and TEST rows with these stats
            dev_std = standardize(dev_rows, stats, CHANNELS)
            test_std = standardize(test_rows, stats, CHANNELS)

            # Build sequences
            dev_seq = make_sequences(dev_std, window_s=window_s)
            test_seq = make_sequences(test_std, window_s=window_s)

            # Bring labels to sequence level via join on window_end
            dev_seq = (
                dev_seq.join(labels_no, on=["window_end"], how="left")
                .join(labels_new, on=["window_end"], how="left")
                .with_columns(
                    [
                        pl.col("label_old").fill_null(0).cast(pl.Int8),
                        pl.col("label_new").fill_null(0).cast(pl.Int8),  # label_new -> label for convenience
                    ]
                )
                .rename({"label_new": "label_shift"})
            )

            test_seq = (
                test_seq.join(labels_no, on=["window_end"], how="left")
                .join(labels_new, on=["window_end"], how="left")
                .with_columns(
                    [pl.col("label_old").fill_null(0).cast(pl.Int8), pl.col("label_new").fill_null(0).cast(pl.Int8)]
                )
                .rename({"label_new": "label_shift"})
            )

            dev_times_all = dev_seq.select("window_end").collect().to_series().sort()
            if len(dev_times_all) == 0:
                print(f"[WARN] No DEV sequences for {window_tag}/{shift_tag}; skipping.")
                continue
            eval_cutoff_idx = int(len(dev_times_all) * 0.8)
            eval_cutoff_ts = dev_times_all.item(min(eval_cutoff_idx, len(dev_times_all) - 1))

            dev_eval = dev_seq.filter(pl.col("window_end") >= eval_cutoff_ts).with_columns(
                pl.col("label_shift").alias("label")
            )

            # TRAIN POOL: first 80% of DEV, then pick normal-only for AE training/val
            train_pool = dev_seq.filter(pl.col("window_end") < eval_cutoff_ts)

            # normal-only windows for AE
            train_norm = train_pool.filter((pl.col("label_old") == 0) & (pl.col("label_shift") == 0))

            # split normal-only into TRAIN / VAL (temporal 80/20)
            times_norm = train_norm.select("window_end").collect().to_series().sort()
            if len(times_norm) == 0:
                print(f"[WARN] No normal windows for training {window_tag}/{shift_tag}; skipping.")
                continue
            cut_idx = int(len(times_norm) * 0.8)
            cut_ts = times_norm.item(min(cut_idx, len(times_norm) - 1))

            train_final = train_norm.filter(pl.col("window_end") < cut_ts)
            val_final = train_norm.filter(pl.col("window_end") >= cut_ts)

            # Output paths
            out_dir = os.path.join(OUT_ROOT, f"{window_tag}_{shift_tag}")
            meta = {
                "channels": CHANNELS,
                "seq_len_s": window_s,
                "stride_s": window_s,
                "standardization": stats,
                "dev_range": [str(DEV_START), str(DEV_END)],
                "test_range": [str(TEST_START), str(TEST_END)],
                "labels": {"old": "current failure", "shift": shift_tag},
            }

            # Save
            write_parquet(train_final.collect(), os.path.join(out_dir, "train.parquet"))
            write_parquet(val_final.collect(), os.path.join(out_dir, "val.parquet"))
            write_parquet(dev_eval.collect(), os.path.join(out_dir, "dev_eval.parquet"))
            write_parquet(test_seq.collect(), os.path.join(out_dir, "test.parquet"))
            write_json(meta, os.path.join(out_dir, "metadata.json"))

            print(
                f"Saved: {out_dir}  (train={train_final.collect().height}, val={val_final.collect().height}, test={test_seq.collect().height})"
            )


# ========= USAGE =========
if __name__ == "__main__":
    raw_lf = (
        (pl.scan_parquet(PARQ))
        .with_columns(
            pl.col(TS_COL).cast(pl.Datetime).alias(TS_COL),
            pl.col("TRAIN_IS_IN_FAILURE").fill_null(False).cast(pl.Int8),
            pl.col("TRAIN_IS_IN_REVISION").fill_null(False).cast(pl.Int8),
        )
        .sort(TS_COL)
    )
    build_all_datasets(raw_lf)

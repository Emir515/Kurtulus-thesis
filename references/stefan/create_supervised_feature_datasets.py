from pathlib import Path

import polars as pl
from tqdm import tqdm

# -----------------------------
# Ranges
# -----------------------------
DEV_START = pl.datetime(2024, 6, 1)
DEV_END = pl.datetime(2025, 2, 11)  # exclusive
TEST_END = pl.datetime(2025, 6, 1)  # exclusive

PARQ = r"C:\Users\neosw\Desktop\WL\V6_base_dataset"
OUT_ROOT_NAME = r"C:\Users\neosw\Desktop\WL\V6_supervised_features"
TS_COL = "timestamp"

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

ANALOG_SENSORS = [
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


# -----------------------------
# Helpers
# -----------------------------
def _compute_asset_lookbacks(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute lookbacks on the FULL time range, then we can time-split later.
    """
    raw = (
        lf.with_columns(
            [
                ((pl.col("TRAIN_IS_IN_FAILURE") == 1) & (pl.col("TRAIN_IS_IN_FAILURE").shift(-1) == 0)).alias(
                    "_fail_end_flag"
                ),
                ((pl.col("TRAIN_IS_IN_REVISION") == 1) & (pl.col("TRAIN_IS_IN_REVISION").shift(-1) == 0)).alias(
                    "_rev_end_flag"
                ),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_fail_end_flag")).then(pl.col(TS_COL)).otherwise(None).alias("_fail_end_ts"),
                pl.when(pl.col("_rev_end_flag")).then(pl.col(TS_COL)).otherwise(None).alias("_rev_end_ts"),
            ]
        )
        .with_columns(
            [
                pl.col("_fail_end_ts").forward_fill().alias("last_failure_end_ts"),
                pl.col("_rev_end_ts").forward_fill().alias("last_revision_end_ts"),
            ]
        )
        .with_columns(
            [
                ((pl.col(TS_COL) - pl.col("last_failure_end_ts")).dt.total_seconds() / 86400).alias(
                    "days_since_last_failure"
                ),
                ((pl.col(TS_COL) - pl.col("last_revision_end_ts")).dt.total_seconds() / 86400).alias(
                    "days_since_last_revision"
                ),
            ]
        )
        .drop(
            [
                "_fail_end_flag",
                "_rev_end_flag",
                "_fail_end_ts",
                "_rev_end_ts",
                "last_failure_end_ts",
                "last_revision_end_ts",
            ]
        )
    )
    return raw


def create_supervised_feature_datasets(lf_full: pl.LazyFrame, window_s: int) -> pl.LazyFrame:
    """
    Build features over the FULL time span, then we'll split into DEV/TEST by window_end.
    """
    raw = _compute_asset_lookbacks(lf_full).filter(pl.col("TRAIN_IS_IN_REVISION") == 0)

    analog_aggs = []
    for c in ANALOG_SENSORS:
        analog_aggs += [
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).sum().alias(f"{c}_sum"),
        ]

    binary_aggs = []
    for c in BINARY_SENSORS:
        binary_aggs += [
            pl.col(c).sum().alias(f"{c}_active_s"),
            pl.col(c).cast(pl.Int8).diff().abs().fill_null(0).sum().alias(f"{c}_flips"),
        ]

    context_aggs = [
        pl.col("days_since_last_failure").last().alias("days_since_last_failure"),
        pl.col("days_since_last_revision").last().alias("days_since_last_revision"),
    ]

    feats_all = (
        raw.group_by_dynamic(
            index_column=TS_COL,
            every=f"{window_s}s",
            period=f"{window_s}s",
            include_boundaries=True,
            closed="left",
        )
        .agg(analog_aggs + binary_aggs + context_aggs)
        .with_columns(pl.col("_upper_boundary").alias("window_end"))
        .drop(["_lower_boundary", "_upper_boundary"])
        .sort("window_end")
    )
    return feats_all


def create_shifted_labels(lf_slice: pl.LazyFrame, horizon_s: int) -> pl.LazyFrame:
    """
    Build labels on a *slice* (DEV or TEST) to avoid cross-boundary leakage.
    """
    labels = (
        lf_slice.group_by_dynamic(
            index_column=TS_COL,
            every="1s",
            period=f"{horizon_s}s",
            closed="left",
        )
        .agg([pl.col("TRAIN_IS_IN_FAILURE").max().alias("label")])
        .with_columns(
            [
                pl.col(TS_COL).alias("window_end"),
                pl.col("label").cast(pl.Int8),
            ]
        )
        .select(["window_end", "label"])
    )
    return labels


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load FULL range once (DEV + TEST)
    lf_all = (
        pl.scan_parquet(PARQ)
        .filter(
            (pl.col("year").is_between(2024, 2025, closed="both")) & (pl.col("month").is_between(1, 12, closed="both"))
        )
        .filter(pl.col(TS_COL).is_between(DEV_START, TEST_END, closed="left"))
        .with_columns(
            pl.col(TS_COL).cast(pl.Datetime).alias(TS_COL),
            pl.col("TRAIN_IS_IN_FAILURE").fill_null(False).cast(pl.Int8),
            pl.col("TRAIN_IS_IN_REVISION").fill_null(False).cast(pl.Int8),
        )
        .sort(TS_COL)
    )

    # DEV and TEST lazy slices for *labels*
    lf_dev = lf_all.filter(pl.col(TS_COL).is_between(DEV_START, DEV_END, closed="left"))
    lf_test = lf_all.filter(pl.col(TS_COL).is_between(DEV_END, TEST_END, closed="left"))

    # Features over FULL range (to get proper lookbacks into TEST)
    feats_6m_all = create_supervised_feature_datasets(lf_all, window_s=350)
    feats_11m_all = create_supervised_feature_datasets(lf_all, window_s=701)
    feats_23m_all = create_supervised_feature_datasets(lf_all, window_s=1402)

    # Labels built per-slice to avoid leakage
    labels_no_dev = create_shifted_labels(lf_dev, horizon_s=1)
    labels_11163_dev = create_shifted_labels(lf_dev, horizon_s=11163)
    labels_22326_dev = create_shifted_labels(lf_dev, horizon_s=22326)

    labels_no_test = create_shifted_labels(lf_test, horizon_s=1)
    labels_11163_test = create_shifted_labels(lf_test, horizon_s=11163)
    labels_22326_test = create_shifted_labels(lf_test, horizon_s=22326)

    # Map helpers
    FEATS_ALL = {
        "6m": feats_6m_all,
        "11m": feats_11m_all,
        "23m": feats_23m_all,
    }
    LABELS_DEV = {
        "no_shift": labels_no_dev,
        "11163_shift": labels_11163_dev,
        "22326_shift": labels_22326_dev,
    }
    LABELS_TEST = {
        "no_shift": labels_no_test,
        "11163_shift": labels_11163_test,
        "22326_shift": labels_22326_test,
    }

    configs = [
        ("6m", "no_shift"),
        ("11m", "no_shift"),
        ("23m", "no_shift"),
        ("6m", "11163_shift"),
        ("11m", "11163_shift"),
        ("23m", "11163_shift"),
        ("6m", "22326_shift"),
        ("11m", "22326_shift"),
        ("23m", "22326_shift"),
    ]

    for window, shift in tqdm(configs, desc="Building DEV+TEST per config"):
        feats_all = FEATS_ALL[window]

        # Split features into DEV/TEST by time
        feats_dev = feats_all.filter(pl.col("window_end").is_between(DEV_START, DEV_END, closed="left"))
        feats_test = feats_all.filter(pl.col("window_end").is_between(DEV_END, TEST_END, closed="left"))

        # Pick matching labels (built on the correct slice)
        lbl_new_dev = LABELS_DEV[shift]
        lbl_old_dev = LABELS_DEV["no_shift"].rename({"label": "old_label"})
        lbl_new_test = LABELS_TEST[shift]
        lbl_old_test = LABELS_TEST["no_shift"].rename({"label": "old_label"})

        # DEV: join + filter rule (keep new==1 OR (new==0 & old==0))
        combined_dev = (
            feats_dev.join(lbl_new_dev, on="window_end", how="left")
            .join(lbl_old_dev, on="window_end", how="left")
            .with_columns(
                [
                    pl.col("label").fill_null(0).cast(pl.Int8),
                    pl.col("old_label").fill_null(0).cast(pl.Int8),
                ]
            )
            .filter((pl.col("label") == 1) | ((pl.col("label") == 0) & (pl.col("old_label") == 0)))
            .drop("old_label")
        )

        # TEST: same rule, but with test labels
        combined_test = (
            feats_test.join(lbl_new_test, on="window_end", how="left")
            .join(lbl_old_test, on="window_end", how="left")
            .with_columns(
                [
                    pl.col("label").fill_null(0).cast(pl.Int8),
                    pl.col("old_label").fill_null(0).cast(pl.Int8),
                ]
            )
            .filter((pl.col("label") == 1) | ((pl.col("label") == 0) & (pl.col("old_label") == 0)))
            .drop("old_label")
        )

        # Save
        out_root = Path(OUT_ROOT_NAME) / f"{window}_{shift}"
        out_root.mkdir(parents=True, exist_ok=True)

        combined_dev.sink_parquet((out_root / "features.parquet").as_posix(), compression="zstd")
        combined_test.sink_parquet((out_root / "test.parquet").as_posix(), compression="zstd")

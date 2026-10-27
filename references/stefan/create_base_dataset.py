import polars as pl
from pathlib import Path
from tqdm import tqdm
import json

# ---------- CONFIG ----------
PARQUET_ROOT = Path(r"C:\Users\neosw\Desktop\WL\V6_parquet")     # input dataset root
OUT_ROOT     = Path(r"C:\Users\neosw\Desktop\WL\V6_base_dataset")  # output dataset root
SELECTED_SENSORS_FILE = r"C:\Users\neosw\Desktop\WL\asset_data\relevant_sensors.json"
PORT_NAME_MAPPING_FILE = r"C:\Users\neosw\Desktop\WL\asset_data\port_name_mapping.json"
STATION_MAPPING_FILE   = r"C:\Users\neosw\Desktop\WL\asset_data\stationsinformation.csv"
REVISION_FILE        = r"C:\Users\neosw\Desktop\WL\asset_data\revision.csv"
FAILURE_FILE        = r"C:\Users\neosw\Desktop\WL\asset_data\failure.csv"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- LOAD LOOKUPS ----------
with open(SELECTED_SENSORS_FILE, "r", encoding="utf-8") as f:
    SELECTED_DICT = json.load(f)  # use a set for fast membership
    SELECTED = set(SELECTED_DICT.keys())

with open(PORT_NAME_MAPPING_FILE, "r", encoding="utf-8") as f:
    PORT_MAP = json.load(f)

stations_lf = pl.scan_csv(
    STATION_MAPPING_FILE,
    separator=";",
    encoding="utf8",
    has_header=True,
    dtypes={
        "Steckencode": pl.Int64,
        "Startstation": pl.Utf8,
        "Endstation": pl.Utf8,
        "Linie": pl.Utf8,
        "special": pl.Utf8,
    },
)

# ---------- REVISION INFO (for joins) ----------
revision_df = pl.scan_csv(
    REVISION_FILE,
    separator=";",
    encoding="utf8"
)

# convert start and end to datetime
revision_df = revision_df.with_columns(
    pl.col("start").str.strptime(pl.Datetime, format="%d.%m.%Y %H:%M").alias("start"),
    pl.col("end").str.strptime(pl.Datetime, format="%d.%m.%Y %H:%M").alias("end")
)

revision_mapping = {
    "AJ": "AJ-Revision",
    "A": "A-Revision",
    "Stichprobeninspektion": "Spot Inspection",
    "K": "K-Revision",
    "E": "Gearbox Oil Change",
    "B": "B-Revision",
    "R5 - Wartung Paket R5": "R5 - Maintenance Package R5",
    "Stichprobeninspektion": "Spot Inspection",
    "B-Revision": "B-Revision",
    "Wartung A": "A-Revision",
    "Revisionskontrolle": "Revision Control",
    "1": "Air Conditioning Maintenance",
    "C": "C-Revision",
    "Inspektion nach §61 STRAB VO ": "Inspection according to §61 STRAB VO",
}

#convert dict to dataframe
revision_mapping_df = pl.LazyFrame(
    {"fzgtext": list(revision_mapping.keys()), "revision_type": list(revision_mapping.values())}
)

# Join the mapping DataFrame with the revision DataFrame
revision_df = revision_df.join(revision_mapping_df, on="fzgtext", how="left")

def add_revision_info(df: pl.DataFrame) -> pl.DataFrame:
    """Add TRAIN_IS_IN_REVISION, TRAIN_REVISION_TYPE column based on revision_df."""
    # Add TRAIN_IS_IN_REVISION and TRAIN_REVISION_TYPE fields
    result = df.join(revision_df.select(["start", "end", "revision_type"]), how="cross").filter(
        (pl.col("timestamp") >= pl.col("start")) & (pl.col("timestamp") <= pl.col("end"))
    ).group_by("timestamp", maintain_order=True).agg(
        [
            pl.lit(True).alias("TRAIN_IS_IN_REVISION"),
            pl.col("revision_type").first().alias("TRAIN_REVISION_TYPE")
        ]
    ).join(df, on="timestamp", how="right").with_columns(
        pl.col("TRAIN_IS_IN_REVISION").fill_null(False),
        pl.col("TRAIN_REVISION_TYPE").fill_null("No Revision")
    )
    return result

# ------------ FAILURE INFO (for joins) ----------
# Load the CSV file as a Polars DataFrame
failure_df = pl.scan_csv(
    FAILURE_FILE,
    separator=";",
    encoding="utf8"
)

# concatinate Störungsbeginn and Störungsbeginn Zeit to start timestamp
failure_df = failure_df.with_columns(
    (pl.col("Störungsbeginn") + " " + pl.col("Störungsbeginn Zeit")).str.strptime(pl.Datetime, format="%d.%m.%Y %H:%M:%S").alias("start_timestamp"),
    (pl.col("Störungsende") + " " + pl.col("Störungsende Zeit")).str.strptime(pl.Datetime, format="%d.%m.%Y %H:%M:%S").alias("end_timestamp")
)

# Drop original columns
failure_df = failure_df.drop(["Störungsbeginn", "Störungsbeginn Zeit", "Störungsende", "Störungsende Zeit"])

failure_df = failure_df.with_columns(
    pl.when(pl.col("Subsystem Bezeichner") == "BREMSSYSTEM").then(pl.lit("Brake System Failure"))
    .when(pl.col("Subsystem Bezeichner") == "NIVEAUSYSTEM").then(pl.lit("Leveling System Failure"))
    .when(pl.col("Subsystem Bezeichner") == "KOMPRESSOR").then(pl.lit("Compressor Module Failure"))
    .otherwise(pl.col("Subsystem Bezeichner"))
    .alias("Subsystem Bezeichner")
)

def add_failure_info(df: pl.DataFrame) -> pl.DataFrame:
    """Add TRAIN_IS_IN_FAILURE, TRAIN_FAILURE_TYPE column based on failure_df."""
    # Add TRAIN_IS_IN_FAILURE and TRAIN_FAILURE_TYPE fields
    result = df.join(failure_df.select(["start_timestamp", "end_timestamp", "Subsystem Bezeichner"]), how="cross").filter(
        (pl.col("timestamp") >= pl.col("start_timestamp")) & (pl.col("timestamp") <= pl.col("end_timestamp"))
    ).group_by("timestamp", maintain_order=True).agg(
        [
            pl.lit(True).alias("TRAIN_IS_IN_FAILURE"),
            pl.col("Subsystem Bezeichner").first().alias("TRAIN_FAILURE_TYPE")
        ]
    ).join(df, on="timestamp", how="right").with_columns(
        pl.col("TRAIN_IS_IN_FAILURE").fill_null(False),
        pl.col("TRAIN_FAILURE_TYPE").fill_null("No Failure")
    )
    return result

# ---------- RENAME LOGIC ----------
SIGNAL_PORTS = {'296','297','808','809','936','937','552','553','1064','1065','1192','1193'}
SPECIAL_PORT_TO_MW = {"798": "MW1", "926": "MW2", "1054": "MW4", "1182": "MW3"}

def port_to_wagon(port: str, name_hint: str | None) -> str:
    # explicit overrides
    if port in SPECIAL_PORT_TO_MW:
        return SPECIAL_PORT_TO_MW[port]
    # by textual mapping hint
    if name_hint:
        if "S1" in name_hint: return "SW1"
        if "S2" in name_hint: return "SW2"
        if "M1" in name_hint and "S1" not in name_hint: return "MW1"
        if "M2" in name_hint and "S1" not in name_hint: return "MW2"
        if "M3" in name_hint and "S2" not in name_hint: return "MW3"
        if "M4" in name_hint and "S2" not in name_hint: return "MW4"
    # by numeric prefix fallback
    if port.startswith("2"):  return "SW1"
    if port.startswith("5"):  return "SW2"
    if port.startswith("8"):  return "MW1"
    if port.startswith("9"):  return "MW2"
    if port.startswith("10"): return "MW4"
    if port.startswith("11"): return "MW3"
    return "TRAIN"

def build_daily_rename_map(columns: list[str]) -> dict[str, str]:
    """
    Build an exact rename map {old_col: new_col} from the day's columns.
    This is schema-driven and works lazily (no collect).
    """
    ren = {}
    for col in columns:
        if col == "timestamp" or col in ("year","month","day"):  # keep helpers
            continue
        if "_" not in col:
            continue
        port = col.split("_", 1)[0]  # e.g., "296" from "296_0_Signal_6958"
        name_hint = PORT_MAP.get(port)

        # signal pressure special four-channel mapping
        if port in SIGNAL_PORTS and "_Signal_" in col:
            dg = "DG1" if int(port) % 2 == 0 else "DG2"
            w  = port_to_wagon(port, name_hint)
            # 0..3 → T, C, CV, FSP
            if col.endswith("_6958") or "_0_Signal_6958" in col:
                ren[col] = f"{w}_T_DRUCK_{dg}"
            elif col.endswith("_6959") or "_1_Signal_6959" in col:
                ren[col] = f"{w}_C_DRUCK_{dg}"
            elif col.endswith("_6960") or "_2_Signal_6960" in col:
                ren[col] = f"{w}_CV_DRUCK_{dg}"
            elif col.endswith("_6963") or "_3_Signal_6963" in col:
                ren[col] = f"{w}_FSP_DRUCK_{dg}"
            else:
                # leave unmapped signal names as-is
                pass
            continue

        # generic port → wagon/group mapping
        w = port_to_wagon(port, name_hint)
        if w != "TRAIN":
            # replace the leading port with the wagon/group tag
            rest = col.split("_", 2)[2]
            ren[col] = f"{w}_{rest}"
        else:
            # keep as TRAIN_...
            rest = col.split("_", 2)[2]
            ren[col] = f"TRAIN_{rest}"
    return ren

# ---------- CONVERSION EXPRESSIONS ----------
CONV = {
    "HBL_DRUCK":              lambda s: (8     * s) / 16384,
    "T_DRUCK":                lambda s: (10    * s) / 16384,
    "C_DRUCK":                lambda s: (10    * s) / 16384,
    "CV_DRUCK":               lambda s: (10    * s) / 16384,
    "FSP_DRUCK":              lambda s: (10    * s) / 16384,
    "P_BREMSKRAFT":           lambda s: (71250 * s) / 16384,
    "V_IST":                  lambda s: (75    * s) / 16384,
    "AUSSENTEMPERATUR":       lambda s: (200   * s) / 16384,
    "ENERGIE_BREMSWIDERSTAND":lambda s: (400   * s) / 16384,
    "LASTSIGNAL":             lambda s: (15887 * s) / 16384,
}

def build_conversion_exprs(cols: list[str], column_classification: dict[str, str]) -> list[pl.Expr]:
    """Return a list of pl.Expr casts/derived values for all non-timestamp cols."""
    exprs: list[pl.Expr] = []
    for c in cols:
        if c == "timestamp" or c in ("year","month","day"):
            continue
        if c in column_classification:
            col_type = column_classification[c]
            if col_type == "binary":
                exprs.append(pl.col(c).cast(pl.Int8).alias(c))
                continue
        applied = False
        for key, fn in CONV.items():
            if key in c:
                exprs.append(fn(pl.col(c)).cast(pl.Float32, strict=False).alias(c))  # overwrite with scaled float
                applied = True
                break
        if not applied:
            # make sure ints are Int32 for compactness
            exprs.append(pl.col(c).cast(pl.Float32, strict=False).alias(c))
    return exprs

# ---------- PROCESS ----------
def process_day_dir(day_dir: Path):
    y = int(day_dir.parts[-3].split("=")[1])
    m = int(day_dir.parts[-2].split("=")[1])
    d = int(day_dir.parts[-1].split("=")[1])

    # scan all parquet in day folder
    lf = pl.scan_parquet(day_dir.as_posix())

    # project to timestamp + selected columns (exact names present in dataset)
    present_cols = list(set(lf.columns))  # available without collect
    wanted = ["timestamp"] + [c for c in SELECTED if c in present_cols]
    lf = lf.select(wanted)

    # --- timestamp dtype (in case) & helpers ---
    lf = lf.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp"),
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.day().alias("day"),
    )

    # --- renames (built from actual schema) ---
    rename_map = build_daily_rename_map(wanted)

    column_classification = SELECTED_DICT
    #rename keys of SELECTED_DICT according to rename_map
    for old_col, new_col in rename_map.items():
        if old_col in column_classification:
            column_classification[new_col] = column_classification.pop(old_col)

    if rename_map:
        lf = lf.rename(rename_map)

    #

    # --- join stations lazily ---
    lf = (lf
          .with_columns(pl.col("TRAIN_HALTESTELLENNUMMER").cast(pl.Int64))
          .join(
                stations_lf.select(["Steckencode","Startstation","Endstation","Linie","special"]),
                left_on="TRAIN_HALTESTELLENNUMMER",
                right_on="Steckencode",
                how="left"
          )
          .with_columns([
              (pl.col("Startstation") + pl.lit(" --> ") + pl.col("Endstation")).alias("TRAIN_CURRENT_SECTION"),
              pl.when(pl.col("special") == "Ja").then(True)
                .when(pl.col("special") == "Nein").then(False)
                .otherwise(None).alias("TRAIN_IS_SPECIAL_SECTION"),
          ])
          .drop(["TRAIN_HALTESTELLENNUMMER","Startstation","Endstation","special"])
    )

    #rename Line to TRAIN_LINE
    if "Linie" in lf.columns:
        lf = lf.rename({"Linie": "TRAIN_LINE"})

    # --- aggregates: AUSSENTEMPERATUR / V_IST (lazy) ---
    # collect matching columns dynamically (from the *renamed* schema)
    schema_cols = lf.columns
    aus_cols = [c for c in schema_cols if "AUSSENTEMPERATUR" in c and c != "TRAIN_AUSSENTEMPERATUR"]
    v_cols   = [c for c in schema_cols if "V_IST" in c and c != "TRAIN_V_IST"]

    if aus_cols:
        lf = lf.with_columns(pl.mean_horizontal([pl.col(c) for c in aus_cols]).alias("TRAIN_AUSSENTEMPERATUR")) \
               .drop(aus_cols)
    if v_cols:
        lf = lf.with_columns(pl.mean_horizontal([pl.col(c) for c in v_cols]).alias("TRAIN_V_IST")) \
               .drop(v_cols)

    # --- unit conversions / downcasts (lazy) ---
    conv_exprs = build_conversion_exprs(lf.columns, column_classification)
    if conv_exprs:
        lf = lf.with_columns(conv_exprs)

    # --- add revision info (lazy) ---
    lf = add_revision_info(lf)

    # --- add failure info (lazy) ---
    lf = add_failure_info(lf)

    #drop duplicate timestamps if any
    lf = lf.unique(subset=["timestamp"], keep="first")

    # --- write (lazy sink) ---
    outdir = OUT_ROOT / f"year={y}" / f"month={m:02d}" / f"day={d:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet((outdir / "day.parquet").as_posix(), compression="zstd")

# ---------- RUN ----------
if __name__ == "__main__":
    day_dirs = sorted(PARQUET_ROOT.glob("year=*/month=*/day=*"))
    for day_dir in tqdm(day_dirs, desc="Processing days"):
        process_day_dir(day_dir)

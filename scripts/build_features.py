"""
Build windowed feature datasets for both train and test splits.

Usage:
    python scripts/build_features.py                  # builds train + test
    python scripts/build_features.py --split train
    python scripts/build_features.py --split test
    python scripts/build_features.py --window 360 --stride 60 --horizon 6.2

Output (in outputs/features/):
    features_train_w6m_h6.2h.parquet
    features_test_w6m_h6.2h.parquet

Each row = one time window.
Columns:
    window_start, window_end    : window boundary timestamps
    label_h0                    : 1 if failure is active at any point in the window
    label_h<H>                  : 1 if failure onset is within H hours after window_end
    failure_type                : type of the approaching failure (or 'normal')
    <feature columns>           : per-sensor + group + asset features
"""

import sys, os, glob, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import ROOT, FEATURES_DIR, SUBSYSTEMS
from src.features import extract_window_features, ALL_SENSOR_COLS, ANALOG_COLS, BINARY_COLS

# ── Paths ──────────────────────────────────────────────────────────────────────
ASSET_DIR    = ROOT / 'data' / 'asset_data'
FAILURE_CSV  = ASSET_DIR / 'failure.csv'
REVISION_CSV = ASSET_DIR / 'revision.csv'

TRAIN_GLOB = str(ROOT / 'train' / '**' / '*.parquet')
TEST_GLOB  = str(ROOT / 'test'  / '**' / '*.parquet')

META_COLS = {
    'TIMESTAMP', 'TRAIN_IS_IN_FAILURE', 'TRAIN_FAILURE_TYPE',
    'TRAIN_IS_IN_MAINTENANCE', 'TRAIN_MAINTENANCE_TYPE',
    'year', 'month', 'day',
}

# ── Failure events per split (verified against sensor data in notebook 01) ────
# Used for labeling — these are the ground-truth onset times.
FAILURE_EVENTS = {
    'train': [
        {'onset': '2024-06-27 15:47:30', 'end': '2024-06-27 17:52:28', 'type': 'Brake'},
        {'onset': '2024-08-19 10:37:51', 'end': '2024-08-19 11:05:44', 'type': 'Brake'},
        {'onset': '2024-08-27 15:58:43', 'end': '2024-08-28 21:29:25', 'type': 'Leveling'},
        {'onset': '2024-09-03 05:04:00', 'end': '2024-09-03 07:33:36', 'type': 'Brake'},
        {'onset': '2024-09-22 05:49:43', 'end': '2024-09-22 08:14:45', 'type': 'Brake'},
        {'onset': '2024-10-06 02:04:15', 'end': '2024-10-08 09:15:36', 'type': 'Brake'},
        {'onset': '2024-10-10 11:19:48', 'end': '2024-10-10 15:35:49', 'type': 'Brake'},
        {'onset': '2024-10-31 15:08:06', 'end': '2024-10-31 18:26:02', 'type': 'Compressor'},
        {'onset': '2024-12-14 18:56:40', 'end': '2024-12-14 23:31:05', 'type': 'Brake'},
        {'onset': '2025-01-21 06:11:15', 'end': '2025-01-21 09:45:44', 'type': 'Brake'},
    ],
    'test': [
        {'onset': '2025-02-18 14:40:40', 'end': '2025-02-19 02:40:57', 'type': 'Brake'},
        {'onset': '2025-02-24 17:39:54', 'end': '2025-02-24 22:52:22', 'type': 'Compressor'},
        {'onset': '2025-02-26 12:40:02', 'end': '2025-02-26 16:23:55', 'type': 'Brake'},
        {'onset': '2025-03-14 07:36:26', 'end': '2025-03-14 08:06:05', 'type': 'Brake'},
        {'onset': '2025-04-10 09:34:42', 'end': '2025-04-10 17:30:11', 'type': 'Brake'},
        {'onset': '2025-04-18 01:16:54', 'end': '2025-04-18 01:26:53', 'type': 'Compressor'},
        {'onset': '2025-05-05 10:04:56', 'end': '2025-05-05 12:41:56', 'type': 'Compressor'},
        {'onset': '2025-05-10 20:25:27', 'end': '2025-05-10 23:36:07', 'type': 'Leveling'},
        {'onset': '2025-05-13 06:52:14', 'end': '2025-05-13 18:36:28', 'type': 'Compressor'},
        {'onset': '2025-05-25 08:11:37', 'end': '2025-05-25 13:04:14', 'type': 'Brake'},
        {'onset': '2025-06-18 22:53:40', 'end': '2025-06-24 18:43:51', 'type': 'Compressor'},
    ],
}
for split in FAILURE_EVENTS:
    for ev in FAILURE_EVENTS[split]:
        ev['onset'] = pd.Timestamp(ev['onset'])
        ev['end']   = pd.Timestamp(ev['end'])


# ── Asset data parsing ─────────────────────────────────────────────────────────
def load_failure_datetimes() -> list:
    """Return sorted list of unique failure onset datetimes from failure.csv."""
    df = pd.read_csv(FAILURE_CSV, sep=';', dtype=str)
    date_col = df.columns[0]   # Störungsbeginn  (dd.mm.yyyy)
    time_col = df.columns[1]   # Störungsbeginn Zeit (HH:MM:SS)
    df = df.dropna(subset=[date_col, time_col])
    combined = df[date_col].str.strip() + ' ' + df[time_col].str.strip()
    dts = pd.to_datetime(combined, dayfirst=True, errors='coerce').dropna()
    return sorted(dts.unique().tolist())


def load_revision_datetimes() -> list:
    """Return sorted list of revision start datetimes from revision.csv."""
    df = pd.read_csv(REVISION_CSV, sep=';', dtype=str)
    dts = pd.to_datetime(df['start'], dayfirst=True, errors='coerce').dropna()
    return sorted(dts.unique().tolist())


def get_asset_features(window_end: pd.Timestamp,
                       failure_dts: list,
                       revision_dts: list) -> dict:
    """Days elapsed since the most recent failure / revision before window_end."""
    past_f = [t for t in failure_dts  if t < window_end]
    past_r = [t for t in revision_dts if t < window_end]

    days_f = (window_end - past_f[-1]).days  if past_f else 999
    days_r = (window_end - past_r[-1]).days  if past_r else 999

    return {
        'days_since_last_failure':  float(days_f),
        'days_since_last_revision': float(days_r),
    }


# ── Labeling ──────────────────────────────────────────────────────────────────
def get_labels(window_end: pd.Timestamp,
               window_start: pd.Timestamp,
               events: list,
               horizon_h: float) -> dict:
    """
    Compute all labels for a window.

    Returns dict with keys:
      label_h0          : 1 if any failure active within [window_start, window_end]
      label_brake       : 1 if Brake onset within horizon OR Brake active in window
      label_compressor  : 1 if Compressor onset within horizon OR active in window
      label_leveling    : 1 if Leveling onset within horizon OR active in window
      label_any         : 1 if any of the above type labels is 1
      failure_type      : type of closest upcoming/active failure ('normal' if none)
    """
    H      = pd.Timedelta(hours=horizon_h)
    result = {'label_h0': 0, 'label_brake': 0, 'label_compressor': 0,
              'label_leveling': 0, 'failure_type': 'normal'}

    type_key = {'Brake': 'label_brake', 'Compressor': 'label_compressor', 'Leveling': 'label_leveling'}

    for ev in events:
        onset = ev['onset']
        end   = ev['end']
        key   = type_key.get(ev['type'])

        active_in_window = onset <= window_end and end >= window_start
        approaching      = window_end < onset <= window_end + H

        if active_in_window:
            result['label_h0'] = 1

        if active_in_window or approaching:
            if key:
                result[key] = 1
            result['failure_type'] = ev['type']

    result['label_any'] = int(
        result['label_brake'] or result['label_compressor'] or result['label_leveling']
    )
    return result


# ── File list helpers ─────────────────────────────────────────────────────────
def sorted_parquet_files(split: str) -> list[str]:
    pattern = TRAIN_GLOB if split == 'train' else TEST_GLOB
    return sorted(glob.glob(pattern, recursive=True))


def date_of_file(path: str):
    """Extract date from partition path (year=.../month=.../day=...)."""
    parts = path.replace('\\', '/').split('/')
    try:
        y = int([p for p in parts if p.startswith('year=')][0].split('=')[1])
        m = int([p for p in parts if p.startswith('month=')][0].split('=')[1])
        d = int([p for p in parts if p.startswith('day=')][0].split('=')[1])
        return pd.Timestamp(year=y, month=m, day=d)
    except (IndexError, ValueError):
        return None


# ── Core build function ───────────────────────────────────────────────────────
def build_features(split: str,
                   window_s:  int   = 360,
                   stride_s:  int   = 60,
                   horizon_h: float = 6.2,
                   min_fill:  float = 0.8) -> pd.DataFrame:
    """
    Stream through all parquet files for `split`, slide a window of `window_s`
    seconds with `stride_s` second stride, extract features and labels.

    Parameters
    ----------
    split      : 'train' or 'test'
    window_s   : window size in seconds (default 360 = 6 min)
    stride_s   : stride in seconds between window starts (default 60 = 1 min)
    horizon_h  : failure label look-ahead in hours (default 6.2)
    min_fill   : minimum fraction of window that must have data (default 0.8)

    Returns
    -------
    pd.DataFrame  one row per window
    """
    print(f'\n{"="*60}')
    print(f'Building features: split={split}, window={window_s}s, '
          f'stride={stride_s}s, horizon={horizon_h}h')
    print(f'{"="*60}')

    files = sorted_parquet_files(split)
    print(f'Parquet files found: {len(files)}')

    events       = FAILURE_EVENTS[split]
    failure_dts  = load_failure_datetimes()
    revision_dts = load_revision_datetimes()
    print(f'Failure records: {len(failure_dts)}, Revision records: {len(revision_dts)}')

    # Needed columns: all sensor cols + meta
    needed_cols = list(ALL_SENSOR_COLS) + [
        'TIMESTAMP', 'TRAIN_IS_IN_FAILURE', 'TRAIN_FAILURE_TYPE',
        'TRAIN_IS_IN_MAINTENANCE',
    ]

    min_rows    = int(window_s * min_fill)
    W           = pd.Timedelta(seconds=window_s)
    S           = pd.Timedelta(seconds=stride_s)
    MAX_GAP     = pd.Timedelta(seconds=window_s * 3)  # reset carry on large gaps

    carry_df    = None
    all_rows    = []
    n_skipped_maintenance = 0
    n_skipped_sparse      = 0
    total_windows         = 0

    for file_idx, fpath in enumerate(files):
        # ── Load one day ──────────────────────────────────────────────────────
        day_df = pd.read_parquet(fpath)
        load_cols = [c for c in needed_cols if c in day_df.columns]
        day_df = day_df[load_cols]
        day_df['TIMESTAMP'] = pd.to_datetime(day_df['TIMESTAMP'])
        day_df.sort_values('TIMESTAMP', inplace=True, ignore_index=True)
        day_df.drop_duplicates('TIMESTAMP', inplace=True, ignore_index=True)

        # ── Prepend carry buffer ──────────────────────────────────────────────
        if carry_df is not None:
            gap = day_df['TIMESTAMP'].iloc[0] - carry_df['TIMESTAMP'].iloc[-1]
            if gap <= MAX_GAP:
                day_df = pd.concat([carry_df, day_df], ignore_index=True)
            # else: gap too large, discard carry (train was not running)

        # ── Slide windows ─────────────────────────────────────────────────────
        ts     = day_df['TIMESTAMP']
        t_min  = ts.iloc[0]
        t_max  = ts.iloc[-1]

        t = t_min
        # Align start to stride boundary relative to epoch (reproducible windows)
        epoch = pd.Timestamp('2024-01-01')
        elapsed_s = int((t_min - epoch).total_seconds())
        t = epoch + pd.Timedelta(seconds=(elapsed_s // stride_s) * stride_s)
        if t < t_min:
            t += S

        while t + W <= t_max:
            mask   = (ts >= t) & (ts < t + W)
            w_rows = day_df[mask]

            if len(w_rows) < min_rows:
                n_skipped_sparse += 1
                t += S
                continue

            # Skip windows that overlap with maintenance
            if w_rows['TRAIN_IS_IN_MAINTENANCE'].any():
                n_skipped_maintenance += 1
                t += S
                continue

            window_end = t + W

            # Labels
            lbl = get_labels(window_end, t, events, horizon_h)

            # Asset features (computed at window_end)
            asset_feats = get_asset_features(window_end, failure_dts, revision_dts)

            # Sensor features
            sensor_data = w_rows[[c for c in ALL_SENSOR_COLS if c in w_rows.columns]]
            feats = extract_window_features(sensor_data, asset_feats)

            # Meta columns
            feats['window_start']       = t
            feats['window_end']         = window_end
            feats['label_h0']           = lbl['label_h0']
            feats['label_brake']        = lbl['label_brake']
            feats['label_compressor']   = lbl['label_compressor']
            feats['label_leveling']     = lbl['label_leveling']
            feats['label_any']          = lbl['label_any']
            feats['failure_type']       = lbl['failure_type']

            all_rows.append(feats)
            total_windows += 1
            t += S

        # ── Save carry: last window_s of data ─────────────────────────────────
        carry_cutoff = t_max - W
        carry_df = day_df[ts >= carry_cutoff].copy()

        # Progress
        if (file_idx + 1) % 20 == 0 or file_idx == len(files) - 1:
            fdate = date_of_file(fpath)
            print(f'  [{file_idx+1:3d}/{len(files)}] {fdate}  '
                  f'windows so far: {total_windows:,}', flush=True)

    print(f'\nWindows extracted : {total_windows:,}')
    print(f'Skipped (maintenance): {n_skipped_maintenance:,}')
    print(f'Skipped (sparse data): {n_skipped_sparse:,}')

    df = pd.DataFrame(all_rows)

    # Move meta columns to front
    meta = ['window_start', 'window_end', 'label_h0',
            'label_brake', 'label_compressor', 'label_leveling', 'label_any',
            'failure_type']
    feat_cols = [c for c in df.columns if c not in meta]
    df = df[meta + feat_cols]

    print(f'\nClass distribution (label_any):')
    vc = df['label_any'].value_counts()
    print(f'  Normal  (0): {vc.get(0, 0):,}  ({100*vc.get(0,0)/len(df):.1f}%)')
    print(f'  Failure (1): {vc.get(1, 0):,}  ({100*vc.get(1,0)/len(df):.1f}%)')
    print(f'\nPer-type label counts:')
    for col in ['label_brake', 'label_compressor', 'label_leveling']:
        n = int(df[col].sum())
        print(f'  {col}: {n:,}')
    print(f'\nTotal features : {len(feat_cols)}')
    print(f'Total windows  : {len(df):,}')

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Build MetroAT feature datasets.')
    parser.add_argument('--split',   default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--window',  type=int,   default=360,  help='Window size in seconds')
    parser.add_argument('--stride',  type=int,   default=60,   help='Stride in seconds')
    parser.add_argument('--horizon', type=float, default=6.2,  help='Label horizon in hours')
    args = parser.parse_args()

    window_min = args.window // 60
    tag = f'w{window_min}m_h{args.horizon}h'

    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    for split in splits:
        df = build_features(
            split=split,
            window_s=args.window,
            stride_s=args.stride,
            horizon_h=args.horizon,
        )
        out_path = FEATURES_DIR / f'features_{split}_{tag}.parquet'
        df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size // (1024 * 1024)
        print(f'\nSaved: {out_path}  ({size_mb} MB)')


if __name__ == '__main__':
    main()

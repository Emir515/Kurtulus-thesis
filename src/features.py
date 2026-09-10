"""
Feature extraction for MetroAT predictive maintenance.

Per-sensor features  (flat baseline, matches Stefan's approach):
  Analog  → mean, std, min, max  over the window
  Binary  → active_s (seconds active), flips (state changes)

Group-level features  (subsystem-aware extension):
  mean, min, spread (std across sensors), trend (slope), chain_asym (CW1-CW2)
  computed per functional sensor group.

Asset features:
  days_since_last_failure, days_since_last_revision

Column naming convention:
  Per-sensor : <SENSOR_NAME>__<stat>
  Group      : GRP__<Subsystem>__<GroupName>__<stat>
  Asset      : days_since_last_failure / days_since_last_revision
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import SUBSYSTEMS

# ── Binary vs analog classification ───────────────────────────────────────────
_BINARY_SUBSTRINGS = (
    '_ACTIVE', '_RUNNING', '_AVAILABLE',
    'MANUAL_MODE', 'AUTOMATIC_MODE', 'EMERGENCY_MODE', 'IS_SPECIAL_SECTION',
)

def _is_binary(col: str) -> bool:
    return any(s in col for s in _BINARY_SUBSTRINGS)

ALL_SENSOR_COLS: list[str] = [c for ss in SUBSYSTEMS.values() for c in ss]
BINARY_COLS:     list[str] = [c for c in ALL_SENSOR_COLS if _is_binary(c)]
ANALOG_COLS:     list[str] = [c for c in ALL_SENSOR_COLS if not _is_binary(c)]

# ── Functional sensor groups ───────────────────────────────────────────────────
# Based on failure analysis: these groups showed consistent Z-score deviations
# before failure events across all 10 train events.
SENSOR_GROUPS: dict[str, list[str]] = {
    # APU — main reservoir pressure (both chains)
    'APU__MainReservoir': [
        'CW1_MAIN_RESERVOIR_PRESSURE',
        'CW2_MAIN_RESERVOIR_PRESSURE',
    ],
    # Brake — four functional sub-groups
    'Brake__SpringPressure': [
        c for c in SUBSYSTEMS['Brake'] if 'SPRING_BRAKE_PRESSURE' in c
    ],
    'Brake__CylinderPressure': [
        c for c in SUBSYSTEMS['Brake'] if 'BRAKE_CYLINDER_PRESSURE' in c
    ],
    'Brake__PropValvePressure': [
        c for c in SUBSYSTEMS['Brake']
        if 'PROPORTIONAL_VALVE_PRESSURE' in c and 'AVAILABLE' not in c
    ],
    'Brake__BrakingForce': [
        c for c in SUBSYSTEMS['Brake'] if 'PNEUMATIC_BRAKING_FORCE' in c
    ],
    # Leveling — load pressure and load signal separately
    'Leveling__LoadPressure': [
        c for c in SUBSYSTEMS['Leveling'] if 'LOAD_PRESSURE' in c
    ],
    'Leveling__LoadSignal': [
        c for c in SUBSYSTEMS['Leveling'] if 'LOAD_SIGNAL' in c
    ],
    # Traction — all four energy resistance sensors
    'Traction__EnergyResistance': list(SUBSYSTEMS['Traction']),
}

# Chain definitions for asymmetry features
_CW1_PREFIXES = ('CW1_', 'MW1_', 'MW2_')
_CW2_PREFIXES = ('CW2_', 'MW3_', 'MW4_')


# ── Internal helpers ───────────────────────────────────────────────────────────
def _slope(arr: np.ndarray) -> float:
    """OLS slope of arr (units per sample). Returns 0 for constant or short arrays."""
    n = len(arr)
    if n < 3:
        return 0.0
    x  = np.arange(n, dtype=np.float64)
    xm = x.mean()
    ym = arr.mean()
    denom = ((x - xm) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - xm) * (arr - ym)).sum() / denom)


# ── Main extraction function ───────────────────────────────────────────────────
def extract_window_features(w: pd.DataFrame, asset_feats: dict) -> dict:
    """
    Extract all features from a single time window.

    Parameters
    ----------
    w : pd.DataFrame
        Window rows containing only sensor columns (no meta/label columns).
        Rows = seconds (1 Hz), columns = sensor names.
    asset_feats : dict
        {'days_since_last_failure': float, 'days_since_last_revision': float}

    Returns
    -------
    dict  feature_name -> scalar value
    """
    feats: dict = {}

    # ── Per-sensor features ────────────────────────────────────────────────────
    for col in ANALOG_COLS:
        if col not in w.columns:
            continue
        v = w[col].to_numpy(dtype=np.float64)
        feats[f'{col}__mean'] = v.mean()
        feats[f'{col}__std']  = v.std()
        feats[f'{col}__min']  = v.min()
        feats[f'{col}__max']  = v.max()

    for col in BINARY_COLS:
        if col not in w.columns:
            continue
        v = w[col].to_numpy(dtype=np.float64)
        feats[f'{col}__active_s'] = v.sum()
        feats[f'{col}__flips']    = float((np.diff(v) != 0).sum())

    # ── Group-level features ───────────────────────────────────────────────────
    for grp, cols in SENSOR_GROUPS.items():
        present = [c for c in cols if c in w.columns]
        if not present:
            continue

        mat = w[present].to_numpy(dtype=np.float64)  # (n_rows, n_sensors)

        col_means = mat.mean(axis=0)   # one value per sensor (cross-sensor mean over time)
        grp_ts    = mat.mean(axis=1)   # group mean at each second (time series)

        feats[f'GRP__{grp}__mean']   = float(col_means.mean())
        feats[f'GRP__{grp}__min']    = float(col_means.min())
        feats[f'GRP__{grp}__spread'] = float(col_means.std()) if len(col_means) > 1 else 0.0
        feats[f'GRP__{grp}__trend']  = _slope(grp_ts)

        # Chain asymmetry: CW1-chain mean minus CW2-chain mean
        cw1 = [c for c in present if c.startswith(_CW1_PREFIXES)]
        cw2 = [c for c in present if c.startswith(_CW2_PREFIXES)]
        if cw1 and cw2:
            feats[f'GRP__{grp}__chain_asym'] = (
                float(w[cw1].to_numpy(dtype=np.float64).mean())
                - float(w[cw2].to_numpy(dtype=np.float64).mean())
            )

    # ── Asset features ─────────────────────────────────────────────────────────
    feats.update(asset_feats)

    return feats

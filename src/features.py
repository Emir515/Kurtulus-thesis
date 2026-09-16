"""
Feature extraction for MetroAT predictive maintenance.

Three feature set modes:
  1. FLAT        — per-sensor features only (Stefan's baseline)
  2. ENGINEERING — per-sensor + group-level features from domain knowledge
  3. DATA_DRIVEN — per-sensor + group-level features from NB02 clusters

Per-sensor features:
  Analog  → mean, std, min, max  over the window
  Binary  → active_s (seconds active), flips (state changes)

Group-level features:
  mean, min, spread (std across sensors), trend (slope), chain_asym (CW1-CW2)
  computed per functional sensor group.

Asset features:
  days_since_last_failure, days_since_last_revision

Column naming convention:
  Per-sensor : <SENSOR_NAME>__<stat>
  Group      : GRP__<GroupName>__<stat>
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

# ── Engineering-based sensor groups (domain knowledge) ────────────────────────
ENGINEERING_GROUPS: dict[str, list[str]] = {
    'APU__MainReservoir': [
        'CW1_MAIN_RESERVOIR_PRESSURE',
        'CW2_MAIN_RESERVOIR_PRESSURE',
    ],
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
    'Leveling__LoadPressure': [
        c for c in SUBSYSTEMS['Leveling'] if 'LOAD_PRESSURE' in c
    ],
    'Leveling__LoadSignal': [
        c for c in SUBSYSTEMS['Leveling'] if 'LOAD_SIGNAL' in c
    ],
    'Traction__EnergyResistance': list(SUBSYSTEMS['Traction']),
}

# ── Data-driven sensor groups (from NB02 anomaly window clustering) ───────────
# Clusters discovered by hierarchical clustering (Ward/Jaccard) of anomalous
# windows across the full train timeline.  Sensors included if activation
# frequency > 30% within the cluster.
#
# C1 (333 windows, 10% failure): Spring brake pressure only
# C2 (83 windows, 11% failure): APU + Leveling + Brake cross-subsystem
# C3 (24 windows, 75% failure): Full system alarm (all subsystems)
# C4 (110 windows, 0% failure): Leveling only (operational)
DATA_DRIVEN_GROUPS: dict[str, list[str]] = {
    # C1: Spring Brake Pattern — pure brake, most common anomaly
    'C1__SpringBrake': [
        'CW1_SPRING_BRAKE_PRESSURE_BOGIE2',
        'CW2_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW1_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW1_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW2_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW2_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW3_SPRING_BRAKE_PRESSURE_BOGIE1',
        'MW4_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW4_SPRING_BRAKE_PRESSURE_BOGIE2',
    ],
    # C2: Cross-Subsystem Pattern — APU pressure + leveling signals + spring brake
    'C2__CrossSystem': [
        'CW1_MAIN_RESERVOIR_PRESSURE', 'CW2_MAIN_RESERVOIR_PRESSURE',  # APU
        'CW1_LOAD_SIGNAL', 'CW2_LOAD_SIGNAL',                          # Leveling
        'MW1_LOAD_SIGNAL', 'MW3_LOAD_SIGNAL',
        'MW3_LOAD_PRESSURE_BOGIE1',
        'MW1_SPRING_BRAKE_PRESSURE_BOGIE1',                             # Brake
        'MW2_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW2_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW3_SPRING_BRAKE_PRESSURE_BOGIE1',
        'MW4_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW4_SPRING_BRAKE_PRESSURE_BOGIE2',
    ],
    # C3: Full System Alarm — all 4 subsystems, strongest failure indicator (75%)
    'C3__FullAlarm': [
        'CW1_MAIN_RESERVOIR_PRESSURE', 'CW2_MAIN_RESERVOIR_PRESSURE',  # APU
        'CW1_SPRING_BRAKE_PRESSURE_BOGIE1', 'CW1_SPRING_BRAKE_PRESSURE_BOGIE2',  # Brake
        'CW2_SPRING_BRAKE_PRESSURE_BOGIE1', 'CW2_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW1_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW1_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW2_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW2_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW3_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW3_SPRING_BRAKE_PRESSURE_BOGIE2',
        'MW4_SPRING_BRAKE_PRESSURE_BOGIE1', 'MW4_SPRING_BRAKE_PRESSURE_BOGIE2',
        'CW1_LOAD_PRESSURE_BOGIE1', 'CW1_LOAD_PRESSURE_BOGIE2',       # Leveling
        'CW2_LOAD_PRESSURE_BOGIE1', 'CW2_LOAD_PRESSURE_BOGIE2',
        'MW1_LOAD_PRESSURE_BOGIE1', 'MW1_LOAD_PRESSURE_BOGIE2',
        'MW2_LOAD_PRESSURE_BOGIE1', 'MW2_LOAD_PRESSURE_BOGIE2',
        'MW3_LOAD_PRESSURE_BOGIE1', 'MW3_LOAD_PRESSURE_BOGIE2',
        'MW4_LOAD_PRESSURE_BOGIE1', 'MW4_LOAD_PRESSURE_BOGIE2',
        'MW1_ENERGY_BRAKING_RESISTANCE', 'MW2_ENERGY_BRAKING_RESISTANCE',  # Traction
        'MW3_ENERGY_BRAKING_RESISTANCE', 'MW4_ENERGY_BRAKING_RESISTANCE',
    ],
    # C4: Leveling Pattern — operational (0% failure), likely passenger loading
    'C4__Leveling': [
        'CW1_LOAD_SIGNAL', 'CW2_LOAD_SIGNAL', 'MW1_LOAD_SIGNAL',
        'CW1_LOAD_PRESSURE_BOGIE1', 'CW1_LOAD_PRESSURE_BOGIE2',
        'CW2_LOAD_PRESSURE_BOGIE1', 'CW2_LOAD_PRESSURE_BOGIE2',
        'MW1_LOAD_PRESSURE_BOGIE1', 'MW1_LOAD_PRESSURE_BOGIE2',
        'MW2_LOAD_PRESSURE_BOGIE1', 'MW2_LOAD_PRESSURE_BOGIE2',
        'MW3_LOAD_PRESSURE_BOGIE1',
        'MW4_LOAD_PRESSURE_BOGIE2',
    ],
}

# Sensors never activated in any anomalous window (66 of 101) — excluded in data-driven mode
_DATA_DRIVEN_ACTIVE = {s for grp in DATA_DRIVEN_GROUPS.values() for s in grp}
C1_NOISE_SENSORS: list[str] = [
    c for c in SUBSYSTEMS.get('Brake', []) + SUBSYSTEMS.get('Context', [])
    + SUBSYSTEMS.get('Traction', []) + SUBSYSTEMS.get('APU', [])
    + SUBSYSTEMS.get('Leveling', [])
    if c not in _DATA_DRIVEN_ACTIVE
]

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
def extract_window_features(
    w: pd.DataFrame,
    asset_feats: dict,
    mode: str = 'flat',
) -> dict:
    """
    Extract all features from a single time window.

    Parameters
    ----------
    w : pd.DataFrame
        Window rows containing only sensor columns (no meta/label columns).
        Rows = seconds (1 Hz), columns = sensor names.
    asset_feats : dict
        {'days_since_last_failure': float, 'days_since_last_revision': float}
    mode : str
        'flat'        — per-sensor features only (all 101 sensors)
        'engineering' — per-sensor + engineering group features
        'data_driven' — per-sensor (excl. C1 noise) + data-driven group features

    Returns
    -------
    dict  feature_name -> scalar value
    """
    feats: dict = {}

    # Determine which sensors to include
    if mode == 'data_driven':
        skip = set(C1_NOISE_SENSORS)
        analog = [c for c in ANALOG_COLS if c not in skip]
        binary = [c for c in BINARY_COLS if c not in skip]
    else:
        analog = ANALOG_COLS
        binary = BINARY_COLS

    # ── Per-sensor features ────────────────────────────────────────────────────
    for col in analog:
        if col not in w.columns:
            continue
        v = w[col].to_numpy(dtype=np.float64)
        feats[f'{col}__mean'] = v.mean()
        feats[f'{col}__std']  = v.std()
        feats[f'{col}__min']  = v.min()
        feats[f'{col}__max']  = v.max()

    for col in binary:
        if col not in w.columns:
            continue
        v = w[col].to_numpy(dtype=np.float64)
        feats[f'{col}__active_s'] = v.sum()
        feats[f'{col}__flips']    = float((np.diff(v) != 0).sum())

    # ── Group-level features ───────────────────────────────────────────────────
    if mode == 'engineering':
        groups = ENGINEERING_GROUPS
    elif mode == 'data_driven':
        groups = DATA_DRIVEN_GROUPS
    else:
        groups = {}  # flat mode: no group features

    for grp, cols in groups.items():
        present = [c for c in cols if c in w.columns]
        if not present:
            continue

        mat = w[present].to_numpy(dtype=np.float64)  # (n_rows, n_sensors)

        col_means = mat.mean(axis=0)
        grp_ts    = mat.mean(axis=1)

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

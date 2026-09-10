# Structured Subsystem-Aware Feature Representations for Predictive Maintenance in Metro Systems

**Author:** Emirhan Kurtulus — TU Wien, MSc Data Science
**Advisors:** Prof. Fazel Ansari, Dipl.-Ing. Andreas Steiner
**Dataset:** MetroAT — Vienna metro (Wiener Linien), pneumatic system, fleet number 3813

---

## Research Question

Does organizing sensor features by subsystem (Brake, APU, Leveling, Traction) — instead of treating all 101 sensors as a flat input — improve predictive maintenance performance and reduce false alarms on the MetroAT dataset?

---

## Dataset: MetroAT

| Property | Value |
|----------|-------|
| Sampling rate | 1 Hz (one row per second) |
| Total rows | ~22.2M (14.3M train + 7.9M test) |
| Total columns | 109 (101 sensor cols + 8 meta/label cols) |
| Format | Apache Parquet, partitioned by year/month/day |
| Missing values | 0 in all 101 sensor columns |

**Sensor subsystems (101 sensors):**
| Subsystem | Count | Description |
|-----------|-------|-------------|
| Brake | 67 | Cylinder/spring/proportional valve pressure, braking force, brake active flags |
| Leveling | 18 | Load pressure and load signal across all 6 wagons |
| APU | 4 | Main reservoir pressure (CW1/CW2), compressor running (CW1/CW2) |
| Traction | 4 | Energy braking resistance MW1-MW4 |
| Context | 8 | Speed, temperature, train line, section, mode flags |

**Data splits:**
| Split | Files | Rows | Date range |
|-------|-------|------|-----------|
| Train | 246 | 14,342,361 | 2024-06-01 to 2025-02-11 |
| Test | 136 | 7,901,921 | 2025-02-12 to 2025-06-30 |

**Failure events:**
| Type | Train | Test | Total |
|------|-------|------|-------|
| Brake System Failure | 8 | 5 | 13 |
| Compressor Module Failure | 1 | 5 | 6 |
| Leveling System Failure | 1 | 1 | 2 |
| **Total** | **10** | **11** | **21** |

**Class imbalance:** ~94.5% normal, ~4.5% maintenance, ~1% failure.

Dataset source: https://researchdata.tuwien.ac.at/records/9ja0q-bq581

---

## Baseline Results (Helm 2025)

Prior work by Stefan Helm (TU Wien, Oct 2025) evaluated two models on MetroAT:

| Model | Dev F1 | Test F1 | Main weakness |
|-------|--------|---------|---------------|
| Random Forest (flat features) | 0.98 | **0.21** | Overfits to `days_since_last_failure` |
| LSTM-AE (unsupervised) | 0.75 | **0.08** | Recall=0.92 but Precision=0.04 |

---

## Key Exploration Findings

Confirmed in `notebooks/01_data_exploration.ipynb`:

- **Precursor signals exist:** 18/20 event-sensor pairs show variance spikes 8x-656x above baseline, median 4.5h before failure onset
- **Subsystem structure is real:** within-subsystem correlation is 3-6x higher than cross-subsystem
- **Window sizes are valid:** tau_corr = 92s median; all candidate windows (6/11.5/23 min) produce independent observations
- **4 failure-maintenance overlaps:** ~20% of failure events overlap with maintenance periods (data quality consideration)

**Anomaly analysis findings** (`notebooks/02_anomaly_analysis.ipynb`):

- **Only 36/101 sensors (36%) show reliable precursor signals** — the rest are binary flags, constant values, or weak responders
- **4 data-driven clusters** found via hierarchical clustering on variance ratio profiles:
  - C3 (11 sensors): pure Brake (spring brake pressure), 100% detection rate, 250-618x variance ratio
  - C4 (9 sensors): cross-subsystem early warning (APU + Leveling + Brake + Traction), 94% detection rate
  - C2 (35 sensors): moderate signal, 34% detection rate
  - C1 (46 sensors): noise, 16% detection rate
- **BOGIE1 vs BOGIE2 asymmetry:** CW brake sensors on BOGIE1 show 100% detection, BOGIE2 only 11%
- **Subsystem grouping is partially validated:** Cluster 3 is 100% pure Brake, but Cluster 4 crosses subsystem boundaries through the shared pneumatic system

---

## Project Structure

```
Kurtulus-thesis/
├── train/                          # Raw parquet data (Jun 2024 - Feb 2025, 246 files)
├── test/                           # Raw parquet data (Feb 2025 - Jun 2025, 136 files)
├── data/
│   └── asset_data/                 # External asset records
│       ├── failure.csv             # 41 failure records
│       ├── revision.csv            # 21 maintenance/revision events
│       ├── relevant_sensors.json   # Stefan's 95-sensor selection
│       ├── port_name_mapping.json  # Port-to-column name mapping
│       ├── stationsinformation.csv
│       └── train_order.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Full EDA (dataset overview, failure events, precursors, correlations)
│   └── 02_anomaly_analysis.ipynb   # Sensor ranking, clustering, subsystem validation
├── scripts/
│   └── build_features.py           # Window-based feature extraction pipeline
├── src/
│   ├── config.py                   # Central paths, sensor schema, hyperparams
│   ├── features.py                 # Feature extraction functions (per-sensor + group-level)
│   └── __init__.py
├── references/
│   └── stefan/                     # Prior work scripts and predictions (Helm 2025)
├── Papers/                         # Reference literature
├── outputs/
│   ├── plots/                      # Generated figures
│   └── sensor_clusters.csv         # Cluster assignments for all 101 sensors
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data Exploration** ✅ — dataset structure, failure events, precursor analysis, subsystem correlations
2. **Anomaly Analysis & Clustering** ✅ — precursor strength for all 101 sensors, hierarchical clustering, subsystem validation
3. **Feature Engineering & Modeling** — build flat vs. subsystem-aware feature sets, train RF, compare performance (next)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Note:** The `train/` and `test/` folders are not in version control (large binary files). All paths are configured in `src/config.py`.

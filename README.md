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

## Baseline Results (Steiner RAMS 2026 / Helm 2025)

Prior work by Stefan Helm (TU Wien) evaluated two models on MetroAT:

| Model | Dev F1 | Test F1 | Main weakness |
|-------|--------|---------|---------------|
| Random Forest (flat features) | 0.98 | **0.21** | Overfits to `days_since_last_failure` |
| LSTM-AE (unsupervised) | 0.75 | **0.08** | Recall=0.92 but Precision=0.04 |

---

## Generated Datasets

**Supervised** (9 configs) — `outputs/supervised_datasets/` (453 MB)
- 3 window sizes (6m/350s, 11m/701s, 23m/1402s) x 3 label horizons (no_shift, 3.1h, 6.2h)
- 344 features per window: analog (min/max/mean/sum) + binary (active_s/flips) + 2 asset features
- Format: train.parquet + test.parquet per config

**Unsupervised** (9 configs) — `outputs/unsupervised_datasets/` (71 GB)
- Same 3x3 grid, raw multivariate sequences as .npy arrays
- Shape: (n_windows, seq_len, 98 channels), float32
- Splits: train (normal-only), val (normal-only), dev_eval (all), test (all)
- Z-score standardized using train-normal statistics only

---

## Key Findings

### Data Exploration (NB01)

- **Precursor signals exist:** variance spikes detected before all 21 failure events, median lead time 5.1h
- **Subsystem structure is real:** within-subsystem correlation is 3-6x higher than cross-subsystem
- **Window sizes are valid:** tau_corr = 92s median; all candidate windows (6/11.5/23 min) produce independent observations
- **Steiner baseline:** RF predicts P(failure) = 0.15-0.60 for above-threshold windows, but almost all are false positives (F1=0.21)

### Anomaly Analysis (NB02)

- Z-score thresholding (|z| > 2.0) on windowed features relative to normal-operation statistics
- 70.7% of windows have at least 1 activated sensor; avg 10.2 sensors activated per window
- 96% of pre-failure windows are anomalous (498/518), confirming detectable patterns exist

### Clustering & Temporal Patterns (NB03 — Phase 1)

Hierarchical clustering (Jaccard distance + Ward linkage) on binary activation matrix:

| Cluster | Sensors | Subsystem | Failure Rate | Interpretation |
|---------|---------|-----------|-------------|----------------|
| C1 | 37 | Brake (valves + forces) | 3.8% | Strongest failure precursor |
| C2 | 12 | Brake (spring brakes) | 1.5% | Spring brake anomalies |
| C3 | 4 | Leveling | 0.5% | Operational patterns |
| C4 | 0 | Mixed/unclustered | 1.8% | Isolated anomalies |

- C1 appears within 0-2h before most failures (strongest precursor)
- Test set validation confirms patterns generalize
- Cluster definitions saved to `outputs/cluster_definitions.json` for Phase 2

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
│       ├── port_name_mapping.json  # Port-to-column name mapping
│       ├── relevant_sensors.json   # Sensor selection reference
│       ├── stationsinformation.csv
│       └── train_order.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Full EDA: timelines, sensor plots, precursors, correlations
│   ├── 02_anomaly_analysis.ipynb   # Z-score anomaly detection, activation matrix
│   └── 03_clustering_temporal_analysis.ipynb  # Phase 1: clustering, temporal patterns, test validation
├── scripts/
│   ├── create_supervised_datasets.py    # Windowed features -> parquet (9 configs)
│   └── create_unsupervised_datasets.py  # Raw sequences -> .npy (9 configs)
├── src/
│   ├── config.py                   # Central paths, sensor schema, hyperparams
│   └── __init__.py
├── references/
│   └── stefan/                     # Prior work scripts and predictions (Helm 2025)
├── Papers/                         # Reference literature
├── outputs/
│   ├── supervised_datasets/        # 9 configs, train/test parquets (453 MB)
│   ├── unsupervised_datasets/      # 9 configs, .npy + .csv (71 GB)
│   ├── plots/                      # Generated figures from NB03
│   └── cluster_definitions.json    # Cluster sensor lists for Phase 2
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data Exploration** -- dataset structure, failure events, full-timeline sensor analysis, precursor detection
2. **Dataset Creation** -- supervised (windowed features) and unsupervised (raw sequences) for 9 window/horizon configs
3. **Anomaly Analysis & Clustering (Phase 1 - RQ1)** -- z-score anomaly detection, hierarchical clustering, temporal pattern analysis, cluster definitions for subsystem-aware representations
4. **Modeling & Evaluation (Phase 2 - RQ2)** -- train supervised (RF, GBM) and unsupervised (LSTM-AE) models on full-feature vs subsystem-aware representations, controlled comparison (next)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Note:** The `train/` and `test/` folders are not in version control (large binary files). The `outputs/` directory is also gitignored. All paths are configured in `src/config.py`.

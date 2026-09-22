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

Re-evaluation of Steiner's RF test predictions: F1=0.25, P=0.14, R=1.00, FAR=1.00 (predicts all windows as failure).

---

## Generated Datasets

**Supervised** — `outputs/supervised/`
- Best baseline config: Wfeat=350s (6 min), Wlabel=22326s (6.2h)
- Train: 41,364 windows (530 failure, 1.28%) | Test: 23,134 windows (667 failure, 2.88%)
- 340 features: 296 analog (74 sensors x min/max/mean/sum) + 42 binary (21 sensors x active_s/flips) + 2 asset (days_since_last_failure, days_since_last_revision)
- Format: train.parquet + test.parquet

**Unsupervised** — `outputs/unsupervised/`
- Best baseline config: Wseq=1402s (23 min), Wlabel=11163s (3.1h)
- 95 channels, z-score standardized using train-normal statistics
- Splits:

| Split | Shape | Type |
|-------|-------|------|
| train | 6,272 x 1,402 x 95 | normal-only |
| val | 1,568 x 1,402 x 95 | normal-only |
| dev_eval | 1,975 x 1,402 x 95 | all (8 failure) |
| test | 5,374 x 1,402 x 95 | all (46 failure) |

- Format: .npy arrays (float32) + metadata.json

---

## Key Findings

### Data Exploration (NB01)

- **Precursor signals exist:** variance spikes detected before all 21 failure events, median lead time 5.1h
- **Subsystem structure is real:** within-subsystem correlation is 3-6x higher than cross-subsystem
- **Window sizes are valid:** tau_corr = 92s median; all candidate windows (6/11.5/23 min) produce independent observations
- **Steiner baseline:** RF predicts P(failure) = 0.15-0.60 for above-threshold windows, but almost all are false positives (F1=0.21)

### Dataset Creation (NB02)

- Reproduces Steiner's exact pipeline: epoch-aligned windowing, maintenance filtering, analog/binary feature aggregation, asset lookback computation, shifted failure labels with filter rule
- Sequential processing (train then test) to stay within memory limits (~16 GB)
- Asset lookback timestamps chained from train to test for continuity

### Dataset Analysis (NB03)

- **Most discriminative subsystems** (Cohen's d, normal vs failure):
  - Brake: mean d=0.747, 203/230 features with d>0.3 (strongest)
  - Leveling: mean d=0.622, top single feature (MW4_LOAD_PRESSURE d=1.46)
  - Context: mean d=0.651 (TRAIN_SPEED_ACTUAL d=1.31)
  - APU/Traction/Asset: weak discriminative power (d<0.25)
- **Within-subsystem correlation** drops during failure (Brake: 0.79 normal -> 0.44 failure), suggesting failure disrupts normal co-variation patterns
- **Standardization verified:** z-score channels well-centered; 4 Traction channels near-constant (low variance)
- **Temporal coverage:** ~177 windows/day, median gap = 350s (one window width), 360/210 operational breaks (>1h) in train/test

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
│   ├── 02_dataset_creation.ipynb   # Supervised & unsupervised dataset creation (Steiner baseline)
│   └── 03_dataset_analysis.ipynb   # Dataset validation, feature analysis, subsystem comparison
├── src/
│   ├── config.py                   # Central paths, sensor schema, hyperparams
│   └── __init__.py
├── references/
│   └── stefan/                     # Prior work scripts and predictions (Helm 2025)
├── Papers/                         # Reference literature
├── outputs/
│   ├── supervised/                 # train.parquet, test.parquet
│   └── unsupervised/              # .npy arrays, labels, metadata.json
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data Exploration (NB01)** — dataset structure, failure events, full-timeline sensor analysis, precursor detection
2. **Dataset Creation (NB02)** — supervised (windowed features) and unsupervised (raw sequences) matching Steiner's best baseline configs
3. **Dataset Analysis (NB03)** — validation against Steiner's predictions, feature discriminability by subsystem, standardization checks
4. **Modeling & Evaluation (Phase 2)** — train supervised (RF, GBM) and unsupervised (LSTM-AE) models on flat vs subsystem-aware representations, controlled comparison (next)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Note:** The `train/` and `test/` folders are not in version control (large binary files). The `outputs/` directory is also gitignored. All paths are configured in `src/config.py`.

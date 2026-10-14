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

## Baseline Results

### Steiner (RAMS 2026 / Helm 2025)

| Model | Dev F1 | Test F1 | Main weakness |
|-------|--------|---------|---------------|
| Random Forest (flat features) | 0.98 | **0.21** | Overfits to `days_since_last_failure` |
| LSTM-AE (unsupervised) | 0.75 | **0.08** | Recall=0.92 but Precision=0.04 |

### Our Reproduction (NB02)

Using Steiner's best hyperparameters (n_estimators=300, max_depth=4, threshold=0.20):

| Model | Test Windows | True Failures | Predicted | F1 | P | R | FAR |
|-------|-------------|---------------|-----------|-----|------|------|------|
| Random Forest | 23,147 | 669 | 879 | 0.120 | 0.106 | 0.139 | 3.5% |
| LSTM-AE | 5,374 seq | 46 | — | — | — | — | — |

LSTM-AE anomaly scores: normal mean=0.773, failure mean=0.857 (weak separation).

**Full-period anomaly timeline (Jun 2024 – Jun 2025):**

![Anomaly Timeline](outputs/anomaly_timeline_full.png)

---

## Key Findings

### Data Exploration (NB01)

- **Precursor signals exist:** variance spikes detected before all 21 failure events, median lead time 5.1h
- **Subsystem structure is real:** within-subsystem correlation is 3-6x higher than cross-subsystem
- **Window sizes are valid:** tau_corr = 92s median; all candidate windows (6/11.5/23 min) produce independent observations
- **Steiner baseline:** RF predicts P(failure) = 0.15-0.60 for above-threshold windows, but almost all are false positives (F1=0.21)

### Dataset Creation & Anomaly Detection (NB02)

- Reproduces Steiner's exact pipeline: epoch-aligned windowing, maintenance filtering, analog/binary feature aggregation, asset lookback computation, shifted failure labels with filter rule
- RF trained with Steiner's best params (no grid search), threshold tuned on 20% temporal holdout
- LSTM-AE trained on normal-only train sequences (10 epochs, CPU)
- Both models predict on ALL test windows — no threshold filtering applied
- Sequential processing (train then test) to stay within memory limits (~16 GB)

### Dataset Analysis — Full Period (NB03)

- **Anomaly hotspots (full period):** RF detects 186 sustained high-score clusters across Jun 2024 – Jun 2025; only 27 overlap with labeled failures, 159 have no labeled failure (potential false alarms or unlabeled degradation)
- **LSTM-AE distribution shift:** train mean reconstruction error = 0.539, test mean = 0.774 — the test period looks systematically "less normal," suggesting concept drift or gradual degradation
- **Score stability:** RF probability distribution is similar across train/test (mean 0.061 vs 0.064); LSTM-AE shows clear shift
- **Most discriminative subsystems** (Cohen's d, normal vs failure, full period):
  - Brake: mean d=0.655, 206/230 features with d>0.3 (strongest)
  - Leveling: mean d=0.552, 51/72 features with d>0.3
  - Context: mean d=0.388 (TRAIN_SPEED_ACTUAL d=0.99)
  - APU: mean d=0.290 | Traction: mean d=0.205 | Asset: weak (d<0.10)
- **Within-subsystem correlation** drops during failure (Brake: 0.58 → 0.44, Traction: 0.74 → 0.52), suggesting failure disrupts normal co-variation patterns
- **Within vs cross-subsystem:** within-subsystem correlation is 2.4x higher than cross-subsystem (0.567 vs 0.236)
- **Standardization:** z-score channels mostly well-centered; 4 Traction channels and several Leveling channels have mean far from 0 (near-constant or skewed distributions)
- **Temporal coverage:** ~178 windows/day, median gap = 350s (one window width), 571 operational breaks (>1h) across full period

### Anomaly Detection & Feature Clustering (NB04)

- **Anomalous windows:** 12,022 / 64,522 (18.6%) identified via z-score (|z|>3, ≥5 features activated per window); 724 overlap with failures (6.0%), 11,298 are normal anomalies
- **Feature activation analysis:** Top 30 most activated features are dominated by brake cylinder pressure; most have failure rates near baseline (4-7%), meaning individual features alone cannot predict failures
- **Feature clustering (not window clustering):** Jaccard distance + average linkage (t=0.7) on binary co-activation profiles → **31 feature clusters (FC1-FC31)**, 130 features total
  - 22 multi-feature clusters, 9 single-feature clusters
  - Single-feature clusters are valid: 8/9 have failure rates above baseline, FC26 reaches 31.8%
- **Cluster cohesion:** All 22 multi-feature clusters have internal Jaccard similarity > 0.3; mean cohesion = 0.54; 11 clusters > 0.5
- **Failure day vs normal day validation** (all 23 failure days vs 336 normal days):
  - FC24 (+28.7pp) and FC28 (+20.0pp) are **failure-specific** — activate far more on failure days
  - FC4 (+17.6pp) is failure-leaning; 23 clusters are similar; FC30 is normal-leaning
- **Two failure mechanisms discovered** via variance lead-up analysis (7 days before 18 failure events):
  - **Gradual degradation (FC28):** MW4_LOAD_SIGNAL_min variance increases 2.84x before failure — a genuine early warning signal detectable 2-3 days ahead
  - **Sudden shift (FC18, 19, 24, 25, 26):** Variance drops or stays flat — sensors jump to abnormal values without gradual build-up, only detectable at the moment of failure
- **Heuristic comparison:** Jaccard-based (Heuristic A) vs correlation-based (Heuristic B) clustering produce similar groupings (ARI, NMI, co-membership agreement), validating robustness
- **Binary vs continuous validation:** Binary activation patterns preserve continuous correlation structure (positive Pearson r between binary cohesion and continuous correlation across clusters)

### Pattern Discovery & Validation (NB05)

- Links feature clusters to failure events and physical subsystems
- Expert verification preparation: cluster summaries with top features and statistics
- Deep dives into critical clusters with raw sensor data

---

## Project Structure

```
Kurtulus-thesis/
├── train/                          # Raw parquet data (Jun 2024 - Feb 2025, 246 files)
├── test/                           # Raw parquet data (Feb 2025 - Jun 2025, 136 files)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Full EDA: timelines, sensor plots, precursors, correlations
│   ├── 02_anomaly_dataset_creation.ipynb  # Supervised & unsupervised pipeline + RF & LSTM-AE
│   ├── 03_dataset_analysis.ipynb   # Full-period analysis: hotspots, distributions, subsystem discriminability
│   ├── 04_anomaly_detection_cluster_analysis.ipynb  # Feature clustering, validation, failure mechanisms
│   └── 05_pattern_discovery_validation.ipynb  # Cluster-to-subsystem mapping, expert verification
├── src/
│   ├── config.py                   # Central paths, sensor schema, hyperparams
│   └── __init__.py
├── references/
│   └── stefan/                     # Prior work scripts and predictions (Helm 2025)
├── Papers/                         # Reference literature
├── outputs/
│   ├── supervised/                 # train.parquet, test.parquet, rf_model, predictions
│   ├── unsupervised/              # .npy arrays, lstm_ae_model, predictions
│   ├── nb04_artifacts/            # Feature clusters, activation matrices, constants
│   ├── anomaly_timeline_full.png
│   ├── heuristic_comparison.png
│   └── binary_vs_continuous_validation.png
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data Exploration (NB01)** — dataset structure, failure events, full-timeline sensor analysis, precursor detection
2. **Anomaly Dataset Creation (NB02)** — supervised (windowed features + RF) and unsupervised (raw sequences + LSTM-AE) matching Steiner's best baseline configs, predict on all test windows
3. **Dataset Analysis (NB03)** — full-period anomaly timeline, score distributions (train vs test), hotspot detection, subsystem discriminability (Cohen's d), correlation analysis, standardization checks, temporal coverage
4. **Anomaly Detection & Feature Clustering (NB04)** — z-score anomaly detection, feature activation profiles, feature clustering via Jaccard distance, failure day vs normal day validation, variance lead-up analysis discovering two failure mechanisms (gradual vs sudden), heuristic comparison, binary vs continuous validation
5. **Pattern Discovery & Validation (NB05)** — link clusters to failure events and subsystems, prepare expert verification sheets, deep dives with raw sensor data
6. **Modeling & Evaluation (Phase 2)** — train supervised (RF, GBM) and unsupervised (LSTM-AE) models on flat vs subsystem-aware representations, controlled comparison (next)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Note:** The `train/` and `test/` folders are not in version control (large binary files). The `outputs/` directory is also gitignored. All paths are configured in `src/config.py`.

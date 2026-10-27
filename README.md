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

### Our Reproduction (NB02a datasets, NB02b baselines)

Current RF outputs (earlier fixed-parameter version, no grid search, threshold = 0.20):

| Model | Test Windows | True Failures | Predicted | F1 | P | R | FAR |
|-------|-------------|---------------|-----------|-----|------|------|------|
| Random Forest | 23,147 | 669 | 879 | 0.120 | 0.106 | 0.139 | 3.5% |
| LSTM-AE | 5,374 seq | 46 | — | — | — | — | — |

LSTM-AE anomaly scores: normal mean=0.773, failure mean=0.857 (weak separation). NB02b does not threshold the LSTM-AE; "F1 ≈ 0.08" is Steiner/Helm's value. NB05 uses the 99th percentile of the validation split (1.123).

> **Datasets match Steiner/Helm exactly.** All 1,674 test windows flagged by Helm's RF (`references/stefan/supervised_test_predictions.csv`) exist in our test set with identical labels. Recomputed on our 669 test positives, his predictions give P = 0.143, R = 0.357, **F1 = 0.204** (reported: 0.21). The gap to our 0.120 is therefore model tuning, not data: the old RF used fixed parameters and refitted on the full undersampled set. NB02b now contains Helm's full procedure (432-combination grid search with 5-fold blocked CV on the dev set, threshold on the holdout, test scored with the dev-fitted model). It is **not executed yet** and is planned for Phase 2; afterwards NB03 and NB05 §4 are re-run.

> **Label definition (checked against Helm's code, `references/stefan/`):** label = 1 if a failure overlaps [window_end, window_end + 6.2 h) (Helm's `create_shifted_labels`: max of the failure flag over that interval). Windows inside a failure are therefore positive (543 of 1,202), and the filter rule removes nothing, exactly as in Helm's code. Helm's thesis text (p87) describes dropping ambiguous in-failure windows, but the code does not. The threshold is tuned on the last 20% of the undersampled train set, also as in Helm's `train_rf_adapted.py`.
>
> A test of the point-shifted variant (failure state 6.2 h ahead, ambiguous windows dropped) gave only 0.5% positives in train, 172 of 218 of them from one long failure, and a test F1 near 0. So it was not adopted.

**Full-period anomaly timeline (Jun 2024 – Jun 2025):**

![Anomaly Timeline](outputs/nb03_artifacts/anomaly_timeline_full.png)

---

## Key Findings

### Data Exploration (NB01)

- **Data:** 22.2M rows at 1 Hz, 101 sensor channels + 8 metadata/flag columns; NB01 loads only the needed columns (the full data set is ~15 GB in memory)
- **Failures:** 21 events (10 train / 11 test; 13 brake, 6 compressor, 2 leveling), identical to the canonical table in NB04; ~1% (train) / ~1.4% (test) of rows in failure, ~4% in maintenance
- **During recorded failures** (train, raw 1 Hz values), large shifts in CW1 spring-brake pressure (Cohen's d = −1.5), CW1 main-reservoir pressure (−1.1) and CW1 load pressure (−1.1)
- **Correlation structure:** within documented sensor groups, mean |r| is clearly higher than across groups (permutation test p = 0.001); within-Brake correlation is partly due to synchronous actuation within a control chain

### Dataset Creation (NB02a)

- Turns the raw MetroAT `train/` and `test/` files into the model-ready representations of the MetroAT framework; the official train/test split is kept (train windows only from `train/`, test windows only from `test/`)
- Follows Helm's code: epoch-aligned 6-min windowing, maintenance filtering, analog (min/max/mean/sum) and binary (active_s/flips) aggregation, asset lookback, 6.2 h failure labels (in-failure windows are positive, so the filter rule removes nothing, as in Helm's code)
- 23-min LSTM-AE sequences of the 95 sensor channels, z-scored with train statistics
- Sequential processing (train then test) to stay within memory limits (~16 GB); features and sequences are cached

### Full-Feature Baselines (NB02b)

- RF: Helm's 432-combination grid search with 5-fold blocked CV on the dev set, threshold tuned on the 20% temporal holdout, test scored with the dev-fitted model (not executed yet, Phase 2)
- LSTM-AE trained on normal-only train sequences (10 epochs, CPU)
- Both models predict on ALL test windows; these are the full-feature side of the RQ2 comparison

### Dataset Analysis — Full Period (NB03)

- **Train/test shift:** LSTM-AE reconstruction error is 0.539 (train) vs 0.774 (test) and rises from ~April 2025, consistent with the baseline shift in RAMS 2026. Pooled train+test comparisons are therefore confounded, because failure windows are concentrated in the test period.
- **RF scores on the train period are in-sample** (the RF was fit on train), so the train-vs-test comparison of RF probabilities (3.43% vs 3.80% above threshold) and hotspots (124 vs 62) is not like-for-like.
- **Anomaly hotspots:** RF finds 186 sustained high-score clusters, 27 overlapping labelled failures. LSTM-AE (train-period p90 threshold) finds 554, only 10 with failures, most of them in the drifting test period.
- **Discriminability per documented sensor group, computed per period** (robust d = min(d_train, d_test)): 234/340 features have robust d > 0.3 (pooled: 268, inflated by drift). Mean robust d is Brake 0.50, Context 0.34, Leveling 0.31, energy-braking resistance 0.17, APU 0.16.
- **Within vs cross-group correlation** (same-sensor aggregation pairs excluded): 0.565 vs 0.236 (2.4×). Within-group correlation drops during failures in train (e.g. Brake 0.61 → 0.37) but much less in test (0.54 → 0.51).
- **Aggregation redundancy:** mean vs sum mean |r| = 0.89 (median 0.97), so dropping `_sum` in NB04 is a simplification. active_s vs flips are only moderately correlated (keep both).
- **Standardization:** with a sample spread over the train period, all LSTM-AE channels are well standardised (overall mean −0.01, std 0.98). The earlier "12 channels off-centre" came from sampling only June 2024.
- **Temporal coverage:** ~178 windows/day, median gap = 350s (one window width), 571 operational breaks (>1h) across full period

### Anomaly Detection & Feature Clustering (NB04)

- **Canonical failure events:** 21 events from the raw `TRAIN_IS_IN_FAILURE` flag (same-type blocks < 1 h apart merged): 10 train, 11 test (matches the 11 test failures in RAMS 2026); 13 brake, 6 compressor, 2 leveling (Wiener Linien records). Shared by NB04 and NB05.
- **Anomalous windows:** 12,022 / 64,522 (18.6%) identified via z-score against the train-normal distribution (|z|>3, ≥5 sensor features activated per window); 724 overlap with failures (6.0%), 11,298 are normal anomalies. Context/asset features (speed, temperature, days-since) are excluded from detection.
- **No test leakage:** the clustering and its validation use only the **5,351 train-period anomalous windows**. The test period never shapes the clusters, so it stays an unseen evaluation set for RQ2. The fixed clusters are then applied to all windows in the analyses below. (An earlier version clustered train and test windows together: 31 FCs. Removing the test windows dropped the clusters that existed only because of test-period behaviour, e.g. the drift-driven old FC30 and old FC31.)
- **Drift:** anomaly rate on windows far from any failure is 10.9% in train vs 27.4% in test (43% in June 2025), in line with the baseline shift reported in RAMS 2026. 9 FCs are flagged as possibly drift-driven (test/train activation ratio ≥ 2), e.g. FC10 (0.2% → 6.6%).
- **Feature activation analysis:** the most activated features are dominated by brake cylinder pressure; most have failure rates near baseline, meaning individual features alone cannot predict failures
- **Feature clustering (not window clustering):** Jaccard distance + average linkage (t=0.7) on binary co-activation profiles → **19 feature clusters (FC1–FC19)**, 126 features total
  - 16 multi-feature clusters, 3 single-feature clusters (all three above the 6.0% baseline failure rate)
- **Cluster cohesion:** all 16 multi-feature clusters have internal Jaccard similarity > 0.3; mean cohesion = 0.63; 10 clusters > 0.5
- **Sensor location per FC:** neutral descriptor (control chain CW1/MW1/MW2 vs CW2/MW3/MW4, bogie, signal types). High cohesion can partly reflect synchronous actuation within a chain.
- **Failure-related day vs normal day** (stratified by period): only **FC2 and FC6** are more active on failure-related days in *both* train and test; FC18's effect comes almost entirely from the train period (+46pp train vs +4pp test).
- **Temporal pattern analysis (drift-controlled, event-level tests):**
  - **No FC is an early warning** in the [−6 d, −1 d] window used by RAMS/Helm.
  - **Two short-term precursors:** **FC5** (CW2 compressor flips) fires in the 24 h before onset in 15/21 events vs 8.2 expected by chance, median lead 14 h; **FC7** (CW1 spring-brake pressure min + load pressure) in 6 events vs 1.7 expected, 63% of its windows are failure windows.
  - 12 FCs are **detection-only** (strongly enriched during failures, e.g. FC7 195×, FC8 49×); 5 are weak/mixed.
- **Volatility lead-up (real within-day variance, same-period baseline):** only FC18 shows a moderate, monotonic pre-onset increase (1.25×, Kendall τ = 0.71, p = 0.03). FC2 rises only on the onset day (1.55×; pre-onset 1.18×).
  - *Correction:* an earlier claim of a "2.96× variance ramp-up before failure" came from a bug. No `_std` features exist, so the old cell plotted the daily mean level and measured the ratio on the failure day itself. It is not supported.
- **Heuristic comparison:** Jaccard-based (Heuristic A, 19 clusters) vs correlation-based (Heuristic B, 13 clusters) clustering agree on 87% of feature pairings (ARI 0.46, NMI 0.77); Heuristic A has higher silhouette (0.55 vs 0.53) and separates failure-prone from benign clusters much better
- **Binary vs continuous validation:** all 16 multi-feature clusters agree (binary cohesion and continuous correlation both above 0.3)

### Pattern Discovery & Validation (NB05)

- **Event-level scorecard per FC** (Helm 2025; RAMS 2026): alarm episodes with persistence filter, early/late hits, false alarms, drift-controlled lift, 24 h lead time.
- **KEEP verdict on train events only, test events as a check:** KEEP if lift ≥ 1.5 and ≥ 3 events covered on the **10 train events**; the same rule is then checked on the **11 test events**. 15 of 19 FCs are KEEP on train, but only **3 hold on the test events: FC7** (train lift 4.4 / test 2.4, 1 false alarm), **FC5** (3.7 / 1.9, lead ~15 h) and **FC8** (spring-brake pressure min, 2.5 / 1.5). The other 12 fall to chance level in the test period (6 of them drift-flagged). Note: 42% of test windows lie near a failure (train: 19%), so test lift is capped near 2.4 and the check is strict.
- **Prior-work flagged intervals** (the "zoom-in" pipeline from RAMS 2026, p7; current fixed-parameter RF): RF reaches 11/11 early hits and LSTM-AE 10/11 on the test events. FC2, FC6 and FC18 are over-represented in RF **false alarms**; FC1, FC14 and FC17 mark RF **early hits**. For LSTM-AE, FC1/2/9/10/14/15/17/18/19 mark early hits. This section is re-run after NB02b's grid search.
- **FC × Wiener Linien failure type:** enrichment in the 24 h before and during failures, per recorded type (evidence for the experts only; no subsystem labels assigned to FCs).
- **Expert verification sheet** (`outputs/nb05_artifacts/expert_verification_sheet.csv/.xlsx`) with all evidence (including `lift_train`, `verdict`, `lift_test`, `test_check`) plus empty expert columns; the RQ1 verification metric and `build_subsystem_datasets()` run once the filled sheet is returned.
- Deep dives are selected from the data (currently FC7 and FC8).

---

## Project Structure

```
Kurtulus-thesis/
├── train/                          # Raw parquet data (Jun 2024 - Feb 2025, 246 files), not in git
├── test/                           # Raw parquet data (Feb 2025 - Jun 2025, 136 files), not in git
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA: events, timelines, sensor shifts during failures, correlation structure
│   ├── 02a_dataset_creation.ipynb  # Windowed features + labels, LSTM-AE sequences from raw MetroAT (Helm/Steiner setup)
│   ├── 02b_baselines.ipynb         # Full-feature RF (grid search) + LSTM-AE baselines; not executed yet (Phase 2)
│   ├── 03_dataset_analysis.ipynb   # Full-period analysis: train/test shift, hotspots, per-period Cohen's d
│   ├── 04_anomaly_detection_cluster_analysis.ipynb  # Z-score anomalies, train-only feature clustering (FC1-FC19), temporal + drift analysis
│   └── 05_pattern_discovery_validation.ipynb  # Event-level FC scoring, prior-model alarms, expert verification sheet
├── src/
│   ├── config.py                   # Paths, documented sensor-group schema, split date
│   └── __init__.py
├── references/
│   └── stefan/                     # Helm's original scripts and predictions (reference implementation)
├── Papers/                         # Reference literature (Helm thesis, RAMS 2026, MetroAT, proposal, ...)
├── outputs/                        # Generated, not in git
│   ├── supervised/                 # Feature caches, train/test parquet (NB02a); RF model + test predictions (NB02b)
│   ├── unsupervised/               # LSTM-AE sequences (.npy) and metadata (NB02a); model and scores (NB02b)
│   ├── nb02_artifacts/             # Test-period anomaly timeline (NB02b)
│   ├── nb03_artifacts/             # Full-period timeline, aggregation redundancy
│   ├── nb04_artifacts/             # FC1-FC19, activations, canonical failure events, temporal/drift/location tables
│   ├── nb05_artifacts/             # Scorecard, prior-model enrichment, failure-type table, expert verification sheet
│   ├── WL_Expert_Review_NB04.pdf   # Expert review document for Wiener Linien
│   └── NB04_Workflow_and_Metrics.pdf
├── thesis-latex/                   # LaTeX thesis (separate git repo, synced with Overleaf)
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data Exploration (NB01)**: dataset structure, failure and maintenance events, full-timeline sensor plots, sensor shifts during failures, correlation structure by documented sensor group
2. **Dataset Creation (NB02a)**: windowed features (6 min) with 6.2 h labels and LSTM-AE sequences (23 min) from the raw MetroAT train/test files, replicating Helm's code
3. **Full-Feature Baselines (NB02b)**: RF (Helm's grid search with blocked CV) and LSTM-AE on all features; predictions on all test windows (run in Phase 2)
4. **Dataset Analysis (NB03)**: full-period score timeline, train/test shift, hotspots, per-period Cohen's d, aggregation redundancy, standardization checks, temporal coverage
5. **Anomaly Detection & Feature Clustering (NB04)**: canonical failure events, z-score anomaly windows, feature clustering via Jaccard distance fitted on the train period only (FC1-FC19), cluster validation, drift-controlled temporal pattern analysis, volatility lead-up, drift checks
6. **Pattern Discovery & Validation (NB05)**: event-level scoring of each FC (KEEP decided on train events, checked on test events), FCs behind the RF/LSTM-AE early hits and false alarms, FC vs. recorded failure type, expert verification sheet and RQ1 metric
7. **Expert verification (RQ1)**: Wiener Linien experts assign subsystems to FCs via the verification sheet
8. **Modeling & Evaluation (Phase 2, RQ2)**: run NB02b, then subsystem-aware vs. full-feature representations under identical conditions; the test period is used only for this final comparison

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline in order (each notebook reads the previous one's outputs)
cd notebooks
for nb in 01_data_exploration 02a_dataset_creation 02b_baselines 03_dataset_analysis \
          04_anomaly_detection_cluster_analysis 05_pattern_discovery_validation; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb --ExecutePreprocessor.timeout=3600
done
```

**Notes**
- The `train/`, `test/` and `outputs/` folders are not in version control.
- Memory: loading all raw data at once needs ~15 GB. NB01 loads only the columns it needs. NB02a caches the windowed features and LSTM-AE sequences (`REBUILD_FEATURES`, `REBUILD_SEQUENCES = True` rebuild them from raw data, ~7 GB RAM). NB02b caches the RF grid search (`outputs/supervised/rf_grid_search.json`, about an hour to compute) and the LSTM-AE weights (`RERUN_GRID_SEARCH`, `RETRAIN_AE = True` recompute them).
- Paths and the documented sensor-group schema are in `src/config.py`.

# Prior work: Stefan Helm (MetroAT baselines)

Reference copies of Stefan Helm's code and results, used to check that this thesis reproduces the MetroAT baselines of Steiner et al. (RAMS 2026) exactly. The full handover (including asset data such as `failure.csv` and `revision.csv`) is kept outside the repository in `../Stefan Work/Handover-Stefan/`.

| File | What it is | Used for |
|---|---|---|
| `create_base_dataset.py` | Builds his base dataset from the raw recordings: sensor selection, German signal names, unit conversion, failure and revision flags | Understanding the original data (public MetroAT uses English names and scaled units) |
| `create_supervised_feature_datasets.py` | Windowed features and shifted labels for the Random Forest | NB02a label check: identical labels, in-failure windows stay positive |
| `create_supervised_feature_datasets_adapted.py` | The same, with paths adapted | |
| `create_unsupervised_sequence_datasets.py` | 23-min sequences, label at the sequence end, normal-only training set, `dev_eval` hold-out | NB02a sequence alignment (prepared for Phase 2) |
| `train_rf_adapted.py` | Random Forest grid search (432 combinations, 5-fold blocked CV) | NB02b |
| `result_analysis_code_only.ipynb` | His result analysis, outputs removed (paths point to his machine) | Shows how the published numbers were made: two-sided Butterworth `filtfilt` smoothing of the scores, cutoff and threshold chosen on the hold-out, early hits and false alarms counted per `failure.csv` record |
| `supervised_test_predictions.csv` | Test windows his Random Forest flagged (1,674) | Label check (100% agreement) and his F1 = 0.206 on our test set |
| `unsupervised_test_predictions.csv` | His LSTM-AE test predictions | |

Key findings from these files are summarised in the main `README.md` (section "Baseline Results").

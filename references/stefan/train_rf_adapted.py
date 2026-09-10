"""
Adapted from Stefan Helm's train_rf.py.
Changes:
  - DATASET_ROOT points to our supervised_features output
  - read_features_parquet uses pandas instead of Polars
  - Everything else is identical to Stefan's original
"""
import glob
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

RNG = np.random.RandomState(42)

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
DATASET_ROOT = os.path.expanduser(
    "~/Desktop/MASTER THESIS/Stefan Work/supervised_features"
)
N_FOLDS      = 5
MIN_POS_RATIO = 0.10

PARAM_GRID: Dict[str, List] = {
    "n_estimators"     : [300, 600],
    "max_depth"        : [None, 2, 4, 16],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2", 0.5],
    "class_weight"     : [None, "balanced"],
}

ID_LIKE_COLS = {"window_end", "timestamp"}
LABEL_COL    = "label"
TIME_COL     = "window_end"


# -----------------------------------------------------------
# Utilities (identical to Stefan)
# -----------------------------------------------------------
def read_features_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected '{LABEL_COL}' in {path} but not found.")
    return df.sort_values(TIME_COL)


def make_blocked_folds(pdf: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    pdf = pdf.sort_values(TIME_COL).reset_index(drop=True)
    n = len(pdf)
    idx_blocks = np.array_split(np.arange(n), n_folds)
    splits = []
    ts_vals = pd.to_datetime(pdf[TIME_COL].values)
    for i in range(n_folds):
        val_idx = idx_blocks[i]
        if val_idx.size == 0:
            continue
        val_start_time = ts_vals[val_idx[0]]
        train_idx = np.where(ts_vals < val_start_time)[0]
        if train_idx.size == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def undersample_majority_to_ratio(pdf: pd.DataFrame, min_pos_ratio: float, rng: np.random.RandomState) -> pd.DataFrame:
    pos_idx = np.flatnonzero(pdf[LABEL_COL] == 1)
    neg_idx = np.flatnonzero(pdf[LABEL_COL] == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0:
        return pdf
    current_ratio = n_pos / (n_pos + n_neg)
    if current_ratio >= min_pos_ratio:
        return pdf
    max_neg_keep = int(np.floor(n_pos * (1.0 / min_pos_ratio - 1.0)))
    neg_keep = min(n_neg, max_neg_keep)
    keep_neg_idx = rng.choice(neg_idx, size=neg_keep, replace=False)
    keep_idx = np.concatenate([pos_idx, keep_neg_idx])
    keep_idx.sort()
    return pdf.iloc[keep_idx]


def build_preprocess_transformer(pdf: pd.DataFrame) -> ColumnTransformer:
    cols = [c for c in pdf.columns if c not in ID_LIKE_COLS | {LABEL_COL}]
    num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
    pre = ColumnTransformer(transformers=[("preprocess", num_pipe, cols)], verbose_feature_names_out=False)
    return pre


def evaluate_at_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float, float, float, int]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_thr, best_p, best_r = -1.0, 0.5, 0.0, 0.0
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
            best_p = precision_score(y_true, y_pred, zero_division=0)
            best_r = recall_score(y_true, y_pred, zero_division=0)
    return best_thr, best_f1, best_p, best_r, int((y_true == 1).sum())


@dataclass
class Result:
    params: Dict
    thr: float
    f1: float
    precision: float
    recall: float
    cm: np.ndarray
    intermediate_results: List[Tuple[Dict, float]]
    dev_holdout: Dict[str, float]


# -----------------------------------------------------------
# Main CV + Grid Search (identical to Stefan)
# -----------------------------------------------------------
def run_cv_grid(pdf_dev_all: pd.DataFrame, param_grid: Dict[str, List]) -> Tuple[Result, Pipeline]:
    pdf = pdf_dev_all.sort_values(TIME_COL).reset_index(drop=True)

    pos_ratio = (pdf[LABEL_COL] == 1).mean()
    if pos_ratio < MIN_POS_RATIO:
        pdf = undersample_majority_to_ratio(pdf, MIN_POS_RATIO, RNG)

    holdout_size = int(0.2 * len(pdf))
    pdf_holdout = pdf.iloc[-holdout_size:].reset_index(drop=True)
    pdf_dev = pdf.iloc[:-holdout_size].reset_index(drop=True)

    splits = make_blocked_folds(pdf_dev, n_folds=N_FOLDS)
    if len(splits) == 0:
        raise RuntimeError("No valid folds constructed. Check development data size.")

    pre = build_preprocess_transformer(pdf_dev)

    keys, values = zip(*param_grid.items(), strict=False)
    combos = [dict(zip(keys, v, strict=False)) for v in itertools.product(*values)]

    feature_cols = [c for c in pdf_dev.columns if c not in ID_LIKE_COLS | {LABEL_COL}]
    X_dev = pdf_dev[feature_cols]
    y_dev = pdf_dev[LABEL_COL].astype(int)

    X_holdout = pdf_holdout[feature_cols]
    y_holdout = pdf_holdout[LABEL_COL].astype(int)

    best_mean_f1 = -1.0
    best_params_overall: Optional[Dict] = None
    intermediate_results: List[Tuple[Dict, float]] = []

    for params_idx, params in enumerate(combos, start=1):
        f1s = []
        print(f"[Combo {params_idx}/{len(combos)}] {params}", end=" ", flush=True)
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr, y_tr = X_dev.iloc[train_idx].copy(), y_dev.iloc[train_idx].copy()
            X_va, y_va = X_dev.iloc[val_idx].copy(), y_dev.iloc[val_idx].copy()

            if y_tr.sum() == 0 or y_va.sum() == 0:
                f1s.append(0.0)
                continue

            base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            pipe = Pipeline(steps=[("pre", pre), ("clf", base_rf)])
            model = Pipeline(steps=pipe.steps)
            model.set_params(**{f"clf__{k}": v for k, v in params.items()})
            model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_va)[:, 1]
            _, f1, _, _, _ = evaluate_at_best_threshold(y_va.values, proba)
            f1s.append(f1)

        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        print(f"=> mean F1={mean_f1:.3f}")
        intermediate_results.append((params, mean_f1))
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_params_overall = params

    if best_params_overall is None:
        raise RuntimeError("Failed to select best params (no valid folds).")

    print(f"\n=== Refit on DEV (80%) with params: {best_params_overall} ===")
    final_model = Pipeline(
        steps=[("pre", pre), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, **best_params_overall))]
    )
    final_model.fit(X_dev, y_dev)

    proba_holdout = final_model.predict_proba(X_holdout)[:, 1]
    thr, f1_h, p_h, r_h, _ = evaluate_at_best_threshold(y_holdout.values, proba_holdout)
    dev_holdout_metrics = {"F1": f1_h, "Precision": p_h, "Recall": r_h, "Threshold": thr}
    print(f"DEV holdout: F1={f1_h:.3f} P={p_h:.3f} R={r_h:.3f} thr={thr:.3f}")

    holdout_res = {
        "scores": proba_holdout.tolist(),
        "labels": y_holdout.tolist(),
    }

    res = Result(
        params=best_params_overall,
        thr=thr,
        f1=f1_h,
        precision=p_h,
        recall=r_h,
        cm=confusion_matrix(y_holdout.values, (proba_holdout >= thr).astype(int)),
        intermediate_results=intermediate_results,
        dev_holdout=dev_holdout_metrics,
    )

    return res, final_model, holdout_res


# -----------------------------------------------------------
# Runner over all feature sets (identical to Stefan)
# -----------------------------------------------------------
def discover_feature_sets(root: str) -> List[Tuple[str, str, Optional[str]]]:
    paths = glob.glob(os.path.join(root, "**", "features.parquet"), recursive=True)
    pairs = []
    for p in sorted(paths):
        tag = os.path.relpath(os.path.dirname(p), root)
        test_path = os.path.join(os.path.dirname(p), "test.parquet")
        pairs.append((tag, p, test_path if os.path.exists(test_path) else None))
    return pairs


def main():
    sets = discover_feature_sets(DATASET_ROOT)
    if not sets:
        raise SystemExit(f"No features.parquet found under {DATASET_ROOT}\nRun create_supervised_feature_datasets.py first.")

    all_summaries = []

    for tag, dev_path, test_path in sets:
        print(f"\n=== Training on: {tag} ===")
        pdf_dev = read_features_parquet(dev_path)
        pdf_dev[TIME_COL] = pd.to_datetime(pdf_dev[TIME_COL])

        res, model, holdout_res = run_cv_grid(pdf_dev, PARAM_GRID)

        test_summary = {}
        scores = {}
        if test_path is not None and os.path.exists(test_path):
            print(f"Evaluating on TEST: {test_path}")
            pdf_test = read_features_parquet(test_path)
            pdf_test[TIME_COL] = pd.to_datetime(pdf_test[TIME_COL])

            feature_cols = [c for c in pdf_dev.columns if c not in ID_LIKE_COLS | {LABEL_COL}]
            X_test = pdf_test[feature_cols]
            y_test = pdf_test[LABEL_COL].astype(int)

            proba_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = (proba_test >= res.thr).astype(int)

            f1_t = f1_score(y_test, y_pred_test, zero_division=0)
            p_t  = precision_score(y_test, y_pred_test, zero_division=0)
            r_t  = recall_score(y_test, y_pred_test, zero_division=0)
            cm_t = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            tn, fp = cm_t[0, 0], cm_t[0, 1]
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            print(f"TEST: F1={f1_t:.3f} P={p_t:.3f} R={r_t:.3f} FAR={far:.4f} | CM={cm_t.tolist()}")

            test_summary = {
                "F1": f1_t, "Precision": p_t, "Recall": r_t,
                "FAR": far, "ConfusionMatrix": cm_t.tolist(), "Threshold": res.thr,
            }
            scores = {
                "dev_eval": holdout_res,
                "test": {"scores": proba_test.tolist(), "labels": y_test.tolist()},
                "threshold": res.thr,
            }
        else:
            print("No test.parquet found; skipping test evaluation.")

        out = {
            "tag": tag,
            "best_params": res.params,
            "dev_holdout": {
                "F1": res.f1, "Precision": res.precision, "Recall": res.recall,
                "ConfusionMatrix": res.cm.tolist(), "Threshold": res.thr,
            },
            "test": test_summary,
            "intermediate_results": res.intermediate_results,
        }
        all_summaries.append(out)

        rf_results_dir = os.path.join(DATASET_ROOT, "_rf_results")
        os.makedirs(rf_results_dir, exist_ok=True)
        safe_tag = tag.replace(os.sep, "__")

        with open(os.path.join(rf_results_dir, f"{safe_tag}_summary.json"), "w") as f:
            json.dump(out, f, indent=2)

        if scores:
            with open(os.path.join(rf_results_dir, f"{safe_tag}_scores.json"), "w") as f:
                json.dump(scores, f, indent=2)

        models_dir = os.path.join(rf_results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(models_dir, f"{safe_tag}_model.joblib"))

    if all_summaries:
        with open(os.path.join(DATASET_ROOT, "_rf_results", "ALL_summaries.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)
        print("\n=== Summary ===")
        for s in sorted(all_summaries, key=lambda x: x["test"].get("F1", 0), reverse=True):
            print(f"  {s['tag']:30s}  TEST F1={s['test'].get('F1', 'N/A'):.3f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Project Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.cf_gen import CFGenAdapter
from src.dlc.ground_truth import GroundTruthGenerator

# ==========================================
# Config
# ==========================================
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# ==========================================
# Helper Functions (Copied from run_baselines_final.py)
# ==========================================

def get_params_count(model):
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
         return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return "N/A"

def compute_delta_cate(model_name, model, X_test, feature_names, treatment_idx=22):
    pm25_idx = 22
    egfr_idx = feature_names.index("EGFR") if "EGFR" in feature_names else 3
    gt_gen = GroundTruthGenerator(pm25_idx=pm25_idx, egfr_idx=egfr_idx)
    true_ite = gt_gen.compute_true_ite(X_test)
    
    est_ite = None
    if hasattr(model, 'predict_ite'):
         est_ite = model.predict_ite(X_test)
         if isinstance(est_ite, torch.Tensor):
            est_ite = est_ite.detach().cpu().numpy()
         est_ite = est_ite.flatten()
    
    pehe = np.sqrt(np.nanmean((true_ite - est_ite) ** 2))
    
    egfr_mask = X_test[:, egfr_idx] > 0
    cate_mut = np.mean(est_ite[egfr_mask])
    cate_wild = np.mean(est_ite[~egfr_mask])
    delta_cate = cate_mut - cate_wild
    
    return pehe, delta_cate

def compute_sensitivity(model_name, model, X_test, feature_names):
    age_idx = feature_names.index("Age") if "Age" in feature_names else 0
    X_perturbed = X_test.copy()
    X_perturbed[:, age_idx] += 10 # Add 10 years
    
    y_orig = model.predict_proba(X_test)[:, 1]
    y_pert = model.predict_proba(X_perturbed)[:, 1]
    
    return np.mean(np.abs(y_orig - y_pert))

def load_clean_pancan():
    csv_path = DATA_DIR / "pancan_synthetic_interaction.csv"
    df = pd.read_csv(csv_path)
    exclude = ['sampleID', 'Outcome_Label', 'True_Prob', 'True_ITE', 'Treatment']
    feats = [c for c in df.columns if c not in exclude]
    X = df[feats].values.astype(np.float32)
    y = df['Outcome_Label'].values.astype(np.int64)
    return X, y, feats

def load_luad_target():
    csv_path = DATA_DIR / "luad_synthetic_interaction.csv"
    df = pd.read_csv(csv_path)
    exclude_cols = ["sampleID", "Outcome_Label", "True_Prob", "True_ITE", "Treatment"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if len(feature_cols) > 23: feature_cols = feature_cols[:23]
    X = df[feature_cols].values.astype(np.float32)
    y = df['Outcome_Label'].values.astype(np.int64)
    return X, y, feature_cols

# ==========================================
# Main
# ==========================================

def run_single_seed(seed, epochs, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Prepare Data
    X_pancan, y_pancan, _ = load_clean_pancan()
    X_luad, y_luad, feature_names = load_luad_target()

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_luad, y_luad, test_size=0.2, random_state=seed, stratify=y_luad
    )

    X_train_pool = np.vstack([X_pancan, X_train_l])
    y_train_pool = np.hstack([y_pancan, y_train_l])

    print(f"[Seed {seed}] Pooled Train: {X_train_pool.shape}, Test: {X_test_l.shape}")

    # 2. CFGen
    model = CFGenAdapter(input_dim=23, epochs=epochs, batch_size=batch_size, outcome_type='binary')

    start_t = time.time()
    model.fit(X_train_pool, y_train_pool)

    # Predict
    y_prob = model.predict_proba(X_test_l)[:, 1]
    y_pred = model.predict(X_test_l)

    infer_time = (time.time() - start_t) / len(X_test_l) * 1000

    res = {
        "AUC": float(roc_auc_score(y_test_l, y_prob)),
        "Acc": float(accuracy_score(y_test_l, y_pred)),
        "F1": float(f1_score(y_test_l, y_pred)),
        "Precision": float(precision_score(y_test_l, y_pred)),
        "Recall": float(recall_score(y_test_l, y_pred)),
        "Time (ms)": float(round(infer_time, 2)),
        "Params": str(get_params_count(model)),
        "Seed": int(seed)
    }

    # Causal
    pehe, delta_cate = compute_delta_cate("CF-Gen", model, X_test_l, feature_names)
    res["PEHE"] = float(pehe)
    res["Delta CATE"] = float(delta_cate)
    res["Sensitivity"] = float(compute_sensitivity("CF-Gen", model, X_test_l, feature_names))

    print(f"  -> AUC: {res['AUC']:.4f}, PEHE: {res['PEHE']:.4f}")
    return res


def summarize_results(results):
    metrics = [
        "AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity", "Time (ms)"
    ]
    summary = {}
    for m in metrics:
        vals = np.array([r[m] for r in results], dtype=np.float64)
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0))
        }
    summary["Params"] = results[0].get("Params", "N/A") if results else "N/A"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-tag", type=str, default="rigorous_20260203_fix5")
    args = parser.parse_args()

    print("🚀 CF-Gen Experiment for Comparison Matrix...")
    print(f"Seeds: {args.seeds}")

    results = []
    for seed in args.seeds:
        results.append(run_single_seed(seed, args.epochs, args.batch_size))

    summary = summarize_results(results)
    output = {
        "per_seed": results,
        "summary": summary
    }

    output_path = RESULTS_DIR / f"cf_gen_results_{args.output_tag}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved: {output_path}")
    print(
        "CF-Gen Summary -> "
        f"AUC {summary['AUC']['mean']:.4f} ± {summary['AUC']['std']:.4f}, "
        f"Acc {summary['Acc']['mean']:.4f} ± {summary['Acc']['std']:.4f}, "
        f"F1 {summary['F1']['mean']:.4f} ± {summary['F1']['std']:.4f}"
    )

if __name__ == "__main__":
    main()

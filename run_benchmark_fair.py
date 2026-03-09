# -*- coding: utf-8 -*-
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

# Project Imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline
from src.baselines.mogonet_baseline import MOGONETBaseline
from src.baselines.transtee_baseline import TransTEEBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_pehe_from_arrays
from src.dlc.ground_truth import GroundTruthGenerator
from src.baselines.utils import set_global_seed

# Reusing imports from run_baselines_final.py
from src.run_baselines_final import (
    load_clean_pancan, load_luad_target, prepare_mogonet_views,
    predict_transtee_proba, DLCWrapper, get_params_count, get_inference_params_count,
    summarize_metrics, format_mean_std, compute_delta_cate, compute_sensitivity
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-tag", type=str, default="fair_benchmark")
    parser.add_argument("--dlc-weights", type=str, default="dlc_final_sota_s_seed_43.pth")
    parser.add_argument("--dlc-d-hidden", type=int, default=128)
    parser.add_argument("--dlc-num-layers", type=int, default=3)
    parser.add_argument("--only-dlc", action="store_true", help="Only evaluate DLC without baselines")
    parser.add_argument("--dlc-strict", action="store_true", default=False)
    args = parser.parse_args()

    print("🚀 Taking off: Fair Baseline Ultimate Re-test...")
    
    # 1. Prepare Data
    X_pancan, y_pancan, _ = load_clean_pancan()
    X_luad, y_luad, feature_names = load_luad_target()
    
    per_seed_results = {}
    PM25_COL_IDX = 22
    
    for seed in args.seeds:
        print(f"\n[Seed {seed}] Preparing data...")
        set_global_seed(seed)

        # Split LUAD
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_luad, y_luad, test_size=0.2, random_state=seed, stratify=y_luad
        )
        
        # Pool Training Data
        X_train_pool = np.vstack([X_pancan, X_train_l])
        y_train_pool = np.hstack([y_pancan, y_train_l])

        if not args.only_dlc:
            models = {
                "XGBoost": XGBBaseline(random_state=seed),
                "TabR": TabRBaseline(random_state=seed, batch_size=256, epochs=20),
                "MOGONET": MOGONETBaseline(random_state=seed, epochs=20),
                "TransTEE": TransTEEBaseline(random_state=seed, epochs=20, batch_size=64),
                "HyperFast": HyperFastBaseline(random_state=seed, epochs=20, batch_size=64)
            }

            for name, model in models.items():
                print(f"\n⚡ Processing {name} (Seed {seed})...")
                
                train_time = 0.0
                infer_time = 0.0
                
                # Training Time
                if name == "TransTEE":
                    pm25_train = X_train_pool[:, PM25_COL_IDX]
                    med_train = np.median(pm25_train)
                    t_train = (pm25_train > med_train).astype(float)
                    X_tr_no_t = np.delete(X_train_pool, PM25_COL_IDX, axis=1)
                    
                    t0 = sync_time()
                    model.fit(X_tr_no_t, t_train, y_train_pool)
                    train_time = sync_time() - t0
                    
                    X_te_no_t = np.delete(X_test_l, PM25_COL_IDX, axis=1)
                    pm25_test = X_test_l[:, PM25_COL_IDX]
                    t_test = (pm25_test > med_train).astype(float)
                    
                    # Inference Time
                    t0 = sync_time()
                    y_prob = predict_transtee_proba(model, X_te_no_t, t=t_test)[:, 1]
                    infer_time = sync_time() - t0
                    y_pred = (y_prob > 0.5).astype(int)
                
                elif name == "MOGONET":
                    views_train = prepare_mogonet_views(X_train_pool, feature_names)
                    t0 = sync_time()
                    model.fit(views_train, y_train_pool)
                    train_time = sync_time() - t0
                    
                    views_test = prepare_mogonet_views(X_test_l, feature_names)
                    t0 = sync_time()
                    y_prob = model.predict_proba(views_test)[:, 1]
                    infer_time = sync_time() - t0
                    y_pred = (y_prob > 0.5).astype(int)
                else:
                    t0 = sync_time()
                    model.fit(X_train_pool, y_train_pool)
                    train_time = sync_time() - t0
                    
                    t0 = sync_time()
                    y_prob = model.predict_proba(X_test_l)[:, 1]
                    infer_time = sync_time() - t0
                    y_pred = model.predict(X_test_l)
                
                res = {
                    "AUC": float(roc_auc_score(y_test_l, y_prob)),
                    "Acc": float(accuracy_score(y_test_l, y_pred)),
                    "F1": float(f1_score(y_test_l, y_pred)),
                    "Precision": float(precision_score(y_test_l, y_pred)),
                    "Recall": float(recall_score(y_test_l, y_pred)),
                    "Training Time (s)": float(train_time),
                    "Inference Time (ms)": float(infer_time / len(X_test_l) * 1000),
                    "Params": str(get_params_count(model)),
                    "Seed": seed
                }
                
                pehe, delta_cate = compute_delta_cate(name, model, X_test_l, feature_names, PM25_COL_IDX)
                res["PEHE"] = float(pehe)
                res["Delta CATE"] = float(delta_cate)
                res["Sensitivity"] = float(compute_sensitivity(name, model, X_test_l, feature_names))

                per_seed_results.setdefault(name, []).append(res)

        # DLC
        print("\n👑 Loading DLC SOTA Results...")
        dlc_model_path = Path(args.dlc_weights)
        if not dlc_model_path.exists():
            dlc_model_path = PROJECT_ROOT / args.dlc_weights
            if not dlc_model_path.exists():
                print(f"Weights {dlc_model_path} not found. Skipping DLC.")
                dlc_model_path = None
        
        if dlc_model_path:
            try:
                pm25_threshold = float(np.median(X_train_l[:, -1]))
                dlc = DLCWrapper(
                    dlc_model_path,
                    input_dim=23,
                    fit_scaler_on=X_pancan,
                    model_config={"d_hidden": args.dlc_d_hidden, "num_layers": args.dlc_num_layers},
                    pm25_threshold=pm25_threshold,
                    strict_load=args.dlc_strict
                )
                
                t0 = sync_time()
                y_prob = dlc.predict_proba(X_test_l)[:, 1]
                infer_time = sync_time() - t0
                
                # Threshold should ideally be fixed at 0.5 or searched on train, but not on test.
                # Since DLC doesn't have an explicit 'predict' logic built-in for threshold, we use 0.5.
                y_pred = (y_prob > 0.5).astype(int)
                
                res = {
                    "AUC": float(roc_auc_score(y_test_l, y_prob)),
                    "Acc": float(accuracy_score(y_test_l, y_pred)),
                    "F1": float(f1_score(y_test_l, y_pred)),
                    "Precision": float(precision_score(y_test_l, y_pred)),
                    "Recall": float(recall_score(y_test_l, y_pred)),
                    "Training Time (s)": 0.0, # Pre-trained
                    "Inference Time (ms)": float(infer_time / len(X_test_l) * 1000),
                    "Params": str(get_inference_params_count(dlc.model)),
                    "Seed": seed
                }
                pehe, delta_cate = compute_delta_cate("DLC", dlc, X_test_l, feature_names, PM25_COL_IDX)
                res["PEHE"] = float(pehe)
                res["Delta CATE"] = float(delta_cate)
                res["Sensitivity"] = float(compute_sensitivity("DLC", dlc, X_test_l, feature_names))
                per_seed_results.setdefault("DLC (SOTA)", []).append(res)
            except Exception as e:
                print(f"Error evaluating DLC: {e}")
                import traceback
                traceback.print_exc()

    # Reporting
    md_path = PROJECT_ROOT / f"revised_benchmark_results.md"
    csv_path = PROJECT_ROOT / f"revised_benchmark_results.csv"
    
    metrics_order = ["AUC", "Acc", "PEHE", "Delta CATE", "Sensitivity", "Training Time (s)", "Inference Time (ms)"]
    
    md_lines = ["# Revised Fair Benchmark Results", "", "| Method | " + " | ".join(metrics_order) + " |"]
    md_lines.append("|-------|" + "|".join(["---"] * len(metrics_order)) + "|")
    
    csv_rows = []
    
    for model_name, records in per_seed_results.items():
        row = f"| {model_name} |"
        summary = summarize_metrics(records, metrics_order)
        
        csv_row = {"Method": model_name}
        
        for m in metrics_order:
            if m in summary:
                val_str = f"{summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}"
                row += f" {val_str} |"
                csv_row[m] = summary[m]['mean']
                csv_row[f"{m}_std"] = summary[m]['std']
            else:
                row += " N/A |"
                csv_row[m] = ""
                csv_row[f"{m}_std"] = ""
        md_lines.append(row)
        csv_rows.append(csv_row)
        
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
        
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        
    print(f"\nTable generated at: {md_path}")
    print(f"CSV generated at: {csv_path}")

if __name__ == "__main__":
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    main()

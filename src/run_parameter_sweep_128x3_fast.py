import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the actual run_single_config logic
from src.run_parameter_sweep_final import run_single_config

def main():
    print(">>> Sweeping HSIC Weight with 128x3 (FAST MODE)...")
    hsic_results = []
    # To save time, we reduce seeds to 3 and epochs significantly
    for hsic in [0.0, 0.01, 0.1, 1.0, 10.0]:
        scores = []
        for seed in [42, 43, 44]:
            try:
                res = run_single_config(seed, {
                    'lambda_hsic': hsic, 
                    'd_hidden': 128, 
                    'num_layers': 3,
                    'epochs_pre': 30,
                    'epochs_fine': 20
                })
                print(f"  -> Seed {seed}: AUC={res['AUC']:.4f}, CATE={res['Delta_CATE']:.4f}", flush=True)
                scores.append(res)
            except Exception as e:
                print(f"  -> Seed {seed}: Failed ({e})", flush=True)
                
        if scores:
            avg = {k: np.mean([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            std = {k: np.std([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            print(f"HSIC: lambda={hsic} -> AUC={avg['AUC']:.4f}±{std['AUC']:.4f}, CATE={avg['Delta_CATE']:.4f}±{std['Delta_CATE']:.4f}", flush=True)
            hsic_results.append({
                'lambda_hsic': hsic, 
                'AUC': avg['AUC'], 'AUC_Std': std['AUC'], 
                'PEHE': avg['PEHE'], 'PEHE_Std': std['PEHE'],
                'Delta CATE': avg['Delta_CATE'], 'Delta_CATE_Std': std['Delta_CATE']
            })
            
    pd.DataFrame(hsic_results).to_csv(PROJECT_ROOT / "lambda_hsic_sweep_results.csv", index=False)

    print("\n>>> Sweeping CATE Weight with 128x3 (FAST MODE)...", flush=True)
    cate_results = []
    for cate in [0.0, 0.5, 1.0, 2.0, 5.0]:
        scores = []
        for seed in [42, 43, 44]:
            try:
                res = run_single_config(seed, {
                    'lambda_cate': cate, 
                    'd_hidden': 128, 
                    'num_layers': 3,
                    'epochs_pre': 30,
                    'epochs_fine': 20
                })
                print(f"  -> Seed {seed}: AUC={res['AUC']:.4f}, CATE={res['Delta_CATE']:.4f}", flush=True)
                scores.append(res)
            except Exception as e:
                print(f"  -> Seed {seed}: Failed ({e})", flush=True)
                
        if scores:
            avg = {k: np.mean([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            std = {k: np.std([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            print(f"CATE Weight: lambda={cate} -> AUC={avg['AUC']:.4f}±{std['AUC']:.4f}, CATE={avg['Delta_CATE']:.4f}±{std['Delta_CATE']:.4f}", flush=True)
            cate_results.append({
                'lambda_cate': cate, 
                'AUC': avg['AUC'], 'AUC_Std': std['AUC'], 
                'PEHE': avg['PEHE'], 'PEHE_Std': std['PEHE'],
                'Delta CATE': avg['Delta_CATE'], 'Delta_CATE_Std': std['Delta_CATE']
            })
    pd.DataFrame(cate_results).to_csv(PROJECT_ROOT / "lambda_cate_sweep_results.csv", index=False)
    print("Done generating CSVs.")

if __name__ == "__main__":
    main()

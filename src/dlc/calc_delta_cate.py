import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.run_final_sota import load_data
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_cate

def calc_delta_cate():
    print("="*60)
    print("Calculating Delta CATE for Exp 1 (PANCAN SOTA)")
    print("="*60)

    # 1. Load Data
    print("[1/3] Loading Data...")
    X, y, true_prob, feature_names = load_data('PANCAN', 'interaction', verbose=False)
    
    # Identify EGFR column index
    try:
        egfr_idx = feature_names.index('EGFR')
        pm25_idx = feature_names.index('Virtual_PM2.5')
        print(f"Feature Indices: EGFR={egfr_idx}, PM2.5={pm25_idx}")
    except ValueError:
        print("Error: Could not find EGFR or Virtual_PM2.5 in features.")
        return

    # 2. Load Model
    print("[2/3] Loading Model...")
    model_path = 'results/dlc_final_sota.pth'
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    best_params = checkpoint['best_params'].copy()
    
    # Filter params
    model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'batch_size', 'epochs']}
    
    model = DLCNet(input_dim=X.shape[1], **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Compute CATE
    print("[3/3] Computing CATE and Delta CATE...")
    X_tensor = torch.FloatTensor(X)
    
    # Calculate CATE for ALL samples
    cate_all = compute_cate(model, X_tensor, treatment_col_idx=pm25_idx)
    
    # Split by EGFR status
    # Note: X is numpy array. feature_names maps directly to columns of X.
    egfr_values = X[:, egfr_idx]
    
    # EGFR = 1 (Mutated)
    mask_mut = egfr_values > 0.5
    cate_mut = cate_all[mask_mut]
    
    # EGFR = 0 (Wildtype)
    mask_wt = egfr_values <= 0.5
    cate_wt = cate_all[mask_wt]
    
    # Calculate Stats
    mean_cate_mut = np.mean(cate_mut)
    mean_cate_wt = np.mean(cate_wt)
    delta_cate = mean_cate_mut - mean_cate_wt
    
    print("-" * 40)
    print(f"CATE (EGFR Mutated): {mean_cate_mut:.4f} (N={len(cate_mut)})")
    print(f"CATE (Wildtype):     {mean_cate_wt:.4f}  (N={len(cate_wt)})")
    print(f"Delta CATE:          {delta_cate:.4f}")
    print("-" * 40)
    
    # Compare with TabR
    print(f"TabR Baseline:       +0.0633")
    if delta_cate > 0.0633:
        print("Result: DLC Exp 1 BEATS TabR! 🚀")
    else:
        print(f"Result: DLC Exp 1 is conservative ({delta_cate:.4f} vs 0.0633). Waiting for Exp 4...")

if __name__ == '__main__':
    calc_delta_cate()
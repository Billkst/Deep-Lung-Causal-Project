import sys
import os
import pandas as pd
import numpy as np
import torch
sys.path.append(os.getcwd())

from src.dlc.ground_truth import GroundTruthGenerator
from src.run_parameter_sweep_final import load_data_rigorous, run_single_config

print(">>> Testing Data Loading and ITE Generation...")
try:
    (X_p, y_p, t_p, ite_p, egfr_p, tp_p, 
     X_l, y_l, t_l, ite_l, egfr_l, tp_l, feats) = load_data_rigorous(42)
    
    print(f"X_p shape: {X_p.shape}")
    print(f"ITE_p mean: {ite_p.mean():.4f}, std: {ite_p.std():.4f}")
    if np.abs(ite_p.mean()) < 1e-5 and ite_p.std() < 1e-5:
        print("FAIL: ITE_p is all zeros!")
        sys.exit(1)
    else:
        print("PASS: ITE_p has non-zero values.")
        
    print(f"ITE_l mean: {ite_l.mean():.4f}, std: {ite_l.std():.4f}")
    if np.abs(ite_l.mean()) < 1e-5 and ite_l.std() < 1e-5:
        print("FAIL: ITE_l is all zeros!")
        sys.exit(1)
    else:
        print("PASS: ITE_l has non-zero values.")

except Exception as e:
    print(f"FAIL: Data loading crashed: {e}")
    sys.exit(1)

print("\n>>> Testing Single Config Run (Smoke Test)...")
try:
    # Run a very short training just to see if it crashes
    res = run_single_config(42, {'epochs_pre': 1, 'epochs_fine': 1, 'bs_pre': 16, 'bs_fine': 16})
    print(f"Run success! Result: {res}")
    if np.abs(res['Delta_CATE']) < 1e-5:
         print("WARNING: Delta_CATE is very close to zero, might still be an issue or just insufficient training.")
    else:
         print(f"PASS: Delta_CATE is {res['Delta_CATE']:.4f}")

except Exception as e:
    print(f"FAIL: Single config run crashed: {e}")
    sys.exit(1)
    
print("\nALL SYSTEM GO.")

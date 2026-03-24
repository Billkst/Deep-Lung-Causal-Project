import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Force unbuffered output
import os
os.environ['PYTHONUNBUFFERED'] = '1'

from src.run_parameter_sweep_final import run_single_config

config = {
    'd_hidden': 128, 'num_layers': 3,
    'lambda_hsic': 0.1, 'lambda_pred': 3.5,
    'lambda_ite': 1.0, 'lambda_cate': 5.0,
    'lambda_prob': 1.0, 'lambda_adv': 0.5
}

seeds = [42, 43, 44]
results = []

for seed in seeds:
    print(f"\n{'='*60}", flush=True)
    print(f"Training lambda_cate=5.0, seed={seed}", flush=True)
    print(f"{'='*60}", flush=True)
    
    t0 = time.time()
    metrics = run_single_config(seed, config)
    train_time = time.time() - t0
    
    results.append({
        'seed': seed,
        'training_time_s': train_time,
        'AUC': metrics['AUC'],
        'ACC': metrics.get('ACC', 0),
        'F1': metrics.get('F1', 0),
        'PEHE': metrics['PEHE'],
        'Delta_CATE': metrics['Delta_CATE'],
        'Sensitivity': metrics.get('Sens_Age', 0)
    })
    
    print(f"\nSeed {seed} completed:", flush=True)
    print(f"  Time: {train_time:.1f}s ({train_time/60:.1f}min)", flush=True)
    print(f"  AUC: {metrics['AUC']:.4f}", flush=True)
    print(f"  Delta CATE: {metrics['Delta_CATE']:.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(results)

print(f"\n{'='*60}", flush=True)
print("Final Results (lambda_cate=5.0)", flush=True)
print(f"{'='*60}", flush=True)
print(f"AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}", flush=True)
print(f"Training Time: {df['training_time_s'].mean():.1f} ± {df['training_time_s'].std():.1f} s", flush=True)

df.to_csv('results/final/benchmark/dlc_lambda5_training_results.csv', index=False)
print("\nSaved to results/final/benchmark/dlc_lambda5_training_results.csv", flush=True)

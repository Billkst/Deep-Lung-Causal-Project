#!/usr/bin/env python3
"""
Lambda CATE密集扫描 - 基于最终PARAMS
扫描点: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_parameter_sweep_final import run_single_config
import numpy as np
import pandas as pd

def main():
    config_base = {
        'd_hidden': 128, 
        'num_layers': 3, 
        'lambda_hsic': 0.1,
        'lambda_pred': 3.5,
        'lambda_ite': 1.0,
        'lambda_prob': 1.0,
        'lambda_adv': 0.5
    }
    
    lambda_cate_values = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
    seeds = [42, 43, 44]
    
    results = []
    
    print("=" * 60)
    print("Lambda CATE密集扫描 (基于最终PARAMS)")
    print("=" * 60)
    
    for lc in lambda_cate_values:
        print(f"\n>>> lambda_cate = {lc}")
        config = {**config_base, 'lambda_cate': lc}
        seed_results = []
        
        for seed in seeds:
            try:
                res = run_single_config(seed, config)
                print(f"  Seed {seed}: AUC={res['AUC']:.4f}, CATE={res['Delta_CATE']:.4f}, PEHE={res['PEHE']:.4f}")
                seed_results.append(res)
            except Exception as e:
                print(f"  Seed {seed}: Failed - {e}")
        
        if seed_results:
            avg = {k: np.mean([s[k] for s in seed_results]) for k in ['AUC', 'PEHE', 'Delta_CATE']}
            std = {k: np.std([s[k] for s in seed_results], ddof=1) for k in ['AUC', 'PEHE', 'Delta_CATE']}
            
            results.append({
                'lambda_cate': lc,
                'AUC': avg['AUC'], 'AUC_Std': std['AUC'],
                'PEHE': avg['PEHE'], 'PEHE_Std': std['PEHE'],
                'Delta CATE': avg['Delta_CATE'], 'Delta_CATE_Std': std['Delta_CATE']
            })
            print(f"  平均: AUC={avg['AUC']:.4f}±{std['AUC']:.4f}, CATE={avg['Delta_CATE']:.4f}±{std['Delta_CATE']:.4f}")
    
    df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "results/final/sweeps/lambda_cate_dense_sweep_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存: {output_path}")

if __name__ == "__main__":
    main()

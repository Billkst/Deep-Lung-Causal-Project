#!/usr/bin/env python3
"""
测量DLC完整时间：Training Time + Inference Time
基于最终SOTA PARAMS (s_seed系列)
"""
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_baselines_final import load_clean_pancan, load_luad_target, DLCWrapper
from src.run_parameter_sweep_final import run_single_config

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def main():
    # 最终SOTA PARAMS
    config = {
        'd_hidden': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'lambda_hsic': 0.1,
        'lambda_pred': 3.5,
        'lambda_ite': 1.0,
        'lambda_cate': 2.0,
        'lambda_prob': 1.0,
        'lambda_adv': 0.5
    }
    
    seeds = [42, 43, 44, 45, 46]
    results = []
    
    print("=" * 70)
    print("DLC 完整时间测量 (Training + Inference)")
    print("=" * 70)
    
    # 加载数据
    X_pancan, y_pancan, _ = load_clean_pancan()
    X_luad, y_luad, feature_names = load_luad_target()
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"[Seed {seed}]")
        print(f"{'='*70}")
        
        # 1. 测量Training Time
        print(f"\n[1/2] 测量 Training Time...")
        t0_train = sync_time()
        
        try:
            metrics = run_single_config(seed, config)
            train_time = sync_time() - t0_train
            
            print(f"  ✓ 训练完成")
            print(f"    时间: {train_time:.2f}s ({train_time/60:.2f}min)")
            print(f"    AUC: {metrics.get('AUC', 0):.4f}")
            print(f"    Delta CATE: {metrics.get('Delta_CATE', 0):.4f}")
            
        except Exception as e:
            print(f"  ✗ 训练失败: {e}")
            continue
        
        # 2. 测量Inference Time
        print(f"\n[2/2] 测量 Inference Time...")
        
        # 准备测试数据
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_luad, y_luad, test_size=0.2, random_state=seed, stratify=y_luad
        )
        
        # 加载刚训练好的checkpoint
        ckpt_path = PROJECT_ROOT / f"results/dlc_final_sota_s_seed_{seed}.pth"
        
        if not ckpt_path.exists():
            print(f"  ✗ Checkpoint不存在: {ckpt_path}")
            continue
        
        try:
            pm25_threshold = float(np.median(X_train_l[:, -1]))
            dlc = DLCWrapper(
                ckpt_path,
                input_dim=23,
                fit_scaler_on=X_pancan,
                model_config={"d_hidden": config['d_hidden'], "num_layers": config['num_layers']},
                pm25_threshold=pm25_threshold,
                strict_load=False
            )
            
            # 预热
            _ = dlc.predict_proba(X_test_l[:10])
            
            # 正式测量
            t0_infer = sync_time()
            y_prob = dlc.predict_proba(X_test_l)[:, 1]
            infer_time_total = sync_time() - t0_infer
            infer_time_per_sample = infer_time_total / len(X_test_l) * 1000  # ms
            
            y_pred = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y_test_l, y_prob)
            acc = accuracy_score(y_test_l, y_pred)
            f1 = f1_score(y_test_l, y_pred)
            
            print(f"  ✓ 推理完成")
            print(f"    总时间: {infer_time_total*1000:.2f}ms")
            print(f"    单样本: {infer_time_per_sample:.4f}ms")
            print(f"    测试集大小: {len(X_test_l)}")
            print(f"    AUC: {auc:.4f}")
            
            results.append({
                'seed': seed,
                'training_time_s': train_time,
                'training_time_min': train_time / 60,
                'inference_time_total_ms': infer_time_total * 1000,
                'inference_time_per_sample_ms': infer_time_per_sample,
                'test_size': len(X_test_l),
                'AUC': auc,
                'ACC': acc,
                'F1': f1
            })
            
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 统计结果
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n{'='*70}")
        print("统计结果")
        print(f"{'='*70}")
        
        print(f"\n【Training Time】")
        print(f"  Mean: {df['training_time_s'].mean():.2f}s ({df['training_time_min'].mean():.2f}min)")
        print(f"  Std:  {df['training_time_s'].std():.2f}s")
        print(f"  Range: [{df['training_time_s'].min():.2f}s, {df['training_time_s'].max():.2f}s]")
        
        print(f"\n【Inference Time (per sample)】")
        print(f"  Mean: {df['inference_time_per_sample_ms'].mean():.4f}ms")
        print(f"  Std:  {df['inference_time_per_sample_ms'].std():.4f}ms")
        
        print(f"\n【性能验证】")
        print(f"  AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}")
        print(f"  ACC: {df['ACC'].mean():.4f} ± {df['ACC'].std():.4f}")
        print(f"  F1:  {df['F1'].mean():.4f} ± {df['F1'].std():.4f}")
        
        # 保存结果
        output_dir = PROJECT_ROOT / "results/final/benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "dlc_complete_time_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存: {csv_path}")
        
    else:
        print("\n✗ 没有成功的测量结果")

if __name__ == "__main__":
    main()

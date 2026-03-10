#!/usr/bin/env python3
"""
基于现有s_seed checkpoints测量DLC Inference Time
从训练日志提取Training Time
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

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def extract_training_time_from_log(seed):
    """从日志中提取训练时间"""
    log_path = PROJECT_ROOT / f"logs/run_s_seed_{seed}.log"
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # 查找训练开始和结束的时间戳
    # 这里简化处理，返回None表示需要实际测量
    return None

def main():
    seeds = [42, 43, 44, 45, 46]
    results = []
    
    print("=" * 70)
    print("DLC Inference Time 测量 (基于现有 s_seed checkpoints)")
    print("=" * 70)
    
    X_pancan, y_pancan, _ = load_clean_pancan()
    X_luad, y_luad, feature_names = load_luad_target()
    
    for seed in seeds:
        print(f"\n[Seed {seed}]")
        
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_luad, y_luad, test_size=0.2, random_state=seed, stratify=y_luad
        )
        
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
                model_config={"d_hidden": 128, "num_layers": 3},
                pm25_threshold=pm25_threshold,
                strict_load=False
            )
            
            _ = dlc.predict_proba(X_test_l[:10])
            
            t0 = sync_time()
            y_prob = dlc.predict_proba(X_test_l)[:, 1]
            infer_time_total = sync_time() - t0
            infer_time_per_sample = infer_time_total / len(X_test_l) * 1000
            
            y_pred = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y_test_l, y_prob)
            acc = accuracy_score(y_test_l, y_pred)
            f1 = f1_score(y_test_l, y_pred)
            
            print(f"  ✓ 推理完成")
            print(f"    总时间: {infer_time_total*1000:.2f}ms")
            print(f"    单样本: {infer_time_per_sample:.4f}ms")
            print(f"    AUC: {auc:.4f}")
            
            results.append({
                'seed': seed,
                'inference_time_total_ms': infer_time_total * 1000,
                'inference_time_per_sample_ms': infer_time_per_sample,
                'test_size': len(X_test_l),
                'AUC': auc,
                'ACC': acc,
                'F1': f1
            })
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n{'='*70}")
        print("Inference Time 统计")
        print(f"{'='*70}")
        print(f"  Mean: {df['inference_time_per_sample_ms'].mean():.4f} ± {df['inference_time_per_sample_ms'].std():.4f} ms")
        print(f"  Range: [{df['inference_time_per_sample_ms'].min():.4f}, {df['inference_time_per_sample_ms'].max():.4f}] ms")
        
        output_dir = PROJECT_ROOT / "results/final/benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "dlc_inference_time_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存: {csv_path}")

if __name__ == "__main__":
    main()

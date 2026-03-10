#!/usr/bin/env python3
"""
Measure DLC Real Training Time
测量DLC真实训练时间 (128x3 SOTA配置)
"""
import sys
import time
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_parameter_sweep_final import run_single_config

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def main():
    # SOTA配置: 128x3, lambda_cate=2.0, lambda_hsic=0.1
    config = {'d_hidden': 128, 'num_layers': 3, 'lambda_cate': 2.0, 'lambda_hsic': 0.1}
    seeds = [42, 43, 44, 45, 46]
    
    results = []
    print("=" * 60)
    print("DLC 真实训练时间测量 (SOTA 128x3)")
    print("=" * 60)
    
    for seed in seeds:
        print(f"\n[Seed {seed}] 开始训练...")
        t0 = sync_time()
        
        try:
            metrics = run_single_config(seed, config)
            train_time = sync_time() - t0
            
            results.append({
                'seed': seed,
                'training_time_s': train_time,
                'AUC': metrics.get('AUC', 0),
                'Delta_CATE': metrics.get('Delta_CATE', 0)
            })
            print(f"  ✓ 完成 | 训练时间: {train_time:.2f}s | AUC: {metrics.get('AUC', 0):.4f}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    if results:
        times = [r['training_time_s'] for r in results]
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        print("\n" + "=" * 60)
        print(f"训练时间统计 (n={len(results)})")
        print(f"  Mean: {mean_time:.2f}s")
        print(f"  Std:  {std_time:.2f}s")
        print(f"  Range: [{min(times):.2f}s, {max(times):.2f}s]")
        print("=" * 60)
        
        # 保存结果
        import pandas as pd
        df = pd.DataFrame(results)
        output_path = PROJECT_ROOT / "results/final/benchmark/dlc_training_time_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存: {output_path}")
        
        # 保存说明
        notes = f"""# DLC 训练时间测量结果

## 测量协议

- **架构**: d_hidden=128, num_layers=3
- **超参数**: lambda_cate=2.0, lambda_hsic=0.1
- **训练流程**: 完整的 pre-train (200 epochs) + fine-tune (100 epochs)
- **计时范围**: 从训练开始到生成最终checkpoint结束
- **硬件同步**: 使用 torch.cuda.synchronize() 确保准确计时

## 结果

- **平均训练时间**: {mean_time:.2f} ± {std_time:.2f} 秒
- **测试种子数**: {len(results)}
- **时间范围**: [{min(times):.2f}s, {max(times):.2f}s]

## 说明

这是DLC模型从零开始训练到生成最终SOTA checkpoint的真实时间成本。
与benchmark中的"Training Time = 0"不同,后者是部署态推理基准(使用预训练权重)。

## 详细数据

见 dlc_training_time_results.csv
"""
        notes_path = PROJECT_ROOT / "results/final/benchmark/dlc_training_time_notes.md"
        with open(notes_path, 'w', encoding='utf-8') as f:
            f.write(notes)
        print(f"说明已保存: {notes_path}")

if __name__ == "__main__":
    main()

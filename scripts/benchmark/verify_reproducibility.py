#!/usr/bin/env python3
"""
验证当前代码能否复现接近SOTA的性能
使用1个seed快速验证
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_parameter_sweep_final import run_single_config

def main():
    config = {'d_hidden': 128, 'num_layers': 3, 'lambda_cate': 2.0, 'lambda_hsic': 0.1}
    seed = 42
    
    print("=" * 60)
    print("验证SOTA性能复现 (128x3, seed=42)")
    print("=" * 60)
    
    try:
        metrics = run_single_config(seed, config)
        
        print("\n结果:")
        print(f"  AUC: {metrics.get('AUC', 0):.4f}")
        print(f"  Delta CATE: {metrics.get('Delta_CATE', 0):.4f}")
        print(f"  PEHE: {metrics.get('PEHE', 0):.4f}")
        
        print("\n历史SOTA参考 (seed=42):")
        print("  AUC: ~0.787")
        print("  Delta CATE: ~0.172")
        print("  PEHE: ~0.118")
        
        auc_close = abs(metrics.get('AUC', 0) - 0.787) < 0.01
        cate_close = abs(metrics.get('Delta_CATE', 0) - 0.172) < 0.02
        
        if auc_close and cate_close:
            print("\n✓ 性能接近历史SOTA,代码可复现")
        else:
            print("\n⚠ 性能与历史SOTA有差异,建议检查")
            
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")

if __name__ == "__main__":
    main()

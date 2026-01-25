"""
Task 5.1: 在合成数据上运行冒烟测试

验证 DLCNet 在小规模合成数据上的完整训练流程。

验收标准:
- 训练完成无错误
- 训练集 AUC > 0.55 (随机数据的合理基线)
- 显存占用 < 2GB
"""

import numpy as np
import torch
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dlc.dlc_net import DLCNet


def main():
    print("="*70)
    print("Task 5.1: 在合成数据上运行冒烟测试")
    print("="*70)
    
    # 生成合成数据
    print("\n[步骤 1] 生成合成数据...")
    np.random.seed(42)
    X_train = np.random.randn(500, 23)
    y_train = np.random.randint(0, 2, 500)
    print(f"  训练集大小: {X_train.shape}")
    print(f"  标签分布: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    
    # 训练模型
    print("\n[步骤 2] 训练模型...")
    model = DLCNet(
        input_dim=23,
        d_conf=8,
        d_effect=16,
        d_hidden=64,
        num_heads=4,
        num_layers=2,
        lambda_hsic=0.1,
        lambda_pred=1.0,
        random_state=42
    )
    
    # 使用较少的 epochs 进行快速测试
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=True)
    
    # 评估模型
    print("\n[步骤 3] 评估模型...")
    metrics = model.evaluate(X_train, y_train)
    
    print("\n" + "="*70)
    print("评估结果:")
    print("="*70)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 验收标准检查
    print("\n" + "="*70)
    print("验收标准检查:")
    print("="*70)
    
    # 检查 1: 训练完成无错误
    print("  ✓ 训练完成无错误")
    
    # 检查 2: 训练集 AUC > 0.55 (随机数据的合理基线)
    auc = metrics['auc_roc']
    if auc > 0.55:
        print(f"  ✓ 训练集 AUC ({auc:.4f}) > 0.55 (随机数据基线)")
    else:
        print(f"  ✗ 训练集 AUC ({auc:.4f}) <= 0.55")
        sys.exit(1)
    
    # 检查 3: 显存占用 < 2GB
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"  显存占用: {memory_allocated:.2f} GB")
        if memory_allocated < 2.0:
            print(f"  ✓ 显存占用 ({memory_allocated:.2f} GB) < 2GB")
        else:
            print(f"  ✗ 显存占用 ({memory_allocated:.2f} GB) >= 2GB")
            sys.exit(1)
    else:
        print("  ℹ CPU 模式，跳过显存检查")
    
    print("\n" + "="*70)
    print("Task 5.1 冒烟测试通过! ✓")
    print("="*70)


if __name__ == "__main__":
    main()

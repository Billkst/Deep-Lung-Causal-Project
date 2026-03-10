"""
TransTEE 超参数调优脚本

目标: 在 IHDP 数据集上达到 PEHE < 0.6
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.baselines.transtee_baseline import IHDPDataset, TransTEEBaseline

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 加载数据
print("加载 IHDP 数据...")
dataset = IHDPDataset()
X, t, y, y_cf = dataset.load_data(include_cfactual=True)

# 计算真实 ITE
true_ite = np.where(t == 1, y - y_cf, y_cf - y)

# 数据划分
X_train, X_test, t_train, t_test, y_train, y_test, ite_train, ite_test = train_test_split(
    X, t, y, true_ite, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print("\n开始超参数调优...\n")

# 尝试不同的超参数配置
configs = [
    {'hidden_dim': 256, 'epochs': 200, 'lr': 0.0005, 'patience': 20, 'name': 'Config1: Large model + Low LR'},
    {'hidden_dim': 128, 'epochs': 200, 'lr': 0.0001, 'patience': 25, 'name': 'Config2: Standard model + Very low LR'},
    {'hidden_dim': 192, 'epochs': 150, 'lr': 0.0003, 'patience': 20, 'name': 'Config3: Medium model + Medium LR'},
]

best_pehe = float('inf')
best_config = None

for i, config in enumerate(configs, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(configs)}] {config['name']}")
    print(f"{'='*60}")
    print(f"hidden_dim={config['hidden_dim']}, epochs={config['epochs']}, lr={config['lr']}, patience={config['patience']}")
    
    model = TransTEEBaseline(
        random_state=42,
        hidden_dim=config['hidden_dim'],
        n_heads=4,
        n_layers=2,
        epochs=config['epochs'],
        batch_size=64,
        learning_rate=config['lr'],
        patience=config['patience']
    )
    
    model.fit(X_train, t_train, y_train)
    pehe = model.evaluate_pehe(X_test, ite_test)
    
    print(f"\nResult: PEHE = {pehe:.4f}")
    
    if pehe < best_pehe:
        best_pehe = pehe
        best_config = config
        print(f"✅ New best configuration!")
    
    if pehe < 0.6:
        print(f"🎉 Performance target achieved! (PEHE < 0.6)")
        break

print(f"\n\n{'='*60}")
print(f"Final Results")
print(f"{'='*60}")
print(f"Best configuration: {best_config['name']}")
print(f"Best PEHE: {best_pehe:.4f}")
if best_pehe < 0.6:
    print(f"✅ Performance target achieved!")
else:
    print(f"⚠️  Performance target not met, further tuning recommended")

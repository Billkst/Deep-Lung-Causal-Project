# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.run_final_sota import load_data
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_cate, compute_sensitivity_score
from sklearn.model_selection import train_test_split

def check_exp1_performance():
    print("="*60)
    print("正在使用修正后的 metrics.py 重新评估 EXP1 模型...")
    print("="*60)

    # 1. 加载数据 (PANCAN)
    print("\n[1/4] 加载数据...")
    X, y, true_prob, feature_names = load_data('PANCAN', 'interaction', verbose=False)
    
    # 划分测试集 (保持随机种子一致，确保是同一批测试数据)
    _, X_test, _, _, _, _ = train_test_split(
        X, y, true_prob, test_size=0.2, random_state=42, stratify=y
    )
    
    # 转为 Tensor
    X_test_tensor = torch.FloatTensor(X_test)
    print(f"测试集样本数: {len(X_test)}")

    # 2. 加载模型
    print("\n[2/4] 加载预训练权重...")
    model_path = 'results/dlc_final_sota.pth'
    if not Path(model_path).exists():
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    best_params = checkpoint['best_params'].copy()

    # 剔除训练专用参数，只保留架构参数
    model_params = {
        k: v for k, v in best_params.items() 
        if k not in ['lr', 'batch_size', 'epochs']
    }
    
    # 初始化模型
    model = DLCNet(input_dim=X.shape[1], **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型加载成功")

    # 3. 重新计算 CATE (关键验证点!)
    print("\n[3/4] 重新计算 CATE (使用 84th/16th 分位数法)...")
    try:
        # 查找 PM2.5 索引
        pm25_idx = feature_names.index('Virtual_PM2.5')
        cate = compute_cate(model, X_test_tensor, treatment_col_idx=pm25_idx)
        
        mean_cate = np.mean(cate)
        std_cate = np.std(cate)
        
        print(f"Feature: Virtual_PM2.5 (Index {pm25_idx})")
        print(f"Mean CATE: {mean_cate:.4f}  <-- 期望: 正数 (表示有害)")
        print(f"Std  CATE: {std_cate:.4f}   <-- 期望: 较大 (表示有异质性)")
        
        if mean_cate < 0:
            print("⚠️ 警告: Mean CATE 仍为负值，可能需要检查标签方向或模型学习情况。")
        else:
            print("✅ 正常: Mean CATE 为正值，符合逻辑。")
            
    except Exception as e:
        print(f"❌ CATE 计算出错: {e}")

    # 4. 重新计算 Sensitivity (关键验证点!)
    print("\n[4/4] 重新计算 Sensitivity (使用 1 Std 扰动)...")
    try:
        # PM2.5 Sensitivity
        sens_pm25 = compute_sensitivity_score(model, X_test_tensor, confounder_idx=pm25_idx)
        print(f"Sensitivity (PM2.5): {sens_pm25:.4f} <-- 期望: 明显大于 0")
        
        # Age Sensitivity
        age_idx = feature_names.index('Age')
        sens_age = compute_sensitivity_score(model, X_test_tensor, confounder_idx=age_idx)
        print(f"Sensitivity (Age):   {sens_age:.4f} <-- 期望: 接近 0")
        
        if sens_pm25 < 0.01:
            print("⚠️ 警告: 模型对 PM2.5 仍然很不敏感，可能在 '偷懒'。")
        else:
            print("✅ 正常: 模型捕捉到了 PM2.5 的影响。")

    except Exception as e:
        print(f"❌ Sensitivity 计算出错: {e}")

if __name__ == '__main__':
    check_exp1_performance()
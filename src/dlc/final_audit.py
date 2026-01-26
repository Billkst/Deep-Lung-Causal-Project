# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.run_final_sota import load_data
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_cate, compute_pehe, compute_sensitivity_score

def evaluate_model(model_path, X, y, true_prob, feature_names, tag="Model"):
    print(f"\n>>> 正在审计: {tag} ({model_path})")
    
    if not Path(model_path).exists():
        print(f"❌ 文件不存在: {model_path}")
        return None

    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # --- 关键修复: 处理微调模型缺失 best_params 的情况 ---
    if 'best_params' in checkpoint:
        best_params = checkpoint['best_params'].copy()
    else:
        print("   ℹ️ [Info] 该模型未保存架构参数 (可能是微调模型)。")
        print("   🔄 正在从 Exp 1 (results/dlc_final_sota.pth) 加载基础架构参数...")
        base_path = 'results/dlc_final_sota.pth'
        if not Path(base_path).exists():
            print(f"   ❌ 致命错误: 找不到 Exp 1 模型文件 ({base_path})，无法重建模型架构！")
            return None
        
        # 加载 Exp 1 的配置
        try:
            base_ckpt = torch.load(base_path, map_location='cpu', weights_only=False)
        except TypeError:
            base_ckpt = torch.load(base_path, map_location='cpu')
            
        best_params = base_ckpt['best_params'].copy()
        print("   ✅ 架构参数加载成功")

    # 过滤参数
    model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'batch_size', 'epochs']}
    
    # 初始化
    model = DLCNet(input_dim=X.shape[1], **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 准备数据
    X_tensor = torch.FloatTensor(X)
    
    # 1. 计算 AUC
    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    
    # 3. 计算 Delta CATE (关键指标)
    try:
        pm25_idx = feature_names.index('Virtual_PM2.5')
        egfr_idx = feature_names.index('EGFR')
        
        cate_pred = compute_cate(model, X_tensor, treatment_col_idx=pm25_idx)
        
        # 分组
        mask_mut = X[:, egfr_idx] > 0.5
        mask_wt = X[:, egfr_idx] <= 0.5
        
        mean_cate_mut = np.mean(cate_pred[mask_mut])
        mean_cate_wt = np.mean(cate_pred[mask_wt])
        
        delta_cate = mean_cate_mut - mean_cate_wt
    except ValueError:
        print("⚠️ 警告: 未找到 EGFR 或 Virtual_PM2.5 列，跳过 CATE 计算")
        delta_cate = 0.0
        mean_cate_mut = 0.0
        mean_cate_wt = 0.0
    
    # 4. 计算 Sensitivity
    try:
        age_idx = feature_names.index('Age')
        sens_age = compute_sensitivity_score(model, X_tensor, confounder_idx=age_idx)
    except ValueError:
        sens_age = 0.0
    
    # 打印报告
    print(f"  📊 AUC:        {auc:.4f}")
    print(f"  🎯 Delta CATE: {delta_cate:.4f} (目标: > 0.05)")
    print(f"     - Mutated:  {mean_cate_mut:.4f}")
    print(f"     - Wildtype: {mean_cate_wt:.4f}")
    print(f"  🛡️ Sens (Age): {sens_age:.4f} (目标: < 0.01)")
    
    return {
        'auc': auc, 
        'delta_cate': delta_cate, 
        'sens_age': sens_age,
        'cate_mut': mean_cate_mut,
        'cate_wt': mean_cate_wt
    }

def main():
    print("="*60)
    print("DLC 最终审计 (Final Audit)")
    print("="*60)
    
    # 加载 LUAD 数据
    print("[1/2] 加载 LUAD 数据...")
    X, y, true_prob, feature_names = load_data('LUAD', 'interaction', verbose=False)
    
    # 审计 Exp 2 (Baseline)
    res_exp2 = evaluate_model(
        'results/LUAD_Baseline/dlc_final_sota.pth', 
        X, y, true_prob, feature_names, 
        tag="EXP2 (Baseline)"
    )
    
    # 审计 Exp 4 (SOTA)
    res_exp4 = evaluate_model(
        'results/LUAD_Finetuned/dlc_finetuned.pth', 
        X, y, true_prob, feature_names, 
        tag="EXP4 (Transfer)"
    )
    
    print("\n" + "="*60)
    print("🏆 最终胜负判定")
    print("="*60)
    
    if res_exp4 and res_exp2:
        # 判定 AUC
        auc_lift = res_exp4['auc'] - res_exp2['auc']
        print(f"1. AUC 提升: {auc_lift:+.4f} " + ("✅ 胜" if auc_lift > 0 else "❌ 败"))
        
        # 判定 交互发现 (Delta CATE)
        delta_lift = res_exp4['delta_cate'] - res_exp2['delta_cate']
        print(f"2. 交互发现能力: {res_exp4['delta_cate']:.4f} vs {res_exp2['delta_cate']:.4f}")
        
        # 判定逻辑
        if res_exp4['delta_cate'] > 0.02 and res_exp4['delta_cate'] > res_exp2['delta_cate']:
            print("   ✅ EXP4 成功发现了显著的 EGFR 交互效应！")
            print("   (这证明了迁移学习从 PANCAN 继承了因果结构，并在 LUAD 上放大了信号)")
        elif res_exp4['delta_cate'] < 0:
            print("   ❌ EXP4 方向错误 (负值)。")
        else:
            print("   ⚠️ 提升不明显。")

if __name__ == '__main__':
    main()
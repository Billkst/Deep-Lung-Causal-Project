#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 21.4 & 21.6: 执行混杂因子敏感性评估

集成已训练的模型，运行完整的敏感性评估流程，生成报告和可视化。

作者：Kiro AI Agent
日期：2026-01-23
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.baselines.mogonet_baseline import MOGONETBaseline
from src.baselines.transtee_baseline import TransTEEBaseline
from src.evaluate_confounding_sensitivity import ConfoundingSensitivityEvaluator

warnings.filterwarnings('ignore')


def train_all_models(data_path: str = "data/luad_synthetic_interaction.csv"):
    """
    训练所有 5 个基线模型
    
    Args:
        data_path: LUAD 数据集路径
        
    Returns:
        models: 训练好的模型字典
    """
    print("="*80)
    print("训练所有基线模型（使用 LUAD 全量数据）")
    print("="*80)
    
    # 加载数据
    print(f"\n[INFO] 加载数据：{data_path}")
    df = pd.read_csv(data_path)
    
    # 剔除 ID 列
    if 'sampleID' in df.columns:
        df = df.drop('sampleID', axis=1)
    
    # 分离特征和标签
    if 'Outcome_Label' in df.columns:
        y = df['Outcome_Label'].values
        X_df = df.drop('Outcome_Label', axis=1)
    else:
        raise ValueError("数据集中未找到 Outcome_Label 列")
    
    # 剔除 True_Prob 列（如果存在）
    if 'True_Prob' in X_df.columns:
        X_df = X_df.drop('True_Prob', axis=1)
    
    # 提取 Virtual_PM2.5 用于 TransTEE
    pm25_values = X_df['Virtual_PM2.5'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)
    
    print(f"[INFO] 数据形状：{X.shape}")
    print(f"[INFO] 标签分布：{np.bincount(y)}")
    
    models = {}
    
    # 1. XGBoost
    print("\n[1/5] 训练 XGBoost...")
    xgb_model = XGBBaseline(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)
    models['XGBoost'] = xgb_model
    print("  - XGBoost 训练完成")
    
    # 2. TabR
    print("\n[2/5] 训练 TabR...")
    tabr_model = TabRBaseline(
        random_state=42, 
        k_neighbors=5, 
        hidden_dim=64, 
        n_heads=4, 
        n_layers=2,
        epochs=30,
        batch_size=32
    )
    tabr_model.fit(X, y)
    models['TabR'] = tabr_model
    print("  - TabR 训练完成")
    
    # 3. HyperFast（带类别权重）
    print("\n[3/5] 训练 HyperFast...")
    # 计算类别权重
    n_samples = len(y)
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    weight_0 = n_samples / (2 * n_class_0)
    weight_1 = n_samples / (2 * n_class_1)
    class_weights = [weight_0, weight_1]
    
    hyperfast_model = HyperFastBaseline(
        random_state=42,
        hidden_dim=128,
        epochs=30,
        batch_size=32,
        class_weights=class_weights,
        prediction_threshold=0.4
    )
    hyperfast_model.fit(X, y)
    models['HyperFast'] = hyperfast_model
    print("  - HyperFast 训练完成")
    
    # 4. MOGONET（需要多视图数据）
    print("\n[4/5] 训练 MOGONET...")
    # 将特征切分为多视图：前3列为临床特征，后面为基因特征
    X_clinical = X[:, :3]  # Age, Gender, Virtual_PM2.5
    X_omics = X[:, 3:]     # Top20 基因
    views = [X_clinical, X_omics]
    
    mogonet_model = MOGONETBaseline(
        random_state=42,
        hidden_dim=64,
        epochs=30,
        batch_size=32
    )
    mogonet_model.fit(views, y)
    models['MOGONET'] = mogonet_model
    print("  - MOGONET 训练完成")
    
    # 5. TransTEE（因果推断模型）
    print("\n[5/5] 训练 TransTEE...")
    # 二值化 PM2.5 为 Treatment
    median_pm25 = np.median(pm25_values)
    treatment = (pm25_values >= median_pm25).astype(int)
    
    print(f"  - PM2.5 中位数：{median_pm25:.2f}")
    print(f"  - Treatment 分布：{np.bincount(treatment)}")
    
    transtee_model = TransTEEBaseline(
        random_state=42,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        epochs=50,
        batch_size=64
    )
    transtee_model.fit(X, treatment, y)
    models['TransTEE'] = transtee_model
    print("  - TransTEE 训练完成")
    
    print("\n[SUCCESS] 所有模型训练完成")
    return models


def main():
    """主函数"""
    print("="*80)
    print("Task 21: Phase 5 - 混杂因子敏感性评估")
    print("="*80)
    
    # 数据路径
    data_path = "data/luad_synthetic_interaction.csv"
    
    # Task 21.4: 训练所有模型
    print("\n[Task 21.4] 集成训练模型...")
    models = train_all_models(data_path)
    
    # Task 21.6: 执行评估
    print("\n[Task 21.6] 执行混杂因子敏感性评估...")
    
    # 初始化评估器
    evaluator = ConfoundingSensitivityEvaluator(
        data_path=data_path,
        models=models
    )
    
    # 运行评估
    results = evaluator.run_evaluation(
        age_col_name='Age',
        pm25_col_name='Virtual_PM2.5'
    )
    
    # 生成报告
    evaluator.generate_report(
        output_path="results/confounding_sensitivity_report.md"
    )
    
    # 生成可视化
    evaluator.plot_ite_age_dependency(
        age_col_name='Age',
        pm25_col_name='Virtual_PM2.5',
        output_path="results/ite_age_dependency.png"
    )
    
    print("\n" + "="*80)
    print("Task 21 完成！")
    print("="*80)
    print("\n[INFO] 输出文件：")
    print("  - results/confounding_sensitivity_report.md")
    print("  - results/ite_age_dependency.png")


if __name__ == '__main__':
    main()

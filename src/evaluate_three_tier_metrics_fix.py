#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 19.4-Fix: 修正与补全三层评估指标

修正内容：
1. 扩大验证样本量：使用 LUAD 全量数据集 (n=513) 作为 Mechanism Validation Set
2. 补全分类模型的 Proxy ΔCATE 计算
3. HyperFast 最终处置

作者：Kiro AI Agent
日期：2026-01-23
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.baselines.mogonet_baseline import MOGONETBaseline
from src.baselines.transtee_baseline import TransTEEBaseline

warnings.filterwarnings('ignore')


def compute_proxy_ite(model, X: np.ndarray, pm25_feature_idx: int, model_name: str = None) -> np.ndarray:
    """
    计算分类模型的 Proxy ITE（反事实干预法）
    
    原理：
    对于每个样本 x，通过控制变量法（Ceteris Paribus）计算：
    1. 构造高暴露样本 x_high：复制 x，强制将 Virtual_PM2.5 设为 +1.0
    2. 构造低暴露样本 x_low：复制 x，强制将 Virtual_PM2.5 设为 -1.0
    3. Proxy ITE(x) = Model.predict_proba(x_high) - Model.predict_proba(x_low)
    
    这种方法剥离了基因背景的干扰，纯粹提取模型眼中的"PM2.5 效应"。
    
    Args:
        model: 训练好的分类模型
        X: 原始特征矩阵 (n_samples, n_features)
        pm25_feature_idx: Virtual_PM2.5 特征在 X 中的索引
        model_name: 模型名称（用于处理 MOGONET 的特殊情况）
    
    Returns:
        proxy_ite: Proxy ITE (n_samples,)
    """
    # 构造高暴露样本（PM2.5 = +1.0）
    X_high = X.copy()
    X_high[:, pm25_feature_idx] = 1.0
    
    # 构造低暴露样本（PM2.5 = -1.0）
    X_low = X.copy()
    X_low[:, pm25_feature_idx] = -1.0
    
    # MOGONET 需要多视图输入
    if model_name == 'MOGONET':
        # 切分为多视图
        X_high_clinical = X_high[:, :3]
        X_high_omics = X_high[:, 3:]
        views_high = [X_high_clinical, X_high_omics]
        
        X_low_clinical = X_low[:, :3]
        X_low_omics = X_low[:, 3:]
        views_low = [X_low_clinical, X_low_omics]
        
        # 预测概率
        y_proba_high = model.predict_proba(views_high)
        y_proba_low = model.predict_proba(views_low)
    else:
        # 预测概率
        y_proba_high = model.predict_proba(X_high)
        y_proba_low = model.predict_proba(X_low)
    
    # 确保是一维数组
    if len(y_proba_high.shape) > 1:
        y_proba_high = y_proba_high[:, 1]
    if len(y_proba_low.shape) > 1:
        y_proba_low = y_proba_low[:, 1]
    
    # 计算 Proxy ITE
    proxy_ite = y_proba_high - y_proba_low
    
    return proxy_ite




class ThreeTierEvaluatorFix:
    """三层评估指标计算器（修正版）"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.luad_full_data = None
        self.models = {}
    
    def load_luad_full_data(self):
        """加载 LUAD 全量数据集 (n=513)"""
        print("[INFO] 加载 LUAD 全量数据集...")
        
        # 加载数据
        data_path = "data/luad_synthetic_interaction.csv"
        self.luad_full_data = pd.read_csv(data_path)
        
        print(f"[SUCCESS] LUAD 全量数据加载完成")
        print(f"  - 样本数: {len(self.luad_full_data)}")
        print(f"  - EGFR 突变样本数: {(self.luad_full_data['EGFR'] == 1).sum()}")
        print(f"  - EGFR 野生型样本数: {(self.luad_full_data['EGFR'] == 0).sum()}")
    
    def prepare_data(self):
        """准备数据用于模型训练和评估"""
        print("\n[INFO] 准备数据...")
        
        # 提取特征和标签
        feature_cols = [col for col in self.luad_full_data.columns 
                       if col not in ['sampleID', 'Virtual_PM2.5', 'True_Prob', 'Outcome_Label']]
        
        X = self.luad_full_data[feature_cols].values  # 转换为 numpy 数组
        y = self.luad_full_data['Outcome_Label'].values  # 转换为 numpy 数组
        
        # 提取 Treatment（二值化 PM2.5）
        pm25_median = self.luad_full_data['Virtual_PM2.5'].median()
        treatment = (self.luad_full_data['Virtual_PM2.5'] > pm25_median).astype(int).values
        
        # 保存特征列名（用于后续的 EGFR 和 TP53 提取）
        self.feature_cols = feature_cols
        
        # 添加标准化的 PM2.5 特征到 X（用于 Proxy ITE 计算）
        # 标准化 PM2.5 到 [-1, 1] 范围
        pm25_values = self.luad_full_data['Virtual_PM2.5'].values
        pm25_mean = pm25_values.mean()
        pm25_std = pm25_values.std()
        pm25_standardized = (pm25_values - pm25_mean) / pm25_std
        
        # 将标准化的 PM2.5 添加到特征矩阵
        X_with_pm25 = np.column_stack([X, pm25_standardized])
        self.pm25_feature_idx = X_with_pm25.shape[1] - 1  # PM2.5 在最后一列
        
        print(f"  - 特征数: {len(feature_cols)} + 1 (PM2.5)")
        print(f"  - 样本数: {len(X_with_pm25)}")
        print(f"  - Treatment=1: {treatment.sum()}")
        print(f"  - Treatment=0: {(treatment == 0).sum()}")
        print(f"  - PM2.5 特征索引: {self.pm25_feature_idx}")
        
        return X_with_pm25, y, treatment
    
    def train_all_models(self, X, y, treatment):
        """训练所有模型"""
        print("\n" + "="*80)
        print("训练所有模型（使用 LUAD 全量数据）")
        print("="*80)
        
        # 1. XGBoost
        print("\n[1/5] 训练 XGBoost...")
        xgb_model = XGBBaseline(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        xgb_model.fit(X, y)
        self.models['XGBoost'] = xgb_model
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
        self.models['TabR'] = tabr_model
        print("  - TabR 训练完成")
        
        # 3. HyperFast
        print("\n[3/5] 训练 HyperFast...")
        hyperfast_model = HyperFastBaseline(
            random_state=42,
            hidden_dim=128,
            epochs=30,
            batch_size=32
        )
        hyperfast_model.fit(X, y)
        self.models['HyperFast'] = hyperfast_model
        print("  - HyperFast 训练完成")
        
        # 4. MOGONET（需要多视图数据）
        print("\n[4/5] 训练 MOGONET...")
        # 将特征切分为多视图：前3列为临床特征，后面为基因特征
        # 临床特征：Age, Gender, (PM2.5)
        # 基因特征：Top20 基因
        X_clinical = X[:, :3]  # Age, Gender, PM2.5
        X_omics = X[:, 3:]     # Top20 基因
        views = [X_clinical, X_omics]
        
        mogonet_model = MOGONETBaseline(
            random_state=42,
            hidden_dim=64,
            epochs=30,
            batch_size=32
        )
        mogonet_model.fit(views, y)
        self.models['MOGONET'] = mogonet_model
        print("  - MOGONET 训练完成")
        
        # 5. TransTEE
        print("\n[5/5] 训练 TransTEE...")
        transtee_model = TransTEEBaseline(
            random_state=42,
            hidden_dim=64,
            n_heads=4,
            n_layers=2,
            epochs=30,
            batch_size=32
        )
        transtee_model.fit(X, y, treatment)
        self.models['TransTEE'] = transtee_model
        print("  - TransTEE 训练完成")
        
        print("\n[SUCCESS] 所有模型训练完成")

    
    def compute_tier1_general_performance(self, X, y) -> Dict:
        """层级 1：通用性能"""
        print("\n" + "="*80)
        print("层级 1：通用性能 (General Performance)")
        print("="*80)
        
        tier1_metrics = {}
        
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        for model_name in classification_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                # MOGONET 需要多视图输入
                if model_name == 'MOGONET':
                    X_clinical = X[:, :3]
                    X_omics = X[:, 3:]
                    views = [X_clinical, X_omics]
                    y_pred = model.predict(views)
                    y_proba = model.predict_proba(views)
                else:
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X)
                
                # 确保 y_proba 是一维数组
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
                
                # 计算指标
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='binary', zero_division=0),
                    'recall': recall_score(y, y_pred, average='binary', zero_division=0),
                    'f1_binary': f1_score(y, y_pred, average='binary', zero_division=0),
                    'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
                    'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
                    'auc_roc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
                }
                
                tier1_metrics[model_name] = metrics
                
                print(f"\n{model_name}:")
                print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  - F1-Score (Macro): {metrics['f1_macro']:.4f}")
                print(f"  - F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        return tier1_metrics
    
    def compute_tier2_egfr_mechanism(self, X, y, treatment) -> Dict:
        """层级 2：机制特异性验证（阳性对照 - EGFR）"""
        print("\n" + "="*80)
        print("层级 2：机制特异性验证（阳性对照 - EGFR）")
        print("="*80)
        
        tier2_metrics = {}
        
        # 获取 EGFR 特征（从特征列名中找到 EGFR 的索引）
        egfr_idx = self.feature_cols.index('EGFR')
        egfr = X[:, egfr_idx]
        
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        # 分类模型的子组 Recall 和 Proxy ΔCATE
        for model_name in classification_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                # MOGONET 需要多视图输入
                if model_name == 'MOGONET':
                    X_clinical = X[:, :3]
                    X_omics = X[:, 3:]
                    views = [X_clinical, X_omics]
                    y_pred = model.predict(views)
                    y_proba = model.predict_proba(views)
                else:
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X)
                
                # 确保 y_proba 是一维数组
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
                
                # 子组 Recall
                egfr_mutant_mask = (egfr == 1)
                egfr_wild_mask = (egfr == 0)
                
                recall_mutant = recall_score(
                    y[egfr_mutant_mask], 
                    y_pred[egfr_mutant_mask], 
                    average='binary', 
                    zero_division=0
                ) if egfr_mutant_mask.sum() > 0 else 0.0
                
                recall_wild = recall_score(
                    y[egfr_wild_mask], 
                    y_pred[egfr_wild_mask], 
                    average='binary', 
                    zero_division=0
                ) if egfr_wild_mask.sum() > 0 else 0.0
                
                # 计算 Proxy ITE（反事实干预法）
                proxy_ite = compute_proxy_ite(model, X, self.pm25_feature_idx, model_name)
                
                # 计算 Proxy ΔCATE
                mean_proxy_ite_mutant = proxy_ite[egfr_mutant_mask].mean()
                mean_proxy_ite_wild = proxy_ite[egfr_wild_mask].mean()
                proxy_delta_cate = mean_proxy_ite_mutant - mean_proxy_ite_wild
                
                tier2_metrics[model_name] = {
                    'recall_egfr_mutant': recall_mutant,
                    'recall_egfr_wild': recall_wild,
                    'delta_recall': recall_mutant - recall_wild,
                    'mean_proxy_ite_egfr_mutant': mean_proxy_ite_mutant,
                    'mean_proxy_ite_egfr_wild': mean_proxy_ite_wild,
                    'proxy_delta_cate_egfr': proxy_delta_cate,
                    'n_egfr_mutant': egfr_mutant_mask.sum(),
                    'n_egfr_wild': egfr_wild_mask.sum()
                }
                
                print(f"\n{model_name}:")
                print(f"  - Recall (EGFR-Mutant): {recall_mutant:.4f} (n={egfr_mutant_mask.sum()})")
                print(f"  - Recall (EGFR-Wild): {recall_wild:.4f} (n={egfr_wild_mask.sum()})")
                print(f"  - Δ Recall: {recall_mutant - recall_wild:+.4f}")
                print(f"  - Proxy ΔCATE (EGFR): {proxy_delta_cate:+.4f}")
        
        # TransTEE 的 ΔCATE
        if 'TransTEE' in self.models:
            model = self.models['TransTEE']
            
            # 预测 ITE（TransTEE 的 predict_ite 只需要 X）
            ite = model.predict_ite(X)
            
            # 计算 ΔCATE
            ite_egfr_mutant = ite[egfr == 1]
            ite_egfr_wild = ite[egfr == 0]
            
            mean_ite_mutant = np.mean(ite_egfr_mutant) if len(ite_egfr_mutant) > 0 else 0.0
            mean_ite_wild = np.mean(ite_egfr_wild) if len(ite_egfr_wild) > 0 else 0.0
            delta_cate = mean_ite_mutant - mean_ite_wild
            
            tier2_metrics['TransTEE'] = {
                'mean_ite_egfr_mutant': mean_ite_mutant,
                'mean_ite_egfr_wild': mean_ite_wild,
                'delta_cate_egfr': delta_cate,
                'n_egfr_mutant': len(ite_egfr_mutant),
                'n_egfr_wild': len(ite_egfr_wild)
            }
            
            print(f"\nTransTEE:")
            print(f"  - Mean ITE (EGFR-Mutant): {mean_ite_mutant:.4f} (n={len(ite_egfr_mutant)})")
            print(f"  - Mean ITE (EGFR-Wild): {mean_ite_wild:.4f} (n={len(ite_egfr_wild)})")
            print(f"  - ΔCATE (EGFR): {delta_cate:+.4f}")
            
            # 验证预期
            if delta_cate > 0:
                print(f"  - ✓ 验证通过：ΔCATE > 0（符合阳性对照预期）")
            else:
                print(f"  - ✗ 验证失败：ΔCATE ≤ 0（不符合阳性对照预期）")
        
        return tier2_metrics

    
    def compute_tier3_tp53_negative_control(self, X, y, treatment) -> Dict:
        """层级 3：虚假关联排查（阴性对照 - TP53）"""
        print("\n" + "="*80)
        print("层级 3：虚假关联排查（阴性对照 - TP53）")
        print("="*80)
        
        tier3_metrics = {}
        
        # 获取 TP53 特征（从特征列名中找到 TP53 的索引）
        tp53_idx = self.feature_cols.index('TP53')
        tp53 = X[:, tp53_idx]
        
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        # 分类模型的子组 Recall 和 Proxy ΔCATE
        for model_name in classification_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                # MOGONET 需要多视图输入
                if model_name == 'MOGONET':
                    X_clinical = X[:, :3]
                    X_omics = X[:, 3:]
                    views = [X_clinical, X_omics]
                    y_pred = model.predict(views)
                    y_proba = model.predict_proba(views)
                else:
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X)
                
                # 确保 y_proba 是一维数组
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
                
                # 子组 Recall
                tp53_mutant_mask = (tp53 == 1)
                tp53_wild_mask = (tp53 == 0)
                
                recall_mutant = recall_score(
                    y[tp53_mutant_mask], 
                    y_pred[tp53_mutant_mask], 
                    average='binary', 
                    zero_division=0
                ) if tp53_mutant_mask.sum() > 0 else 0.0
                
                recall_wild = recall_score(
                    y[tp53_wild_mask], 
                    y_pred[tp53_wild_mask], 
                    average='binary', 
                    zero_division=0
                ) if tp53_wild_mask.sum() > 0 else 0.0
                
                # 计算 Proxy ITE（反事实干预法）
                proxy_ite = compute_proxy_ite(model, X, self.pm25_feature_idx, model_name)
                
                # 计算 Proxy ΔCATE
                mean_proxy_ite_mutant = proxy_ite[tp53_mutant_mask].mean()
                mean_proxy_ite_wild = proxy_ite[tp53_wild_mask].mean()
                proxy_delta_cate = mean_proxy_ite_mutant - mean_proxy_ite_wild
                
                tier3_metrics[model_name] = {
                    'recall_tp53_mutant': recall_mutant,
                    'recall_tp53_wild': recall_wild,
                    'delta_recall': recall_mutant - recall_wild,
                    'mean_proxy_ite_tp53_mutant': mean_proxy_ite_mutant,
                    'mean_proxy_ite_tp53_wild': mean_proxy_ite_wild,
                    'proxy_delta_cate_tp53': proxy_delta_cate,
                    'n_tp53_mutant': tp53_mutant_mask.sum(),
                    'n_tp53_wild': tp53_wild_mask.sum()
                }
                
                print(f"\n{model_name}:")
                print(f"  - Recall (TP53-Mutant): {recall_mutant:.4f} (n={tp53_mutant_mask.sum()})")
                print(f"  - Recall (TP53-Wild): {recall_wild:.4f} (n={tp53_wild_mask.sum()})")
                print(f"  - Δ Recall: {recall_mutant - recall_wild:+.4f}")
                print(f"  - Proxy ΔCATE (TP53): {proxy_delta_cate:+.4f}")
        
        # TransTEE 的 ΔCATE（阴性对照）
        if 'TransTEE' in self.models:
            model = self.models['TransTEE']
            
            # 预测 ITE（TransTEE 的 predict_ite 只需要 X）
            ite = model.predict_ite(X)
            
            # 计算 ΔCATE
            ite_tp53_mutant = ite[tp53 == 1]
            ite_tp53_wild = ite[tp53 == 0]
            
            mean_ite_mutant = np.mean(ite_tp53_mutant) if len(ite_tp53_mutant) > 0 else 0.0
            mean_ite_wild = np.mean(ite_tp53_wild) if len(ite_tp53_wild) > 0 else 0.0
            delta_cate = mean_ite_mutant - mean_ite_wild
            
            tier3_metrics['TransTEE'] = {
                'mean_ite_tp53_mutant': mean_ite_mutant,
                'mean_ite_tp53_wild': mean_ite_wild,
                'delta_cate_tp53': delta_cate,
                'n_tp53_mutant': len(ite_tp53_mutant),
                'n_tp53_wild': len(ite_tp53_wild)
            }
            
            print(f"\nTransTEE:")
            print(f"  - Mean ITE (TP53-Mutant): {mean_ite_mutant:.4f} (n={len(ite_tp53_mutant)})")
            print(f"  - Mean ITE (TP53-Wild): {mean_ite_wild:.4f} (n={len(ite_tp53_wild)})")
            print(f"  - ΔCATE (TP53): {delta_cate:+.4f}")
            
            # 验证预期（阴性对照应接近 0）
            threshold = 0.1
            if abs(delta_cate) < threshold:
                print(f"  - ✓ 验证通过：|ΔCATE| < {threshold}（符合阴性对照预期）")
            else:
                print(f"  - ✗ 验证失败：|ΔCATE| ≥ {threshold}（模型可能在'幻觉'交互效应）")
        
        return tier3_metrics
    
    def generate_comprehensive_report(self, tier1_metrics: Dict, 
                                      tier2_metrics: Dict, 
                                      tier3_metrics: Dict):
        """生成综合评估报告"""
        print("\n" + "="*80)
        print("综合评估报告：三层指标体系（修正版）")
        print("="*80)
        
        # 层级 1：通用性能对比表
        print("\n### 层级 1：通用性能对比")
        print("\n| Model | AUC-ROC | F1-Macro | F1-Weighted |")
        print("|-------|---------|----------|-------------|")
        
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        for model_name in classification_models:
            if model_name in tier1_metrics:
                m = tier1_metrics[model_name]
                status = "⚠️" if m['auc_roc'] <= 0.5 else "✅" if m['auc_roc'] >= 0.85 else ""
                print(f"| {model_name} | {m['auc_roc']:.4f} {status} | "
                      f"{m['f1_macro']:.4f} | {m['f1_weighted']:.4f} |")
        
        # 层级 2：EGFR 机制验证
        print("\n### 层级 2：机制特异性验证（阳性对照 - EGFR）")
        print("\n#### 分类模型 Proxy ΔCATE (EGFR)")
        print("\n| Model | Proxy ΔCATE | 验证结果 |")
        print("|-------|-------------|----------|")
        
        for model_name in classification_models:
            if model_name in tier2_metrics:
                m = tier2_metrics[model_name]
                validation = "✓ 通过" if m['proxy_delta_cate_egfr'] > 0 else "✗ 失败"
                print(f"| {model_name} | {m['proxy_delta_cate_egfr']:+.4f} | {validation} |")
        
        print("\n#### TransTEE ΔCATE (EGFR)")
        print("\n| Model | Mean ITE (EGFR-Mutant) | Mean ITE (EGFR-Wild) | ΔCATE | 验证结果 |")
        print("|-------|------------------------|----------------------|-------|----------|")
        
        if 'TransTEE' in tier2_metrics:
            m = tier2_metrics['TransTEE']
            validation = "✓ 通过" if m['delta_cate_egfr'] > 0 else "✗ 失败"
            print(f"| TransTEE | {m['mean_ite_egfr_mutant']:.4f} | "
                  f"{m['mean_ite_egfr_wild']:.4f} | {m['delta_cate_egfr']:+.4f} | {validation} |")
        
        # 层级 3：TP53 阴性对照
        print("\n### 层级 3：虚假关联排查（阴性对照 - TP53）")
        print("\n#### 分类模型 Proxy ΔCATE (TP53)")
        print("\n| Model | Proxy ΔCATE | 验证结果 |")
        print("|-------|-------------|----------|")
        
        threshold = 0.1
        
        for model_name in classification_models:
            if model_name in tier3_metrics:
                m = tier3_metrics[model_name]
                validation = "✓ 通过" if abs(m['proxy_delta_cate_tp53']) < threshold else "✗ 失败"
                print(f"| {model_name} | {m['proxy_delta_cate_tp53']:+.4f} | {validation} |")
        
        print("\n#### TransTEE ΔCATE (TP53 - 阴性对照)")
        print("\n| Model | Mean ITE (TP53-Mutant) | Mean ITE (TP53-Wild) | ΔCATE | 验证结果 |")
        print("|-------|------------------------|----------------------|-------|----------|")
        
        if 'TransTEE' in tier3_metrics:
            m = tier3_metrics['TransTEE']
            validation = "✓ 通过" if abs(m['delta_cate_tp53']) < threshold else "✗ 失败"
            print(f"| TransTEE | {m['mean_ite_tp53_mutant']:.4f} | "
                  f"{m['mean_ite_tp53_wild']:.4f} | {m['delta_cate_tp53']:+.4f} | {validation} |")
    
    def save_results(self, tier1_metrics: Dict, tier2_metrics: Dict, tier3_metrics: Dict):
        """保存结果到 JSON 文件"""
        output = {
            'tier1_general_performance': tier1_metrics,
            'tier2_egfr_mechanism': tier2_metrics,
            'tier3_tp53_negative_control': tier3_metrics
        }
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        output_serializable = convert_to_serializable(output)
        
        output_path = self.results_dir / 'three_tier_metrics_fix.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] 修正后的三层指标已保存到：{output_path}")


def main():
    """主函数"""
    print("="*80)
    print("Task 19.4-Fix: 修正与补全三层评估指标")
    print("="*80)
    
    # 初始化评估器
    evaluator = ThreeTierEvaluatorFix()
    
    # 加载 LUAD 全量数据
    evaluator.load_luad_full_data()
    
    # 准备数据
    X, y, treatment = evaluator.prepare_data()
    
    # 训练所有模型
    evaluator.train_all_models(X, y, treatment)
    
    # 计算三层指标
    tier1_metrics = evaluator.compute_tier1_general_performance(X, y)
    tier2_metrics = evaluator.compute_tier2_egfr_mechanism(X, y, treatment)
    tier3_metrics = evaluator.compute_tier3_tp53_negative_control(X, y, treatment)
    
    # 生成综合报告
    evaluator.generate_comprehensive_report(tier1_metrics, tier2_metrics, tier3_metrics)
    
    # 保存结果
    evaluator.save_results(tier1_metrics, tier2_metrics, tier3_metrics)
    
    print("\n" + "="*80)
    print("Task 19.4-Fix 完成！")
    print("="*80)


if __name__ == "__main__":
    main()

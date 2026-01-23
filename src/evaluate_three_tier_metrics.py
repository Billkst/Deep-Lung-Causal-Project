#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 19.4: 评估指标体系（三层指标）

三层评估指标：
1. 层级 1：通用性能 (General Performance)
   - AUC-ROC, F1-Score (Macro/Weighted)
   - 对所有 5 个模型在实验 A 和 B 下分别计算

2. 层级 2：机制特异性验证（阳性对照 - EGFR）
   - Subgroup Recall: 在测试集中，将人群分为 EGFR-Mutant 和 EGFR-Wild 两组，分别报告 Recall
   - TransTEE ΔCATE: 计算 Mean(ITE | EGFR=1) - Mean(ITE | EGFR=0)
   - 预期：ΔCATE 应显著大于 0

3. 层级 3：虚假关联排查（阴性对照 - TP53）
   - 选择 TP53（或其他高频但无交互的基因）作为对照
   - TransTEE ΔCATE (Negative): 计算 Mean(ITE | TP53=1) - Mean(ITE | TP53=0)
   - 预期：该值应接近 0
   - 如果该值很大，说明模型在"幻觉"交互效应，实验失败

作者：Kiro AI Agent
日期：2026-01-22
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

warnings.filterwarnings('ignore')


class ThreeTierEvaluator:
    """三层评估指标计算器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.predictions_a = None
        self.predictions_b = None
        self.results_a = None
        self.results_b = None
    
    def load_experiment_results(self):
        """加载实验 A 和 B 的结果"""
        print("[INFO] 加载实验结果...")
        
        # 加载预测结果
        with open(self.results_dir / 'experiment_a_predictions.pkl', 'rb') as f:
            self.predictions_a = pickle.load(f)
        
        with open(self.results_dir / 'experiment_b_predictions.pkl', 'rb') as f:
            self.predictions_b = pickle.load(f)
        
        # 加载评估结果
        with open(self.results_dir / 'experiment_a_results.json', 'r') as f:
            self.results_a = json.load(f)
        
        with open(self.results_dir / 'experiment_b_results.json', 'r') as f:
            self.results_b = json.load(f)
        
        print("[SUCCESS] 实验结果加载完成")
        print(f"  - 实验 A 模型数：{len(self.predictions_a)}")
        print(f"  - 实验 B 模型数：{len(self.predictions_b)}")
    
    def compute_tier1_general_performance(self) -> Dict:
        """
        层级 1：通用性能 (General Performance)
        
        计算 AUC-ROC, F1-Score (Macro/Weighted)
        对所有 5 个模型在实验 A 和 B 下分别计算
        
        Returns:
            tier1_metrics: 包含所有模型的通用性能指标
        """
        print("\n" + "="*80)
        print("层级 1：通用性能 (General Performance)")
        print("="*80)
        
        tier1_metrics = {
            'experiment_a': {},
            'experiment_b': {}
        }
        
        # 分类模型列表
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        # 计算实验 A 的指标
        print("\n### 实验 A：临床现状基线 (LUAD Only)")
        for model_name in classification_models:
            if model_name in self.predictions_a:
                pred_data = self.predictions_a[model_name]
                y_true = pred_data['y_true']
                y_pred = pred_data['y_pred']
                y_proba = pred_data['y_proba']
                
                # 计算指标
                metrics = self._compute_classification_metrics(y_true, y_pred, y_proba)
                tier1_metrics['experiment_a'][model_name] = metrics
                
                print(f"\n{model_name}:")
                print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  - F1-Score (Macro): {metrics['f1_macro']:.4f}")
                print(f"  - F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        # 计算实验 B 的指标
        print("\n### 实验 B：数据增强基线 (PANCAN + LUAD)")
        for model_name in classification_models:
            if model_name in self.predictions_b:
                pred_data = self.predictions_b[model_name]
                y_true = pred_data['y_true']
                y_pred = pred_data['y_pred']
                y_proba = pred_data['y_proba']
                
                # 计算指标
                metrics = self._compute_classification_metrics(y_true, y_pred, y_proba)
                tier1_metrics['experiment_b'][model_name] = metrics
                
                print(f"\n{model_name}:")
                print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  - F1-Score (Macro): {metrics['f1_macro']:.4f}")
                print(f"  - F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        return tier1_metrics
    
    def _compute_classification_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """计算分类指标"""
        # 确保 y_proba 是二维数组
        if len(y_proba.shape) == 1:
            y_proba_binary = y_proba
        else:
            y_proba_binary = y_proba[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_binary': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba_binary) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        return metrics
    
    def compute_tier2_egfr_mechanism(self) -> Dict:
        """
        层级 2：机制特异性验证（阳性对照 - EGFR）
        
        1. Subgroup Recall: 在测试集中，将人群分为 EGFR-Mutant 和 EGFR-Wild 两组，分别报告 Recall
        2. TransTEE ΔCATE: 计算 Mean(ITE | EGFR=1) - Mean(ITE | EGFR=0)
        3. 预期：ΔCATE 应显著大于 0
        
        Returns:
            tier2_metrics: 包含 EGFR 机制验证的指标
        """
        print("\n" + "="*80)
        print("层级 2：机制特异性验证（阳性对照 - EGFR）")
        print("="*80)
        
        tier2_metrics = {
            'experiment_a': {},
            'experiment_b': {}
        }
        
        # 分类模型列表
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        # 实验 A
        print("\n### 实验 A：临床现状基线 (LUAD Only)")
        tier2_metrics['experiment_a'] = self._compute_egfr_subgroup_metrics(
            self.predictions_a, classification_models, experiment_name='A'
        )
        
        # 实验 B
        print("\n### 实验 B：数据增强基线 (PANCAN + LUAD)")
        tier2_metrics['experiment_b'] = self._compute_egfr_subgroup_metrics(
            self.predictions_b, classification_models, experiment_name='B'
        )
        
        return tier2_metrics
    
    def _compute_egfr_subgroup_metrics(self, predictions: Dict, 
                                       classification_models: List[str],
                                       experiment_name: str) -> Dict:
        """计算 EGFR 子组指标"""
        subgroup_metrics = {}
        
        # 分类模型的子组 Recall
        for model_name in classification_models:
            if model_name in predictions:
                pred_data = predictions[model_name]
                y_true = pred_data['y_true']
                y_pred = pred_data['y_pred']
                
                # 获取 EGFR 特征（从 TransTEE 的 X_test 中提取）
                if 'TransTEE' in predictions and 'X_test' in predictions['TransTEE']:
                    X_test = predictions['TransTEE']['X_test']
                    
                    if 'EGFR' in X_test.columns:
                        egfr = X_test['EGFR'].values
                        
                        # 分组计算 Recall
                        egfr_mutant_mask = (egfr == 1)
                        egfr_wild_mask = (egfr == 0)
                        
                        recall_mutant = recall_score(
                            y_true[egfr_mutant_mask], 
                            y_pred[egfr_mutant_mask], 
                            average='binary', 
                            zero_division=0
                        ) if egfr_mutant_mask.sum() > 0 else 0.0
                        
                        recall_wild = recall_score(
                            y_true[egfr_wild_mask], 
                            y_pred[egfr_wild_mask], 
                            average='binary', 
                            zero_division=0
                        ) if egfr_wild_mask.sum() > 0 else 0.0
                        
                        subgroup_metrics[model_name] = {
                            'recall_egfr_mutant': recall_mutant,
                            'recall_egfr_wild': recall_wild,
                            'delta_recall': recall_mutant - recall_wild,
                            'n_egfr_mutant': egfr_mutant_mask.sum(),
                            'n_egfr_wild': egfr_wild_mask.sum()
                        }
                        
                        print(f"\n{model_name}:")
                        print(f"  - Recall (EGFR-Mutant): {recall_mutant:.4f} (n={egfr_mutant_mask.sum()})")
                        print(f"  - Recall (EGFR-Wild): {recall_wild:.4f} (n={egfr_wild_mask.sum()})")
                        print(f"  - Δ Recall: {recall_mutant - recall_wild:+.4f}")
        
        # TransTEE 的 ΔCATE
        if 'TransTEE' in predictions and 'X_test' in predictions['TransTEE']:
            transtee_data = predictions['TransTEE']
            ite = transtee_data['ite']
            X_test = transtee_data['X_test']
            
            if 'EGFR' in X_test.columns:
                egfr = X_test['EGFR'].values
                
                # 计算 ΔCATE
                ite_egfr_mutant = ite[egfr == 1]
                ite_egfr_wild = ite[egfr == 0]
                
                mean_ite_mutant = np.mean(ite_egfr_mutant) if len(ite_egfr_mutant) > 0 else 0.0
                mean_ite_wild = np.mean(ite_egfr_wild) if len(ite_egfr_wild) > 0 else 0.0
                delta_cate = mean_ite_mutant - mean_ite_wild
                
                subgroup_metrics['TransTEE'] = {
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
        
        return subgroup_metrics
    
    def compute_tier3_tp53_negative_control(self) -> Dict:
        """
        层级 3：虚假关联排查（阴性对照 - TP53）
        
        1. 选择 TP53（或其他高频但无交互的基因）作为对照
        2. TransTEE ΔCATE (Negative): 计算 Mean(ITE | TP53=1) - Mean(ITE | TP53=0)
        3. 预期：该值应接近 0
        4. 如果该值很大，说明模型在"幻觉"交互效应，实验失败
        
        Returns:
            tier3_metrics: 包含 TP53 阴性对照的指标
        """
        print("\n" + "="*80)
        print("层级 3：虚假关联排查（阴性对照 - TP53）")
        print("="*80)
        
        tier3_metrics = {
            'experiment_a': {},
            'experiment_b': {}
        }
        
        # 分类模型列表
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        # 实验 A
        print("\n### 实验 A：临床现状基线 (LUAD Only)")
        tier3_metrics['experiment_a'] = self._compute_tp53_subgroup_metrics(
            self.predictions_a, classification_models, experiment_name='A'
        )
        
        # 实验 B
        print("\n### 实验 B：数据增强基线 (PANCAN + LUAD)")
        tier3_metrics['experiment_b'] = self._compute_tp53_subgroup_metrics(
            self.predictions_b, classification_models, experiment_name='B'
        )
        
        return tier3_metrics
    
    def _compute_tp53_subgroup_metrics(self, predictions: Dict, 
                                       classification_models: List[str],
                                       experiment_name: str) -> Dict:
        """计算 TP53 子组指标（阴性对照）"""
        subgroup_metrics = {}
        
        # 分类模型的子组 Recall
        for model_name in classification_models:
            if model_name in predictions:
                pred_data = predictions[model_name]
                y_true = pred_data['y_true']
                y_pred = pred_data['y_pred']
                
                # 获取 TP53 特征（从 TransTEE 的 X_test 中提取）
                if 'TransTEE' in predictions and 'X_test' in predictions['TransTEE']:
                    X_test = predictions['TransTEE']['X_test']
                    
                    if 'TP53' in X_test.columns:
                        tp53 = X_test['TP53'].values
                        
                        # 分组计算 Recall
                        tp53_mutant_mask = (tp53 == 1)
                        tp53_wild_mask = (tp53 == 0)
                        
                        recall_mutant = recall_score(
                            y_true[tp53_mutant_mask], 
                            y_pred[tp53_mutant_mask], 
                            average='binary', 
                            zero_division=0
                        ) if tp53_mutant_mask.sum() > 0 else 0.0
                        
                        recall_wild = recall_score(
                            y_true[tp53_wild_mask], 
                            y_pred[tp53_wild_mask], 
                            average='binary', 
                            zero_division=0
                        ) if tp53_wild_mask.sum() > 0 else 0.0
                        
                        subgroup_metrics[model_name] = {
                            'recall_tp53_mutant': recall_mutant,
                            'recall_tp53_wild': recall_wild,
                            'delta_recall': recall_mutant - recall_wild,
                            'n_tp53_mutant': tp53_mutant_mask.sum(),
                            'n_tp53_wild': tp53_wild_mask.sum()
                        }
                        
                        print(f"\n{model_name}:")
                        print(f"  - Recall (TP53-Mutant): {recall_mutant:.4f} (n={tp53_mutant_mask.sum()})")
                        print(f"  - Recall (TP53-Wild): {recall_wild:.4f} (n={tp53_wild_mask.sum()})")
                        print(f"  - Δ Recall: {recall_mutant - recall_wild:+.4f}")
        
        # TransTEE 的 ΔCATE（阴性对照）
        if 'TransTEE' in predictions and 'X_test' in predictions['TransTEE']:
            transtee_data = predictions['TransTEE']
            ite = transtee_data['ite']
            X_test = transtee_data['X_test']
            
            if 'TP53' in X_test.columns:
                tp53 = X_test['TP53'].values
                
                # 计算 ΔCATE
                ite_tp53_mutant = ite[tp53 == 1]
                ite_tp53_wild = ite[tp53 == 0]
                
                mean_ite_mutant = np.mean(ite_tp53_mutant) if len(ite_tp53_mutant) > 0 else 0.0
                mean_ite_wild = np.mean(ite_tp53_wild) if len(ite_tp53_wild) > 0 else 0.0
                delta_cate = mean_ite_mutant - mean_ite_wild
                
                subgroup_metrics['TransTEE'] = {
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
                threshold = 0.1  # 阈值：如果 |ΔCATE| < 0.1，认为接近 0
                if abs(delta_cate) < threshold:
                    print(f"  - ✓ 验证通过：|ΔCATE| < {threshold}（符合阴性对照预期）")
                else:
                    print(f"  - ✗ 验证失败：|ΔCATE| ≥ {threshold}（模型可能在'幻觉'交互效应）")
        
        return subgroup_metrics
    
    def generate_comprehensive_report(self, tier1_metrics: Dict, 
                                      tier2_metrics: Dict, 
                                      tier3_metrics: Dict):
        """生成综合评估报告"""
        print("\n" + "="*80)
        print("综合评估报告：三层指标体系")
        print("="*80)
        
        # 层级 1：通用性能对比表
        print("\n### 层级 1：通用性能对比")
        print("\n| Model | Experiment | AUC-ROC | F1-Macro | F1-Weighted |")
        print("|-------|------------|---------|----------|-------------|")
        
        classification_models = ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']
        
        for model_name in classification_models:
            # 实验 A
            if model_name in tier1_metrics['experiment_a']:
                m = tier1_metrics['experiment_a'][model_name]
                print(f"| {model_name} | A (LUAD Only) | {m['auc_roc']:.4f} | "
                      f"{m['f1_macro']:.4f} | {m['f1_weighted']:.4f} |")
            
            # 实验 B
            if model_name in tier1_metrics['experiment_b']:
                m = tier1_metrics['experiment_b'][model_name]
                print(f"| {model_name} | B (PANCAN+LUAD) | {m['auc_roc']:.4f} | "
                      f"{m['f1_macro']:.4f} | {m['f1_weighted']:.4f} |")
        
        # 层级 2：EGFR 机制验证
        print("\n### 层级 2：机制特异性验证（阳性对照 - EGFR）")
        print("\n#### 分类模型子组 Recall")
        print("\n| Model | Experiment | Recall (EGFR-Mutant) | Recall (EGFR-Wild) | Δ Recall |")
        print("|-------|------------|----------------------|--------------------|----------|")
        
        for model_name in classification_models:
            # 实验 A
            if model_name in tier2_metrics['experiment_a']:
                m = tier2_metrics['experiment_a'][model_name]
                print(f"| {model_name} | A | {m['recall_egfr_mutant']:.4f} | "
                      f"{m['recall_egfr_wild']:.4f} | {m['delta_recall']:+.4f} |")
            
            # 实验 B
            if model_name in tier2_metrics['experiment_b']:
                m = tier2_metrics['experiment_b'][model_name]
                print(f"| {model_name} | B | {m['recall_egfr_mutant']:.4f} | "
                      f"{m['recall_egfr_wild']:.4f} | {m['delta_recall']:+.4f} |")
        
        print("\n#### TransTEE ΔCATE (EGFR)")
        print("\n| Experiment | Mean ITE (EGFR-Mutant) | Mean ITE (EGFR-Wild) | ΔCATE | 验证结果 |")
        print("|------------|------------------------|----------------------|-------|----------|")
        
        # 实验 A
        if 'TransTEE' in tier2_metrics['experiment_a']:
            m = tier2_metrics['experiment_a']['TransTEE']
            validation = "✓ 通过" if m['delta_cate_egfr'] > 0 else "✗ 失败"
            print(f"| A (LUAD Only) | {m['mean_ite_egfr_mutant']:.4f} | "
                  f"{m['mean_ite_egfr_wild']:.4f} | {m['delta_cate_egfr']:+.4f} | {validation} |")
        
        # 实验 B
        if 'TransTEE' in tier2_metrics['experiment_b']:
            m = tier2_metrics['experiment_b']['TransTEE']
            validation = "✓ 通过" if m['delta_cate_egfr'] > 0 else "✗ 失败"
            print(f"| B (PANCAN+LUAD) | {m['mean_ite_egfr_mutant']:.4f} | "
                  f"{m['mean_ite_egfr_wild']:.4f} | {m['delta_cate_egfr']:+.4f} | {validation} |")
        
        # 层级 3：TP53 阴性对照
        print("\n### 层级 3：虚假关联排查（阴性对照 - TP53）")
        print("\n#### 分类模型子组 Recall")
        print("\n| Model | Experiment | Recall (TP53-Mutant) | Recall (TP53-Wild) | Δ Recall |")
        print("|-------|------------|----------------------|--------------------|----------|")
        
        for model_name in classification_models:
            # 实验 A
            if model_name in tier3_metrics['experiment_a']:
                m = tier3_metrics['experiment_a'][model_name]
                print(f"| {model_name} | A | {m['recall_tp53_mutant']:.4f} | "
                      f"{m['recall_tp53_wild']:.4f} | {m['delta_recall']:+.4f} |")
            
            # 实验 B
            if model_name in tier3_metrics['experiment_b']:
                m = tier3_metrics['experiment_b'][model_name]
                print(f"| {model_name} | B | {m['recall_tp53_mutant']:.4f} | "
                      f"{m['recall_tp53_wild']:.4f} | {m['delta_recall']:+.4f} |")
        
        print("\n#### TransTEE ΔCATE (TP53 - 阴性对照)")
        print("\n| Experiment | Mean ITE (TP53-Mutant) | Mean ITE (TP53-Wild) | ΔCATE | 验证结果 |")
        print("|------------|------------------------|----------------------|-------|----------|")
        
        threshold = 0.1
        
        # 实验 A
        if 'TransTEE' in tier3_metrics['experiment_a']:
            m = tier3_metrics['experiment_a']['TransTEE']
            validation = "✓ 通过" if abs(m['delta_cate_tp53']) < threshold else "✗ 失败"
            print(f"| A (LUAD Only) | {m['mean_ite_tp53_mutant']:.4f} | "
                  f"{m['mean_ite_tp53_wild']:.4f} | {m['delta_cate_tp53']:+.4f} | {validation} |")
        
        # 实验 B
        if 'TransTEE' in tier3_metrics['experiment_b']:
            m = tier3_metrics['experiment_b']['TransTEE']
            validation = "✓ 通过" if abs(m['delta_cate_tp53']) < threshold else "✗ 失败"
            print(f"| B (PANCAN+LUAD) | {m['mean_ite_tp53_mutant']:.4f} | "
                  f"{m['mean_ite_tp53_wild']:.4f} | {m['delta_cate_tp53']:+.4f} | {validation} |")
    
    def save_three_tier_metrics(self, tier1_metrics: Dict, 
                                tier2_metrics: Dict, 
                                tier3_metrics: Dict):
        """保存三层指标到 JSON 文件"""
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
        
        output_path = self.results_dir / 'three_tier_metrics.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] 三层指标已保存到：{output_path}")


def main():
    """主函数"""
    print("="*80)
    print("Task 19.4: 评估指标体系（三层指标）")
    print("="*80)
    
    # 初始化评估器
    evaluator = ThreeTierEvaluator()
    
    # 加载实验结果
    evaluator.load_experiment_results()
    
    # 计算三层指标
    tier1_metrics = evaluator.compute_tier1_general_performance()
    tier2_metrics = evaluator.compute_tier2_egfr_mechanism()
    tier3_metrics = evaluator.compute_tier3_tp53_negative_control()
    
    # 生成综合报告
    evaluator.generate_comprehensive_report(tier1_metrics, tier2_metrics, tier3_metrics)
    
    # 保存结果
    evaluator.save_three_tier_metrics(tier1_metrics, tier2_metrics, tier3_metrics)
    
    print("\n" + "="*80)
    print("三层指标评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

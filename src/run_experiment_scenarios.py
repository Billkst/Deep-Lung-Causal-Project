#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 19.3: 实验设计 - 两组严格的场景化测试（完整版 - 5 个模型）

实验 A：临床现状基线 (Small Sample Baseline)
- 训练集：仅使用 luad_synthetic_interaction.csv 的训练部分
- 测试集：luad_synthetic_interaction.csv 的测试部分
- 数据划分：8:1:1 (Train:Val:Test, Seed=42)
- 目的：模拟数据匮乏的现状，确立性能下限

实验 B：简单数据增强基线 (Merged Data Baseline)
- 训练集：pancan_synthetic_interaction.csv + luad_synthetic_interaction.csv (训练部分) 的物理拼接
- 测试集：仅使用 luad_synthetic_interaction.csv 的测试部分
- 严禁事项：绝对不能把 PANCAN 数据混入测试集
- 目的：验证简单混合数据是否因分布偏移导致收益递减

关键修复：
1. 包含全部 5 个模型：XGBoost, TabR, HyperFast, MOGONET, TransTEE
2. 强制修复 HyperFast 的 F1=0 问题（类别权重 + 阈值调整）
3. 保存预测结果，为 Task 19.4 做准备

作者：Kiro AI Agent
日期：2026-01-21（重跑版本）
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入基线模型
from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.baselines.mogonet_baseline import MOGONETBaseline
from src.baselines.transtee_baseline import TransTEEBaseline

warnings.filterwarnings('ignore')



class ExperimentDataLoader:
    """实验数据加载器"""
    
    def __init__(self, 
                 luad_path: str = "data/luad_synthetic_interaction.csv",
                 pancan_path: str = "data/pancan_synthetic_interaction.csv"):
        self.luad_path = Path(luad_path)
        self.pancan_path = Path(pancan_path)
        self._check_files_exist()
    
    def _check_files_exist(self):
        """检查数据文件是否存在"""
        if not self.luad_path.exists():
            raise FileNotFoundError(f"LUAD 数据文件不存在：{self.luad_path}")
        if not self.pancan_path.exists():
            raise FileNotFoundError(f"PANCAN 数据文件不存在：{self.pancan_path}")
    
    def load_luad_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """加载 LUAD 数据集"""
        print(f"[INFO] 正在加载 LUAD 数据：{self.luad_path}")
        df = pd.read_csv(self.luad_path)
        
        print(f"[INFO] LUAD 数据形状：{df.shape}")
        
        # 剔除 ID 列
        if 'sampleID' in df.columns:
            df = df.drop('sampleID', axis=1)
        
        # 分离特征和标签
        if 'Outcome_Label' in df.columns:
            y = df['Outcome_Label']
            X = df.drop('Outcome_Label', axis=1)
        else:
            raise ValueError("数据集中未找到 Outcome_Label 列")
        
        # 剔除 True_Prob 列（如果存在）
        if 'True_Prob' in X.columns:
            X = X.drop('True_Prob', axis=1)
        
        print(f"[INFO] LUAD 特征维度：{X.shape[1]}, 样本数：{X.shape[0]}")
        print(f"[INFO] LUAD 标签分布：{y.value_counts().to_dict()}")
        
        return X, y
    
    def load_pancan_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """加载 PANCAN 数据集"""
        print(f"[INFO] 正在加载 PANCAN 数据：{self.pancan_path}")
        df = pd.read_csv(self.pancan_path)
        
        print(f"[INFO] PANCAN 数据形状：{df.shape}")
        
        # 剔除 ID 列
        if 'sampleID' in df.columns:
            df = df.drop('sampleID', axis=1)
        
        # 分离特征和标签
        if 'Outcome_Label' in df.columns:
            y = df['Outcome_Label']
            X = df.drop('Outcome_Label', axis=1)
        else:
            raise ValueError("数据集中未找到 Outcome_Label 列")
        
        # 剔除 True_Prob 列（如果存在）
        if 'True_Prob' in X.columns:
            X = X.drop('True_Prob', axis=1)
        
        print(f"[INFO] PANCAN 特征维度：{X.shape[1]}, 样本数：{X.shape[0]}")
        print(f"[INFO] PANCAN 标签分布：{y.value_counts().to_dict()}")
        
        return X, y

    
    def split_luad_data(self, X: pd.DataFrame, y: pd.Series, 
                        random_state: int = 42) -> Dict[str, Any]:
        """
        LUAD 数据划分：Train:Val:Test = 8:1:1
        
        用于实验 A（小样本基线）
        """
        print(f"\n[INFO] LUAD 数据划分（Train:Val:Test = 8:1:1, Seed={random_state}）")
        
        # 第一次划分：80% 训练集，20% 临时集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # 第二次划分：将临时集平分为验证集和测试集（各 10%）
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
        
        print(f"[INFO] LUAD 训练集：{X_train.shape[0]} 样本")
        print(f"[INFO] LUAD 验证集：{X_val.shape[0]} 样本")
        print(f"[INFO] LUAD 测试集：{X_test.shape[0]} 样本")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def merge_and_split_data(self, X_luad: pd.DataFrame, y_luad: pd.Series,
                             X_pancan: pd.DataFrame, y_pancan: pd.Series,
                             random_state: int = 42) -> Dict[str, Any]:
        """
        合并 PANCAN 和 LUAD 数据，用于实验 B（数据增强基线）
        
        关键规则：
        1. 先将 LUAD 数据划分为 Train:Val:Test = 8:1:1
        2. 将 PANCAN 全部数据与 LUAD 训练集物理拼接
        3. 测试集仅使用 LUAD 测试集（严禁混入 PANCAN 数据）
        
        Returns:
            data_dict: 包含训练集、验证集、测试集的字典
        """
        print(f"\n[INFO] 实验 B：合并 PANCAN 和 LUAD 数据")
        
        # 步骤 1：先划分 LUAD 数据
        print("[INFO] 步骤 1：划分 LUAD 数据（8:1:1）")
        X_luad_train, X_luad_temp, y_luad_train, y_luad_temp = train_test_split(
            X_luad, y_luad, test_size=0.2, random_state=random_state, stratify=y_luad
        )
        
        X_luad_val, X_luad_test, y_luad_val, y_luad_test = train_test_split(
            X_luad_temp, y_luad_temp, test_size=0.5, random_state=random_state, stratify=y_luad_temp
        )
        
        print(f"  - LUAD 训练集：{X_luad_train.shape[0]} 样本")
        print(f"  - LUAD 验证集：{X_luad_val.shape[0]} 样本")
        print(f"  - LUAD 测试集：{X_luad_test.shape[0]} 样本")
        
        # 步骤 2：确保特征对齐
        print("\n[INFO] 步骤 2：特征对齐")
        common_features = X_luad.columns.intersection(X_pancan.columns).tolist()
        print(f"  - 共同特征数：{len(common_features)}")
        
        X_luad_train_aligned = X_luad_train[common_features]
        X_luad_val_aligned = X_luad_val[common_features]
        X_luad_test_aligned = X_luad_test[common_features]
        X_pancan_aligned = X_pancan[common_features]
        
        # 步骤 3：物理拼接 PANCAN 和 LUAD 训练集
        print("\n[INFO] 步骤 3：物理拼接 PANCAN 和 LUAD 训练集")
        X_train_merged = pd.concat([X_pancan_aligned, X_luad_train_aligned], axis=0, ignore_index=True)
        y_train_merged = pd.concat([y_pancan, y_luad_train], axis=0, ignore_index=True)
        
        print(f"  - PANCAN 样本数：{X_pancan_aligned.shape[0]}")
        print(f"  - LUAD 训练样本数：{X_luad_train_aligned.shape[0]}")
        print(f"  - 合并后训练集：{X_train_merged.shape[0]} 样本")
        
        # 步骤 4：验证测试集纯净性
        print("\n[INFO] 步骤 4：验证测试集纯净性")
        print(f"  - 测试集仅包含 LUAD 数据：{X_luad_test_aligned.shape[0]} 样本")
        print(f"  - ✓ 确认：PANCAN 数据未混入测试集")
        
        return {
            'X_train': X_train_merged,
            'X_val': X_luad_val_aligned,
            'X_test': X_luad_test_aligned,
            'y_train': y_train_merged,
            'y_val': y_luad_val,
            'y_test': y_luad_test
        }



class ExperimentEvaluator:
    """实验评估器 - 支持全部 5 个模型"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        # 存储预测结果，为 Task 19.4 做准备
        self.predictions = {}
    
    def train_and_evaluate_model(self, model_name: str, model_class, 
                                 X_train, y_train, X_test, y_test,
                                 **model_kwargs) -> Dict:
        """训练并评估单个模型（标准分类模型）"""
        print(f"\n[INFO] 训练 {model_name}...")
        
        start_time = time.time()
        
        try:
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            metrics = model.evaluate(X_test, y_test)
            inference_time = time.time() - inference_start
            
            # 保存预测概率（为 Task 19.4 做准备）
            y_proba = model.predict_proba(X_test)
            self.predictions[model_name] = {
                'y_proba': y_proba,
                'y_pred': model.predict(X_test),
                'y_true': y_test
            }
            
            metrics['train_time'] = train_time
            metrics['inference_time'] = inference_time
            metrics['status'] = 'success'
            
            print(f"[SUCCESS] {model_name} 训练完成")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] {model_name} 训练失败：{str(e)}")
            import traceback
            traceback.print_exc()
            metrics = {'status': 'failed', 'error': str(e)}
        
        return metrics

    
    def run_experiment_a(self, data_dict: Dict) -> Dict:
        """
        运行实验 A：临床现状基线 (Small Sample Baseline)
        
        包含全部 5 个模型：XGBoost, TabR, HyperFast, MOGONET, TransTEE
        """
        print("\n" + "="*80)
        print("实验 A：临床现状基线 (Small Sample Baseline)")
        print("="*80)
        print("[INFO] 训练集：LUAD 训练部分")
        print("[INFO] 测试集：LUAD 测试部分")
        print("[INFO] 目的：确立性能下限")
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data_dict['X_train'])
        X_test_scaled = scaler.transform(data_dict['X_test'])
        
        y_train = data_dict['y_train'].values
        y_test = data_dict['y_test'].values
        
        results = {}
        
        # 1. XGBoost
        results['XGBoost'] = self.train_and_evaluate_model(
            'XGBoost', XGBBaseline, X_train_scaled, y_train, X_test_scaled, y_test,
            random_state=42
        )
        
        # 2. TabR
        results['TabR'] = self.train_and_evaluate_model(
            'TabR', TabRBaseline, X_train_scaled, y_train, X_test_scaled, y_test,
            random_state=42, epochs=30, batch_size=32
        )
        
        # 3. HyperFast（强制修复 F1=0 问题）
        print(f"\n[INFO] 训练 HyperFast（强制修复版本）...")
        print(f"[INFO] 类别分布：{np.bincount(y_train)}")
        
        # 计算类别权重
        n_samples = len(y_train)
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        weight_0 = n_samples / (2 * n_class_0)
        weight_1 = n_samples / (2 * n_class_1)
        class_weights = [weight_0, weight_1]
        
        print(f"[INFO] 类别权重：[{weight_0:.4f}, {weight_1:.4f}]")
        
        results['HyperFast'] = self.train_and_evaluate_model(
            'HyperFast', HyperFastBaseline, X_train_scaled, y_train, X_test_scaled, y_test,
            random_state=42, epochs=30, batch_size=32, 
            class_weights=class_weights, prediction_threshold=0.4
        )
        
        # 4. MOGONET（多视图）
        print(f"\n[INFO] 训练 MOGONET（多视图图网络）...")
        try:
            # 准备多视图数据
            clinical_features = ['Age', 'Gender', 'Virtual_PM2.5']
            omics_features = [col for col in data_dict['X_train'].columns if col not in clinical_features]
            
            # 训练集视图
            clinical_train = data_dict['X_train'][clinical_features].values
            omics_train = data_dict['X_train'][omics_features].values
            views_train = [clinical_train, omics_train]
            
            # 测试集视图
            clinical_test = data_dict['X_test'][clinical_features].values
            omics_test = data_dict['X_test'][omics_features].values
            views_test = [clinical_test, omics_test]
            
            start_time = time.time()
            mogonet_model = MOGONETBaseline(random_state=42)
            mogonet_model.fit(views_train, y_train)
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            mogonet_metrics = mogonet_model.evaluate(views_test, y_test)
            inference_time = time.time() - inference_start
            
            # 保存预测结果
            y_proba_mogonet = mogonet_model.predict_proba(views_test)
            self.predictions['MOGONET'] = {
                'y_proba': y_proba_mogonet,
                'y_pred': mogonet_model.predict(views_test),
                'y_true': y_test
            }
            
            mogonet_metrics['train_time'] = train_time
            mogonet_metrics['inference_time'] = inference_time
            mogonet_metrics['status'] = 'success'
            
            print(f"[SUCCESS] MOGONET 训练完成")
            print(f"  - Accuracy: {mogonet_metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {mogonet_metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {mogonet_metrics['auc_roc']:.4f}")
            
            results['MOGONET'] = mogonet_metrics
            
        except Exception as e:
            print(f"[ERROR] MOGONET 训练失败：{str(e)}")
            import traceback
            traceback.print_exc()
            results['MOGONET'] = {'status': 'failed', 'error': str(e)}
        
        # 5. TransTEE（因果推断）
        print(f"\n[INFO] 训练 TransTEE（因果推断模型）...")
        try:
            # 准备 TransTEE 数据：将 Virtual_PM2.5 二值化为 Treatment
            X_train_df = data_dict['X_train']
            X_test_df = data_dict['X_test']
            
            # 提取 Virtual_PM2.5 并二值化
            pm25_train = X_train_df['Virtual_PM2.5'].values
            pm25_test = X_test_df['Virtual_PM2.5'].values
            
            median_pm25 = np.median(pm25_train)
            t_train = (pm25_train >= median_pm25).astype(int)
            t_test = (pm25_test >= median_pm25).astype(int)
            
            print(f"[INFO] Virtual_PM2.5 中位数：{median_pm25:.2f}")
            print(f"[INFO] Treatment 分布（训练集）：{np.bincount(t_train)}")
            
            # 移除 Virtual_PM2.5，保留其他特征作为协变量
            X_cov_train = X_train_df.drop('Virtual_PM2.5', axis=1).values
            X_cov_test = X_test_df.drop('Virtual_PM2.5', axis=1).values
            
            # 标准化协变量
            scaler_transtee = StandardScaler()
            X_cov_train_scaled = scaler_transtee.fit_transform(X_cov_train)
            X_cov_test_scaled = scaler_transtee.transform(X_cov_test)
            
            start_time = time.time()
            transtee_model = TransTEEBaseline(random_state=42, epochs=50, batch_size=64)
            transtee_model.fit(X_cov_train_scaled, t_train, y_train)
            train_time = time.time() - start_time
            
            # 评估（因果推断指标）
            inference_start = time.time()
            ite_pred = transtee_model.predict_ite(X_cov_test_scaled)
            inference_time = time.time() - inference_start
            
            # 保存 ITE 预测结果（为 Task 19.4 做准备）
            self.predictions['TransTEE'] = {
                'ite': ite_pred,
                't': t_test,
                'y_true': y_test,
                'X_test': X_test_df  # 保留完整特征，用于 EGFR/TP53 分组
            }
            
            # 计算 ATE
            ate_pred = np.mean(ite_pred)
            
            transtee_metrics = {
                'ate': ate_pred,
                'ite_mean': np.mean(ite_pred),
                'ite_std': np.std(ite_pred),
                'train_time': train_time,
                'inference_time': inference_time,
                'status': 'success'
            }
            
            print(f"[SUCCESS] TransTEE 训练完成")
            print(f"  - ATE: {ate_pred:.4f}")
            print(f"  - ITE Mean: {transtee_metrics['ite_mean']:.4f}")
            print(f"  - ITE Std: {transtee_metrics['ite_std']:.4f}")
            
            results['TransTEE'] = transtee_metrics
            
        except Exception as e:
            print(f"[ERROR] TransTEE 训练失败：{str(e)}")
            import traceback
            traceback.print_exc()
            results['TransTEE'] = {'status': 'failed', 'error': str(e)}
        
        return results

    
    def run_experiment_b(self, data_dict: Dict) -> Dict:
        """
        运行实验 B：简单数据增强基线 (Merged Data Baseline)
        
        包含全部 5 个模型：XGBoost, TabR, HyperFast, MOGONET, TransTEE
        """
        print("\n" + "="*80)
        print("实验 B：简单数据增强基线 (Merged Data Baseline)")
        print("="*80)
        print("[INFO] 训练集：PANCAN + LUAD 训练部分（物理拼接）")
        print("[INFO] 测试集：LUAD 测试部分")
        print("[INFO] 目的：验证简单混合数据是否因分布偏移导致收益递减")
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data_dict['X_train'])
        X_test_scaled = scaler.transform(data_dict['X_test'])
        
        y_train = data_dict['y_train'].values
        y_test = data_dict['y_test'].values
        
        results = {}
        
        # 1. XGBoost
        results['XGBoost'] = self.train_and_evaluate_model(
            'XGBoost', XGBBaseline, X_train_scaled, y_train, X_test_scaled, y_test,
            random_state=42
        )
        
        # 2. TabR
        results['TabR'] = self.train_and_evaluate_model(
            'TabR', TabRBaseline, X_train_scaled, y_train, X_test_scaled, y_test,
            random_state=42, epochs=30, batch_size=32
        )
        
        # 3. HyperFast（强制修复 F1=0 问题）
        print(f"\n[INFO] 训练 HyperFast（强制修复版本）...")
        print(f"[INFO] 类别分布：{np.bincount(y_train)}")
        
        # 计算类别权重
        n_samples = len(y_train)
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        weight_0 = n_samples / (2 * n_class_0)
        weight_1 = n_samples / (2 * n_class_1)
        class_weights = [weight_0, weight_1]
        
        print(f"[INFO] 类别权重：[{weight_0:.4f}, {weight_1:.4f}]")
        
        results['HyperFast'] = self.train_and_evaluate_model(
            'HyperFast', HyperFastBaseline, X_train_scaled, y_train, X_test_scaled, y_test,
            random_state=42, epochs=30, batch_size=32,
            class_weights=class_weights, prediction_threshold=0.4
        )
        
        # 4. MOGONET（多视图）
        print(f"\n[INFO] 训练 MOGONET（多视图图网络）...")
        try:
            # 准备多视图数据
            clinical_features = ['Age', 'Gender', 'Virtual_PM2.5']
            omics_features = [col for col in data_dict['X_train'].columns if col not in clinical_features]
            
            # 训练集视图
            clinical_train = data_dict['X_train'][clinical_features].values
            omics_train = data_dict['X_train'][omics_features].values
            views_train = [clinical_train, omics_train]
            
            # 测试集视图
            clinical_test = data_dict['X_test'][clinical_features].values
            omics_test = data_dict['X_test'][omics_features].values
            views_test = [clinical_test, omics_test]
            
            start_time = time.time()
            mogonet_model = MOGONETBaseline(random_state=42)
            mogonet_model.fit(views_train, y_train)
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            mogonet_metrics = mogonet_model.evaluate(views_test, y_test)
            inference_time = time.time() - inference_start
            
            # 保存预测结果
            y_proba_mogonet = mogonet_model.predict_proba(views_test)
            self.predictions['MOGONET'] = {
                'y_proba': y_proba_mogonet,
                'y_pred': mogonet_model.predict(views_test),
                'y_true': y_test
            }
            
            mogonet_metrics['train_time'] = train_time
            mogonet_metrics['inference_time'] = inference_time
            mogonet_metrics['status'] = 'success'
            
            print(f"[SUCCESS] MOGONET 训练完成")
            print(f"  - Accuracy: {mogonet_metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {mogonet_metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {mogonet_metrics['auc_roc']:.4f}")
            
            results['MOGONET'] = mogonet_metrics
            
        except Exception as e:
            print(f"[ERROR] MOGONET 训练失败：{str(e)}")
            import traceback
            traceback.print_exc()
            results['MOGONET'] = {'status': 'failed', 'error': str(e)}
        
        # 5. TransTEE（因果推断）
        print(f"\n[INFO] 训练 TransTEE（因果推断模型）...")
        try:
            # 准备 TransTEE 数据：将 Virtual_PM2.5 二值化为 Treatment
            X_train_df = data_dict['X_train']
            X_test_df = data_dict['X_test']
            
            # 提取 Virtual_PM2.5 并二值化
            pm25_train = X_train_df['Virtual_PM2.5'].values
            pm25_test = X_test_df['Virtual_PM2.5'].values
            
            median_pm25 = np.median(pm25_train)
            t_train = (pm25_train >= median_pm25).astype(int)
            t_test = (pm25_test >= median_pm25).astype(int)
            
            print(f"[INFO] Virtual_PM2.5 中位数：{median_pm25:.2f}")
            print(f"[INFO] Treatment 分布（训练集）：{np.bincount(t_train)}")
            
            # 移除 Virtual_PM2.5，保留其他特征作为协变量
            X_cov_train = X_train_df.drop('Virtual_PM2.5', axis=1).values
            X_cov_test = X_test_df.drop('Virtual_PM2.5', axis=1).values
            
            # 标准化协变量
            scaler_transtee = StandardScaler()
            X_cov_train_scaled = scaler_transtee.fit_transform(X_cov_train)
            X_cov_test_scaled = scaler_transtee.transform(X_cov_test)
            
            start_time = time.time()
            transtee_model = TransTEEBaseline(random_state=42, epochs=50, batch_size=64)
            transtee_model.fit(X_cov_train_scaled, t_train, y_train)
            train_time = time.time() - start_time
            
            # 评估（因果推断指标）
            inference_start = time.time()
            ite_pred = transtee_model.predict_ite(X_cov_test_scaled)
            inference_time = time.time() - inference_start
            
            # 保存 ITE 预测结果（为 Task 19.4 做准备）
            self.predictions['TransTEE'] = {
                'ite': ite_pred,
                't': t_test,
                'y_true': y_test,
                'X_test': X_test_df  # 保留完整特征，用于 EGFR/TP53 分组
            }
            
            # 计算 ATE
            ate_pred = np.mean(ite_pred)
            
            transtee_metrics = {
                'ate': ate_pred,
                'ite_mean': np.mean(ite_pred),
                'ite_std': np.std(ite_pred),
                'train_time': train_time,
                'inference_time': inference_time,
                'status': 'success'
            }
            
            print(f"[SUCCESS] TransTEE 训练完成")
            print(f"  - ATE: {ate_pred:.4f}")
            print(f"  - ITE Mean: {transtee_metrics['ite_mean']:.4f}")
            print(f"  - ITE Std: {transtee_metrics['ite_std']:.4f}")
            
            results['TransTEE'] = transtee_metrics
            
        except Exception as e:
            print(f"[ERROR] TransTEE 训练失败：{str(e)}")
            import traceback
            traceback.print_exc()
            results['TransTEE'] = {'status': 'failed', 'error': str(e)}
        
        return results

    
    def save_results(self, results: Dict, filename: str):
        """保存结果到 JSON 文件"""
        output_path = self.results_dir / filename
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] 结果已保存到：{output_path}")
    
    def save_predictions(self, experiment_name: str):
        """保存预测结果（为 Task 19.4 做准备）"""
        output_path = self.results_dir / f"{experiment_name}_predictions.pkl"
        
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.predictions, f)
        
        print(f"[INFO] 预测结果已保存到：{output_path}")
    
    def print_comparison_table(self, results_a: Dict, results_b: Dict):
        """打印实验 A 和 B 的对比表格（包含全部 5 个模型）"""
        print("\n" + "="*80)
        print("实验结果对比：实验 A vs 实验 B（全部 5 个模型）")
        print("="*80)
        
        print("\n### 分类模型性能对比")
        print("| Model | Experiment | Accuracy | Precision | Recall | F1 | AUC-ROC |")
        print("|-------|------------|----------|-----------|--------|----|---------| ")
        
        for model_name in ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']:
            # 实验 A
            if model_name in results_a and results_a[model_name].get('status') == 'success':
                m = results_a[model_name]
                print(f"| {model_name} | A (LUAD Only) | {m.get('accuracy', 0):.4f} | "
                      f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                      f"{m.get('f1', 0):.4f} | {m.get('auc_roc', 0):.4f} |")
            
            # 实验 B
            if model_name in results_b and results_b[model_name].get('status') == 'success':
                m = results_b[model_name]
                print(f"| {model_name} | B (PANCAN+LUAD) | {m.get('accuracy', 0):.4f} | "
                      f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                      f"{m.get('f1', 0):.4f} | {m.get('auc_roc', 0):.4f} |")
        
        # TransTEE 因果推断指标
        print("\n### TransTEE 因果推断指标")
        print("| Experiment | ATE | ITE Mean | ITE Std | Train Time (s) |")
        print("|------------|-----|----------|---------|----------------|")
        
        if 'TransTEE' in results_a and results_a['TransTEE'].get('status') == 'success':
            m = results_a['TransTEE']
            print(f"| A (LUAD Only) | {m.get('ate', 0):.4f} | {m.get('ite_mean', 0):.4f} | "
                  f"{m.get('ite_std', 0):.4f} | {m.get('train_time', 0):.2f} |")
        
        if 'TransTEE' in results_b and results_b['TransTEE'].get('status') == 'success':
            m = results_b['TransTEE']
            print(f"| B (PANCAN+LUAD) | {m.get('ate', 0):.4f} | {m.get('ite_mean', 0):.4f} | "
                  f"{m.get('ite_std', 0):.4f} | {m.get('train_time', 0):.2f} |")
        
        # 计算性能提升
        print("\n### 性能提升分析（实验 B 相对于实验 A）")
        print("| Model | Δ Accuracy | Δ F1 | Δ AUC-ROC |")
        print("|-------|------------|------|-----------|")
        
        for model_name in ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']:
            if (model_name in results_a and results_a[model_name].get('status') == 'success' and
                model_name in results_b and results_b[model_name].get('status') == 'success'):
                
                delta_acc = results_b[model_name]['accuracy'] - results_a[model_name]['accuracy']
                delta_f1 = results_b[model_name]['f1'] - results_a[model_name]['f1']
                delta_auc = results_b[model_name]['auc_roc'] - results_a[model_name]['auc_roc']
                
                print(f"| {model_name} | {delta_acc:+.4f} | {delta_f1:+.4f} | {delta_auc:+.4f} |")


def main():
    """主函数"""
    print("="*80)
    print("Task 19.3: 实验设计 - 两组严格的场景化测试（完整版 - 5 个模型）")
    print("="*80)
    
    # 1. 加载数据
    loader = ExperimentDataLoader()
    X_luad, y_luad = loader.load_luad_data()
    X_pancan, y_pancan = loader.load_pancan_data()
    
    # 2. 初始化评估器
    evaluator = ExperimentEvaluator()
    
    # 3. 运行实验 A：临床现状基线（仅 LUAD）
    print("\n" + "="*80)
    print("准备实验 A 数据")
    print("="*80)
    data_dict_a = loader.split_luad_data(X_luad, y_luad, random_state=42)
    results_a = evaluator.run_experiment_a(data_dict_a)
    
    # 保存实验 A 结果
    experiment_a_output = {
        'experiment': 'A',
        'description': 'Small Sample Baseline (LUAD Only)',
        'dataset': {
            'train': 'luad_synthetic_interaction (train split)',
            'test': 'luad_synthetic_interaction (test split)',
            'split_ratio': '8:1:1',
            'seed': 42
        },
        'models': results_a
    }
    evaluator.save_results(experiment_a_output, 'experiment_a_results.json')
    evaluator.save_predictions('experiment_a')
    
    # 4. 运行实验 B：数据增强基线（PANCAN + LUAD）
    print("\n" + "="*80)
    print("准备实验 B 数据")
    print("="*80)
    data_dict_b = loader.merge_and_split_data(X_luad, y_luad, X_pancan, y_pancan, random_state=42)
    results_b = evaluator.run_experiment_b(data_dict_b)
    
    # 保存实验 B 结果
    experiment_b_output = {
        'experiment': 'B',
        'description': 'Merged Data Baseline (PANCAN + LUAD)',
        'dataset': {
            'train': 'pancan_synthetic_interaction (all) + luad_synthetic_interaction (train split)',
            'test': 'luad_synthetic_interaction (test split)',
            'split_ratio': '8:1:1 (for LUAD)',
            'seed': 42
        },
        'models': results_b
    }
    evaluator.save_results(experiment_b_output, 'experiment_b_results.json')
    evaluator.save_predictions('experiment_b')
    
    # 5. 打印对比表格
    evaluator.print_comparison_table(results_a, results_b)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n[INFO] 结果文件：")
    print("  - results/experiment_a_results.json")
    print("  - results/experiment_b_results.json")
    print("  - results/experiment_a_predictions.pkl（为 Task 19.4 准备）")
    print("  - results/experiment_b_predictions.pkl（为 Task 19.4 准备）")


if __name__ == "__main__":
    main()

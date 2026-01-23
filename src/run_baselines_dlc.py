#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLC 数据集集成与评估主脚本

功能：
1. 加载 DLC 数据集 (pancan_synthetic_interaction.csv)
2. 数据预处理（剔除 ID 列，特征对齐，8:1:1 划分）
3. 批量训练 5 个 SOTA 模型
4. 评估并保存结果

作者：Kiro AI Agent
日期：2026-01-19
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


class DLCDataLoader:
    """DLC 数据集加载器"""
    
    def __init__(self, data_path: str = "data/pancan_synthetic_interaction.csv"):
        self.data_path = Path(data_path)
        self._check_file_exists()
    
    def _check_file_exists(self):
        """检查数据文件是否存在"""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"DLC 数据文件不存在：{self.data_path}\n"
                "请确保数据文件存在于正确的路径"
            )
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载 DLC 数据集
        
        Returns:
            X: 特征矩阵 (DataFrame)
            y: 标签向量 (Series)
        """
        print(f"[INFO] 正在加载数据：{self.data_path}")
        df = pd.read_csv(self.data_path)
        
        print(f"[INFO] 数据形状：{df.shape}")
        print(f"[INFO] 列名：{df.columns.tolist()}")
        
        # 剔除 ID 列
        if 'sampleID' in df.columns:
            df = df.drop('sampleID', axis=1)
            print("[INFO] 已剔除 sampleID 列")
        
        # 分离特征和标签
        if 'Outcome_Label' in df.columns:
            y = df['Outcome_Label']
            X = df.drop('Outcome_Label', axis=1)
        else:
            raise ValueError("数据集中未找到 Outcome_Label 列")
        
        # 剔除 True_Prob 列（如果存在）
        if 'True_Prob' in X.columns:
            X = X.drop('True_Prob', axis=1)
            print("[INFO] 已剔除 True_Prob 列")
        
        print(f"[INFO] 特征维度：{X.shape[1]}, 样本数：{X.shape[0]}")
        print(f"[INFO] 标签分布：{y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   random_state: int = 42) -> Dict[str, Any]:
        """
        数据划分：Train:Val:Test = 8:1:1
        
        Args:
            X: 特征矩阵
            y: 标签向量
            random_state: 随机种子
            
        Returns:
            data_dict: 包含训练集、验证集、测试集的字典
        """
        print(f"\n[INFO] 数据划分（Train:Val:Test = 8:1:1, Seed={random_state}）")
        
        # 第一次划分：80% 训练集，20% 临时集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # 第二次划分：将临时集平分为验证集和测试集（各 10%）
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
        
        print(f"[INFO] 训练集：{X_train.shape[0]} 样本")
        print(f"[INFO] 验证集：{X_val.shape[0]} 样本")
        print(f"[INFO] 测试集：{X_test.shape[0]} 样本")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }


class DLCDataPreprocessor:
    """DLC 数据预处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clinical_features = ['Age', 'Gender']
        self.omics_features = None
    
    def fit_transform_standard(self, X_train: pd.DataFrame, 
                               X_val: pd.DataFrame, 
                               X_test: pd.DataFrame) -> Tuple:
        """
        标准化特征（用于 XGBoost, TabR, HyperFast）
        
        Returns:
            X_train_scaled, X_val_scaled, X_test_scaled
        """
        print("\n[INFO] 特征标准化（StandardScaler）")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def prepare_transtee_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        为 TransTEE 准备数据：将 Virtual_PM2.5 二值化为 Treatment
        
        Returns:
            X_covariates: 协变量矩阵（不含 Virtual_PM2.5）
            t: 治疗变量（二值化的 Virtual_PM2.5）
            y: 结局变量
        """
        print("\n[INFO] 为 TransTEE 准备数据")
        
        if 'Virtual_PM2.5' not in X.columns:
            raise ValueError("数据集中未找到 Virtual_PM2.5 列")
        
        # 提取 Virtual_PM2.5 并二值化（使用中位数分割）
        pm25_values = X['Virtual_PM2.5'].values
        median_pm25 = np.median(pm25_values)
        t = (pm25_values >= median_pm25).astype(int)
        
        print(f"[INFO] Virtual_PM2.5 中位数：{median_pm25:.2f}")
        print(f"[INFO] Treatment 分布：{np.bincount(t)}")
        
        # 移除 Virtual_PM2.5，保留其他特征作为协变量
        X_covariates = X.drop('Virtual_PM2.5', axis=1)
        
        return X_covariates, t, y.values
    
    def prepare_mogonet_data(self, X: pd.DataFrame) -> List[np.ndarray]:
        """
        为 MOGONET 准备多视图数据
        
        Returns:
            views: [clinical_view, omics_view]
        """
        print("\n[INFO] 为 MOGONET 准备多视图数据")
        
        # 识别临床特征和基因特征
        clinical_cols = [col for col in X.columns if col in self.clinical_features]
        omics_cols = [col for col in X.columns if col not in self.clinical_features]
        
        # 如果 Virtual_PM2.5 存在，将其归类为临床特征
        if 'Virtual_PM2.5' in omics_cols:
            clinical_cols.append('Virtual_PM2.5')
            omics_cols.remove('Virtual_PM2.5')
        
        self.omics_features = omics_cols
        
        print(f"[INFO] Clinical View: {len(clinical_cols)} 特征")
        print(f"[INFO] Omics View: {len(omics_cols)} 特征")
        
        clinical_view = X[clinical_cols].values
        omics_view = X[omics_cols].values
        
        return [clinical_view, omics_view]


class BaselineEvaluator:
    """基线模型评估器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def train_and_evaluate_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """训练并评估 XGBoost"""
        print("\n" + "="*60)
        print("Phase 1: XGBoost (经典基线)")
        print("="*60)
        
        start_time = time.time()
        
        try:
            model = XGBBaseline(random_state=42)
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            metrics = model.evaluate(X_test, y_test)
            inference_time = time.time() - inference_start
            
            metrics['train_time'] = train_time
            metrics['inference_time'] = inference_time
            metrics['status'] = 'success'
            
            print(f"[SUCCESS] XGBoost 训练完成")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  - 训练时间: {train_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] XGBoost 训练失败：{str(e)}")
            metrics = {'status': 'failed', 'error': str(e)}
        
        return metrics
    
    def train_and_evaluate_tabr(self, X_train, y_train, X_test, y_test) -> Dict:
        """训练并评估 TabR"""
        print("\n" + "="*60)
        print("Phase 2: TabR (检索增强表格学习, 2024 ICLR)")
        print("="*60)
        
        start_time = time.time()
        
        try:
            model = TabRBaseline(random_state=42, epochs=30, batch_size=32)
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            metrics = model.evaluate(X_test, y_test)
            inference_time = time.time() - inference_start
            
            metrics['train_time'] = train_time
            metrics['inference_time'] = inference_time
            metrics['status'] = 'success'
            
            print(f"[SUCCESS] TabR 训练完成")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  - 训练时间: {train_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] TabR 训练失败：{str(e)}")
            metrics = {'status': 'failed', 'error': str(e)}
        
        return metrics
    
    def train_and_evaluate_hyperfast(self, X_train, y_train, X_test, y_test) -> Dict:
        """训练并评估 HyperFast"""
        print("\n" + "="*60)
        print("Phase 2: HyperFast (Hypernetwork 快速推理, 2024 NeurIPS)")
        print("="*60)
        
        start_time = time.time()
        
        try:
            model = HyperFastBaseline(random_state=42, epochs=30, batch_size=32)
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            metrics = model.evaluate(X_test, y_test)
            inference_time = time.time() - inference_start
            
            metrics['train_time'] = train_time
            metrics['inference_time'] = inference_time
            metrics['status'] = 'success'
            
            print(f"[SUCCESS] HyperFast 训练完成")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  - 训练时间: {train_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] HyperFast 训练失败：{str(e)}")
            metrics = {'status': 'failed', 'error': str(e)}
        
        return metrics
    
    def train_and_evaluate_mogonet(self, views_train, y_train, 
                                   views_test, y_test) -> Dict:
        """训练并评估 MOGONET"""
        print("\n" + "="*60)
        print("Phase 2: MOGONET (多组学图网络)")
        print("="*60)
        
        start_time = time.time()
        
        try:
            model = MOGONETBaseline(random_state=42)
            model.fit(views_train, y_train)
            
            train_time = time.time() - start_time
            
            # 评估
            inference_start = time.time()
            metrics = model.evaluate(views_test, y_test)
            inference_time = time.time() - inference_start
            
            metrics['train_time'] = train_time
            metrics['inference_time'] = inference_time
            metrics['status'] = 'success'
            
            print(f"[SUCCESS] MOGONET 训练完成")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  - 训练时间: {train_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] MOGONET 训练失败：{str(e)}")
            metrics = {'status': 'failed', 'error': str(e)}
        
        return metrics
    
    def train_and_evaluate_transtee(self, X_train, t_train, y_train,
                                    X_test, t_test, y_test) -> Dict:
        """训练并评估 TransTEE"""
        print("\n" + "="*60)
        print("Phase 3: TransTEE (Transformer 因果推断, 2022 ICLR)")
        print("="*60)
        
        start_time = time.time()
        
        try:
            model = TransTEEBaseline(random_state=42, epochs=50, batch_size=64)
            model.fit(X_train, t_train, y_train)
            
            train_time = time.time() - start_time
            
            # 评估（因果推断指标）
            inference_start = time.time()
            ite_pred = model.predict_ite(X_test)
            inference_time = time.time() - inference_start
            
            # 计算 ATE (Average Treatment Effect)
            ate_pred = np.mean(ite_pred)
            
            metrics = {
                'ate': ate_pred,
                'ite_mean': np.mean(ite_pred),
                'ite_std': np.std(ite_pred),
                'train_time': train_time,
                'inference_time': inference_time,
                'status': 'success'
            }
            
            print(f"[SUCCESS] TransTEE 训练完成")
            print(f"  - ATE: {ate_pred:.4f}")
            print(f"  - ITE Mean: {metrics['ite_mean']:.4f}")
            print(f"  - ITE Std: {metrics['ite_std']:.4f}")
            print(f"  - 训练时间: {train_time:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] TransTEE 训练失败：{str(e)}")
            metrics = {'status': 'failed', 'error': str(e)}
        
        return metrics
    
    def save_results(self, results: Dict, filename: str = "baseline_metrics.json"):
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
    
    def print_summary_table(self, results: Dict):
        """打印 Markdown 格式的结果对比表格"""
        print("\n" + "="*80)
        print("实验结果汇总")
        print("="*80)
        
        # 分类模型表格
        print("\n### 分类模型性能对比")
        print("| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Train Time (s) |")
        print("|-------|----------|-----------|--------|----|---------| --------------|")
        
        for model_name in ['XGBoost', 'TabR', 'HyperFast', 'MOGONET']:
            if model_name in results['models']:
                m = results['models'][model_name]
                if m.get('status') == 'success':
                    print(f"| {model_name} | {m.get('accuracy', 0):.4f} | "
                          f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                          f"{m.get('f1', 0):.4f} | {m.get('auc_roc', 0):.4f} | "
                          f"{m.get('train_time', 0):.2f} |")
                else:
                    print(f"| {model_name} | FAILED | - | - | - | - | - |")
        
        # TransTEE 因果推断指标
        if 'TransTEE' in results['models']:
            m = results['models']['TransTEE']
            if m.get('status') == 'success':
                print("\n### TransTEE 因果推断指标")
                print(f"- ATE (Average Treatment Effect): {m.get('ate', 0):.4f}")
                print(f"- ITE Mean: {m.get('ite_mean', 0):.4f}")
                print(f"- ITE Std: {m.get('ite_std', 0):.4f}")
                print(f"- Train Time: {m.get('train_time', 0):.2f}s")


def main():
    """主函数"""
    print("="*80)
    print("DLC 数据集集成与评估")
    print("="*80)
    
    # 1. 加载数据
    loader = DLCDataLoader()
    X, y = loader.load_data()
    
    # 2. 数据划分
    data_dict = loader.split_data(X, y, random_state=42)
    
    # 3. 数据预处理
    preprocessor = DLCDataPreprocessor()
    
    # 标准化特征（用于 XGBoost, TabR, HyperFast）
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.fit_transform_standard(
        data_dict['X_train'], data_dict['X_val'], data_dict['X_test']
    )
    
    # 4. 初始化评估器
    evaluator = BaselineEvaluator()
    
    # 5. 训练和评估所有模型
    results = {
        'dataset': 'pancan_synthetic_interaction',
        'split': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1,
            'seed': 42
        },
        'models': {}
    }
    
    # 5.1 XGBoost
    xgb_metrics = evaluator.train_and_evaluate_xgboost(
        X_train_scaled, data_dict['y_train'].values,
        X_test_scaled, data_dict['y_test'].values
    )
    results['models']['XGBoost'] = xgb_metrics
    
    # 5.2 TabR
    tabr_metrics = evaluator.train_and_evaluate_tabr(
        X_train_scaled, data_dict['y_train'].values,
        X_test_scaled, data_dict['y_test'].values
    )
    results['models']['TabR'] = tabr_metrics
    
    # 5.3 HyperFast
    hyperfast_metrics = evaluator.train_and_evaluate_hyperfast(
        X_train_scaled, data_dict['y_train'].values,
        X_test_scaled, data_dict['y_test'].values
    )
    results['models']['HyperFast'] = hyperfast_metrics
    
    # 5.4 MOGONET（多视图数据）
    views_train = preprocessor.prepare_mogonet_data(data_dict['X_train'])
    views_test = preprocessor.prepare_mogonet_data(data_dict['X_test'])
    
    mogonet_metrics = evaluator.train_and_evaluate_mogonet(
        views_train, data_dict['y_train'].values,
        views_test, data_dict['y_test'].values
    )
    results['models']['MOGONET'] = mogonet_metrics
    
    # 5.5 TransTEE（因果推断）
    X_cov_train, t_train, y_train_transtee = preprocessor.prepare_transtee_data(
        data_dict['X_train'], data_dict['y_train']
    )
    X_cov_test, t_test, y_test_transtee = preprocessor.prepare_transtee_data(
        data_dict['X_test'], data_dict['y_test']
    )
    
    # 标准化协变量
    scaler_transtee = StandardScaler()
    X_cov_train_scaled = scaler_transtee.fit_transform(X_cov_train)
    X_cov_test_scaled = scaler_transtee.transform(X_cov_test)
    
    transtee_metrics = evaluator.train_and_evaluate_transtee(
        X_cov_train_scaled, t_train, y_train_transtee,
        X_cov_test_scaled, t_test, y_test_transtee
    )
    results['models']['TransTEE'] = transtee_metrics
    
    # 6. 保存结果
    evaluator.save_results(results)
    
    # 7. 打印汇总表格
    evaluator.print_summary_table(results)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()

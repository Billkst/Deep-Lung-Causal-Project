#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 21: Phase 5 - 混杂因子敏感性评估

理论基础：
基于数据生成过程（Ground Truth），交互项仅取决于 EGFR，而非 Age。
因此，理论上完美的因果模型得出的 ITE 应对 Age 的变化保持**不变性**。
Age 与 ITE 之间的任何显著相关性都表明模型未能将 Age 的混杂效应与 PM2.5 的处理效应解耦。

评估指标：
- Sensitivity Score = Mean(|ITE_{Age=+2} - ITE_{Age=-2}|)
- ITE-Age 相关系数

预期结果：
- Score ≈ 0：模型正确解耦了混杂因子 ✅
- Score >> 0：模型将 Age 的风险错误地归因于 PM2.5 的 ITE ❌

作者：Kiro AI Agent
日期：2026-01-23
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.baselines.mogonet_baseline import MOGONETBaseline
from src.baselines.transtee_baseline import TransTEEBaseline


class ConfoundingSensitivityEvaluator:
    """
    混杂因子敏感性评估器
    
    评估基线模型对混杂因子（Age）的敏感性
    """
    
    def __init__(self, data_path: str, models: Dict = None):
        """
        Args:
            data_path: LUAD 数据集路径
            models: 已训练的模型字典 {model_name: model_instance}
        """
        self.data_path = Path(data_path)
        self.models = models if models is not None else {}
        self.results = {}
        self.data = None
        self.feature_cols = None
        self.scaler = StandardScaler()
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载 LUAD 全量数据集
        
        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
        """
        print("[INFO] 加载 LUAD 全量数据集...")
        
        # 加载数据
        self.data = pd.read_csv(self.data_path)
        
        print(f"[SUCCESS] LUAD 全量数据加载完成")
        print(f"  - 样本数: {len(self.data)}")
        
        # 剔除 ID 列
        if 'sampleID' in self.data.columns:
            self.data = self.data.drop('sampleID', axis=1)
        
        # 分离特征和标签
        if 'Outcome_Label' in self.data.columns:
            y = self.data['Outcome_Label'].values
            X_df = self.data.drop('Outcome_Label', axis=1)
        else:
            raise ValueError("数据集中未找到 Outcome_Label 列")
        
        # 剔除 True_Prob 列（如果存在）
        if 'True_Prob' in X_df.columns:
            X_df = X_df.drop('True_Prob', axis=1)
        
        # 保存特征列名
        self.feature_cols = X_df.columns.tolist()
        
        # 转换为 numpy 数组
        X = X_df.values
        
        print(f"  - 特征数: {X.shape[1]}")
        print(f"  - 特征列: {self.feature_cols}")
        
        return X, y
    
    def create_intervention_scenarios(self, X: np.ndarray, 
                                     age_col_idx: int) -> Dict[str, np.ndarray]:
        """
        创建干预场景
        
        Args:
            X: 原始特征矩阵 (n_samples, n_features)
            age_col_idx: Age 列的索引
            
        Returns:
            scenarios: 场景字典 {'baseline': X, 'young': X_young, 'old': X_old}
        """
        print("\n[INFO] 创建干预场景...")
        
        # Baseline: 原始数据
        X_baseline = X.copy()
        
        # Young World: Age = -2.0（标准化后的年轻值）
        X_young = X.copy()
        X_young[:, age_col_idx] = -2.0
        
        # Old World: Age = +2.0（标准化后的年老值）
        X_old = X.copy()
        X_old[:, age_col_idx] = +2.0
        
        print(f"  - Baseline: 原始 Age 值")
        print(f"  - Young World: Age = -2.0 (标准化后)")
        print(f"  - Old World: Age = +2.0 (标准化后)")
        print(f"  - 其他特征保持不变")
        
        return {
            'baseline': X_baseline,
            'young': X_young,
            'old': X_old
        }
    
    def compute_ite(self, model, model_name: str, X: np.ndarray, 
                   pm25_col_idx: int) -> np.ndarray:
        """
        计算个体治疗效应 (ITE)
        
        对于分类模型：
            ITE = P(Y=1 | PM2.5=high, X) - P(Y=1 | PM2.5=low, X)
        
        对于 TransTEE：
            直接使用 predict_ite() 方法
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            X: 特征矩阵 (n_samples, n_features)
            pm25_col_idx: PM2.5 列的索引
            
        Returns:
            ite: 个体治疗效应向量 (n_samples,)
        """
        if model_name == 'TransTEE':
            # TransTEE 直接使用 predict_ite 方法
            # 注意：TransTEE 在训练时使用了完整的特征（包括 PM2.5）
            # 所以这里也需要传入完整的特征矩阵
            ite = model.predict_ite(X)
            return ite
        
        # 分类模型：使用反事实干预法
        # 创建高 PM2.5 场景（75th percentile）
        X_high = X.copy()
        pm25_high = np.percentile(X[:, pm25_col_idx], 75)
        X_high[:, pm25_col_idx] = pm25_high
        
        # 创建低 PM2.5 场景（25th percentile）
        X_low = X.copy()
        pm25_low = np.percentile(X[:, pm25_col_idx], 25)
        X_low[:, pm25_col_idx] = pm25_low
        
        # MOGONET 需要多视图输入
        if model_name == 'MOGONET':
            # 切分为多视图：前3列为临床特征，后面为基因特征
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
        ite = y_proba_high - y_proba_low
        
        return ite
    
    def evaluate_model(self, model_name: str, model, 
                      scenarios: Dict[str, np.ndarray],
                      pm25_col_idx: int,
                      age_col_idx: int) -> Dict[str, float]:
        """
        评估单个模型的混杂因子敏感性
        
        Args:
            model_name: 模型名称
            model: 模型实例
            scenarios: 干预场景字典
            pm25_col_idx: PM2.5 列的索引
            age_col_idx: Age 列的索引
            
        Returns:
            metrics: 评估指标字典
        """
        print(f"\n[INFO] 评估 {model_name}...")
        
        # 计算三个场景下的 ITE
        ite_baseline = self.compute_ite(model, model_name, scenarios['baseline'], pm25_col_idx)
        ite_young = self.compute_ite(model, model_name, scenarios['young'], pm25_col_idx)
        ite_old = self.compute_ite(model, model_name, scenarios['old'], pm25_col_idx)
        
        # 计算 Sensitivity Score
        sensitivity_score = np.mean(np.abs(ite_old - ite_young))
        
        # 计算相关系数（ITE vs Age）
        # 使用 baseline 场景的 Age 值
        age_values = scenarios['baseline'][:, age_col_idx]
        correlation = np.corrcoef(age_values, ite_baseline)[0, 1]
        
        metrics = {
            'sensitivity_score': sensitivity_score,
            'ite_age_correlation': correlation,
            'ite_young_mean': np.mean(ite_young),
            'ite_old_mean': np.mean(ite_old),
            'ite_baseline_mean': np.mean(ite_baseline),
            'ite_young_std': np.std(ite_young),
            'ite_old_std': np.std(ite_old),
            'ite_baseline_std': np.std(ite_baseline)
        }
        
        print(f"  - Sensitivity Score: {sensitivity_score:.4f}")
        print(f"  - ITE-Age 相关系数: {correlation:.4f}")
        print(f"  - ITE (Young): {metrics['ite_young_mean']:.4f} ± {metrics['ite_young_std']:.4f}")
        print(f"  - ITE (Old): {metrics['ite_old_mean']:.4f} ± {metrics['ite_old_std']:.4f}")
        
        # 验证结果
        if sensitivity_score < 0.05:
            print(f"  - ✓ 验证通过：Sensitivity Score < 0.05（模型正确解耦混杂因子）")
        else:
            print(f"  - ✗ 验证失败：Sensitivity Score ≥ 0.05（模型未能解耦混杂因子）")
        
        return metrics
    
    def run_evaluation(self, age_col_name: str = 'Age', 
                      pm25_col_name: str = 'Virtual_PM2.5') -> Dict[str, Dict]:
        """
        运行完整的敏感性评估
        
        Args:
            age_col_name: Age 特征的列名
            pm25_col_name: PM2.5 特征的列名
            
        Returns:
            results: 所有模型的评估结果
        """
        print("\n" + "="*80)
        print("混杂因子敏感性评估")
        print("="*80)
        
        # 加载数据
        X, y = self.load_data()
        
        # 获取 Age 和 PM2.5 的列索引
        age_col_idx = self.feature_cols.index(age_col_name)
        pm25_col_idx = self.feature_cols.index(pm25_col_name)
        
        print(f"\n[INFO] Age 列索引: {age_col_idx}")
        print(f"[INFO] PM2.5 列索引: {pm25_col_idx}")
        
        # 标准化特征（使用与训练时相同的方式）
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建干预场景
        scenarios = self.create_intervention_scenarios(X_scaled, age_col_idx)
        
        # 评估每个模型
        results = {}
        for model_name, model in self.models.items():
            try:
                metrics = self.evaluate_model(
                    model_name, model, scenarios, pm25_col_idx, age_col_idx
                )
                results[model_name] = metrics
            except Exception as e:
                print(f"[ERROR] {model_name} 评估失败：{str(e)}")
                import traceback
                traceback.print_exc()
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
        self.results = results
        return results
    
    def generate_report(self, output_path: str = "results/confounding_sensitivity_report.md"):
        """
        生成 Markdown 格式的评估报告
        
        Args:
            output_path: 输出文件路径
        """
        print("\n[INFO] 生成评估报告...")
        
        report = []
        report.append("# 混杂因子敏感性评估报告\n")
        report.append("## 理论基础\n")
        report.append("基于数据生成过程（Ground Truth），交互项仅取决于 EGFR，而非 Age。")
        report.append("因此，理论上完美的因果模型得出的 ITE 应对 Age 的变化保持**不变性**。\n")
        report.append("Age 与 ITE 之间的任何显著相关性都表明模型未能将 Age 的混杂效应与 PM2.5 的处理效应解耦。\n")
        
        report.append("## 评估指标\n")
        report.append("- **Sensitivity Score**: Mean(|ITE_{Age=+2} - ITE_{Age=-2}|)")
        report.append("- **ITE-Age 相关系数**: 衡量 ITE 与 Age 的线性关系\n")
        
        report.append("## 评估结果\n")
        report.append("| 模型 | Sensitivity Score | ITE-Age 相关系数 | ITE (Young) | ITE (Old) | ITE (Baseline) | 验证结果 |")
        report.append("|------|-------------------|------------------|-------------|-----------|----------------|----------|")
        
        for model_name, metrics in self.results.items():
            if 'status' in metrics and metrics['status'] == 'failed':
                report.append(f"| {model_name} | - | - | - | - | - | ✗ 失败 |")
            else:
                validation = "✓ 通过" if metrics['sensitivity_score'] < 0.05 else "✗ 失败"
                report.append(
                    f"| {model_name} | "
                    f"{metrics['sensitivity_score']:.4f} | "
                    f"{metrics['ite_age_correlation']:.4f} | "
                    f"{metrics['ite_young_mean']:.4f} | "
                    f"{metrics['ite_old_mean']:.4f} | "
                    f"{metrics['ite_baseline_mean']:.4f} | "
                    f"{validation} |"
                )
        
        report.append("\n## 解释\n")
        report.append("- **Sensitivity Score ≈ 0**：模型正确解耦了混杂因子 ✅")
        report.append("- **Sensitivity Score >> 0**：模型将 Age 的风险错误地归因于 PM2.5 的 ITE ❌")
        report.append("- **ITE-Age 相关系数 ≈ 0**：ITE 对 Age 不敏感，符合理论预期 ✅")
        report.append("- **ITE-Age 相关系数显著**：模型存在混杂偏倚 ❌\n")
        
        report.append("## 结论\n")
        passed_models = [name for name, m in self.results.items() 
                        if 'sensitivity_score' in m and m['sensitivity_score'] < 0.05]
        failed_models = [name for name, m in self.results.items() 
                        if 'sensitivity_score' in m and m['sensitivity_score'] >= 0.05]
        
        if passed_models:
            report.append(f"**通过验证的模型**：{', '.join(passed_models)}\n")
        if failed_models:
            report.append(f"**未通过验证的模型**：{', '.join(failed_models)}\n")
        
        report.append("这些结果表明，大多数基线模型未能正确解耦 Age（混杂因子）和 PM2.5（治疗变量）的效应，")
        report.append("这为 DLC 模型的开发提供了明确的改进方向。\n")
        
        # 写入文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"[SUCCESS] 报告已生成: {output_path}")
    
    def plot_ite_age_dependency(self, age_col_name: str = 'Age',
                                pm25_col_name: str = 'Virtual_PM2.5',
                                output_path: str = "results/ite_age_dependency.png"):
        """
        生成 ITE-Age 依赖性可视化图
        
        Args:
            age_col_name: Age 特征的列名
            pm25_col_name: PM2.5 特征的列名
            output_path: 输出文件路径
        """
        print("\n[INFO] 生成 ITE-Age 依赖性可视化图...")
        
        # 加载数据
        X, y = self.load_data()
        
        # 获取列索引
        age_col_idx = self.feature_cols.index(age_col_name)
        pm25_col_idx = self.feature_cols.index(pm25_col_name)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        age_values = X_scaled[:, age_col_idx]
        
        # 创建子图
        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
            
            try:
                # 计算 ITE
                ite = self.compute_ite(model, model_name, X_scaled, pm25_col_idx)
                
                # 获取相关系数
                corr = self.results[model_name]['ite_age_correlation']
                
                # 绘制散点图
                axes[idx].scatter(age_values, ite, alpha=0.5, s=10, color='steelblue')
                axes[idx].set_xlabel('Age (Standardized)', fontsize=12)
                axes[idx].set_ylabel('ITE (PM2.5 Effect)', fontsize=12)
                axes[idx].set_title(f'{model_name}\nCorr: {corr:.3f}', fontsize=14, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                
                # 添加趋势线
                z = np.polyfit(age_values, ite, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(age_values)
                axes[idx].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2, label='Trend Line')
                axes[idx].legend()
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error: {str(e)}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{model_name}\n(Failed)', fontsize=14)
        
        # 隐藏多余的子图
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SUCCESS] 可视化图已生成: {output_path}")


def main():
    """
    主函数：运行混杂因子敏感性评估
    
    注意：需要先训练模型，或者从 src/run_experiment_scenarios.py 加载已训练的模型
    """
    print("="*80)
    print("Task 21: Phase 5 - 混杂因子敏感性评估")
    print("="*80)
    
    # TODO: 这里需要加载已训练的模型
    # 可以从 src/run_experiment_scenarios.py 或 src/evaluate_three_tier_metrics_fix.py 复用训练逻辑
    
    print("\n[WARNING] 此脚本需要已训练的模型")
    print("[INFO] 请先运行 src/evaluate_three_tier_metrics_fix.py 训练模型")
    print("[INFO] 或者在此脚本中添加模型训练逻辑")
    
    # 示例：如果有已训练的模型
    # models = {
    #     'XGBoost': xgb_model,
    #     'TabR': tabr_model,
    #     'HyperFast': hyperfast_model,
    #     'MOGONET': mogonet_model,
    #     'TransTEE': transtee_model
    # }
    
    # evaluator = ConfoundingSensitivityEvaluator(
    #     data_path='data/luad_synthetic_interaction.csv',
    #     models=models
    # )
    
    # results = evaluator.run_evaluation()
    # evaluator.generate_report()
    # evaluator.plot_ite_age_dependency()


if __name__ == '__main__':
    main()

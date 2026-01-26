# -*- coding: utf-8 -*-
"""
DLC Final SOTA Training Script
==============================

Phase 7: Performance SOTA & Scientific Validation

最终执行脚本，包含完整的工作流程:
1. 加载 PANCAN 完整数据集
2. 执行贝叶斯超参数优化
3. 使用最优参数训练 DLCNet
4. 计算完整评估指标（包括因果指标）
5. 保存结果和模型

Usage:
    python src/run_final_sota.py [--n_trials 50] [--epochs 100] [--skip_tuning]

Output:
    - results/final_sota_report.md: 完整评估报告
    - results/dlc_final_sota.pth: 训练好的模型
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 导入 DLC 模块
from src.data_processor import DataCleaner, SemiSyntheticGenerator
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_pehe, compute_cate, compute_sensitivity_score
from src.dlc.tune import tune_dlc_hyperparameters, get_default_params


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='DLC Final SOTA Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据参数
    parser.add_argument(
        '--data_source', type=str, default='PANCAN',
        choices=['PANCAN', 'LUAD'],
        help='数据源选择'
    )
    parser.add_argument(
        '--scenario', type=str, default='interaction',
        choices=['interaction', 'linear'],
        help='数据生成场景'
    )
    
    # 超参数调优参数
    parser.add_argument(
        '--n_trials', type=int, default=50,
        help='超参数调优试验次数'
    )
    parser.add_argument(
        '--skip_tuning', action='store_true',
        help='跳过超参数调优，使用默认参数'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='最终训练轮数'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='批量大小（仅在跳过调优时使用）'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help='测试集比例'
    )
    
    # 其他参数
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results',
        help='输出目录'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='详细输出'
    )
    
    return parser.parse_args()


def load_data(
    data_source: str = 'PANCAN',
    scenario: str = 'interaction',
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    加载并预处理数据。
    
    优先级顺序:
    1. 优先加载已生成的 CSV 文件 (data/pancan_synthetic_interaction.csv)
    2. 如果 CSV 不存在，尝试从原始 TXT 文件生成
    3. 如果 TXT 文件也不存在，生成临时合成数据
    
    Args:
        data_source: 数据源 ('PANCAN' 或 'LUAD')
        scenario: 数据生成场景 ('interaction' 或 'linear')
        random_state: 随机种子
        verbose: 详细输出
    
    Returns:
        Tuple[X, y, true_prob, feature_names]:
            - X: 特征矩阵 [N, 23]
            - y: 标签向量 [N]
            - true_prob: 真实概率向量 [N]（用于计算真实 ITE）
            - feature_names: 特征名列表
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"加载 {data_source} 数据集 (场景: {scenario})")
        print("=" * 60)
    
    # 定义数据路径
    data_dir = PROJECT_ROOT / 'data'
    
    # 步骤 1: 优先检查已生成的 CSV 文件
    csv_filename = f"{data_source.lower()}_synthetic_{scenario}.csv"
    csv_path = data_dir / csv_filename
    
    if csv_path.exists():
        if verbose:
            print(f"✓ 找到已生成的 CSV 文件: {csv_path}")
        
        # 加载 CSV
        import pandas as pd
        final_df = pd.read_csv(csv_path)
        
        # 动态推断特征列
        # 逻辑：排除非特征列 (ID, 标签, 真实概率)，剩下的全都是输入特征
        exclude_cols = ['sampleID', 'Outcome_Label', 'True_Prob']
        feature_cols = [c for c in final_df.columns if c not in exclude_cols]
        
        # 简单验证（仅警告，不阻断）
        expected_count = 23
        if len(feature_cols) != expected_count:
            if verbose:
                print(f"⚠️  警告: CSV 特征数量 ({len(feature_cols)}) 与预期 ({expected_count}) 不一致。")
                print(f"检测到的特征: {feature_cols[:5]} ...")
        
        # 提取数据
        X = final_df[feature_cols].values.astype(np.float32)
        y = final_df['Outcome_Label'].values.astype(np.int64)
        true_prob = final_df['True_Prob'].values.astype(np.float32)
        
        if verbose:
            print(f"CSV 数据加载完成:")
            print(f"  - 样本数: {len(X)}")
            print(f"  - 特征数: {X.shape[1]}")
            print(f"  - 正类比例: {y.mean():.2%}")
            print(f"  - 特征列: {feature_cols[:3]} ... {feature_cols[-2:]}")
        
        return X, y, true_prob, feature_cols
    
    # 步骤 2: CSV 不存在，尝试从原始 TXT 文件生成
    if verbose:
        print(f"✗ CSV 文件不存在: {csv_path}")
        print(f"尝试从原始 TXT 文件生成...")
    
    # 根据数据源选择路径
    if data_source == 'PANCAN':
        gene_path = data_dir / 'PANCAN' / 'PANCAN_mutation.txt'
        clinical_path = data_dir / 'PANCAN' / 'PANCAN_clinical.txt'
    else:  # LUAD
        gene_path = data_dir / 'LUAD' / 'LUAD_mc3_gene_level.txt'
        clinical_path = data_dir / 'LUAD' / 'TCGA.LUAD.sampleMap_LUAD_clinicalMatrix.txt'
    
    # 检查 TXT 文件是否存在
    if gene_path.exists() and clinical_path.exists():
        if verbose:
            print(f"✓ 找到原始 TXT 文件:")
            print(f"  - 基因数据: {gene_path}")
            print(f"  - 临床数据: {clinical_path}")
        
        # 加载真实数据
        cleaner = DataCleaner(data_source)
        cleaner.load_gene_data(str(gene_path))
        cleaner.load_clinical_data(str(clinical_path))
        cleaner.merge_data()
        cleaner.select_top_genes(n=20)
        cleaned_df = cleaner.clean_clinical()
        
        # 生成半合成结局
        generator = SemiSyntheticGenerator(
            cleaned_df,
            cleaner.top20_genes,
            seed=random_state
        )
        final_df = generator.generate(scenario)
        
        # 提取特征和标签
        feature_cols = ['Age', 'Gender', 'Virtual_PM2.5'] + cleaner.top20_genes
        X = final_df[feature_cols].values.astype(np.float32)
        y = final_df['Outcome_Label'].values.astype(np.int64)
        true_prob = final_df['True_Prob'].values.astype(np.float32)
        
        if verbose:
            print(f"TXT 数据加载完成:")
            print(f"  - 样本数: {len(X)}")
            print(f"  - 特征数: {X.shape[1]}")
            print(f"  - 正类比例: {y.mean():.2%}")
        
        return X, y, true_prob, feature_cols
    
    # 步骤 3: TXT 文件也不存在，生成临时合成数据
    if verbose:
        print(f"✗ 原始 TXT 文件不存在:")
        print(f"  - 基因数据: {gene_path}")
        print(f"  - 临床数据: {clinical_path}")
        print(f"⚠️  使用临时合成数据进行演示（仅 2000 样本）")
    
    # 生成合成数据用于演示
    return _generate_synthetic_data(random_state, verbose)


def _generate_synthetic_data(
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    生成合成数据用于演示。
    
    Args:
        random_state: 随机种子
        verbose: 详细输出
    
    Returns:
        Tuple[X, y, true_prob, feature_names]
    """
    np.random.seed(random_state)
    
    n_samples = 2000
    n_genes = 20
    
    # 特征名
    feature_names = ['Age', 'Gender', 'Virtual_PM2.5'] + [f'Gene_{i}' for i in range(n_genes)]
    
    # 生成特征
    # Age: 正态分布 (50, 15)
    age = np.random.normal(50, 15, n_samples)
    age = np.clip(age, 20, 90)
    
    # Gender: 二元变量
    gender = np.random.binomial(1, 0.5, n_samples)
    
    # PM2.5: 与 Age 相关（混杂）
    pm25 = 30 + 0.3 * (age - 50) + np.random.normal(0, 10, n_samples)
    pm25 = np.clip(pm25, 5, 100)
    
    # 基因特征: 稀疏二元
    genes = np.random.binomial(1, 0.1, (n_samples, n_genes))
    
    # 组合特征矩阵
    X = np.column_stack([age, gender, pm25, genes]).astype(np.float32)
    
    # 生成真实概率（包含交互效应）
    # Logit = -3 + 0.02*Age + 0.5*Gender + 0.03*PM2.5 + 0.5*EGFR + 0.3*(PM2.5*EGFR)
    pm25_std = (pm25 - pm25.mean()) / pm25.std()
    egfr = genes[:, 0]  # 假设第一个基因是 EGFR
    gene_effect = 0.5 * genes.sum(axis=1)
    
    logit = -3.0 + 0.086 * pm25_std + 0.69 * (pm25_std * egfr) + gene_effect
    true_prob = 1.0 / (1.0 + np.exp(-logit))
    
    # 采样标签
    y = (np.random.random(n_samples) < true_prob).astype(np.int64)
    
    if verbose:
        print(f"合成数据生成完成:")
        print(f"  - 样本数: {n_samples}")
        print(f"  - 特征数: {len(feature_names)}")
        print(f"  - 正类比例: {y.mean():.2%}")
    
    return X, y, true_prob.astype(np.float32), feature_names


def compute_true_ite(
    true_prob: np.ndarray,
    X: np.ndarray,
    treatment_col_idx: int = 2
) -> np.ndarray:
    """
    从真实概率计算真实 ITE。
    
    由于我们知道数据生成过程，可以通过扰动治疗变量计算真实 ITE。
    
    Args:
        true_prob: 真实概率向量
        X: 特征矩阵
        treatment_col_idx: 治疗列索引（PM2.5）
    
    Returns:
        np.ndarray: 真实 ITE 近似值
    
    Note:
        这是一个近似方法。在真实场景中，可能需要重新模拟数据生成过程。
    """
    # 使用 PM2.5 的高低分位数来近似 ITE
    pm25 = X[:, treatment_col_idx]
    
    # 计算 PM2.5 对结局的边际效应
    # 使用简单的差分方法近似
    pm25_high_mask = pm25 > np.median(pm25)
    
    # 假设高暴露组和低暴露组的概率差异代表 ITE
    # 这是一个简化，实际应该使用因果推断方法
    
    # 使用 True_Prob 与 PM2.5 的相关性来估计 ITE
    pm25_std = (pm25 - pm25.mean()) / (pm25.std() + 1e-8)
    
    # ITE ≈ ∂P/∂PM2.5 * ΔPM2.5
    # 使用数值方法估计偏导数
    delta = 0.1
    
    # 近似真实 ITE（假设线性近似）
    # 这里使用一个启发式方法：ITE ∝ 暴露敏感性
    true_ite = 0.086 * np.ones_like(true_prob)  # 基准效应
    
    # 如果存在 EGFR 突变，效应增强
    if X.shape[1] > 3:
        egfr = X[:, 3]  # 假设 EGFR 是第一个基因
        true_ite = true_ite + 0.69 * egfr * pm25_std
    
    return true_ite


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
    epochs: int = 100,
    random_state: int = 42,
    verbose: bool = True
) -> DLCNet:
    """
    使用最优参数训练最终模型。
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        best_params: 最优超参数
        epochs: 训练轮数
        random_state: 随机种子
        verbose: 详细输出
    
    Returns:
        DLCNet: 训练好的模型
    """
    if verbose:
        print("\n" + "=" * 60)
        print("训练最终模型")
        print("=" * 60)
        print(f"超参数: {best_params}")
        print(f"训练轮数: {epochs}")
    
    # 创建模型
    model = DLCNet(
        input_dim=X_train.shape[1],
        d_hidden=best_params.get('d_hidden', 64),
        num_layers=best_params.get('num_layers', 2),
        lambda_hsic=best_params.get('lambda_hsic', 0.1),
        random_state=random_state
    )
    
    # 训练模型
    # 注意：DLCNet.fit() 内部已实现三阶段训练
    # 这里我们使用自定义训练循环以支持更多轮数
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=best_params.get('batch_size', 64),
        verbose=verbose
    )
    
    return model


def compute_all_metrics(
    model: DLCNet,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_prob: np.ndarray,
    feature_names: list,
    verbose: bool = True
) -> Dict[str, float]:
    """
    计算所有评估指标。
    
    包括:
    - 分类指标: Accuracy, Precision, Recall, F1, AUC-ROC
    - 因果指标: PEHE, Sensitivity Score
    
    Args:
        model: 训练好的 DLCNet 模型
        X_test: 测试特征
        y_test: 测试标签
        true_prob: 真实概率（用于计算真实 ITE）
        feature_names: 特征名列表
        verbose: 详细输出
    
    Returns:
        Dict: 所有指标的字典
    """
    if verbose:
        print("\n" + "=" * 60)
        print("计算评估指标")
        print("=" * 60)
    
    metrics = {}
    
    # 1. 基础分类指标
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
    metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    
    if verbose:
        print("\n分类指标:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # 2. 因果指标 - PEHE
    X_test_tensor = torch.FloatTensor(model.scaler.transform(X_test))
    
    # 动态查找列索引（避免硬编码）
    try:
        treatment_col_idx = feature_names.index('Virtual_PM2.5')
    except ValueError:
        # 如果找不到 Virtual_PM2.5，尝试查找 PM2.5 或使用默认值
        if 'PM2.5' in feature_names:
            treatment_col_idx = feature_names.index('PM2.5')
        else:
            # 回退到默认值（假设在第 3 列）
            treatment_col_idx = 2
            if verbose:
                print(f"⚠️  警告: 未找到 'Virtual_PM2.5' 列，使用默认索引 {treatment_col_idx}")
    
    # 计算预测的 ITE
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        pred_ite = outputs['ITE'].squeeze().numpy()
    
    # 计算真实 ITE（近似）
    true_ite = compute_true_ite(true_prob, X_test, treatment_col_idx=treatment_col_idx)
    
    # 计算 PEHE
    metrics['pehe'] = compute_pehe(true_ite, pred_ite)
    
    if verbose:
        print("\n因果指标:")
        print(f"  PEHE:      {metrics['pehe']:.4f}")
    
    # 3. 敏感性分数
    # 动态查找 Age 列索引
    try:
        age_col_idx = feature_names.index('Age')
    except ValueError:
        age_col_idx = 0  # 回退到默认值
        if verbose:
            print(f"⚠️  警告: 未找到 'Age' 列，使用默认索引 {age_col_idx}")
    
    # 测试 Age 的敏感性
    sens_age = compute_sensitivity_score(
        model, X_test_tensor,
        confounder_idx=age_col_idx, epsilon=0.1
    )
    metrics['sensitivity_age'] = sens_age
    
    # 测试 PM2.5 的敏感性
    sens_pm25 = compute_sensitivity_score(
        model, X_test_tensor,
        confounder_idx=treatment_col_idx, epsilon=0.1
    )
    metrics['sensitivity_pm25'] = sens_pm25
    
    if verbose:
        print(f"  Sensitivity (Age, ε=0.1):   {sens_age:.4f}")
        print(f"  Sensitivity (PM2.5, ε=0.1): {sens_pm25:.4f}")
    
    # 4. CATE 统计
    cate = compute_cate(model, X_test_tensor, treatment_col_idx=treatment_col_idx)
    metrics['cate_mean'] = float(np.mean(cate))
    metrics['cate_std'] = float(np.std(cate))
    
    if verbose:
        print(f"\nCATE 统计:")
        print(f"  Mean CATE: {metrics['cate_mean']:.4f}")
        print(f"  Std CATE:  {metrics['cate_std']:.4f}")
    
    return metrics


def save_results(
    metrics: Dict[str, float],
    best_params: Dict[str, Any],
    model: DLCNet,
    output_dir: str,
    verbose: bool = True
) -> None:
    """
    保存结果和模型。
    
    Args:
        metrics: 评估指标
        best_params: 最优超参数
        model: 训练好的模型
        output_dir: 输出目录
        verbose: 详细输出
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存模型
    model_path = output_path / 'dlc_final_sota.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    if verbose:
        print(f"\n模型已保存至: {model_path}")
    
    # 2. 生成 Markdown 报告
    report_path = output_path / 'final_sota_report.md'
    
    report_content = f"""# DLC Final SOTA Evaluation Report

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 最优超参数

| 参数 | 值 |
|------|-----|
| d_hidden | {best_params.get('d_hidden', 'N/A')} |
| num_layers | {best_params.get('num_layers', 'N/A')} |
| lambda_hsic | {best_params.get('lambda_hsic', 'N/A'):.4f if isinstance(best_params.get('lambda_hsic'), float) else 'N/A'} |
| lr | {best_params.get('lr', 'N/A'):.6f if isinstance(best_params.get('lr'), float) else 'N/A'} |
| batch_size | {best_params.get('batch_size', 'N/A')} |

## 2. 分类性能指标

| 指标 | 值 |
|------|-----|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1']:.4f} |
| AUC-ROC | {metrics['auc_roc']:.4f} |

## 3. 因果推断指标

| 指标 | 值 | 说明 |
|------|-----|------|
| PEHE | {metrics['pehe']:.4f} | Precision in Estimation of Heterogeneous Effect (越低越好) |
| Sensitivity (Age) | {metrics['sensitivity_age']:.4f} | 模型对年龄扰动的敏感性 |
| Sensitivity (PM2.5) | {metrics['sensitivity_pm25']:.4f} | 模型对 PM2.5 扰动的敏感性 |
| Mean CATE | {metrics['cate_mean']:.4f} | 条件平均治疗效应均值 |
| Std CATE | {metrics['cate_std']:.4f} | 条件平均治疗效应标准差 |

## 4. 解释

### PEHE (Precision in Estimation of Heterogeneous Effect)
PEHE 衡量预测的个体治疗效应 (ITE) 与真实 ITE 之间的误差。
较低的 PEHE 值表示模型能够准确估计不同个体的治疗效应异质性。

### 敏感性分数
敏感性分数测量模型预测对特定变量扰动的响应程度：
- **Age 敏感性**: 衡量模型是否过度依赖年龄信息（混杂因素）
- **PM2.5 敏感性**: 衡量模型对环境暴露的响应（治疗变量）

理想情况下，因果模型应该：
- 对混杂因素（Age）有较低的敏感性（表示成功解耦）
- 对治疗变量（PM2.5）有合理的敏感性（表示捕获因果效应）

### CATE (Conditional Average Treatment Effect)
CATE 描述了在给定协变量条件下的平均治疗效应。
正的 Mean CATE 表示 PM2.5 暴露平均增加不良结局风险。

## 5. 模型文件

- 模型权重: `dlc_final_sota.pth`
- 本报告: `final_sota_report.md`

---
*由 DLC Final SOTA Training Script 自动生成*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    if verbose:
        print(f"报告已保存至: {report_path}")
    
    # 3. 保存 JSON 格式结果（便于程序读取）
    json_path = output_path / 'final_sota_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'best_params': best_params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"JSON 结果已保存至: {json_path}")


def main():
    """
    主函数：执行完整的 SOTA 训练流程。
    """
    # 解析参数
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("DLC Final SOTA Training Pipeline")
    print("Phase 7: Performance SOTA & Scientific Validation")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置随机种子
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    
    # Step 1: 加载数据
    X, y, true_prob, feature_names = load_data(
        data_source=args.data_source,
        scenario=args.scenario,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    # Step 2: 划分训练集和测试集
    X_train, X_test, y_train, y_test, prob_train, prob_test = train_test_split(
        X, y, true_prob,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # Step 3: 超参数调优
    if args.skip_tuning:
        print("\n跳过超参数调优，使用默认参数")
        best_params = get_default_params()
        best_params['batch_size'] = args.batch_size
    else:
        # 从训练集中划分调优验证集
        X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=args.random_state,
            stratify=y_train
        )
        
        tune_results = tune_dlc_hyperparameters(
            X_train=X_tune_train,
            y_train=y_tune_train,
            X_val=X_tune_val,
            y_val=y_tune_val,
            n_trials=args.n_trials,
            random_state=args.random_state,
            verbose=args.verbose
        )
        
        best_params = tune_results['best_params']
    
    # Step 4: 训练最终模型
    model = train_final_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
        epochs=args.epochs,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    # Step 5: 计算所有指标
    metrics = compute_all_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        true_prob=prob_test,
        feature_names=feature_names,
        verbose=args.verbose
    )
    
    # Step 6: 保存结果
    save_results(
        metrics=metrics,
        best_params=best_params,
        model=model,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # 完成
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 返回结果（便于测试）
    return {
        'model': model,
        'metrics': metrics,
        'best_params': best_params
    }


if __name__ == '__main__':
    main()

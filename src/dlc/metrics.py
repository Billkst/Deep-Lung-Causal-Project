# -*- coding: utf-8 -*-
"""
DLC Causal Metrics Module
=========================

高级因果推断评估指标模块。

提供 PEHE (Precision in Estimation of Heterogeneous Effect)、
CATE (Conditional Average Treatment Effect) 计算和
敏感性分析等因果推断专用指标。

Functions:
    compute_pehe: 计算个体治疗效应估计精度
    compute_cate: 通过反事实干预计算条件平均治疗效应
    compute_sensitivity_score: 计算模型对混杂因素扰动的敏感性分数

满足需求: Phase 7 - Performance SOTA & Scientific Validation
"""

import numpy as np
import torch
from typing import Union, Optional, Any
import warnings

from .ground_truth import GroundTruthGenerator


def compute_pehe(
    model_pred_ite: Union[np.ndarray, torch.Tensor],
    X_features: Optional[Union[np.ndarray, torch.Tensor, Any]] = None
) -> float:
    """
    计算 PEHE (Precision in Estimation of Heterogeneous Effect)。
    
    PEHE 是评估个体治疗效应 (ITE) 估计精度的标准指标，
    衡量预测 ITE 与真实 ITE 之间的均方根误差。
    
    Args:
        model_pred_ite: 模型预测 ITE [N] 或 [N, 1]
        X_features: 特征矩阵，用于生成 Ground Truth ITE
    
    Returns:
        float: PEHE 值，即 sqrt(mean((y_true - y_pred)^2))
               若所有值均为 NaN，返回 np.nan
    
    Formula:
        PEHE = sqrt(1/N * Σ(τ_true_i - τ_pred_i)^2)
        
        其中 τ 表示 ITE (Individual Treatment Effect)
    
    Note:
        - 自动处理 NaN 值（使用 nanmean）
        - 自动将 torch.Tensor 转换为 np.ndarray
        - 自动展平为一维数组
    
    Example:
        >>> y_pred = np.array([0.4, 0.35, 0.65, 0.25])
        >>> X = np.random.randn(4, 23)
        >>> pehe = compute_pehe(y_pred, X)
        >>> print(f"PEHE: {pehe:.4f}")
    
    References:
        Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference.
        Journal of Computational and Graphical Statistics, 20(1), 217-240.
    """
    if X_features is None:
        raise ValueError("X_features 不能为空，需用于生成 Ground Truth ITE")

    feature_columns = getattr(X_features, 'columns', None)

    if isinstance(model_pred_ite, torch.Tensor):
        model_pred_ite = model_pred_ite.detach().cpu().numpy()

    if isinstance(X_features, torch.Tensor):
        X_features = X_features.detach().cpu().numpy()

    model_pred_ite = np.asarray(model_pred_ite)
    X_features_np = np.asarray(X_features)

    # 兼容旧接口: compute_pehe(true_ite, pred_ite)
    if model_pred_ite.shape == X_features_np.shape:
        if model_pred_ite.ndim in (1, 2):
            if model_pred_ite.ndim == 2 and model_pred_ite.shape[1] == 1:
                model_pred_ite = model_pred_ite.flatten()
                X_features_np = X_features_np.flatten()
            return compute_pehe_from_arrays(model_pred_ite, X_features_np)

    generator = GroundTruthGenerator(feature_names=feature_columns)
    true_ite = generator.compute_true_ite(X_features_np)
    return compute_pehe_from_arrays(true_ite, model_pred_ite)


def compute_pehe_from_arrays(
    y_true_ite: Union[np.ndarray, torch.Tensor],
    y_pred_ite: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    计算 PEHE (Precision in Estimation of Heterogeneous Effect)。

    Args:
        y_true_ite: 真实个体治疗效应 [N] 或 [N, 1]
        y_pred_ite: 预测个体治疗效应 [N] 或 [N, 1]

    Returns:
        float: PEHE 值
    """
    # 类型转换: torch.Tensor -> np.ndarray
    if isinstance(y_true_ite, torch.Tensor):
        y_true_ite = y_true_ite.detach().cpu().numpy()
    if isinstance(y_pred_ite, torch.Tensor):
        y_pred_ite = y_pred_ite.detach().cpu().numpy()
    
    # 确保为 numpy 数组
    y_true_ite = np.asarray(y_true_ite, dtype=np.float64)
    y_pred_ite = np.asarray(y_pred_ite, dtype=np.float64)
    
    # 展平为一维数组
    y_true_ite = y_true_ite.flatten()
    y_pred_ite = y_pred_ite.flatten()
    
    # 检查形状一致性
    if y_true_ite.shape != y_pred_ite.shape:
        raise ValueError(
            f"形状不匹配: y_true_ite {y_true_ite.shape} vs y_pred_ite {y_pred_ite.shape}"
        )
    
    # 检查空数组
    if len(y_true_ite) == 0:
        warnings.warn("输入数组为空，返回 NaN")
        return np.nan
    
    # 计算平方误差
    squared_error = (y_true_ite - y_pred_ite) ** 2
    
    # 使用 nanmean 处理可能的 NaN 值
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mse = np.nanmean(squared_error)
    
    # 处理全 NaN 情况
    if np.isnan(mse):
        warnings.warn("所有值均为 NaN，返回 NaN")
        return np.nan
    
    # 返回 RMSE（即 sqrt(MSE)）
    pehe = np.sqrt(mse)
    
    return float(pehe)


def compute_cate(
    model: Any,
    X: torch.Tensor,
    treatment_col_idx: int,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    通过反事实干预计算 CATE (Conditional Average Treatment Effect)。
    
    对于连续变量（如 PM2.5），使用稳健的分位数法：
    1. 计算治疗变量的 84th 分位数（高暴露，约 Mean + 1 Std）
    2. 计算治疗变量的 16th 分位数（低暴露，约 Mean - 1 Std）
    3. 分别预测 Y(high) 和 Y(low)
    4. 计算 CATE = Y(high) - Y(low)
    
    这种方法对应于正态分布下的 ±1σ，是连续变量因果效应计算的稳健方式。
    
    Args:
        model: 预测模型，必须实现 forward() 方法，
               返回包含 'pred' 键的字典（logits），
               或者实现 predict_proba() 方法
        X: 输入特征张量 [N, D]
        treatment_col_idx: 治疗列的索引（例如 PM2.5 列索引为 2）
        device: 计算设备，默认为 None（自动检测）
    
    Returns:
        np.ndarray: CATE 估计值 [N]，表示每个样本的条件平均治疗效应
                    CATE = P(Y=1|High) - P(Y=1|Low)
    
    Note:
        - 模型会自动设置为 eval() 模式
        - 使用 torch.no_grad() 禁用梯度计算
        - 支持 DLCNet 的双头预测器结构
        - 对于连续变量，使用分位数法比二元干预更合理
    
    Example:
        >>> from src.dlc.dlc_net import DLCNet
        >>> model = DLCNet(input_dim=23)
        >>> X = torch.randn(100, 23)
        >>> cate = compute_cate(model, X, treatment_col_idx=2)
        >>> print(f"CATE mean: {cate.mean():.4f}")
    
    Raises:
        ValueError: 如果 treatment_col_idx 超出范围
        TypeError: 如果模型不支持所需的接口
    """
    # 输入验证
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    # 检查 treatment_col_idx 范围
    if treatment_col_idx < 0 or treatment_col_idx >= X.shape[1]:
        raise ValueError(
            f"treatment_col_idx {treatment_col_idx} 超出范围 [0, {X.shape[1]-1}]"
        )
    
    # 确定设备
    if device is None:
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    
    # 将数据移动到设备
    X = X.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 获取治疗变量的分布统计
        treatment_vals = X[:, treatment_col_idx].detach().cpu().numpy()
        
        # --- 关键修改：使用分位数法 (Robust) ---
        # 84th percentile ≈ Mean + 1 Std
        high_val = np.percentile(treatment_vals, 84)
        # 16th percentile ≈ Mean - 1 Std
        low_val = np.percentile(treatment_vals, 16)
        
        # 创建高暴露组数据 (treatment = 84th percentile)
        X_treat = X.clone()
        X_treat[:, treatment_col_idx] = float(high_val)
        
        # 创建低暴露组数据 (treatment = 16th percentile)
        X_control = X.clone()
        X_control[:, treatment_col_idx] = float(low_val)
        
        # 预测 Y(high) 和 Y(low)
        # 优先使用 DLCNet 的 forward 方法（返回字典）
        if hasattr(model, 'forward'):
            try:
                # 尝试 DLCNet 风格的 forward
                outputs_treat = model(X_treat)
                outputs_control = model(X_control)
                
                # 检查返回值类型
                if isinstance(outputs_treat, dict):
                    # DLCNet 风格: 返回字典包含 'Y_1' 概率
                    if 'Y_1' in outputs_treat:
                        prob_treat = outputs_treat['Y_1'].squeeze().cpu().numpy()
                        prob_control = outputs_control['Y_1'].squeeze().cpu().numpy()
                    else:
                        # 兼容返回 logits 的情况
                        prob_treat = torch.sigmoid(outputs_treat['pred']).squeeze().cpu().numpy()
                        prob_control = torch.sigmoid(outputs_control['pred']).squeeze().cpu().numpy()
                else:
                    # 普通模型: forward 直接返回预测值
                    # 假设已经是概率或需要 sigmoid
                    prob_treat = torch.sigmoid(outputs_treat).squeeze().cpu().numpy()
                    prob_control = torch.sigmoid(outputs_control).squeeze().cpu().numpy()
            except (KeyError, TypeError):
                # 回退到 predict_proba
                if hasattr(model, 'predict_proba'):
                    # 使用 predict_proba[:, 1] 作为正类概率
                    prob_treat = model.predict_proba(X_treat.cpu().numpy())[:, 1]
                    prob_control = model.predict_proba(X_control.cpu().numpy())[:, 1]
                else:
                    raise TypeError(
                        "模型必须实现返回字典的 forward() 方法或 predict_proba() 方法"
                    )
        elif hasattr(model, 'predict_proba'):
            prob_treat = model.predict_proba(X_treat.cpu().numpy())[:, 1]
            prob_control = model.predict_proba(X_control.cpu().numpy())[:, 1]
        else:
            raise TypeError(
                "模型必须实现 forward() 或 predict_proba() 方法"
            )
        
        # 计算 CATE = P(Y=1|High) - P(Y=1|Low)
        cate = prob_treat - prob_control
    
    return cate.astype(np.float64)


def compute_sensitivity_score(
    model: Any,
    X: torch.Tensor,
    confounder_idx: int,
    epsilon: Optional[float] = None,
    device: Optional[torch.device] = None,
    treatment_col_idx: int = 2
) -> float:
    """
    计算模型对混杂因素扰动的敏感性分数。
    
    通过扰动指定的混杂因素列，衡量模型预测的变化程度。
    敏感性分数越高，说明模型对该混杂因素越敏感。
    
    如果不指定 epsilon，默认使用该特征的 1 个标准差作为扰动幅度，
    这对于不同量纲的特征更加稳健。
    
    Args:
        model: 预测模型，必须实现 forward() 或 predict_proba() 方法
        X: 输入特征张量 [N, D]
        confounder_idx: 混杂因素列的索引（例如 Age 列索引为 0）
        epsilon: 扰动幅度，默认为 None（自动使用该特征的 1 个标准差）
                 如果指定，则使用绝对值扰动（而非比例扰动）
        device: 计算设备，默认为 None（自动检测）
    
    Returns:
        float: 敏感性分数 = mean(|ITE_original - ITE_perturbed|)
               值域 [0, 1]，越高表示越敏感
    
    Formula:
        如果 epsilon 为 None:
            epsilon = std(X[:, confounder_idx])
        X_perturbed[:, confounder_idx] = X[:, confounder_idx] + epsilon
        sensitivity = mean(|ITE(X) - ITE(X_perturbed)|)
    
    Note:
        - 用于评估模型是否正确捕获混杂效应
        - 较低的敏感性分数可能表示模型成功解耦了混杂因素
        - 使用标准差作为扰动幅度对不同量纲的特征更加公平
        - 对于 Age (范围 40-80)，1 Std ≈ 10 年
        - 对于 PM2.5 (范围 20-90)，1 Std ≈ 15 μg/m³
    
    Example:
        >>> from src.dlc.dlc_net import DLCNet
        >>> model = DLCNet(input_dim=23)
        >>> X = torch.randn(100, 23)
        >>> # 自动使用标准差
        >>> sens = compute_sensitivity_score(model, X, confounder_idx=0)
        >>> print(f"Sensitivity Score (Age, ε=1 Std): {sens:.4f}")
        >>> # 手动指定扰动
        >>> sens = compute_sensitivity_score(model, X, confounder_idx=0, epsilon=5.0)
        >>> print(f"Sensitivity Score (Age, ε=5.0): {sens:.4f}")
    
    Raises:
        ValueError: 如果 confounder_idx 超出范围或 epsilon < 0
    """
    # 输入验证
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    # 检查 confounder_idx 范围
    if confounder_idx < 0 or confounder_idx >= X.shape[1]:
        raise ValueError(
            f"confounder_idx {confounder_idx} 超出范围 [0, {X.shape[1]-1}]"
        )
    
    # 检查 epsilon（如果指定）
    if epsilon is not None and epsilon < 0:
        raise ValueError(f"epsilon 必须为非负数，收到: {epsilon}")
    
    # 确定设备
    if device is None:
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    
    # 将数据移动到设备
    X = X.to(device)
    
    # --- 关键修改：动态确定 epsilon ---
    if epsilon is None:
        # 使用特征的标准差作为扰动幅度
        feat_std = X[:, confounder_idx].std().item()
        # 防止 std 为 0
        epsilon = feat_std if feat_std > 1e-6 else 1.0
    
    # 设置模型为评估模式
    model.eval()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 创建扰动数据（使用深拷贝）
        X_perturbed = X.clone()
        X_perturbed[:, confounder_idx] = X[:, confounder_idx] + epsilon

        # 计算原始 ITE
        ite_original = _compute_ite_from_model(model, X, treatment_col_idx)

        # 计算扰动后 ITE
        ite_perturbed = _compute_ite_from_model(model, X_perturbed, treatment_col_idx)

        # 计算敏感性分数: mean(|ITE_original - ITE_perturbed|)
        sensitivity = np.mean(np.abs(ite_original - ite_perturbed))
    
    return float(sensitivity)


def _get_prediction(
    model: Any,
    X: torch.Tensor
) -> np.ndarray:
    """
    内部辅助函数：从模型获取预测值。
    
    支持多种模型接口:
    1. DLCNet 风格: forward() 返回字典，使用 'Y_1'
    2. 标准 PyTorch: forward() 返回张量
    3. Sklearn 风格: predict_proba() 方法
    
    Args:
        model: 预测模型
        X: 输入特征张量 [N, D]
    
    Returns:
        np.ndarray: 预测概率或值 [N]
    """
    if hasattr(model, 'forward'):
        try:
            outputs = model(X)
            
            if isinstance(outputs, dict):
                # DLCNet 风格
                return outputs['Y_1'].squeeze().cpu().numpy()
            else:
                # 标准 PyTorch 模型
                return outputs.squeeze().cpu().numpy()
        except (KeyError, TypeError):
            pass
    
    if hasattr(model, 'predict_proba'):
        # Sklearn 风格
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        return model.predict_proba(X_np)[:, 1]
    
    raise TypeError("模型必须实现 forward() 或 predict_proba() 方法")


def _get_probability_prediction(
    model: Any,
    X: torch.Tensor
) -> np.ndarray:
    """
    获取概率预测值，用于 ITE 计算。
    """
    if hasattr(model, 'forward'):
        outputs = model(X)
        if isinstance(outputs, dict):
            if 'pred' in outputs:
                return torch.sigmoid(outputs['pred']).squeeze().cpu().numpy()
            if 'Y_1' in outputs:
                return outputs['Y_1'].squeeze().cpu().numpy()
        return torch.sigmoid(outputs).squeeze().cpu().numpy()

    if hasattr(model, 'predict_proba'):
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        return model.predict_proba(X_np)[:, 1]

    raise TypeError("模型必须实现 forward() 或 predict_proba() 方法")


def _compute_ite_from_model(
    model: Any,
    X: torch.Tensor,
    treatment_col_idx: int,
    high_val: float = 1.0,
    low_val: float = -1.0
) -> np.ndarray:
    X_treat = X.clone()
    X_control = X.clone()
    X_treat[:, treatment_col_idx] = float(high_val)
    X_control[:, treatment_col_idx] = float(low_val)

    prob_treat = _get_probability_prediction(model, X_treat)
    prob_control = _get_probability_prediction(model, X_control)
    return prob_treat - prob_control


def compute_ate(
    model: Any,
    X: torch.Tensor,
    treatment_col_idx: int,
    device: Optional[torch.device] = None
) -> float:
    """
    计算 ATE (Average Treatment Effect)。
    
    ATE 是所有样本 CATE 的平均值。
    
    Args:
        model: 预测模型
        X: 输入特征张量 [N, D]
        treatment_col_idx: 治疗列的索引
        device: 计算设备
    
    Returns:
        float: ATE = mean(CATE)
    
    Example:
        >>> ate = compute_ate(model, X, treatment_col_idx=2)
        >>> print(f"ATE: {ate:.4f}")
    """
    cate = compute_cate(model, X, treatment_col_idx, device)
    ate = float(np.mean(cate))
    return ate


def compute_att(
    model: Any,
    X: torch.Tensor,
    treatment_col_idx: int,
    device: Optional[torch.device] = None
) -> float:
    """
    计算 ATT (Average Treatment Effect on the Treated)。
    
    ATT 是治疗组（treatment=1）样本的 CATE 平均值。
    
    Args:
        model: 预测模型
        X: 输入特征张量 [N, D]
        treatment_col_idx: 治疗列的索引
        device: 计算设备
    
    Returns:
        float: ATT = mean(CATE | T=1)
    
    Example:
        >>> att = compute_att(model, X, treatment_col_idx=2)
        >>> print(f"ATT: {att:.4f}")
    """
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    # 获取治疗组掩码
    treated_mask = X[:, treatment_col_idx] > 0.5
    
    if treated_mask.sum() == 0:
        warnings.warn("没有治疗组样本，返回 NaN")
        return np.nan
    
    # 计算所有 CATE
    cate = compute_cate(model, X, treatment_col_idx, device)
    
    # 仅取治疗组的 CATE
    att = float(np.mean(cate[treated_mask.cpu().numpy()]))
    
    return att


# 模块元信息
__all__ = [
    'compute_pehe',
    'compute_cate',
    'compute_sensitivity_score',
    'compute_ate',
    'compute_att',
]

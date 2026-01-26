# -*- coding: utf-8 -*-
"""
DLC Hyperparameter Tuning Module
================================

基于 Optuna 的贝叶斯超参数优化模块。

使用高效的贝叶斯优化算法搜索 DLCNet 的最优超参数配置，
包括网络结构参数、损失权重和训练参数。

Functions:
    tune_dlc_hyperparameters: 执行超参数搜索并返回最优配置
    create_study: 创建 Optuna 研究对象
    objective: 目标函数，用于单次试验评估

满足需求: Phase 7 - Performance SOTA & Scientific Validation
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import logging

# 条件导入 Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "Optuna 未安装。请运行 'pip install optuna' 以启用超参数调优功能。"
    )

# 设置日志
logger = logging.getLogger(__name__)


def tune_dlc_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    study_name: str = "dlc_optimization",
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    random_state: int = 42,
    hsic_threshold: float = 0.05,
    n_epochs_per_trial: int = 15,
    verbose: bool = True,
    custom_search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    使用 Optuna 进行 DLCNet 贝叶斯超参数优化。
    
    搜索空间包括:
    - d_hidden: 隐藏层维度 [32, 256]
    - num_layers: 超图卷积层数 [1, 4]
    - lambda_hsic: HSIC 损失权重 (LogUniform) [0.1, 10.0]
    - lr: 学习率 (LogUniform) [1e-5, 1e-2]
    - batch_size: 批量大小 {32, 64, 128}
    
    Args:
        X_train: 训练特征矩阵 [N_train, D]
        y_train: 训练标签向量 [N_train]
        X_val: 验证特征矩阵 [N_val, D]
        y_val: 验证标签向量 [N_val]
        n_trials: 试验次数，默认 50
        study_name: 研究名称，用于日志和数据库存储
        timeout: 超时时间（秒），默认 None（不限时）
        n_jobs: 并行作业数，默认 1（单线程）
        random_state: 随机种子，默认 42
        hsic_threshold: HSIC 损失约束阈值，默认 0.05
                        若验证集 HSIC > 阈值，该试验被惩罚
        n_epochs_per_trial: 每次试验的训练轮数，默认 15
        verbose: 是否打印详细日志，默认 True
        custom_search_space: 自定义搜索空间字典，默认 None
    
    Returns:
        Dict 包含:
            - best_params: 最优超参数字典
            - best_score: 最优验证 AUC
            - study: Optuna Study 对象（可用于进一步分析）
            - n_trials_completed: 完成的试验数
            - optimization_history: 优化历史记录
    
    Raises:
        ImportError: 如果 Optuna 未安装
        ValueError: 如果输入数据格式不正确
    
    Example:
        >>> from src.dlc.tune import tune_dlc_hyperparameters
        >>> results = tune_dlc_hyperparameters(
        ...     X_train, y_train, X_val, y_val,
        ...     n_trials=30,
        ...     verbose=True
        ... )
        >>> print(f"Best AUC: {results['best_score']:.4f}")
        >>> print(f"Best params: {results['best_params']}")
    
    Note:
        - 使用 MedianPruner 进行早期停止不良试验
        - 使用 TPE (Tree-structured Parzen Estimator) 采样器
        - HSIC 约束确保因果解耦质量
    """
    # 检查 Optuna 是否可用
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna 未安装。请运行 'pip install optuna' 以启用超参数调优功能。"
        )
    
    # 输入验证
    _validate_inputs(X_train, y_train, X_val, y_val)
    
    # 设置随机种子
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    
    # 确定输入维度
    input_dim = X_train.shape[1]
    
    # 创建目标函数
    objective_fn = _create_objective(
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        input_dim=input_dim,
        hsic_threshold=hsic_threshold,
        n_epochs=n_epochs_per_trial,
        random_state=random_state,
        custom_search_space=custom_search_space
    )
    
    # 配置日志级别
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # 创建 Study
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(
        n_startup_trials=5,      # 前 5 次试验不剪枝
        n_warmup_steps=5,        # 前 5 个 epoch 不剪枝
        interval_steps=1         # 每个 epoch 检查一次
    )
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",    # 最大化 AUC
        sampler=sampler,
        pruner=pruner
    )
    
    # 打印优化开始信息
    if verbose:
        print("\n" + "=" * 60)
        print("DLCNet 超参数优化")
        print("=" * 60)
        print(f"试验次数: {n_trials}")
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"每试验训练轮数: {n_epochs_per_trial}")
        print(f"HSIC 约束阈值: {hsic_threshold}")
        print("=" * 60 + "\n")
    
    # 执行优化
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=verbose
    )
    
    # 提取结果
    best_params = study.best_params
    best_score = study.best_value
    
    # 收集优化历史
    optimization_history = []
    for trial in study.trials:
        optimization_history.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state)
        })
    
    # 打印最终结果
    if verbose:
        print("\n" + "=" * 60)
        print("优化完成!")
        print("=" * 60)
        print(f"完成试验数: {len(study.trials)}")
        print(f"最佳验证 AUC: {best_score:.4f}")
        print("最佳超参数:")
        for key, value in best_params.items():
            print(f"  - {key}: {value}")
        print("=" * 60 + "\n")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'study': study,
        'n_trials_completed': len(study.trials),
        'optimization_history': optimization_history,
        'scaler': scaler  # 返回 scaler 以便后续使用
    }


def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> None:
    """
    验证输入数据格式。
    
    Args:
        X_train, y_train, X_val, y_val: 训练和验证数据
    
    Raises:
        ValueError: 如果数据格式不正确
    """
    # 检查类型
    if not isinstance(X_train, np.ndarray):
        raise ValueError("X_train 必须是 numpy.ndarray")
    if not isinstance(y_train, np.ndarray):
        raise ValueError("y_train 必须是 numpy.ndarray")
    if not isinstance(X_val, np.ndarray):
        raise ValueError("X_val 必须是 numpy.ndarray")
    if not isinstance(y_val, np.ndarray):
        raise ValueError("y_val 必须是 numpy.ndarray")
    
    # 检查形状
    if X_train.ndim != 2:
        raise ValueError(f"X_train 必须是 2D 数组，收到 {X_train.ndim}D")
    if X_val.ndim != 2:
        raise ValueError(f"X_val 必须是 2D 数组，收到 {X_val.ndim}D")
    
    # 检查特征维度一致性
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"特征维度不一致: X_train {X_train.shape[1]} vs X_val {X_val.shape[1]}"
        )
    
    # 检查样本数一致性
    if len(X_train) != len(y_train):
        raise ValueError(
            f"训练集大小不一致: X_train {len(X_train)} vs y_train {len(y_train)}"
        )
    if len(X_val) != len(y_val):
        raise ValueError(
            f"验证集大小不一致: X_val {len(X_val)} vs y_val {len(y_val)}"
        )
    
    # 检查标签值
    unique_labels_train = np.unique(y_train)
    unique_labels_val = np.unique(y_val)
    
    if not np.all(np.isin(unique_labels_train, [0, 1])):
        raise ValueError(f"y_train 必须只包含 0 和 1，收到: {unique_labels_train}")
    if not np.all(np.isin(unique_labels_val, [0, 1])):
        raise ValueError(f"y_val 必须只包含 0 和 1，收到: {unique_labels_val}")


def _create_objective(
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    input_dim: int,
    hsic_threshold: float,
    n_epochs: int,
    random_state: int,
    custom_search_space: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    创建 Optuna 目标函数。
    
    Args:
        X_train_tensor, y_train_tensor: 训练数据张量
        X_val_tensor, y_val_tensor: 验证数据张量
        input_dim: 输入特征维度
        hsic_threshold: HSIC 约束阈值
        n_epochs: 训练轮数
        random_state: 随机种子
        custom_search_space: 自定义搜索空间
    
    Returns:
        Callable: Optuna 目标函数
    """
    # 延迟导入 DLCNet 以避免循环导入
    from src.dlc.dlc_net import DLCNet
    from src.dlc.causal_vae import CausalVAE
    
    # 默认搜索空间
    default_search_space = {
        'd_hidden': {'type': 'int', 'low': 32, 'high': 256, 'step': 32},
        'num_layers': {'type': 'int', 'low': 1, 'high': 4},
        'lambda_hsic': {'type': 'float', 'low': 0.1, 'high': 10.0, 'log': True},
        'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128]}
    }
    
    # 合并自定义搜索空间
    search_space = default_search_space.copy()
    if custom_search_space:
        search_space.update(custom_search_space)
    
    def objective(trial: optuna.Trial) -> float:
        """
        单次试验的目标函数。
        
        Args:
            trial: Optuna Trial 对象
        
        Returns:
            float: 验证集 AUC（若违反 HSIC 约束则返回 0.0）
        """
        # 采样超参数
        d_hidden = trial.suggest_int(
            'd_hidden',
            search_space['d_hidden']['low'],
            search_space['d_hidden']['high'],
            step=search_space['d_hidden'].get('step', 1)
        )
        
        num_layers = trial.suggest_int(
            'num_layers',
            search_space['num_layers']['low'],
            search_space['num_layers']['high']
        )
        
        lambda_hsic = trial.suggest_float(
            'lambda_hsic',
            search_space['lambda_hsic']['low'],
            search_space['lambda_hsic']['high'],
            log=search_space['lambda_hsic'].get('log', True)
        )
        
        lr = trial.suggest_float(
            'lr',
            search_space['lr']['low'],
            search_space['lr']['high'],
            log=search_space['lr'].get('log', True)
        )
        
        batch_size = trial.suggest_categorical(
            'batch_size',
            search_space['batch_size']['choices']
        )
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        try:
            model = DLCNet(
                input_dim=input_dim,
                d_hidden=d_hidden,
                num_layers=num_layers,
                lambda_hsic=lambda_hsic,
                random_state=random_state
            ).to(device)
        except Exception as e:
            logger.warning(f"模型创建失败: {e}")
            return 0.0
        
        # 创建 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 训练循环
        best_val_auc = 0.0
        
        for epoch in range(n_epochs):
            # 训练
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 前向传播
                outputs = model(X_batch)
                
                # 计算损失
                losses = model.compute_loss(X_batch, y_batch, outputs)
                loss = losses['loss_total']
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                X_val_device = X_val_tensor.to(device)
                y_val_device = y_val_tensor.to(device)
                
                outputs_val = model(X_val_device)
                
                # 计算验证集 HSIC
                val_hsic = CausalVAE.compute_hsic_loss(
                    outputs_val['Z_conf'],
                    outputs_val['Z_effect']
                ).item()
                
                # HSIC 约束检查
                if val_hsic > hsic_threshold:
                    # 违反约束，惩罚该试验
                    logger.info(
                        f"Trial {trial.number}: HSIC={val_hsic:.4f} > {hsic_threshold}, 惩罚"
                    )
                    return 0.0
                
                # 计算验证集 AUC
                y_pred = outputs_val['Y_1'].squeeze().cpu().numpy()
                y_true = y_val_tensor.numpy()
                
                try:
                    val_auc = roc_auc_score(y_true, y_pred)
                except ValueError:
                    # 处理单类别情况
                    val_auc = 0.5
                
                best_val_auc = max(best_val_auc, val_auc)
            
            # 报告中间结果（用于剪枝）
            trial.report(val_auc, epoch)
            
            # 检查是否应该剪枝
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_auc
    
    return objective


def quick_tune(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    快速超参数调优（便捷函数）。
    
    自动划分训练/验证集并执行超参数搜索。
    
    Args:
        X: 特征矩阵 [N, D]
        y: 标签向量 [N]
        n_trials: 试验次数，默认 20
        test_size: 验证集比例，默认 0.2
        random_state: 随机种子，默认 42
    
    Returns:
        Dict: 同 tune_dlc_hyperparameters 的返回值
    
    Example:
        >>> results = quick_tune(X, y, n_trials=10)
        >>> print(f"Best params: {results['best_params']}")
    """
    from sklearn.model_selection import train_test_split
    
    # 划分数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # 执行调优
    return tune_dlc_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=n_trials,
        random_state=random_state,
        verbose=True
    )


def get_default_params() -> Dict[str, Any]:
    """
    获取 DLCNet 的默认超参数。
    
    这些参数是根据初步实验确定的合理默认值。
    
    Returns:
        Dict: 默认超参数字典
    """
    return {
        'd_hidden': 64,
        'num_layers': 2,
        'lambda_hsic': 0.1,
        'lr': 1e-3,
        'batch_size': 64
    }


# 模块元信息
__all__ = [
    'tune_dlc_hyperparameters',
    'quick_tune',
    'get_default_params',
]

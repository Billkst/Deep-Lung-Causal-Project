"""
DLC (Deep Lung Causal) 模型模块

该模块实现了基于因果推断的肺癌预测模型,包含以下核心组件:
- CausalVAE: 因果变分自编码器,用于解耦混杂因素和效应因素
- DynamicHypergraphNN: 动态超图神经网络,用于建模基因交互
- DLCNet: 主模型,整合上述组件进行端到端训练和预测
- metrics: 因果推断评估指标 (PEHE, CATE, Sensitivity)
- tune: 贝叶斯超参数优化
"""

from .causal_vae import CausalVAE
from .hypergraph_nn import DynamicHypergraphNN
from .dlc_net import DLCNet
from .metrics import (
    compute_pehe,
    compute_cate,
    compute_sensitivity_score,
    compute_ate,
    compute_att
)

# 条件导入 tune 模块（需要 optuna）
try:
    from .tune import tune_dlc_hyperparameters, quick_tune, get_default_params
    _TUNE_AVAILABLE = True
except ImportError:
    _TUNE_AVAILABLE = False

__all__ = [
    # 核心模型
    'CausalVAE',
    'DynamicHypergraphNN',
    'DLCNet',
    # 因果指标
    'compute_pehe',
    'compute_cate',
    'compute_sensitivity_score',
    'compute_ate',
    'compute_att',
]

# 如果 tune 模块可用，添加到导出列表
if _TUNE_AVAILABLE:
    __all__.extend([
        'tune_dlc_hyperparameters',
        'quick_tune',
        'get_default_params',
    ])

__version__ = '1.1.0'

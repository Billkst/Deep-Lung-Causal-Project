"""
DLC (Deep Lung Causal) 模型模块

该模块实现了基于因果推断的肺癌预测模型,包含以下核心组件:
- CausalVAE: 因果变分自编码器,用于解耦混杂因素和效应因素
- DynamicHypergraphNN: 动态超图神经网络,用于建模基因交互
- DLCNet: 主模型,整合上述组件进行端到端训练和预测
"""

from .causal_vae import CausalVAE
# from .hypergraph_nn import DynamicHypergraphNN  # 待实现
# from .dlc_net import DLCNet  # 待实现

__all__ = [
    'CausalVAE',
    # 'DynamicHypergraphNN',
    # 'DLCNet',
]

__version__ = '1.0.0'

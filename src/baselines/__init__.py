"""
对比算法实验平台 (Baseline Comparison Platform)

该模块实现了多种经典机器学习、深度学习和因果推断算法，
作为 DLC 模型的性能基准对照。

所有算法必须首先在官方基准数据集上通过验证，确保实现正确性。
"""

from .utils import set_global_seed, preprocess_data

__all__ = [
    'set_global_seed',
    'preprocess_data',
]

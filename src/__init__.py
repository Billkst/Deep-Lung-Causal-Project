# -*- coding: utf-8 -*-
"""
DLC Data Engineering Module
===========================

Deep-Lung-Causal (DLC) 模型数据工程模块，负责处理 PANCAN 和 LUAD 数据集，
构建特征对齐的半合成验证数据集。

主要组件:
- DataCleaner: 数据清洗类
- SemiSyntheticGenerator: 半合成数据生成器
"""

from .data_processor import DataCleaner, SemiSyntheticGenerator

__all__ = ['DataCleaner', 'SemiSyntheticGenerator']
__version__ = '0.1.0'

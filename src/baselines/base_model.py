"""
对比算法抽象基类模块

本模块定义了所有对比算法的统一接口规范。
所有对比模型必须继承 BaseModel 类并实现所有抽象方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseModel(ABC):
    """
    对比算法抽象基类
    
    所有对比模型必须继承此类并实现所有抽象方法。
    
    Attributes:
        random_state (int): 随机种子，默认 42，用于确保结果可复现
        model: 具体的模型实例，由子类初始化
    
    Methods:
        fit(X, y): 训练模型
        predict(X): 预测标签
        predict_proba(X): 预测概率
        evaluate(X, y): 评估模型性能
        get_params(): 获取模型参数
        set_params(**params): 设置模型参数
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化模型
        
        Args:
            random_state (int): 随机种子，默认 42
        """
        self.random_state = random_state
        self.model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        训练模型
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            y (np.ndarray): 标签向量，shape (n_samples,)
            
        Returns:
            BaseModel: 训练后的模型实例（返回 self 以支持链式调用）
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            
        Returns:
            np.ndarray: 预测标签，shape (n_samples,)
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            
        Returns:
            np.ndarray: 预测概率，shape (n_samples, n_classes)
                       对于二分类问题，返回 shape (n_samples, 2)
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            y (np.ndarray): 真实标签，shape (n_samples,)
            
        Returns:
            Dict[str, float]: 评估指标字典，至少包含以下键：
                - 'accuracy': 准确率
                - 'precision': 精确率
                - 'recall': 召回率
                - 'f1': F1 分数
                - 'auc_roc': ROC 曲线下面积
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass

    def count_parameters(self) -> str:
        """
        计算模型参数量
        
        Returns:
            str: 参数量统计字符串
        """
        return "N/A"
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            Dict[str, Any]: 模型参数字典，至少包含 'random_state'
        """
        return {'random_state': self.random_state}
    
    def set_params(self, **params) -> 'BaseModel':
        """
        设置模型参数
        
        Args:
            **params: 要设置的参数键值对
            
        Returns:
            BaseModel: 返回 self 以支持链式调用
            
        Example:
            >>> model = SomeModel()
            >>> model.set_params(random_state=123, max_iter=1000)
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

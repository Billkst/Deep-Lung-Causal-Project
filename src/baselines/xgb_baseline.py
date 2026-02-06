"""
XGBoost 基线模型模块

论文: "XGBoost: A Scalable Tree Boosting System" (KDD 2016)
作者: Tianqi Chen, Carlos Guestrin
ArXiv: https://arxiv.org/abs/1603.02754
ACM DL: https://dl.acm.org/doi/10.1145/2939672.2939785

本模块实现了基于 XGBoost 的分类模型，作为对比算法之一。
模型内部集成 Early Stopping 机制，防止过拟合。

关键特性：
1. 内部验证集划分：从训练集中自动划分 15% 作为验证集
2. Early Stopping：使用内部验证集监控，严禁使用外部测试集
3. 可复现性：固定随机种子确保结果可重复

性能基准：UCI Breast Cancer Accuracy > 0.95

硬件要求：
- CPU: 多核 CPU（XGBoost 主要在 CPU 上运行）
- GPU: 可选，XGBoost 支持 GPU 加速（tree_method='gpu_hist'）
- 内存: 约 1-2 GB（取决于数据集大小）
- 显存使用: 极少（< 1 GB，如果使用 GPU 加速）

性能优化策略：
1. 多线程：设置 n_jobs=-1 使用所有 CPU 核心
2. GPU 加速：设置 tree_method='gpu_hist' 启用 GPU 训练
3. 树深度：调整 max_depth (默认 5) 平衡性能和过拟合
4. 学习率：调整 learning_rate (默认 0.1) 和 n_estimators 平衡训练速度
5. Early Stopping：设置 early_stopping_rounds (默认 10) 避免过度训练

使用示例：
```python
from src.baselines.xgb_baseline import XGBBaseline

# 基本使用
model = XGBBaseline(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# GPU 加速（可选）
model_gpu = XGBBaseline(
    random_state=42,
    tree_method='gpu_hist'  # 启用 GPU 加速
)
model_gpu.fit(X_train, y_train)
```
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)
from typing import Dict
from .base_model import BaseModel


class XGBBaseline(BaseModel):
    """
    XGBoost 基线模型
    
    使用 XGBoost 算法进行二分类任务，内部集成 Early Stopping 机制。
    
    Attributes:
        random_state (int): 随机种子，默认 42
        n_estimators (int): 最大迭代次数，默认 100
        learning_rate (float): 学习率，默认 0.1
        max_depth (int): 树的最大深度，默认 6
        model (xgb.XGBClassifier): XGBoost 分类器实例
    
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> 
        >>> # 加载数据
        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42, stratify=y
        ... )
        >>> 
        >>> # 训练模型
        >>> model = XGBBaseline(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # 评估模型
        >>> metrics = model.evaluate(X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    
    def __init__(
        self, 
        random_state: int = 42, 
        n_estimators: int = 200, 
        learning_rate: float = 0.1, 
        max_depth: int = 5,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        """
        初始化 XGBoost 模型
        
        Args:
            random_state (int): 随机种子，默认 42
            n_estimators (int): 最大迭代次数，默认 200
            learning_rate (float): 学习率，默认 0.1
            max_depth (int): 树的最大深度，默认 5
            min_child_weight (int): 最小子节点权重，默认 1
            subsample (float): 样本采样比例，默认 0.8
            colsample_bytree (float): 特征采样比例，默认 0.8
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBBaseline':
        """
        训练 XGBoost 模型
        
        内部自动划分 15% 作为验证集用于 Early Stopping。
        严禁使用外部测试集，确保防止数据泄露。
        
        Args:
            X (np.ndarray): 训练集特征矩阵，shape (n_samples, n_features)
            y (np.ndarray): 训练集标签向量，shape (n_samples,)
            
        Returns:
            XGBBaseline: 训练后的模型实例
            
        Note:
            - 内部验证集从训练集中划分，占比 15%
            - 使用 Stratified Split 保持类别比例
            - Early Stopping rounds 设置为 10
            - 评估指标使用 logloss
        """
        # 内部验证集划分 (15%)
        # 使用 stratify 参数确保训练集和验证集的类别比例一致
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.15, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # 初始化 XGBoost 分类器
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric='logloss',  # 使用 log loss 作为评估指标
            early_stopping_rounds=10  # Early Stopping 轮数
        )
        
        # 使用内部验证集进行 Early Stopping
        # 严禁使用外部测试集
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False  # 关闭训练过程输出
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            
        Returns:
            np.ndarray: 预测标签，shape (n_samples,)
            
        Raises:
            ValueError: 如果模型尚未训练
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            
        Returns:
            np.ndarray: 预测概率，shape (n_samples, 2)
                       第一列为类别 0 的概率，第二列为类别 1 的概率
            
        Raises:
            ValueError: 如果模型尚未训练
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        计算多个分类指标，包括准确率、精确率、召回率、F1 分数和 AUC-ROC。
        
        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            y (np.ndarray): 真实标签，shape (n_samples,)
            
        Returns:
            Dict[str, float]: 评估指标字典，包含：
                - 'accuracy': 准确率
                - 'precision': 精确率（二分类）
                - 'recall': 召回率（二分类）
                - 'f1': F1 分数（二分类）
                - 'auc_roc': ROC 曲线下面积
            
        Raises:
            ValueError: 如果模型尚未训练
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        # 预测标签和概率
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # 取正类（类别 1）的概率
        
        # 计算评估指标
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y, y_proba)
        }

    def count_parameters(self) -> str:
        """
        计算模型参数量 (估算树的总节点数)
        """
        try:
            if self.model is None:
                return "Not Trained"
            # Get number of nodes as proxy for parameters
            # Each node has a split condition (feature + threshold) or leaf value
            df = self.model.get_booster().trees_to_dataframe()
            n_nodes = df.shape[0]
            return f"{n_nodes} (Nodes)"
        except Exception:
            return "N/A (Tree Ensemble)"
    
    def get_params(self) -> Dict[str, any]:
        """
        获取模型参数
        
        Returns:
            Dict[str, any]: 模型参数字典
        """
        return {
            'random_state': self.random_state,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree
        }


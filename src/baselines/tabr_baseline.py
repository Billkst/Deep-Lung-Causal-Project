"""
TabR (Tabular Deep Learning via Retrieval) Baseline Model

论文: "TabR: Tabular Deep Learning Meets Nearest Neighbors" (ICLR 2024)
ArXiv: https://arxiv.org/abs/2307.14338
OpenReview: https://openreview.net/forum?id=rhgIgTSSxW

核心思想：
使用检索增强机制，从训练集中检索相似样本作为上下文，提升表格数据的预测性能。

关键组件：
1. Feature Encoder: 编码输入特征
2. Context Retrieval: k-NN 检索相似样本
3. Transformer Encoder: 处理查询和上下文的交互
4. Prediction Head: 输出最终预测

性能基准：UCI Breast Cancer Accuracy > 0.95

硬件要求：
- GPU: 单卡 RTX 3090 (24GB) 或更高
- 推荐批量大小: 32-64
- 显存使用: 约 2-4 GB（取决于数据集大小和模型配置）

性能优化策略：
1. 批量大小调整：如果显存不足，减小 batch_size 到 16 或 8
2. 梯度累积：使用 GradientAccumulationTrainer 保持有效批量大小
3. 模型配置：减小 hidden_dim (128 -> 64) 或 n_layers (2 -> 1)
4. 混合精度训练：使用 torch.cuda.amp 进行 FP16 训练
5. 检索池优化：减小 k_neighbors (5 -> 3) 降低计算开销

使用示例：
```python
from src.baselines.utils import GPUMemoryMonitor

# 使用显存监控
with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
    model = TabRBaseline(
        random_state=42,
        k_neighbors=5,
        hidden_dim=128,
        batch_size=32
    )
    model.fit(X_train, y_train)
    monitor.check_memory(context="TabR Training")
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict
import logging

from .base_model import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabRNet(nn.Module):
    """
    TabR 网络架构
    
    组件：
    1. Feature Encoder: 编码特征到隐藏空间
    2. Transformer Encoder: 处理查询和上下文的交互
    3. Prediction Head: 输出分类预测
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        """
        初始化 TabR 网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            n_heads: Transformer 注意力头数
            n_layers: Transformer 层数
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer 编码器（用于处理上下文）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 二分类
        )
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询样本 (batch_size, input_dim)
            context: 上下文样本 (batch_size, k, input_dim)
            
        Returns:
            logits: 预测 logits (batch_size, 2)
        """
        batch_size, k, _ = context.shape
        
        # 编码查询和上下文
        query_emb = self.feature_encoder(query)  # (batch_size, hidden_dim)
        context_emb = self.feature_encoder(context.view(-1, context.size(-1)))  # (batch_size*k, hidden_dim)
        context_emb = context_emb.view(batch_size, k, -1)  # (batch_size, k, hidden_dim)
        
        # 拼接查询和上下文
        combined = torch.cat([query_emb.unsqueeze(1), context_emb], dim=1)  # (batch_size, k+1, hidden_dim)
        
        # Transformer 处理
        attended = self.transformer(combined)  # (batch_size, k+1, hidden_dim)
        
        # 使用查询位置的输出进行预测
        query_output = attended[:, 0, :]  # (batch_size, hidden_dim)
        
        # 预测
        logits = self.prediction_head(query_output)
        
        return logits


class TabRBaseline(BaseModel):
    """
    TabR 基线模型 (2024 ICLR)
    
    检索增强表格学习模型
    性能基准：UCI Breast Cancer Accuracy > 0.95
    
    使用方法：
    ```python
    model = TabRBaseline(random_state=42, k_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
    ```
    """
    
    def __init__(self, random_state: int = 42, k_neighbors: int = 5, 
                 hidden_dim: int = 128, n_heads: int = 4, n_layers: int = 2,
                 epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
                 patience: int = 10):
        """
        初始化 TabR 模型
        
        Args:
            random_state: 随机种子
            k_neighbors: 检索的上下文样本数量
            hidden_dim: 隐藏层维度
            n_heads: Transformer 注意力头数
            n_layers: Transformer 层数
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            patience: Early Stopping 的耐心值
        """
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 检索池和模型
        self.context_X = None
        self.context_y = None
        self.knn = None
        self.model = None
    
    def _build_context_set(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        构建检索池 (Context Set)
        
        使用 k-NN 为每个样本检索最相似的 k 个训练样本
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        self.context_X = X_train
        self.context_y = y_train
        
        # 构建 k-NN 索引
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='euclidean')
        self.knn.fit(X_train)
        
        logger.info(f"检索池构建完成：{len(X_train)} 个样本，k={self.k_neighbors}")
    
    def _retrieve_context(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        为查询样本检索上下文
        
        Args:
            X: 查询样本 (n_samples, n_features)
            
        Returns:
            context: 上下文样本 (n_samples, k, n_features)
            context_labels: 上下文标签 (n_samples, k)
        """
        if self.knn is None:
            raise ValueError("检索池尚未构建，请先调用 fit() 方法")
        
        # 检索最近邻（排除自身）
        distances, indices = self.knn.kneighbors(X)
        indices = indices[:, 1:]  # 排除第一个（自身）
        
        # 获取上下文样本和标签
        context = self.context_X[indices]  # (n_samples, k, n_features)
        context_labels = self.context_y[indices]  # (n_samples, k)
        
        return context, context_labels
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TabRBaseline':
        """
        训练 TabR 模型
        
        内部自动划分 15% 作为验证集用于 Early Stopping
        
        Args:
            X: 训练集特征 (n_samples, n_features)
            y: 训练集标签 (n_samples,)
            
        Returns:
            self: 训练后的模型
        """
        logger.info("开始训练 TabR 模型...")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 内部验证集划分 (15%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.15, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"数据划分：训练集 {len(X_train)} 样本，验证集 {len(X_val)} 样本")
        
        # 构建检索池
        self._build_context_set(X_train, y_train)
        
        # 初始化模型
        input_dim = X_train.shape[1]
        self.model = TabRNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers
        ).to(self.device)
        
        # 优化器和学习率调度器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs), desc="Training TabR")
        
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            # 批量训练
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                
                # 检索上下文
                context, _ = self._retrieve_context(batch_X)
                
                # 转换为 Tensor
                batch_X_t = torch.FloatTensor(batch_X).to(self.device)
                batch_y_t = torch.LongTensor(batch_y).to(self.device)
                context_t = torch.FloatTensor(context).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                logits = self.model(batch_X_t, context_t)
                loss = criterion(logits, batch_y_t)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # 验证
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i:i+self.batch_size]
                    batch_y = y_val[i:i+self.batch_size]
                    
                    # 检索上下文
                    context, _ = self._retrieve_context(batch_X)
                    
                    # 转换为 Tensor
                    batch_X_t = torch.FloatTensor(batch_X).to(self.device)
                    batch_y_t = torch.LongTensor(batch_y).to(self.device)
                    context_t = torch.FloatTensor(context).to(self.device)
                    
                    # 前向传播
                    logits = self.model(batch_X_t, context_t)
                    loss = criterion(logits, batch_y_t)
                    
                    val_loss += loss.item()
                    n_val_batches += 1
            
            val_loss /= n_val_batches
            
            # 更新进度条
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                logger.info(f"Early Stopping 触发，在 Epoch {epoch+1} 停止训练")
                break
        
        pbar.close()
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        logger.info("TabR 模型训练完成")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            X: 测试集特征 (n_samples, n_features)
            
        Returns:
            predictions: 预测标签 (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 测试集特征 (n_samples, n_features)
            
        Returns:
            probabilities: 预测概率 (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        X_scaled = self.scaler.transform(X)
        
        # 检索上下文
        context, _ = self._retrieve_context(X_scaled)
        
        # 转换为 Tensor
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        context_t = torch.FloatTensor(context).to(self.device)
        
        # 预测
        self.model.eval()
        all_proba = []
        
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch_X = X_t[i:i+self.batch_size]
                batch_context = context_t[i:i+self.batch_size]
                
                logits = self.model(batch_X, batch_context)
                proba = F.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())
        
        return np.vstack(all_proba)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 测试集特征
            y: 测试集标签
            
        Returns:
            metrics: 包含各项指标的字典
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, average='binary', zero_division=0),
            'auc_roc': roc_auc_score(y, y_proba)
        }
        
        return metrics

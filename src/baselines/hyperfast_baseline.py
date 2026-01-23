"""
HyperFast 基线模型 (AAAI 2024)

论文: "HyperFast: Instant Classification for Tabular Data" (AAAI 2024)
ArXiv: https://arxiv.org/abs/2402.14335
OpenReview: https://openreview.net/forum?id=VRBhaU8IDz

核心思想: 使用 Hypernetwork 动态生成任务特定的分类器权重，实现快速推理。

性能基准: UCI Breast Cancer Accuracy > 0.93

硬件要求：
- GPU: 单卡 RTX 3090 (24GB) 或更高
- 推荐批量大小: 32-64
- 显存使用: 约 1-3 GB（取决于 hidden_dim 配置）

性能优化策略：
1. 批量大小调整：如果显存不足，减小 batch_size 到 16 或 8
2. 梯度累积：使用 GradientAccumulationTrainer 保持有效批量大小
3. Hypernetwork 配置：减小 hidden_dim (256 -> 128) 降低参数量
4. 混合精度训练：使用 torch.cuda.amp 进行 FP16 训练
5. 权重生成优化：缓存数据集统计信息，避免重复计算

使用示例：
```python
from src.baselines.utils import GPUMemoryMonitor, GradientAccumulationTrainer

# 使用显存监控和梯度累积
with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
    model = HyperFastBaseline(
        random_state=42,
        hidden_dim=256,
        batch_size=32
    )
    model.fit(X_train, y_train)
    monitor.check_memory(context="HyperFast Training")
```
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict

from .base_model import BaseModel


class Hypernetwork(nn.Module):
    """
    Hypernetwork: 生成任务特定的分类器权重
    
    输入：数据集统计信息（均值、标准差、特征维度等）
    输出：分类器的权重参数
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        初始化 Hypernetwork
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 统计信息编码器
        # 输入: [mean (input_dim), std (input_dim), n_features (1), n_samples (1)]
        stat_dim = input_dim * 2 + 2
        self.stat_encoder = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 权重生成器（生成分类器的所有层权重）
        # fc1: input_dim -> 128
        # fc2: 128 -> 64
        # fc3: 64 -> 2 (二分类)
        self.weight_generator = nn.ModuleDict({
            'fc1_weight': nn.Linear(hidden_dim, input_dim * 128),
            'fc1_bias': nn.Linear(hidden_dim, 128),
            'fc2_weight': nn.Linear(hidden_dim, 128 * 64),
            'fc2_bias': nn.Linear(hidden_dim, 64),
            'fc3_weight': nn.Linear(hidden_dim, 64 * 2),
            'fc3_bias': nn.Linear(hidden_dim, 2)
        })
    
    def forward(self, dataset_stats):
        """
        前向传播：生成分类器权重
        
        Args:
            dataset_stats: 数据集统计信息 (batch_size, stat_dim)
            
        Returns:
            weights: 分类器权重字典
        """
        # 编码统计信息
        stat_emb = self.stat_encoder(dataset_stats)
        
        # 生成分类器权重
        weights = {}
        for name, generator in self.weight_generator.items():
            weights[name] = generator(stat_emb)
        
        return weights


class DynamicClassifier(nn.Module):
    """
    动态分类器：使用 Hypernetwork 生成的权重进行分类
    
    支持批量推理
    """
    
    def __init__(self, input_dim: int):
        """
        初始化动态分类器
        
        Args:
            input_dim: 输入特征维度
        """
        super().__init__()
        self.input_dim = input_dim
    
    def forward(self, x, weights):
        """
        前向传播：使用动态权重进行分类
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            weights: Hypernetwork 生成的权重字典
            
        Returns:
            logits: 分类 logits (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # 第一层: input_dim -> 128
        w1 = weights['fc1_weight'].view(-1, self.input_dim, 128)
        b1 = weights['fc1_bias'].view(-1, 128)
        
        # 处理批量推理：如果 weights 是单个样本，需要扩展
        if w1.size(0) == 1 and batch_size > 1:
            w1 = w1.expand(batch_size, -1, -1)
            b1 = b1.expand(batch_size, -1)
        
        h1 = torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1
        h1 = F.relu(h1)
        
        # 第二层: 128 -> 64
        w2 = weights['fc2_weight'].view(-1, 128, 64)
        b2 = weights['fc2_bias'].view(-1, 64)
        
        if w2.size(0) == 1 and batch_size > 1:
            w2 = w2.expand(batch_size, -1, -1)
            b2 = b2.expand(batch_size, -1)
        
        h2 = torch.bmm(h1.unsqueeze(1), w2).squeeze(1) + b2
        h2 = F.relu(h2)
        
        # 输出层: 64 -> 2
        w3 = weights['fc3_weight'].view(-1, 64, 2)
        b3 = weights['fc3_bias'].view(-1, 2)
        
        if w3.size(0) == 1 and batch_size > 1:
            w3 = w3.expand(batch_size, -1, -1)
            b3 = b3.expand(batch_size, -1)
        
        logits = torch.bmm(h2.unsqueeze(1), w3).squeeze(1) + b3
        
        return logits


class HyperFastBaseline(BaseModel):
    """
    HyperFast 基线模型 (2024 NeurIPS)
    
    基于 Hypernetwork 的快速推理模型
    性能基准：UCI Breast Cancer Accuracy > 0.93
    
    硬件要求：单卡 RTX 3090 (24GB)
    推荐批量大小：32-64
    """
    
    def __init__(self, random_state: int = 42, hidden_dim: int = 256, 
                 epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
                 threshold: float = 0.5, class_weights=None, prediction_threshold: float = None):
        """
        初始化 HyperFast 模型
        
        Args:
            random_state: 随机种子
            hidden_dim: Hypernetwork 隐藏层维度
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            threshold: 分类阈值（默认 0.5，可调整以应对类别不平衡）
            class_weights: 类别权重列表 [weight_class_0, weight_class_1]，用于处理类别不平衡
            prediction_threshold: 预测阈值（如果提供，会覆盖 threshold 参数）
        """
        super().__init__(random_state)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 处理阈值参数（prediction_threshold 优先级更高）
        if prediction_threshold is not None:
            self.threshold = prediction_threshold
        else:
            self.threshold = threshold
        
        # 处理类别权重参数
        if class_weights is not None:
            if isinstance(class_weights, list):
                self.class_weights = torch.FloatTensor(class_weights)
            elif isinstance(class_weights, np.ndarray):
                self.class_weights = torch.FloatTensor(class_weights)
            elif isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights
            else:
                raise ValueError(f"class_weights 必须是 list, np.ndarray 或 torch.Tensor，当前类型：{type(class_weights)}")
        else:
            self.class_weights = None
        
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        
        # 模型组件
        self.hypernetwork = None
        self.classifier = None
        self.dataset_stats = None
    
    def _compute_dataset_stats(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """
        计算数据集统计信息
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            
        Returns:
            stats: 统计信息 Tensor (1, stat_dim)
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        n_features = float(X.shape[1])
        n_samples = float(X.shape[0])
        
        # 拼接统计信息: [mean, std, n_features, n_samples]
        stats = np.concatenate([mean, std, [n_features, n_samples]])
        return torch.FloatTensor(stats).unsqueeze(0).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HyperFastBaseline':
        """
        训练 HyperFast 模型
        
        内部自动划分 15% 作为验证集用于 Early Stopping
        严禁使用外部测试集
        
        Args:
            X: 训练集特征 (n_samples, n_features)
            y: 训练集标签 (n_samples,)
            
        Returns:
            self
        """
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 内部验证集划分 (15%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.15, random_state=self.random_state, stratify=y
        )
        
        # 计算类别权重（应对类别不平衡）
        # 如果外部已提供类别权重，则使用外部权重；否则自动计算
        if self.class_weights is None:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights_array = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            self.class_weights = torch.FloatTensor(class_weights_array)
            print(f"[HyperFast] 自动计算类别权重: {class_weights_array}")
        else:
            print(f"[HyperFast] 使用外部提供的类别权重: {self.class_weights.cpu().numpy()}")
        
        # 确保类别权重在正确的设备上
        self.class_weights = self.class_weights.to(self.device)
        
        print(f"[HyperFast] 类别分布: {np.bincount(y_train)}")
        
        # 初始化 Hypernetwork 和分类器
        input_dim = X_train.shape[1]
        self.hypernetwork = Hypernetwork(input_dim, self.hidden_dim).to(self.device)
        self.classifier = DynamicClassifier(input_dim).to(self.device)
        
        # 计算数据集统计信息
        self.dataset_stats = self._compute_dataset_stats(X_train, y_train)
        
        # 优化器
        optimizer = torch.optim.Adam(
            list(self.hypernetwork.parameters()) + list(self.classifier.parameters()),
            lr=self.learning_rate
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 损失函数（使用类别权重）
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs), desc="Training HyperFast")
        
        for epoch in pbar:
            # 训练模式
            self.hypernetwork.train()
            self.classifier.train()
            
            # 批量训练
            n_batches = int(np.ceil(len(X_train) / self.batch_size))
            train_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_train))
                
                X_batch = torch.FloatTensor(X_train[start_idx:end_idx]).to(self.device)
                y_batch = torch.LongTensor(y_train[start_idx:end_idx]).to(self.device)
                
                # 前向传播
                weights = self.hypernetwork(self.dataset_stats)
                logits = self.classifier(X_batch, weights)
                loss = criterion(logits, y_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= n_batches
            
            # 验证
            self.hypernetwork.eval()
            self.classifier.eval()
            
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                y_val_t = torch.LongTensor(y_val).to(self.device)
                
                weights = self.hypernetwork(self.dataset_stats)
                logits = self.classifier(X_val_t, weights)
                val_loss = criterion(logits, y_val_t).item()
            
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
                # 保存最佳模型状态
                self.best_hypernetwork_state = self.hypernetwork.state_dict()
                self.best_classifier_state = self.classifier.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # 恢复最佳模型
                    self.hypernetwork.load_state_dict(self.best_hypernetwork_state)
                    self.classifier.load_state_dict(self.best_classifier_state)
                    print(f"\n[HyperFast] Early stopping at epoch {epoch+1}")
                    break
        
        pbar.close()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            y_pred: 预测标签 (n_samples,)
        """
        if self.hypernetwork is None or self.classifier is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        proba = self.predict_proba(X)
        # 使用可调整的阈值（默认 0.5，可通过 threshold 参数调整）
        return (proba[:, 1] > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            proba: 预测概率 (n_samples, 2)
        """
        if self.hypernetwork is None or self.classifier is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        # 使用 Hypernetwork 生成权重
        self.hypernetwork.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            weights = self.hypernetwork(self.dataset_stats)
            logits = self.classifier(X_t, weights)
            proba = F.softmax(logits, dim=1)
        
        return proba.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples,)
            
        Returns:
            metrics: 评估指标字典
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'auc_roc': roc_auc_score(y, y_proba)
        }

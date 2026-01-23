"""
MOGONET 多组学图网络基线模型

论文: "MOGONET integrates multi-omics data using graph convolutional networks 
       allowing patient classification and biomarker identification"
期刊: Nature Communications, Volume 12, Article 3445 (2021)
DOI: https://doi.org/10.1038/s41467-021-23774-w
GitHub: https://github.com/txWang/MOGONET

用途: 处理多视图数据（mRNA, DNA, miRNA）的图网络模型
性能基准: ROSMAP Accuracy > 0.80

硬件要求：
- GPU: 单卡 RTX 3090 (24GB) 或更高
- 推荐批量大小: 16-32（图网络显存占用较大）
- 显存使用: 约 4-8 GB（取决于图的大小和模型配置）

性能优化策略：
1. 批量大小调整：图网络显存占用大，建议使用较小的 batch_size (16-32)
2. 梯度累积：使用 GradientAccumulationTrainer 保持有效批量大小
3. 图构建优化：减小邻接矩阵的稠密度，使用稀疏图表示
4. 视图融合：优化多视图融合策略，减少中间特征维度
5. 混合精度训练：使用 torch.cuda.amp 进行 FP16 训练

使用示例：
```python
from src.baselines.mogonet_baseline import ROSMAPDataset, MOGONETBaseline
from src.baselines.utils import GPUMemoryMonitor

# 加载数据
dataset = ROSMAPDataset()
views, labels = dataset.load_data()

# 使用显存监控
with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
    model = MOGONETBaseline(random_state=42)
    model.fit(views, labels)
    monitor.check_memory(context="MOGONET Training")
```
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from src.baselines.base_model import BaseModel


class ROSMAPDataset:
    """
    ROSMAP 多组学数据集加载器
    
    负责读取 ROSMAP 目录下的多视图数据并构建图结构
    
    数据文件结构:
    - 1_tr.csv, 1_te.csv: 视图 1 (mRNA)
    - 2_tr.csv, 2_te.csv: 视图 2 (DNA)
    - 3_tr.csv, 3_te.csv: 视图 3 (miRNA)
    - labels_tr.csv, labels_te.csv: 标签
    
    使用方法:
        dataset = ROSMAPDataset()
        views, labels = dataset.load_data()
    """
    
    def __init__(self, data_dir: str = "data/baselines_official/ROSMAP/"):
        """
        初始化 ROSMAP 数据集加载器
        
        Args:
            data_dir: ROSMAP 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self._check_files_exist()
    
    def _check_files_exist(self):
        """
        检查必需的数据文件是否存在
        
        如果任何文件缺失，抛出 FileNotFoundError 并提供下载指引
        
        Raises:
            FileNotFoundError: 当任何必需文件不存在时
        """
        required_files = [
            '1_tr.csv', '1_te.csv',
            '2_tr.csv', '2_te.csv',
            '3_tr.csv', '3_te.csv',
            'labels_tr.csv', 'labels_te.csv'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = self.data_dir / file
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            error_message = (
                f"ROSMAP 数据文件缺失。请下载以下文件：\n" +
                "\n".join(missing_files) +
                "\n\n下载指引：\n" +
                "1. 访问 ROSMAP 官方数据源获取多组学数据\n" +
                "2. 将文件放置在 data/baselines_official/ROSMAP/ 目录下\n" +
                "3. 确保文件名与上述列表完全一致"
            )
            raise FileNotFoundError(error_message)
    
    def load_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        加载并合并 ROSMAP 数据
        
        强制要求：使用 pd.concat 合并 _tr.csv 和 _te.csv 文件
        
        处理流程:
        1. 读取三个视图的训练集和测试集
        2. 使用 pd.concat 合并训练集和测试集
        3. 确保样本顺序一致
        4. 返回三个视图的特征矩阵和标签向量
        
        Returns:
            views: 三个视图的特征矩阵列表 [view1, view2, view3]
                   每个视图的形状为 (n_samples, n_features_i)
            labels: 标签向量，形状为 (n_samples,)
        
        Raises:
            FileNotFoundError: 当数据文件不存在时（由 _check_files_exist 抛出）
        """
        # 读取视图 1 (mRNA) - 不使用 index_col，数据文件没有样本 ID 列
        view1_tr = pd.read_csv(self.data_dir / '1_tr.csv')
        view1_te = pd.read_csv(self.data_dir / '1_te.csv')
        view1 = pd.concat([view1_tr, view1_te], axis=0, ignore_index=True)
        
        # 读取视图 2 (DNA)
        view2_tr = pd.read_csv(self.data_dir / '2_tr.csv')
        view2_te = pd.read_csv(self.data_dir / '2_te.csv')
        view2 = pd.concat([view2_tr, view2_te], axis=0, ignore_index=True)
        
        # 读取视图 3 (miRNA)
        view3_tr = pd.read_csv(self.data_dir / '3_tr.csv')
        view3_te = pd.read_csv(self.data_dir / '3_te.csv')
        view3 = pd.concat([view3_tr, view3_te], axis=0, ignore_index=True)
        
        # 读取标签 - 标签数据在第一列（作为索引读取）
        labels_tr = pd.read_csv(self.data_dir / 'labels_tr.csv', index_col=0)
        labels_te = pd.read_csv(self.data_dir / 'labels_te.csv', index_col=0)
        
        # 标签在索引中，需要提取索引作为标签值
        labels_tr_values = labels_tr.index.values
        labels_te_values = labels_te.index.values
        labels = np.concatenate([labels_tr_values, labels_te_values])
        
        # 确保所有视图和标签的样本数一致
        n_samples = len(labels)
        assert view1.shape[0] == n_samples, f"视图 1 样本数 {view1.shape[0]} 与标签数 {n_samples} 不一致"
        assert view2.shape[0] == n_samples, f"视图 2 样本数 {view2.shape[0]} 与标签数 {n_samples} 不一致"
        assert view3.shape[0] == n_samples, f"视图 3 样本数 {view3.shape[0]} 与标签数 {n_samples} 不一致"
        
        # 转换为 numpy 数组
        view1 = view1.values
        view2 = view2.values
        view3 = view3.values
        labels = labels.astype(int)
        
        return [view1, view2, view3], labels


class MOGONETBaseline(BaseModel):
    """
    MOGONET 多组学图网络基线模型
    
    论文: Multi-Omics Graph Convolutional Networks for Survival Prediction
    实现多视图图卷积网络用于多组学数据融合
    性能基准：ROSMAP Accuracy > 0.80
    
    架构:
    - 视图特定的图卷积网络 (View-specific GCN)
    - 跨视图注意力融合 (Cross-view Attention Fusion)
    - 分类预测头 (Classification Head)
    """
    
    def __init__(self, random_state: int = 42, hidden_dim: int = 128, 
                 n_gcn_layers: int = 3, epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 0.001, k_neighbors: int = 10,
                 scaler_type: str = 'robust'):
        """
        初始化 MOGONET 模型
        
        Args:
            random_state: 随机种子，默认 42
            hidden_dim: 隐藏层维度（优化：从 64 提升到 128）
            n_gcn_layers: GCN 层数（优化：从 2 提升到 3）
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            k_neighbors: 构建图时的 k 近邻数量（可调整：5, 10, 15）
            scaler_type: 特征归一化方法 ('standard', 'minmax', 'robust')
        """
        super().__init__(random_state)
        self.hidden_dim = hidden_dim
        self.n_gcn_layers = n_gcn_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.k_neighbors = k_neighbors
        self.scaler_type = scaler_type
        
        # 设置随机种子
        import torch
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scalers = []
    
    def _get_scaler(self):
        """
        根据配置返回相应的特征缩放器
        
        Returns:
            scaler: sklearn 缩放器实例
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:  # 'standard'
            return StandardScaler()
    
    def _build_knn_graph(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于 k-NN 构建图结构
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
        
        Returns:
            edge_index: 边索引 (2, n_edges)
            edge_weight: 边权重 (n_edges,)
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics.pairwise import cosine_similarity
        
        n_samples = len(features)
        # 确保 k_neighbors 不超过样本数
        k = min(self.k_neighbors, n_samples - 1)
        
        if k <= 0:
            # 如果样本数太少，返回空图
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)
        
        # 构建 k-NN 图
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nbrs.fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        # 构建边列表
        edge_list = []
        edge_weights = []
        
        for i in range(len(features)):
            for j, neighbor_idx in enumerate(indices[i][1:]):  # 跳过自身
                edge_list.append([i, neighbor_idx])
                # 使用余弦相似度作为边权重
                similarity = 1 - distances[i][j + 1]
                edge_weights.append(max(similarity, 0))  # 确保非负
        
        edge_index = np.array(edge_list).T
        edge_weight = np.array(edge_weights)
        
        return edge_index, edge_weight
    
    def _create_mogonet_model(self, view_dims: List[int], n_classes: int):
        """
        创建 MOGONET 模型
        
        Args:
            view_dims: 每个视图的特征维度列表
            n_classes: 类别数量
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        
        class ViewSpecificGCN(nn.Module):
            """视图特定的图卷积网络"""
            def __init__(self, input_dim, hidden_dim, n_layers):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(input_dim, hidden_dim))
                for _ in range(n_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x, edge_index, edge_weight=None):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index, edge_weight)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                        x = self.dropout(x)
                return x
        
        class MOGONETModel(nn.Module):
            """MOGONET 完整模型"""
            def __init__(self, view_dims, hidden_dim, n_gcn_layers, n_classes):
                super().__init__()
                self.n_views = len(view_dims)
                
                # 为每个视图创建 GCN
                self.view_gcns = nn.ModuleList([
                    ViewSpecificGCN(view_dim, hidden_dim, n_gcn_layers)
                    for view_dim in view_dims
                ])
                
                # 跨视图注意力融合
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=4,
                    dropout=0.1,
                    batch_first=True
                )
                
                # 分类头
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, n_classes)
                )
            
            def forward(self, view_features, edge_indices, edge_weights):
                # 对每个视图应用 GCN
                view_embeddings = []
                for i, gcn in enumerate(self.view_gcns):
                    emb = gcn(view_features[i], edge_indices[i], edge_weights[i])
                    view_embeddings.append(emb)
                
                # 堆叠视图嵌入 (n_samples, n_views, hidden_dim)
                stacked_embeddings = torch.stack(view_embeddings, dim=1)
                
                # 跨视图注意力融合
                fused_embedding, _ = self.attention(
                    stacked_embeddings, 
                    stacked_embeddings, 
                    stacked_embeddings
                )
                
                # 平均池化
                pooled = fused_embedding.mean(dim=1)  # (n_samples, hidden_dim)
                
                # 分类
                logits = self.classifier(pooled)
                
                return logits
        
        self.model = MOGONETModel(
            view_dims, 
            self.hidden_dim, 
            self.n_gcn_layers, 
            n_classes
        ).to(self.device)
    
    def fit(self, views: List[np.ndarray], y: np.ndarray) -> 'MOGONETBaseline':
        """
        训练 MOGONET 模型
        
        Args:
            views: 多视图特征列表 [view1, view2, view3]
                   每个视图的形状为 (n_samples, n_features_i)
            y: 标签向量，形状为 (n_samples,)
        
        Returns:
            self: 训练后的模型实例
        """
        import torch
        import torch.nn.functional as F
        from sklearn.model_selection import train_test_split
        from tqdm import tqdm
        
        print(f"\n[MOGONET] 开始训练...")
        print(f"[MOGONET] 配置: hidden_dim={self.hidden_dim}, n_gcn_layers={self.n_gcn_layers}, k_neighbors={self.k_neighbors}, scaler={self.scaler_type}")
        
        # 特征标准化（优化：支持多种归一化方法）
        print(f"[MOGONET] 正在标准化特征...")
        self.scalers = []
        scaled_views = []
        for i, view in enumerate(views):
            # 处理 NaN 值：用列均值填充
            view_clean = np.nan_to_num(view, nan=np.nanmean(view))
            
            scaler = self._get_scaler()
            scaled_view = scaler.fit_transform(view_clean)
            self.scalers.append(scaler)
            scaled_views.append(scaled_view)
            print(f"[MOGONET] 视图 {i+1}: {view.shape} -> 标准化完成")
        
        # 内部验证集划分 (15%)
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.15, random_state=self.random_state, stratify=y
        )
        
        # 构建图结构
        print(f"[MOGONET] 正在构建 k-NN 图 (k={self.k_neighbors})...")
        self.train_edge_indices = []
        self.train_edge_weights = []
        for i, view in enumerate(scaled_views):
            print(f"[MOGONET] 构建视图 {i+1} 的图结构...")
            edge_index, edge_weight = self._build_knn_graph(view[train_idx])
            self.train_edge_indices.append(
                torch.LongTensor(edge_index).to(self.device)
            )
            self.train_edge_weights.append(
                torch.FloatTensor(edge_weight).to(self.device)
            )
            print(f"[MOGONET] 视图 {i+1}: {edge_index.shape[1]} 条边")
        
        # 创建模型
        view_dims = [view.shape[1] for view in scaled_views]
        n_classes = len(np.unique(y))
        self._create_mogonet_model(view_dims, n_classes)
        
        # 准备训练数据
        train_views = [
            torch.FloatTensor(view[train_idx]).to(self.device) 
            for view in scaled_views
        ]
        train_labels = torch.LongTensor(y[train_idx]).to(self.device)
        
        val_views = [
            torch.FloatTensor(view[val_idx]).to(self.device) 
            for view in scaled_views
        ]
        val_labels = torch.LongTensor(y[val_idx]).to(self.device)
        
        # 为验证集构建图
        print(f"[MOGONET] 正在为验证集构建图结构...")
        val_edge_indices = []
        val_edge_weights = []
        for i, view in enumerate(scaled_views):
            edge_index, edge_weight = self._build_knn_graph(view[val_idx])
            val_edge_indices.append(
                torch.LongTensor(edge_index).to(self.device)
            )
            val_edge_weights.append(
                torch.FloatTensor(edge_weight).to(self.device)
            )
        
        # 训练
        print(f"[MOGONET] 开始训练循环 (epochs={self.epochs})...")
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.model.train()
        pbar = tqdm(range(self.epochs), desc="Training MOGONET")
        for epoch in pbar:
            # 前向传播
            logits = self.model(train_views, self.train_edge_indices, self.train_edge_weights)
            loss = F.cross_entropy(logits, train_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
            
            # 验证
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_views, val_edge_indices, val_edge_weights)
                    val_loss = F.cross_entropy(val_logits, val_labels).item()
                    
                    # 计算验证准确率
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = (val_preds == val_labels).float().mean().item()
                    
                    pbar.set_postfix({
                        'train_loss': f'{loss.item():.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'val_acc': f'{val_acc:.4f}'
                    })
                    
                    # Early Stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"\n[MOGONET] Early stopping at epoch {epoch + 1}")
                        break
                
                self.model.train()
        
        pbar.close()
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f"[MOGONET] 训练完成，已加载最佳模型")
        
        return self
    
    def predict(self, views: List[np.ndarray]) -> np.ndarray:
        """
        预测标签
        
        Args:
            views: 多视图特征列表 [view1, view2, view3]
        
        Returns:
            predictions: 预测标签，形状为 (n_samples,)
        """
        proba = self.predict_proba(views)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray:
        """
        预测概率
        
        Args:
            views: 多视图特征列表 [view1, view2, view3]
        
        Returns:
            probabilities: 预测概率，形状为 (n_samples, n_classes)
        """
        import torch
        import torch.nn.functional as F
        
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        print(f"[MOGONET] 正在预测 {len(views[0])} 个样本...")
        
        # 特征标准化
        scaled_views = []
        for i, view in enumerate(views):
            # 处理 NaN 值：用列均值填充
            view_clean = np.nan_to_num(view, nan=np.nanmean(view))
            scaled_view = self.scalers[i].transform(view_clean)
            scaled_views.append(scaled_view)
        
        # 构建图结构
        print(f"[MOGONET] 正在构建测试集图结构...")
        edge_indices = []
        edge_weights = []
        for view in scaled_views:
            edge_index, edge_weight = self._build_knn_graph(view)
            edge_indices.append(
                torch.LongTensor(edge_index).to(self.device)
            )
            edge_weights.append(
                torch.FloatTensor(edge_weight).to(self.device)
            )
        
        # 转换为 Tensor
        test_views = [
            torch.FloatTensor(view).to(self.device) 
            for view in scaled_views
        ]
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_views, edge_indices, edge_weights)
            proba = F.softmax(logits, dim=1)
        
        print(f"[MOGONET] 预测完成")
        return proba.cpu().numpy()
    
    def evaluate(self, views: List[np.ndarray], y: np.ndarray) -> dict:
        """
        评估模型性能
        
        Args:
            views: 多视图特征列表 [view1, view2, view3]
            y: 真实标签，形状为 (n_samples,)
        
        Returns:
            metrics: 评估指标字典，包含 accuracy, precision, recall, f1, auc_roc
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        y_pred = self.predict(views)
        y_proba = self.predict_proba(views)
        
        # 处理多分类情况
        if y_proba.shape[1] == 2:
            # 二分类
            y_proba_pos = y_proba[:, 1]
            average = 'binary'
        else:
            # 多分类
            y_proba_pos = y_proba
            average = 'macro'
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1': f1_score(y, y_pred, average=average, zero_division=0),
        }
        
        # 计算 AUC-ROC
        try:
            if y_proba.shape[1] == 2:
                metrics['auc_roc'] = roc_auc_score(y, y_proba_pos)
            else:
                metrics['auc_roc'] = roc_auc_score(y, y_proba, multi_class='ovr')
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        return metrics

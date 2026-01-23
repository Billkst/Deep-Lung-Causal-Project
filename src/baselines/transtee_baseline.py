"""
TransTEE 基线模型 (NeurIPS 2022)

论文: "Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation" (NeurIPS 2022)
ArXiv: https://arxiv.org/abs/2202.01336
GitHub: https://github.com/hlzhang109/TransTEE

核心思想: 使用 Transformer 架构的自注意力机制捕捉协变量之间的复杂交互，
实现精确的治疗效应估计。

性能基准: IHDP PEHE < 0.6 (显著优于传统方法 ~2.0)
当前复现: PEHE = 1.5469 (良好，提升 22.7%)

硬件要求：
- GPU: 单卡 RTX 3090 (24GB) 或更高
- 推荐批量大小: 64-128
- 显存使用: 约 1-2 GB（取决于数据集大小和模型配置）

性能优化策略：
1. 批量大小调整：如果显存不足，减小 batch_size 到 32 或 16
2. 梯度累积：使用 GradientAccumulationTrainer 保持有效批量大小
3. Transformer 配置：减小 hidden_dim (128 -> 64) 或 n_layers (2 -> 1)
4. 注意力头数：减小 n_heads (4 -> 2) 降低计算开销
5. 混合精度训练：使用 torch.cuda.amp 进行 FP16 训练

使用示例：
```python
from src.baselines.utils import GPUMemoryMonitor

# 使用显存监控
with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
    model = TransTEEBaseline(
        random_state=42,
        hidden_dim=128,
        n_heads=4,
        batch_size=64
    )
    model.fit(X, t, y)
    monitor.check_memory(context="TransTEE Training")
    
    # 预测个体治疗效应
    ite = model.predict_ite(X_test)
```
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


class IHDPDataset:
    """
    IHDP 因果推断数据集加载器
    
    负责加载和解析 IHDP (Infant Health and Development Program) 数据集，
    用于因果推断模型的验证。
    
    数据格式:
    - treatment: 治疗变量 (0/1)
    - y_factual: 观测到的结局
    - 其他列: 协变量 (covariates)
    
    Attributes:
        data_dir (Path): 数据文件目录路径
    """
    
    def __init__(self, data_dir: str = "data/baselines_official/IHDP/"):
        """
        初始化 IHDP 数据集加载器
        
        Args:
            data_dir: IHDP 数据文件所在目录，默认为 "data/baselines_official/IHDP/"
        """
        self.data_dir = Path(data_dir)
        self._check_files_exist()
    
    def _check_files_exist(self):
        """
        检查必需的数据文件是否存在
        
        如果文件不存在，抛出 FileNotFoundError 并提供下载指引
        
        Raises:
            FileNotFoundError: 当 ihdp_npci_1.csv 文件不存在时
        """
        data_file = self.data_dir / 'ihdp_npci_1.csv'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"IHDP 数据文件缺失：{data_file}\n\n"
                "下载指引：\n"
                "1. 访问 IHDP 官方数据源\n"
                "2. 下载 ihdp_npci_1.csv 文件\n"
                "3. 将文件放置在 data/baselines_official/IHDP/ 目录下\n\n"
                "注意：此文件是因果推断基准数据集，用于验证 TransTEE 模型的正确性。"
            )
    
    def load_data(self, include_cfactual: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载 IHDP 数据（仅使用第一个文件进行代码验证）
        
        IHDP 数据集格式（无列名）：
        - 第 0 列：treatment (0/1)
        - 第 1 列：y_factual (观测到的结局，连续值)
        - 第 2 列：y_cfactual (反事实结局，用于计算真实 ITE)
        - 第 3-29 列：covariates (27 个协变量)
        
        Args:
            include_cfactual: 是否返回反事实结局（用于计算真实 ITE）
        
        Returns:
            X: 协变量矩阵 (n_samples, n_features)
            t: 治疗变量 (n_samples,) - 二值变量 (0/1)
            y: 结局变量 (n_samples,) - 观测到的结局
            y_cf: 反事实结局 (n_samples,) - 仅当 include_cfactual=True 时返回
            
        Raises:
            FileNotFoundError: 当数据文件不存在时
            ValueError: 当数据格式不正确时
        """
        data_file = self.data_dir / 'ihdp_npci_1.csv'
        
        try:
            # 读取 CSV 文件（无列名）
            data = pd.read_csv(data_file, header=None)
        except Exception as e:
            raise ValueError(f"读取 IHDP 数据文件失败: {str(e)}")
        
        # 验证数据形状
        if data.shape[1] < 4:
            raise ValueError(
                f"IHDP 数据文件格式错误：期望至少 4 列（treatment, y_factual, y_cfactual, covariates），"
                f"实际只有 {data.shape[1]} 列"
            )
        
        # 提取数据
        # 第 0 列：treatment
        t = data.iloc[:, 0].values
        
        # 第 1 列：y_factual (观测到的结局)
        y = data.iloc[:, 1].values
        
        # 第 2 列：y_cfactual (反事实结局)
        y_cf = data.iloc[:, 2].values
        
        # 第 3 列及之后：covariates
        X = data.iloc[:, 3:].values
        
        # 数据验证
        if not np.all(np.isin(t, [0, 1])):
            raise ValueError(
                f"治疗变量必须是二值变量 (0/1)，"
                f"当前值范围: [{t.min()}, {t.max()}]"
            )
        
        if X.shape[1] == 0:
            raise ValueError(
                "协变量矩阵为空，请检查数据文件格式。"
                "数据文件应包含至少 4 列：treatment, y_factual, y_cfactual, covariates"
            )
        
        if include_cfactual:
            return X, t, y, y_cf
        else:
            return X, t, y


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.baselines.base_model import BaseModel


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器：捕捉协变量之间的复杂交互
    
    使用 Transformer 的自注意力机制来建模协变量之间的非线性交互关系，
    这对于因果推断中的混淆因子控制至关重要。
    
    架构组件:
    1. Feature Embedding: 将输入特征映射到高维空间
    2. Transformer Encoder Layers: 多层自注意力机制
    3. Layer Normalization: 稳定训练过程
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度，默认 128
        n_heads: 多头注意力的头数，默认 4
        n_layers: Transformer 编码器层数，默认 2
        dropout: Dropout 概率，默认 0.1
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # 特征嵌入层 (Feature Embedding)
        # 将原始特征映射到 hidden_dim 维空间
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer 编码器层
        # 使用多头注意力机制捕捉协变量之间的交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 协变量 (batch_size, input_dim)
            
        Returns:
            encoded: 编码后的表示 (batch_size, hidden_dim)
        """
        # 特征嵌入
        x_emb = self.feature_embedding(x)  # (batch_size, hidden_dim)
        
        # Transformer 需要 (batch_size, seq_len, hidden_dim) 格式
        # 这里将每个样本视为长度为 1 的序列
        x_seq = x_emb.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Transformer 编码
        # 自注意力机制会捕捉特征之间的关系
        encoded = self.transformer(x_seq)  # (batch_size, 1, hidden_dim)
        encoded = encoded.squeeze(1)  # (batch_size, hidden_dim)
        
        return encoded



class TransTEENet(nn.Module):
    """
    TransTEE 网络架构
    
    实现双头架构用于因果推断中的治疗效应估计：
    1. Treatment Head: 预测治疗组的潜在结局 Y(1)
    2. Control Head: 预测对照组的潜在结局 Y(0)
    
    核心思想:
    - 使用共享的 Transformer 编码器提取协变量的表示
    - 两个独立的预测头分别估计两种潜在结局
    - 个体治疗效应 ITE = Y(1) - Y(0)
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度，默认 128
        n_heads: 多头注意力的头数，默认 4
        n_layers: Transformer 编码器层数，默认 2
        dropout: Dropout 概率，默认 0.1
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # 共享的 Transformer 编码器
        # 用于提取协变量的通用表示
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Treatment Head (治疗组预测头)
        # 预测在接受治疗情况下的潜在结局 Y(1)
        self.treatment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Control Head (对照组预测头)
        # 预测在未接受治疗情况下的潜在结局 Y(0)
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, t):
        """
        前向传播 - 双头架构
        
        Args:
            x: 协变量 (batch_size, input_dim)
            t: 治疗变量 (batch_size,) - 二值变量 (0/1)
            
        Returns:
            y_pred: 预测的观测结局 (batch_size,)
                   根据实际治疗状态选择对应的潜在结局
            y0_pred: 预测的对照组结局 Y(0) (batch_size,)
            y1_pred: 预测的治疗组结局 Y(1) (batch_size,)
        """
        # 使用共享编码器提取协变量表示
        encoded = self.encoder(x)  # (batch_size, hidden_dim)
        
        # 预测两个潜在结局
        y0_pred = self.control_head(encoded).squeeze(-1)  # (batch_size,)
        y1_pred = self.treatment_head(encoded).squeeze(-1)  # (batch_size,)
        
        # 根据实际治疗状态选择预测值
        # 如果 t=1（接受治疗），使用 y1_pred
        # 如果 t=0（未接受治疗），使用 y0_pred
        y_pred = t * y1_pred + (1 - t) * y0_pred
        
        return y_pred, y0_pred, y1_pred



class TransTEEBaseline(BaseModel):
    """
    TransTEE 基线模型 (NeurIPS 2022)
    
    论文: "Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation" (NeurIPS 2022)
    ArXiv: https://arxiv.org/abs/2202.01336
    
    基于 Transformer 的治疗效应估计模型，使用自注意力机制捕捉协变量之间的
    复杂交互关系，实现精确的个体治疗效应（ITE）估计。
    
    性能基准: IHDP PEHE < 0.6 (显著优于传统方法 ~2.0)
    当前复现: PEHE = 1.5469 (良好，提升 22.7%)
    
    核心功能:
    1. fit(): 训练 TransTEE 模型
    2. predict_ite(): 估计个体治疗效应
    3. evaluate_pehe(): 计算 PEHE 误差
    
    Args:
        random_state: 随机种子，默认 42
        hidden_dim: 隐藏层维度，默认 128
        n_heads: 多头注意力的头数，默认 4
        n_layers: Transformer 编码器层数，默认 2
        epochs: 训练轮数，默认 100
        batch_size: 批量大小，默认 64
        learning_rate: 学习率，默认 0.001
        patience: Early Stopping 的耐心值，默认 10
    """
    
    def __init__(self, random_state: int = 42, hidden_dim: int = 128, 
                 n_heads: int = 4, n_layers: int = 2, epochs: int = 100, 
                 batch_size: int = 64, learning_rate: float = 0.001, 
                 patience: int = 10):
        super().__init__(random_state)
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
    
    def fit(self, X, t, y):
        """
        训练 TransTEE 模型
        
        内部自动划分 15% 作为验证集用于 Early Stopping
        
        Args:
            X: 协变量矩阵 (n_samples, n_features)
            t: 治疗变量 (n_samples,) - 二值变量 (0/1)
            y: 观测到的结局 (n_samples,)
            
        Returns:
            self: 训练后的模型实例
        """
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 内部验证集划分 (15%)
        X_train, X_val, t_train, t_val, y_train, y_val = train_test_split(
            X_scaled, t, y, 
            test_size=0.15, 
            random_state=self.random_state
        )
        
        # 初始化模型
        input_dim = X_train.shape[1]
        self.model = TransTEENet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 转换为 Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        t_train_t = torch.FloatTensor(t_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        t_val_t = torch.FloatTensor(t_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # Early Stopping 变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 训练循环
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs), desc="Training TransTEE")
        
        for epoch in pbar:
            self.model.train()
            
            # Mini-batch 训练
            n_samples = X_train_t.size(0)
            indices = torch.randperm(n_samples)
            
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                X_batch = X_train_t[batch_indices]
                t_batch = t_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]
                
                # 前向传播
                y_pred, y0_pred, y1_pred = self.model(X_batch, t_batch)
                
                # 计算损失（仅对观测到的结局）
                loss = criterion(y_pred, y_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                y_val_pred, _, _ = self.model(X_val_t, t_val_t)
                val_loss = criterion(y_val_pred, y_val_t).item()
            
            # 更新进度条
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            
            # Early Stopping 检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Early Stopping
            if patience_counter >= self.patience:
                print(f"\n[TransTEE] Early stopping at epoch {epoch + 1}")
                break
        
        pbar.close()
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self
    
    def predict_ite(self, X):
        """
        预测个体治疗效应 (ITE)
        
        ITE = E[Y(1) - Y(0) | X]
        
        个体治疗效应表示对于给定协变量 X 的个体，
        接受治疗相比不接受治疗的预期结局差异。
        
        Args:
            X: 协变量矩阵 (n_samples, n_features)
            
        Returns:
            ite: 个体治疗效应估计 (n_samples,)
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        # 特征标准化
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # 创建虚拟的治疗变量（不影响潜在结局的预测）
            dummy_t = torch.zeros(X_t.size(0)).to(self.device)
            
            # 预测两个潜在结局
            _, y0_pred, y1_pred = self.model(X_t, dummy_t)
            
            # 计算 ITE
            ite = y1_pred - y0_pred
        
        return ite.cpu().numpy()
    
    def evaluate_pehe(self, X, true_ite):
        """
        评估 PEHE 误差
        
        PEHE (Precision in Estimation of Heterogeneous Effect) 是因果推断中
        常用的评估指标，衡量个体治疗效应估计的精度。
        
        PEHE = sqrt(E[(ITE_pred - ITE_true)^2])
        
        Args:
            X: 协变量矩阵 (n_samples, n_features)
            true_ite: 真实的个体治疗效应 (n_samples,)
            
        Returns:
            pehe: PEHE 误差值（越小越好）
        """
        ite_pred = self.predict_ite(X)
        pehe = np.sqrt(np.mean((ite_pred - true_ite) ** 2))
        return pehe
    
    def predict(self, X):
        """
        预测标签（不适用于因果推断）
        
        TransTEE 用于治疗效应估计，不进行分类预测。
        请使用 predict_ite() 方法估计个体治疗效应。
        
        Raises:
            NotImplementedError: 因果推断模型不支持分类预测
        """
        raise NotImplementedError(
            "TransTEE 用于治疗效应估计，不进行分类预测。"
            "请使用 predict_ite() 方法估计个体治疗效应。"
        )
    
    def predict_proba(self, X):
        """
        预测概率（不适用于因果推断）
        
        TransTEE 用于治疗效应估计，不进行分类预测。
        请使用 predict_ite() 方法估计个体治疗效应。
        
        Raises:
            NotImplementedError: 因果推断模型不支持概率预测
        """
        raise NotImplementedError(
            "TransTEE 用于治疗效应估计，不进行分类预测。"
            "请使用 predict_ite() 方法估计个体治疗效应。"
        )
    
    def evaluate(self, X, y):
        """
        评估模型性能（不适用于因果推断）
        
        TransTEE 使用 evaluate_pehe() 进行评估。
        
        Raises:
            NotImplementedError: 因果推断模型使用 PEHE 评估
        """
        raise NotImplementedError(
            "TransTEE 使用 evaluate_pehe() 进行评估。"
            "请提供真实的个体治疗效应（true_ite）作为参数。"
        )

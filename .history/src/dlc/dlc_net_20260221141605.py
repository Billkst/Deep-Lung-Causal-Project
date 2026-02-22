"""
DLCNet: Deep-Lung-Causal 主模型

该模块实现了 DLC 模型的主要架构，整合 CausalVAE 和 DynamicHypergraphNN。

满足需求: REQ-F-007, REQ-F-008, REQ-F-009, REQ-F-010
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from typing import Dict
from src.baselines.base_model import BaseModel
from src.dlc.causal_vae import CausalVAE
from src.dlc.hypergraph_nn import DynamicHypergraphNN


class GradientReversalFn(Function):
    """
    Gradient Reversal Layer (GRL) Function.
    Forward: identity.
    Backward: reverse gradient (multiply by -alpha).
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) Module.
    """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class DLCNet(BaseModel, nn.Module):
    """
    Deep-Lung-Causal 主模型。
    
    满足需求: REQ-F-007, REQ-F-008, REQ-F-009, REQ-F-010
    
    继承自 BaseModel 和 nn.Module，实现所有抽象方法。
    
    Attributes:
        causal_vae (CausalVAE): 因果变分自编码器
        hypergraph_nn (DynamicHypergraphNN): 动态超图神经网络
        outcome_head_0 (nn.Module): Y(0) 预测头
        outcome_head_1 (nn.Module): Y(1) 预测头
    """
    
    def __init__(
        self,
        input_dim: int = 23,
        d_conf: int = 8,
        d_effect: int = 16,
        d_hidden: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        lambda_hsic: float = 0.1,
        lambda_pred: float = 1.0,
        dropout: float = 0.1,
        random_state: int = 42
    ):
        """
        初始化 DLCNet
        
        Args:
            input_dim: 输入特征维度 (默认 23)
            d_conf: 混杂表征维度 (默认 8)
            d_effect: 效应表征维度 (默认 16)
            d_hidden: 隐藏层维度 (默认 64)
            num_heads: 注意力头数 (默认 4)
            num_layers: 超图卷积层数 (默认 2)
            lambda_hsic: HSIC 损失权重 (默认 0.1)
            lambda_pred: 预测损失权重 (默认 1.0)
            dropout: Dropout 概率 (默认 0.1)
            random_state: 随机种子 (默认 42)
        """
        # 初始化两个父类
        BaseModel.__init__(self, random_state=random_state)
        nn.Module.__init__(self)
        
        # 保存超参数
        self.input_dim = input_dim
        self.d_conf = d_conf
        self.d_effect = d_effect
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lambda_hsic = lambda_hsic
        self.lambda_pred = lambda_pred
        self.dropout = dropout
        
        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # 子任务 3.2.1: 初始化 CausalVAE 和 DynamicHypergraphNN
        # CausalVAE: 输入维度为 23（所有特征）
        self.causal_vae = CausalVAE(
            input_dim=input_dim,
            d_conf=d_conf,
            d_effect=d_effect,
            hidden_dim=d_hidden
        )
        
        # DynamicHypergraphNN: 处理 20 个基因特征
        self.hypergraph_nn = DynamicHypergraphNN(
            num_genes=20,
            d_effect=d_effect,
            d_hidden=d_hidden,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # Gene Skip Connection: 保留原始基因信号用于 ITE 预测
        # 超图卷积的均值池化会平滑 EGFR 等关键基因的极化信号,
        # skip connection 让原始基因特征直接参与预测,
        # 确保模型同时利用高阶交互(HGNN)和个体基因信号(skip)。
        self.gene_skip = nn.Linear(20, d_hidden)

        # v25: 结构化修正 - Age 对抗头 (Adversarial Head)
        # 目标: 强制 Z_effect 不包含 Age 信息
        # 结构: Z_effect -> GRL -> MLP -> Predicted Age (Classification via Bins)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.adv_age_head = nn.Sequential(
            nn.Linear(d_effect, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, 3) # 3-class Classification
        )
        
        # 子任务 3.2.2: 实现双头预测器
        # 预测头输入: 全局池化后的超图表征 [B, d_hidden]
        # 预测头输出: 结局概率 [B, 1]
        
        # Y(0) 预测头 (未暴露于环境风险)
        self.outcome_head_0 = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )
        
        # Y(1) 预测头 (暴露于环境风险)
        self.outcome_head_1 = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )
    
    def forward(
        self, 
        X: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 (满足 REQ-F-010)。
        
        Args:
            X: 输入特征 [B, 23]
               假设输入顺序为: [Age, Gender, Genes(20), PM2.5]
               - X[:, :2]: 混杂特征 (Age, Gender)
               - X[:, 2:-1]: 基因特征 (Top 20 Genes)
               - X[:, -1]: 环境特征 (PM2.5)
        
        Returns:
            字典包含:
                - Y_0: Y(0) 预测 [B, 1]
                - Y_1: Y(1) 预测 [B, 1]
                - ITE: 个体治疗效应 [B, 1]
                - Z_conf: 混杂表征 [B, d_conf]
                - Z_effect: 效应表征 [B, d_effect]
                - X_recon: 重构特征 [B, 23]
        """
        # 子任务 3.2.3: 实现 forward 方法，整合特征融合流程
        
        # 步骤 1: 特征分割
        # X_conf = X[:, :2]      # Age, Gender [B, 2]
        
        # 修正: 适配实际数据顺序 [Age, Gender, Genes... , PM2.5]
        # 之前错误假设: [Age, Gender, PM2.5, Genes...]
        X_gene = X[:, 2:-1]      # Top 20 Genes [B, 20]
        # X_env = X[:, -1:]      # PM2.5 [B, 1]
        
        # 步骤 2: 因果解耦
        # CausalVAE 接收所有特征作为输入
        vae_outputs = self.causal_vae(X)  # 输入 [B, 23]
        
        Z_conf = vae_outputs['Z_conf']      # [B, d_conf]
        Z_effect = vae_outputs['Z_effect']  # [B, d_effect]
        X_recon = vae_outputs['X_recon']    # [B, 23]
        
        # 步骤 3: 超图交互建模 (同时也是 v25 的对抗输入)
        # 3a. 使用基因特征和效应表征构建动态超图
        H_out = self.hypergraph_nn(X_gene, Z_effect)  # [B, 20, d_hidden]

        # v25: 对抗性预测 Age
        # 我们希望 Z_effect 无法预测 Age，从而使 grad(Age_pred, Z_effect) 经过 GRL 翻转后，
        # 迫使 Encoder 去除 Z_effect 中的 Age 信息。
        Z_effect_rev = self.grl(Z_effect)
        Age_pred = self.adv_age_head(Z_effect_rev) # [B, 1]
        
        # 步骤 4: 全局池化 + Gene Skip Connection
        # 对所有基因节点的表征进行平均池化
        H_global = torch.mean(H_out, dim=1)  # [B, d_hidden]
        # 残差: 加入原始基因信号，保留 EGFR 等关键基因的极化特征
        H_global = H_global + self.gene_skip(X_gene)  # [B, d_hidden]
        
        # 步骤 5: 双头预测
        # 使用两个独立的预测头分别预测 Y(0) 和 Y(1)
        Y_0 = self.outcome_head_0(H_global)  # [B, 1]
        Y_1 = self.outcome_head_1(H_global)  # [B, 1]
        
        # 步骤 6: 计算个体治疗效应 (ITE)
        ITE = Y_1 - Y_0  # [B, 1]
        
        # 返回所有输出
        return {
            'Y_0': Y_0,
            'Y_1': Y_1,
            'ITE': ITE,
            'Z_conf': Z_conf,
            'Z_effect': Z_effect,
            'X_recon': X_recon,
            'Age_pred': Age_pred  # v25 Output
        }
    
    def compute_loss(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        outputs: Dict[str, torch.Tensor],
        t: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失 (满足 REQ-F-008)。
        
        Args:
            X: 输入特征 [B, 23]
            y: 真实标签 [B]
            outputs: forward() 的输出字典
            t: 治疗变量 [B] (0/1)。如果未提供，尝试从 X[:, 2] 推断 (需确保 X 排列正确)
        
        Returns:
            字典包含:
                - loss_total: 总损失
                - loss_recon: 重构损失
                - loss_hsic: HSIC 独立性损失
                - loss_pred: 预测损失
        
        公式:
            L_total = L_recon + λ_hsic * L_hsic + λ_pred * L_pred
            L_pred = t * BCE(Y1, y) + (1-t) * BCE(Y0, y)
        """
        # 1. 计算重构损失
        # CausalVAE 重构所有 23 维特征
        loss_recon = CausalVAE.compute_recon_loss(X, outputs['X_recon'])
        
        # 2. 计算 HSIC 独立性损失
        loss_hsic = CausalVAE.compute_hsic_loss(
            outputs['Z_conf'], 
            outputs['Z_effect']
        )
        
        # 3. 计算预测损失 (Causal Loss: DragonNet style)
        Y_0 = outputs['Y_0'].squeeze()
        Y_1 = outputs['Y_1'].squeeze()
        
        # 确定治疗变量 t
        if t is None:
            # 尝试从 X[:, 2] (Env 特征) 推断
            # 注意：这里假设 X[:, 2] 是标准化后的 PM2.5
            # 如果 PM2.5 > 0 (Standardized) => T=1 (High Risk)
            # 这是一个强假设，建议显示传递 t
            t = (X[:, 2] > 0).float()
        else:
            t = t.float()
        
        # 确保 y 是 float
        y = y.float()
        
        # 计算两头的损失
        # 当 t=1 时，只监督 Y_1; 当 t=0 时，只监督 Y_0
        loss_y1 = F.binary_cross_entropy(Y_1, y, reduction='none')
        loss_y0 = F.binary_cross_entropy(Y_0, y, reduction='none')
        
        loss_pred_vec = t * loss_y1 + (1 - t) * loss_y0
        loss_pred = loss_pred_vec.mean()
        
        # 4. 组合损失
        loss_total = (
            loss_recon + 
            self.lambda_hsic * loss_hsic + 
            self.lambda_pred * loss_pred
        )
        
        # 返回所有损失项
        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_hsic': loss_hsic,
            'loss_pred': loss_pred
        }
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> 'DLCNet':
        """
        训练模型 (满足 REQ-F-009)。
        
        Args:
            X: 特征矩阵 [N, 23]
            y: 标签向量 [N]
            epochs: 训练轮数
            batch_size: 批量大小
            verbose: 是否打印训练信息
        
        Returns:
            self: 训练后的模型实例
        
        实现:
            1. 数据预处理 (标准化)
            2. 创建 DataLoader
            3. 三阶段训练 (见设计文档 4.2 节)
            4. 保存最佳模型
        """
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # 子任务 3.4.1: 数据预处理和 DataLoader 创建
        
        # 1. 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 保存 scaler 用于预测
        self.scaler = scaler
        
        # 2. 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )
        
        # 3. 转换为 PyTorch 张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 4. 创建 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # 5. 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        if verbose:
            print(f"训练设备: {device}")
            print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # 子任务 3.4.2: 阶段 1 - VAE Warmup (20 epochs)
        if verbose:
            print("\n" + "="*50)
            print("阶段 1: VAE Warmup (20 epochs)")
            print("="*50)
        
        # 冻结超图和预测头
        for param in self.hypergraph_nn.parameters():
            param.requires_grad = False
        for param in self.outcome_head_0.parameters():
            param.requires_grad = False
        for param in self.outcome_head_1.parameters():
            param.requires_grad = False
        
        # 优化器 (仅优化 VAE)
        optimizer_stage1 = torch.optim.Adam(
            self.causal_vae.parameters(), 
            lr=1e-3
        )
        
        # 训练 20 轮
        for epoch in range(20):
            self.train()
            train_loss = 0.0
            train_hsic = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 前向传播
                outputs = self(X_batch)
                
                # 计算损失 (仅 VAE 损失)
                loss_recon = CausalVAE.compute_recon_loss(X_batch, outputs['X_recon'])
                loss_hsic = CausalVAE.compute_hsic_loss(
                    outputs['Z_conf'], 
                    outputs['Z_effect']
                )
                loss = loss_recon + self.lambda_hsic * loss_hsic
                
                # 反向传播
                optimizer_stage1.zero_grad()
                loss.backward()
                optimizer_stage1.step()
                
                train_loss += loss.item()
                train_hsic += loss_hsic.item()
            
            # 验证
            self.eval()
            val_hsic = 0.0
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = self(X_batch)
                    hsic = CausalVAE.compute_hsic_loss(
                        outputs['Z_conf'], 
                        outputs['Z_effect']
                    )
                    val_hsic += hsic.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_hsic = train_hsic / len(train_loader)
            avg_val_hsic = val_hsic / len(val_loader)
            
            if verbose:
                print(f"Epoch {epoch+1}/20 - "
                      f"Loss: {avg_train_loss:.4f}, "
                      f"Train HSIC: {avg_train_hsic:.4f}, "
                      f"Val HSIC: {avg_val_hsic:.4f}")
        
        # 子任务 3.4.3: 阶段 2 - Interaction Learning (30 epochs)
        if verbose:
            print("\n" + "="*50)
            print("阶段 2: Interaction Learning (30 epochs)")
            print("="*50)
        
        # 冻结 VAE，解冻超图和预测头
        for param in self.causal_vae.parameters():
            param.requires_grad = False
        for param in self.hypergraph_nn.parameters():
            param.requires_grad = True
        for param in self.outcome_head_0.parameters():
            param.requires_grad = True
        for param in self.outcome_head_1.parameters():
            param.requires_grad = True
        
        # 优化器
        optimizer_stage2 = torch.optim.Adam([
            {'params': self.hypergraph_nn.parameters()},
            {'params': self.outcome_head_0.parameters()},
            {'params': self.outcome_head_1.parameters()}
        ], lr=5e-4)
        
        # 训练 30 轮
        best_val_auc = 0.0
        
        for epoch in range(30):
            self.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 前向传播
                outputs = self(X_batch)
                
                # 计算损失 (仅预测损失)
                y_pred = outputs['Y_1'].squeeze()
                loss_pred = F.binary_cross_entropy(y_pred, y_batch.float())
                loss = self.lambda_pred * loss_pred
                
                # 反向传播
                optimizer_stage2.zero_grad()
                loss.backward()
                optimizer_stage2.step()
                
                train_loss += loss.item()
            
            # 验证
            self.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = self(X_batch)
                    val_preds.append(outputs['Y_1'].cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())
            
            val_preds = np.concatenate(val_preds).flatten()
            val_labels = np.concatenate(val_labels)
            
            # 计算 AUC
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_labels, val_preds)
            
            avg_train_loss = train_loss / len(train_loader)
            
            if verbose:
                print(f"Epoch {epoch+1}/30 - "
                      f"Loss: {avg_train_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}")
            
            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
        
        # 子任务 3.4.4: 阶段 3 - Joint Fine-tuning (50 epochs)
        if verbose:
            print("\n" + "="*50)
            print("阶段 3: Joint Fine-tuning (50 epochs)")
            print("="*50)
        
        # 解冻所有模块
        for param in self.parameters():
            param.requires_grad = True
        
        # 优化器 (使用更小的学习率)
        optimizer_stage3 = torch.optim.Adam(
            self.parameters(), 
            lr=1e-4
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stage3, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # 子任务 3.4.5: Early Stopping 和模型保存
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(50):
            self.train()
            train_loss = 0.0
            train_recon = 0.0
            train_hsic = 0.0
            train_pred = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 前向传播
                outputs = self(X_batch)
                
                # 计算完整损失
                losses = self.compute_loss(X_batch, y_batch, outputs)
                loss = losses['loss_total']
                
                # 反向传播
                optimizer_stage3.zero_grad()
                loss.backward()
                optimizer_stage3.step()
                
                train_loss += loss.item()
                train_recon += losses['loss_recon'].item()
                train_hsic += losses['loss_hsic'].item()
                train_pred += losses['loss_pred'].item()
            
            # 验证
            self.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = self(X_batch)
                    losses = self.compute_loss(X_batch, y_batch, outputs)
                    val_loss += losses['loss_total'].item()
                    
                    val_preds.append(outputs['Y_1'].cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())
            
            val_preds = np.concatenate(val_preds).flatten()
            val_labels = np.concatenate(val_labels)
            val_auc = roc_auc_score(val_labels, val_preds)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/50 - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型状态
                self.best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if hasattr(self, 'best_state_dict'):
            self.load_state_dict(self.best_state_dict)
            if verbose:
                print(f"\n已加载最佳模型 (Val Loss: {best_val_loss:.4f})")
        
        # 移回 CPU
        self.to('cpu')
        
        return self
    
    def predict_proba(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        预测概率 (满足 REQ-F-009)。
        
        子任务 3.5.1: 实现 predict_proba (返回 [N, 2] 概率矩阵)
        
        Args:
            X: 特征矩阵 [N, 23]
        
        Returns:
            proba: 预测概率 [N, 2]
                   proba[:, 0] = 1 - Y(1)  # 负类概率
                   proba[:, 1] = Y(1)      # 正类概率
        
        实现:
            1. 数据标准化（使用训练时的 scaler）
            2. 前向传播获取 Y(1)
            3. 构造二分类概率矩阵
        """
        # 1. 数据标准化
        if hasattr(self, 'scaler'):
            X_scaled = self.scaler.transform(X)
        else:
            # 如果模型未训练，直接使用原始数据
            X_scaled = X
        
        # 2. 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X_scaled)
        
        # 3. 设置为评估模式
        self.eval()
        
        # 4. 前向传播
        with torch.no_grad():
            outputs = self(X_tensor)
            y1_proba = outputs['Y_1'].squeeze().numpy()  # [N]
        
        # 5. 构造二分类概率矩阵
        # proba[:, 0] = P(y=0) = 1 - P(y=1)
        # proba[:, 1] = P(y=1) = Y(1)
        proba = np.stack([1 - y1_proba, y1_proba], axis=1)  # [N, 2]
        
        return proba
    
    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        预测标签 (满足 REQ-F-009)。
        
        子任务 3.5.2: 实现 predict (阈值化 predict_proba)
        
        Args:
            X: 特征矩阵 [N, 23]
        
        Returns:
            y_pred: 预测标签 [N]，值为 0 或 1
        
        实现:
            1. 调用 predict_proba()
            2. 阈值化: y_pred = (proba[:, 1] > 0.5).astype(int)
        """
        # 1. 获取预测概率
        proba = self.predict_proba(X)  # [N, 2]
        
        # 2. 阈值化（使用 0.5 作为阈值）
        y_pred = (proba[:, 1] > 0.5).astype(int)  # [N]
        
        return y_pred
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        评估模型性能 (满足 REQ-F-009)。
        
        子任务 3.5.3: 实现 evaluate (计算 Accuracy, Precision, Recall, F1, AUC)
        
        Args:
            X: 特征矩阵 [N, 23]
            y: 真实标签 [N]
        
        Returns:
            字典包含:
                - accuracy: 准确率
                - precision: 精确率
                - recall: 召回率
                - f1: F1 分数
                - auc_roc: ROC AUC
        
        实现:
            1. 调用 predict() 获取预测标签
            2. 调用 predict_proba() 获取预测概率
            3. 使用 sklearn.metrics 计算各项指标
        """
        from sklearn.metrics import (
            accuracy_score, 
            precision_score, 
            recall_score, 
            f1_score, 
            roc_auc_score
        )
        
        # 1. 获取预测标签
        y_pred = self.predict(X)  # [N]
        
        # 2. 获取预测概率
        y_proba = self.predict_proba(X)  # [N, 2]
        
        # 3. 计算各项指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y, y_proba[:, 1])
        }
        
        return metrics

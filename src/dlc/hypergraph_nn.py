"""
DynamicHypergraphNN: 动态超图神经网络

该模块实现了基于注意力机制的动态超图构建和卷积操作。
"""

import torch
import torch.nn as nn
from typing import Tuple


class DynamicHypergraphNN(nn.Module):
    """
    动态超图神经网络，建模基因间的高阶交互。
    
    满足需求: REQ-F-004, REQ-F-005, REQ-F-006
    
    Attributes:
        num_genes (int): 基因节点数量 (20)
        d_effect (int): 环境表征维度
        d_hidden (int): 隐藏层维度
        num_heads (int): 注意力头数
        num_layers (int): 超图卷积层数
    """
    
    def __init__(
        self,
        num_genes: int = 20,
        d_effect: int = 16,
        d_hidden: int = 64,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        """
        初始化 DynamicHypergraphNN
        
        Args:
            num_genes: 基因节点数量
            d_effect: 环境效应表征维度
            d_hidden: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 超图卷积层数
        """
        super().__init__()
        
        self.num_genes = num_genes
        self.d_effect = d_effect
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 基因嵌入层：将基因特征嵌入到高维空间
        self.gene_embedding = nn.Linear(1, d_hidden)
        
        # 多头自注意力机制
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 门控网络：使用 Z_effect 计算门控系数
        # 输出维度为 num_edges，这里先设置为 num_heads * num_genes
        self.num_edges = num_heads * num_genes
        self.gate_fc = nn.Linear(d_effect, self.num_edges)
        
        # 超图卷积层
        self.conv_layers = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden) for _ in range(num_layers)
        ])
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def build_hypergraph(
        self, 
        X_gene: torch.Tensor
    ) -> torch.Tensor:
        """
        通过多头自注意力构建超图邻接矩阵 (满足 REQ-F-004)。
        
        Args:
            X_gene: 基因特征 [B, num_genes]
        
        Returns:
            H: 超图关联矩阵 [B, num_edges, num_genes]
               H[b, e, v] = 1 表示节点 v 属于超边 e
        
        算法:
            1. 将基因特征嵌入到高维空间: X_emb = Embedding(X_gene)  # [B, num_genes, d_hidden]
            2. 计算多头注意力分数: Attn = MultiHeadAttention(X_emb, X_emb)  # [B, num_heads, num_genes, num_genes]
            3. 对每个注意力头，选择 Top-K 相似节点构成超边
            4. 合并所有头的超边，去重
        """
        B, num_genes = X_gene.shape
        
        # 1. 嵌入基因特征
        # X_gene: [B, num_genes] -> [B, num_genes, 1] -> [B, num_genes, d_hidden]
        X_emb = self.gene_embedding(X_gene.unsqueeze(-1))  # [B, num_genes, d_hidden]
        
        # 2. 多头自注意力 [B, num_heads, num_genes, num_genes]
        _, attn_weights = self.multi_head_attention(X_emb, X_emb, X_emb, average_attn_weights=False)
        
        # 3. 向量化构建超图
        # 目标: H [B, num_edges, num_genes]
        # num_edges = num_heads * num_genes
        
        # (1) 将 heads 和 genes 维度合并为 edges 维度
        # attn_weights: [B, num_heads, target_genes, source_genes]
        # reshape -> [B, num_heads * num_genes, num_genes]
        attn_flat = attn_weights.view(B, self.num_heads * num_genes, num_genes)
        
        # (2) 计算 Top-K
        k = min(5, num_genes)
        _, topk_indices = torch.topk(attn_flat, k=k, dim=-1)  # [B, num_edges, k]
        
        # (3) 初始化 H 并使用 scatter_ 填充
        H = torch.zeros(B, self.num_edges, num_genes, device=X_gene.device)
        src = torch.ones_like(topk_indices, dtype=torch.float)
        H.scatter_(2, topk_indices, src)
        
        # (4) 确保自环 (Self-loops): 每个 edge e (对应 node v) 必须包含 node v
        # edge e 对应于头 h 和节点 v，关系是: e = h * num_genes + v
        # 所以 v = e % num_genes
        
        # 构造需要强制置 1 的索引
        # 我们需要设置 H[b, e, e % num_genes] = 1
        
        # 生成 e 的索引 [0, 1, ..., num_edges-1]
        e_indices = torch.arange(self.num_edges, device=X_gene.device)
        # 计算对应的 v 索引
        v_indices = e_indices % num_genes
        
        # 对所有 batch 执行赋值
        # 使用切片广播: H[:, e_indices, v_indices] = 1.0 
        # 注意: 高级索引 H[:, tensor, tensor] 会导致维度坍缩或不对齐，
        # H[:, e, v] = 1 会广播到所有 B
        H[:, e_indices, v_indices] = 1.0
        
        return H  # [B, num_edges, num_genes]
    
    def apply_environment_gating(
        self, 
        H: torch.Tensor, 
        Z_effect: torch.Tensor
    ) -> torch.Tensor:
        """
        应用环境门控机制 (满足 REQ-F-005) 🔑。
        
        Args:
            H: 超图关联矩阵 [B, num_edges, num_genes]
            Z_effect: 环境效应表征 [B, d_effect]
        
        Returns:
            H_gated: 门控后的超图矩阵 [B, num_edges, num_genes]
        
        算法:
            1. 计算门控系数: gate = sigmoid(W_gate @ Z_effect + b_gate)  # [B, num_edges]
            2. 应用门控: H_gated[b, e, v] = gate[b, e] * H[b, e, v]
        
        生物学解释:
            - gate[b, e] 表示环境暴露对超边 e 的激活强度
            - 模拟"环境促癌"机制：环境暴露增强基因间的协同致病效应
        """
        B, num_edges, num_genes = H.shape
        
        # 1. 计算门控系数
        # Z_effect: [B, d_effect] -> gate_logits: [B, num_edges]
        gate_logits = self.gate_fc(Z_effect)  # [B, num_edges]
        
        # 应用 sigmoid 激活，确保门控系数在 [0, 1] 范围内
        gate = torch.sigmoid(gate_logits)  # [B, num_edges]
        
        # 2. 应用门控到超图矩阵
        # gate: [B, num_edges] -> [B, num_edges, 1]
        # H: [B, num_edges, num_genes]
        # H_gated: [B, num_edges, num_genes]
        gate_expanded = gate.unsqueeze(-1)  # [B, num_edges, 1]
        H_gated = gate_expanded * H  # 广播乘法: [B, num_edges, 1] * [B, num_edges, num_genes]
        
        return H_gated
    
    def hypergraph_conv(
        self, 
        X: torch.Tensor, 
        H: torch.Tensor
    ) -> torch.Tensor:
        """
        超图卷积操作 (满足 REQ-F-006)。
        
        Args:
            X: 节点特征 [B, num_genes, d_hidden]
            H: 超图关联矩阵 [B, num_edges, num_genes]
        
        Returns:
            X_out: 更新后的节点特征 [B, num_genes, d_hidden]
        
        公式 (基于 Feng et al., 2019):
            X_out = D_v^(-1/2) @ H^T @ W @ D_e^(-1) @ H @ D_v^(-1/2) @ X @ Theta
        
        其中:
            - D_v: 节点度矩阵 (对角矩阵)，D_v[i,i] = sum_e H[e,i]
            - D_e: 超边度矩阵 (对角矩阵)，D_e[e,e] = sum_i H[e,i]
            - W: 超边权重矩阵 (对角矩阵)，这里简化为单位矩阵
            - Theta: 可学习权重矩阵 (通过 self.conv_layers 实现)
        """
        B, num_edges, num_genes = H.shape
        _, _, d_hidden = X.shape
        
        # 1. 计算节点度矩阵 D_v
        # D_v[b, i, i] = sum_e H[b, e, i]
        D_v = H.sum(dim=1)  # [B, num_genes]
        
        # 避免除零，添加小的 epsilon
        D_v = D_v + 1e-8
        
        # 计算 D_v^(-1/2)
        D_v_inv_sqrt = torch.pow(D_v, -0.5)  # [B, num_genes]
        
        # 2. 计算超边度矩阵 D_e
        # D_e[b, e, e] = sum_i H[b, e, i]
        D_e = H.sum(dim=2)  # [B, num_edges]
        
        # 避免除零
        D_e = D_e + 1e-8
        
        # 计算 D_e^(-1)
        D_e_inv = torch.pow(D_e, -1.0)  # [B, num_edges]
        
        # 3. 实现超图卷积公式
        # X_out = D_v^(-1/2) @ H^T @ W @ D_e^(-1) @ H @ D_v^(-1/2) @ X @ Theta
        
        # 步骤 1: D_v^(-1/2) @ X
        # D_v_inv_sqrt: [B, num_genes] -> [B, num_genes, 1]
        # X: [B, num_genes, d_hidden]
        X_scaled = D_v_inv_sqrt.unsqueeze(-1) * X  # [B, num_genes, d_hidden]
        
        # 步骤 2: H @ (D_v^(-1/2) @ X)
        # H: [B, num_edges, num_genes]
        # X_scaled: [B, num_genes, d_hidden]
        # 结果: [B, num_edges, d_hidden]
        X_edge = torch.bmm(H, X_scaled)  # [B, num_edges, d_hidden]
        
        # 步骤 3: D_e^(-1) @ (H @ D_v^(-1/2) @ X)
        # D_e_inv: [B, num_edges] -> [B, num_edges, 1]
        # X_edge: [B, num_edges, d_hidden]
        X_edge_scaled = D_e_inv.unsqueeze(-1) * X_edge  # [B, num_edges, d_hidden]
        
        # 步骤 4: W @ D_e^(-1) @ H @ D_v^(-1/2) @ X
        # 这里 W 简化为单位矩阵，所以跳过
        
        # 步骤 5: H^T @ (W @ D_e^(-1) @ H @ D_v^(-1/2) @ X)
        # H^T: [B, num_genes, num_edges]
        # X_edge_scaled: [B, num_edges, d_hidden]
        # 结果: [B, num_genes, d_hidden]
        H_T = H.transpose(1, 2)  # [B, num_genes, num_edges]
        X_aggregated = torch.bmm(H_T, X_edge_scaled)  # [B, num_genes, d_hidden]
        
        # 步骤 6: D_v^(-1/2) @ (H^T @ W @ D_e^(-1) @ H @ D_v^(-1/2) @ X)
        # D_v_inv_sqrt: [B, num_genes] -> [B, num_genes, 1]
        # X_aggregated: [B, num_genes, d_hidden]
        X_normalized = D_v_inv_sqrt.unsqueeze(-1) * X_aggregated  # [B, num_genes, d_hidden]
        
        # 步骤 7: 应用可学习的线性变换 Theta (通过 conv_layers)
        # 这里我们只应用第一层卷积层作为 Theta
        # 注意：在 forward 方法中会多次调用 hypergraph_conv，每次使用不同的层
        # 所以这里不应该在 hypergraph_conv 内部应用 conv_layers
        # 而是返回归一化后的特征，让 forward 方法来应用线性变换
        
        X_out = X_normalized  # [B, num_genes, d_hidden]
        
        return X_out
    
    def forward(
        self, 
        X_gene: torch.Tensor, 
        Z_effect: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            X_gene: 基因特征 [B, num_genes]
            Z_effect: 环境效应表征 [B, d_effect]
        
        Returns:
            H_out: 节点表征 [B, num_genes, d_hidden]
        
        流程:
            1. 构建超图: H = build_hypergraph(X_gene)
            2. 应用门控: H_gated = apply_environment_gating(H, Z_effect)
            3. 多层卷积: 
               for layer in layers:
                   X = hypergraph_conv(X, H_gated)
                   X = ReLU(X)
            4. 返回最终节点表征
        """
        B, num_genes = X_gene.shape
        
        # 1. 构建超图
        H = self.build_hypergraph(X_gene)  # [B, num_edges, num_genes]
        
        # 2. 应用环境门控
        H_gated = self.apply_environment_gating(H, Z_effect)  # [B, num_edges, num_genes]
        
        # 3. 初始化节点特征（使用基因嵌入）
        X = self.gene_embedding(X_gene.unsqueeze(-1))  # [B, num_genes, d_hidden]
        
        # 4. 多层超图卷积
        for i, conv_layer in enumerate(self.conv_layers):
            # 超图卷积
            X = self.hypergraph_conv(X, H_gated)  # [B, num_genes, d_hidden]
            
            # 应用线性变换（可学习的 Theta）
            X = conv_layer(X)  # [B, num_genes, d_hidden]
            
            # 激活函数（最后一层不使用激活）
            if i < len(self.conv_layers) - 1:
                X = self.relu(X)
        
        return X  # [B, num_genes, d_hidden]

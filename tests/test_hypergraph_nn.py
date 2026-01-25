"""
DynamicHypergraphNN 单元测试

测试 DynamicHypergraphNN 模块的各项功能。
"""

import pytest
import torch
import numpy as np
from src.dlc.hypergraph_nn import DynamicHypergraphNN


class TestDynamicHypergraphNN:
    """DynamicHypergraphNN 单元测试类"""
    
    def test_hypergraph_initialization(self):
        """
        测试 DynamicHypergraphNN 初始化
        
        验证:
        - 模型能够正确初始化
        - 所有参数维度正确
        """
        model = DynamicHypergraphNN(
            num_genes=20,
            d_effect=16,
            d_hidden=64,
            num_heads=4,
            num_layers=2
        )
        
        # 验证模型是 nn.Module 的实例
        assert isinstance(model, torch.nn.Module)
        
        # 验证超参数设置正确
        assert model.num_genes == 20
        assert model.d_effect == 16
        assert model.d_hidden == 64
        assert model.num_heads == 4
        assert model.num_layers == 2
    
    def test_build_hypergraph_shape(self):
        """
        测试超图构建输出形状
        
        验证:
        - 超图关联矩阵 H 的形状为 [B, num_edges, num_genes]
        """
        model = DynamicHypergraphNN(num_genes=20, num_heads=4)
        
        # 生成随机基因特征
        batch_size = 32
        X_gene = torch.randn(batch_size, 20)
        
        # 构建超图
        H = model.build_hypergraph(X_gene)
        
        # 验证形状
        assert H.shape[0] == batch_size, f"Batch size mismatch: {H.shape[0]} != {batch_size}"
        assert H.shape[2] == 20, f"Gene dimension mismatch: {H.shape[2]} != 20"
        # num_edges 应该是一个合理的数值（由注意力机制决定）
        assert H.shape[1] > 0, "Number of hyperedges should be positive"
    
    def test_build_hypergraph_edge_degrees(self):
        """
        测试超边至少连接 2 个节点
        
        验证:
        - 每个超边至少连接 2 个基因节点
        """
        model = DynamicHypergraphNN(num_genes=20, num_heads=4)
        
        # 生成随机基因特征
        batch_size = 16
        X_gene = torch.randn(batch_size, 20)
        
        # 构建超图
        H = model.build_hypergraph(X_gene)
        
        # 计算每个超边的度数（连接的节点数）
        edge_degrees = H.sum(dim=-1)  # [B, num_edges]
        
        # 验证每个超边至少连接 2 个节点
        assert (edge_degrees >= 2).all(), \
            f"Some hyperedges connect < 2 nodes. Min degree: {edge_degrees.min().item()}"
    
    def test_build_hypergraph_node_degrees(self):
        """
        测试每个节点至少属于 1 个超边
        
        验证:
        - 每个基因节点至少属于 1 个超边
        """
        model = DynamicHypergraphNN(num_genes=20, num_heads=4)
        
        # 生成随机基因特征
        batch_size = 16
        X_gene = torch.randn(batch_size, 20)
        
        # 构建超图
        H = model.build_hypergraph(X_gene)
        
        # 计算每个节点的度数（属于的超边数）
        node_degrees = H.sum(dim=1)  # [B, num_genes]
        
        # 验证每个节点至少属于 1 个超边
        assert (node_degrees >= 1).all(), \
            f"Some nodes belong to 0 hyperedges. Min degree: {node_degrees.min().item()}"
    
    def test_apply_environment_gating_shape(self):
        """
        测试门控输出形状
        
        验证:
        - 门控后的超图矩阵形状与输入一致
        """
        model = DynamicHypergraphNN(num_genes=20, d_effect=16)
        
        batch_size = 32
        num_edges = 80
        
        # 生成随机输入
        H = torch.ones(batch_size, num_edges, 20)
        Z_effect = torch.randn(batch_size, 16)
        
        # 应用门控
        H_gated = model.apply_environment_gating(H, Z_effect)
        
        # 验证形状
        assert H_gated.shape == H.shape, \
            f"Shape mismatch: {H_gated.shape} != {H.shape}"
    
    def test_apply_environment_gating_range(self):
        """
        测试门控系数在 [0, 1] 范围
        
        验证:
        - 门控系数通过 sigmoid 激活，应在 [0, 1] 范围内
        - 门控后的超图矩阵值也应在合理范围内
        """
        model = DynamicHypergraphNN(num_genes=20, d_effect=16)
        
        batch_size = 32
        num_edges = 80
        
        # 生成随机输入（H 为二值矩阵）
        H = torch.randint(0, 2, (batch_size, num_edges, 20)).float()
        Z_effect = torch.randn(batch_size, 16)
        
        # 应用门控
        H_gated = model.apply_environment_gating(H, Z_effect)
        
        # 验证门控后的值在 [0, 1] 范围内（因为 H 是 0/1，gate 是 sigmoid）
        assert (H_gated >= 0).all(), "Gated values should be >= 0"
        assert (H_gated <= 1).all(), "Gated values should be <= 1"
        
        # 验证无 NaN 或 Inf
        assert not torch.isnan(H_gated).any(), "NaN detected in gated output"
        assert not torch.isinf(H_gated).any(), "Inf detected in gated output"
    
    def test_hypergraph_conv_shape(self):
        """
        测试超图卷积输出形状
        
        验证:
        - 卷积后的节点特征形状为 [B, num_genes, d_hidden]
        """
        model = DynamicHypergraphNN(num_genes=20, d_hidden=64)
        
        batch_size = 32
        num_edges = 80
        
        # 生成随机输入
        X = torch.randn(batch_size, 20, 64)
        H = torch.randint(0, 2, (batch_size, num_edges, 20)).float()
        
        # 超图卷积
        X_out = model.hypergraph_conv(X, H)
        
        # 验证形状
        assert X_out.shape == (batch_size, 20, 64), \
            f"Shape mismatch: {X_out.shape} != ({batch_size}, 20, 64)"
    
    def test_forward_output_shape(self):
        """
        测试 forward 输出形状
        
        验证:
        - 最终输出的节点表征形状为 [B, num_genes, d_hidden]
        """
        model = DynamicHypergraphNN(
            num_genes=20,
            d_effect=16,
            d_hidden=64,
            num_heads=4,
            num_layers=2
        )
        
        batch_size = 32
        
        # 生成随机输入
        X_gene = torch.randn(batch_size, 20)
        Z_effect = torch.randn(batch_size, 16)
        
        # 前向传播
        H_out = model.forward(X_gene, Z_effect)
        
        # 验证形状
        assert H_out.shape == (batch_size, 20, 64), \
            f"Shape mismatch: {H_out.shape} != ({batch_size}, 20, 64)"
        
        # 验证无 NaN 或 Inf
        assert not torch.isnan(H_out).any(), "NaN detected in forward output"
        assert not torch.isinf(H_out).any(), "Inf detected in forward output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

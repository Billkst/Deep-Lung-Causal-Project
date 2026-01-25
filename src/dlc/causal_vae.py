"""
CausalVAE: 因果变分自编码器

该模块实现了用于解耦混杂因素和效应因素的变分自编码器。

满足需求: REQ-F-001, REQ-F-002, REQ-F-003
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class CausalVAE(nn.Module):
    """
    因果变分自编码器，实现混杂因子与效应表征的解耦。
    
    Attributes:
        input_dim (int): 输入特征维度 (23)
        d_conf (int): 混杂表征维度
        d_effect (int): 效应表征维度
        hidden_dim (int): 隐藏层维度
        encoder_conf (nn.Module): 混杂编码器
        encoder_effect (nn.Module): 效应编码器
        decoder (nn.Module): 解码器
    """
    
    def __init__(
        self,
        input_dim: int = 23,
        d_conf: int = 8,
        d_effect: int = 16,
        hidden_dim: int = 64
    ):
        """
        初始化 CausalVAE
        
        Args:
            input_dim: 输入特征维度 (默认 23)
            d_conf: 混杂表征维度 (默认 8)
            d_effect: 效应表征维度 (默认 16)
            hidden_dim: 隐藏层维度 (默认 64)
        """
        super(CausalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.d_conf = d_conf
        self.d_effect = d_effect
        self.hidden_dim = hidden_dim
        
        # 混杂编码器 (Encoder for Confounders)
        # 输入: [B, 23] -> 隐藏层: [B, hidden_dim] -> 输出: [B, 2*d_conf]
        self.encoder_conf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * d_conf)  # 输出 mu 和 logvar
        )
        
        # 效应编码器 (Encoder for Effect)
        # 输入: [B, 23] -> 隐藏层: [B, hidden_dim] -> 输出: [B, 2*d_effect]
        self.encoder_effect = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * d_effect)  # 输出 mu 和 logvar
        )
        
        # 解码器 (Decoder)
        # 输入: [B, d_conf + d_effect] -> 隐藏层: [B, hidden_dim] -> 输出: [B, 23]
        self.decoder = nn.Sequential(
            nn.Linear(d_conf + d_effect, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(
        self, 
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码输入为潜变量分布参数。
        
        Args:
            X: 输入特征 [B, 23]
        
        Returns:
            mu_conf: Z_conf 均值 [B, d_conf]
            logvar_conf: Z_conf 对数方差 [B, d_conf]
            mu_effect: Z_effect 均值 [B, d_effect]
            logvar_effect: Z_effect 对数方差 [B, d_effect]
        """
        # 编码混杂因素
        h_conf = self.encoder_conf(X)  # [B, 2*d_conf]
        mu_conf = h_conf[:, :self.d_conf]  # [B, d_conf]
        logvar_conf = h_conf[:, self.d_conf:]  # [B, d_conf]
        
        # 编码效应因素
        h_effect = self.encoder_effect(X)  # [B, 2*d_effect]
        mu_effect = h_effect[:, :self.d_effect]  # [B, d_effect]
        logvar_effect = h_effect[:, self.d_effect:]  # [B, d_effect]
        
        return mu_conf, logvar_conf, mu_effect, logvar_effect
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick)。
        
        Args:
            mu: 均值 [B, d]
            logvar: 对数方差 [B, d]
        
        Returns:
            z: 采样的潜变量 [B, d]
        
        公式:
            z = mu + eps * exp(0.5 * logvar)
            其中 eps ~ N(0, I)
        """
        # 计算标准差
        std = torch.exp(0.5 * logvar)
        
        # 采样标准正态分布
        eps = torch.randn_like(std)
        
        # 重参数化
        z = mu + eps * std
        
        return z
    
    def decode(
        self, 
        Z_conf: torch.Tensor, 
        Z_effect: torch.Tensor
    ) -> torch.Tensor:
        """
        从潜变量重构输入特征。
        
        Args:
            Z_conf: 混杂表征 [B, d_conf]
            Z_effect: 效应表征 [B, d_effect]
        
        Returns:
            X_recon: 重构特征 [B, 23]
        """
        # 拼接潜变量
        Z = torch.cat([Z_conf, Z_effect], dim=1)  # [B, d_conf + d_effect]
        
        # 解码
        X_recon = self.decoder(Z)  # [B, 23]
        
        return X_recon
    
    def forward(
        self, 
        X: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        Args:
            X: 输入特征 [B, 23]
        
        Returns:
            字典包含:
                - Z_conf: 混杂表征 [B, d_conf]
                - Z_effect: 效应表征 [B, d_effect]
                - X_recon: 重构特征 [B, 23]
                - mu_conf, logvar_conf: 混杂分布参数
                - mu_effect, logvar_effect: 效应分布参数
        """
        # 编码
        mu_conf, logvar_conf, mu_effect, logvar_effect = self.encode(X)
        
        # 重参数化采样
        Z_conf = self.reparameterize(mu_conf, logvar_conf)
        Z_effect = self.reparameterize(mu_effect, logvar_effect)
        
        # 解码
        X_recon = self.decode(Z_conf, Z_effect)
        
        return {
            'Z_conf': Z_conf,
            'Z_effect': Z_effect,
            'X_recon': X_recon,
            'mu_conf': mu_conf,
            'logvar_conf': logvar_conf,
            'mu_effect': mu_effect,
            'logvar_effect': logvar_effect
        }
    
    @staticmethod
    def compute_hsic_loss(
        Z_conf: torch.Tensor, 
        Z_effect: torch.Tensor,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """
        计算 HSIC 独立性损失 (满足 REQ-F-002)。
        
        Args:
            Z_conf: 混杂表征 [B, d_conf]
            Z_effect: 效应表征 [B, d_effect]
            sigma: RBF 核带宽参数
        
        Returns:
            hsic_loss: HSIC 损失值 (标量)
        
        算法:
            1. 计算 RBF 核矩阵 K_conf = exp(-||Z_conf_i - Z_conf_j||^2 / (2*sigma^2))
            2. 计算 RBF 核矩阵 K_effect = exp(-||Z_effect_i - Z_effect_j||^2 / (2*sigma^2))
            3. 中心化核矩阵: H = I - 1/B * 11^T
            4. HSIC = (1/(B-1)^2) * Tr(K_conf @ H @ K_effect @ H)
        
        理论依据:
            HSIC = 0 当且仅当 Z_conf 与 Z_effect 统计独立
        """
        B = Z_conf.shape[0]
        
        # 步骤 1: 计算 RBF 核矩阵 K_conf
        # 计算成对距离矩阵: ||Z_conf_i - Z_conf_j||^2
        # 使用公式: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
        Z_conf_norm = (Z_conf ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        dist_conf = Z_conf_norm + Z_conf_norm.T - 2 * torch.mm(Z_conf, Z_conf.T)  # [B, B]
        K_conf = torch.exp(-dist_conf / (2 * sigma ** 2))  # [B, B]
        
        # 步骤 2: 计算 RBF 核矩阵 K_effect
        Z_effect_norm = (Z_effect ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        dist_effect = Z_effect_norm + Z_effect_norm.T - 2 * torch.mm(Z_effect, Z_effect.T)  # [B, B]
        K_effect = torch.exp(-dist_effect / (2 * sigma ** 2))  # [B, B]
        
        # 步骤 3: 构建中心化矩阵 H = I - 1/B * 11^T
        H = torch.eye(B, device=Z_conf.device) - torch.ones(B, B, device=Z_conf.device) / B  # [B, B]
        
        # 步骤 4: 计算 HSIC = (1/(B-1)^2) * Tr(K_conf @ H @ K_effect @ H)
        # 使用 Tr(ABCD) = sum(A * (BCD)^T) 的性质优化计算
        K_conf_H = torch.mm(K_conf, H)  # [B, B]
        K_effect_H = torch.mm(K_effect, H)  # [B, B]
        hsic = torch.trace(torch.mm(K_conf_H, K_effect_H)) / ((B - 1) ** 2)
        
        return hsic
    
    @staticmethod
    def compute_recon_loss(
        X: torch.Tensor, 
        X_recon: torch.Tensor
    ) -> torch.Tensor:
        """
        计算重构损失 (满足 REQ-F-003)。
        
        Args:
            X: 原始特征 [B, 23]
            X_recon: 重构特征 [B, 23]
        
        Returns:
            recon_loss: 重构损失 (标量)
        
        实现:
            - 所有变量使用 MSE Loss (简化实现)
            
        注意: 在实际应用中，可以根据特征类型使用不同的损失函数。
        这里为了简化和稳定性，统一使用 MSE。
        """
        # 使用 MSE Loss 计算所有特征的重构损失
        recon_loss = torch.nn.functional.mse_loss(X_recon, X, reduction='mean')
        
        return recon_loss

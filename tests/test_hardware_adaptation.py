"""
硬件适配测试

测试所有模型在 RTX 3090 (24GB) 上的运行情况，
包括显存监控和梯度累积功能。

Requirements:
    - 13.1: 确保所有模型在单卡 RTX 3090 (24GB) 上可运行
    - 13.2: 为大模型实现梯度累积机制
    - 13.3: 在模型训练时监控 GPU 显存使用情况
"""

import pytest
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.baselines.utils import (
    GPUMemoryMonitor,
    GradientAccumulationTrainer,
    auto_adjust_batch_size,
    get_gpu_memory_info
)
from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline, TabRNet
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.baselines.transtee_baseline import TransTEEBaseline


class TestGPUMemoryMonitor:
    """测试 GPU 显存监控功能"""
    
    def test_monitor_initialization(self):
        """测试显存监控器初始化"""
        monitor = GPUMemoryMonitor(warning_threshold_gb=20.0)
        
        # 验证初始化参数
        assert monitor.warning_threshold_gb == 20.0
        assert monitor.device_id == 0
        assert monitor.peak_memory_bytes == 0
        assert monitor.warning_count == 0
    
    def test_check_memory(self):
        """测试显存检查功能"""
        monitor = GPUMemoryMonitor(warning_threshold_gb=20.0)
        
        # 检查显存
        memory_info = monitor.check_memory(context="Test")
        
        # 验证返回的信息
        assert 'allocated_gb' in memory_info
        assert 'reserved_gb' in memory_info
        assert 'total_gb' in memory_info
        assert 'utilization' in memory_info
        
        # 验证数值范围
        assert memory_info['allocated_gb'] >= 0
        assert memory_info['utilization'] >= 0
        assert memory_info['utilization'] <= 1
    
    def test_get_memory_stats(self):
        """测试获取显存统计信息"""
        monitor = GPUMemoryMonitor(warning_threshold_gb=20.0)
        
        # 执行一些操作
        monitor.check_memory()
        
        # 获取统计信息
        stats = monitor.get_memory_stats()
        
        # 验证统计信息
        assert 'peak_memory_gb' in stats
        assert 'current_memory_gb' in stats
        assert 'total_memory_gb' in stats
        assert 'warning_count' in stats
        
        assert stats['peak_memory_gb'] >= 0
        assert stats['current_memory_gb'] >= 0
    
    def test_context_manager(self):
        """测试上下文管理器功能"""
        with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
            # 在上下文中执行操作
            monitor.check_memory()
            stats = monitor.get_memory_stats()
            
            assert stats['peak_memory_gb'] >= 0
    
    def test_reset_peak_memory(self):
        """测试重置峰值显存"""
        monitor = GPUMemoryMonitor(warning_threshold_gb=20.0)
        
        # 执行一些操作
        monitor.check_memory()
        
        # 重置峰值
        monitor.reset_peak_memory()
        
        # 验证重置后的状态
        assert monitor.peak_memory_bytes == 0
        assert monitor.warning_count == 0


class TestGradientAccumulation:
    """测试梯度累积功能"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
    def test_gradient_accumulation_trainer(self):
        """测试梯度累积训练器"""
        # 创建简单模型
        model = torch.nn.Linear(10, 2).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建梯度累积训练器
        trainer = GradientAccumulationTrainer(
            model=model,
            optimizer=optimizer,
            accumulation_steps=4
        )
        
        # 模拟训练步骤
        criterion = torch.nn.CrossEntropyLoss()
        
        for step in range(8):
            X = torch.randn(16, 10).cuda()
            y = torch.randint(0, 2, (16,)).cuda()
            
            # 前向传播
            output = model(X)
            loss = criterion(output, y)
            
            # 反向传播（带梯度累积）
            trainer.backward_step(loss, step)
        
        # 获取统计信息
        stats = trainer.get_stats()
        
        assert stats['total_steps'] == 8
        assert stats['update_count'] == 2  # 8 步 / 4 累积步数 = 2 次更新
        assert stats['accumulation_steps'] == 4
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
    def test_auto_adjust_batch_size(self):
        """测试自动批量大小调整"""
        # 创建简单模型
        model = torch.nn.Linear(30, 2)
        sample_input = torch.randn(1, 30)
        
        # 自动调整批量大小
        batch_size, accum_steps = auto_adjust_batch_size(
            model=model,
            sample_input=sample_input,
            initial_batch_size=64,
            memory_threshold_gb=20.0,
            min_batch_size=8
        )
        
        # 验证结果
        assert batch_size >= 8
        assert accum_steps >= 1
        assert batch_size * accum_steps <= 64  # 有效批量大小不超过初始值


class TestModelHardwareCompatibility:
    """测试模型在 RTX 3090 上的兼容性"""
    
    @pytest.fixture
    def breast_cancer_data(self):
        """加载 UCI Breast Cancer 数据集"""
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_xgboost_memory_usage(self, breast_cancer_data):
        """测试 XGBoost 模型的显存使用"""
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 使用显存监控器
        with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
            # 训练模型
            model = XGBBaseline(random_state=42)
            model.fit(X_train, y_train)
            
            # 检查显存
            monitor.check_memory(context="XGBoost Training")
            
            # 预测
            predictions = model.predict(X_test)
            
            # 获取统计信息
            stats = monitor.get_memory_stats()
            
            # XGBoost 主要在 CPU 上运行，GPU 显存使用应该很少
            assert stats['peak_memory_gb'] < 5.0, \
                f"XGBoost 显存使用过高: {stats['peak_memory_gb']:.2f} GB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
    def test_tabr_memory_usage(self, breast_cancer_data):
        """测试 TabR 模型的显存使用"""
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 使用显存监控器
        with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
            # 训练模型（使用较小的配置）
            model = TabRBaseline(
                random_state=42,
                k_neighbors=3,
                hidden_dim=64,
                epochs=5,
                batch_size=32
            )
            model.fit(X_train[:100], y_train[:100])  # 使用小数据集快速测试
            
            # 检查显存
            monitor.check_memory(context="TabR Training")
            
            # 获取统计信息
            stats = monitor.get_memory_stats()
            
            # TabR 应该在 RTX 3090 (24GB) 上可运行
            assert stats['peak_memory_gb'] < 24.0, \
                f"TabR 显存使用超过 24GB: {stats['peak_memory_gb']:.2f} GB"
            
            # 验证没有触发警告
            print(f"TabR 峰值显存使用: {stats['peak_memory_gb']:.2f} GB")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
    def test_hyperfast_memory_usage(self, breast_cancer_data):
        """测试 HyperFast 模型的显存使用"""
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 使用显存监控器
        with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
            # 训练模型（使用较小的配置）
            model = HyperFastBaseline(
                random_state=42,
                hidden_dim=128,
                epochs=5,
                batch_size=32
            )
            model.fit(X_train[:100], y_train[:100])  # 使用小数据集快速测试
            
            # 检查显存
            monitor.check_memory(context="HyperFast Training")
            
            # 获取统计信息
            stats = monitor.get_memory_stats()
            
            # HyperFast 应该在 RTX 3090 (24GB) 上可运行
            assert stats['peak_memory_gb'] < 24.0, \
                f"HyperFast 显存使用超过 24GB: {stats['peak_memory_gb']:.2f} GB"
            
            print(f"HyperFast 峰值显存使用: {stats['peak_memory_gb']:.2f} GB")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
    def test_transtee_memory_usage(self):
        """测试 TransTEE 模型的显存使用"""
        # 生成模拟因果推断数据
        np.random.seed(42)
        n_samples = 100
        n_features = 25
        
        X = np.random.randn(n_samples, n_features)
        t = np.random.randint(0, 2, n_samples)
        y = np.random.randn(n_samples)
        
        # 使用显存监控器
        with GPUMemoryMonitor(warning_threshold_gb=20.0) as monitor:
            # 训练模型（使用较小的配置）
            model = TransTEEBaseline(
                random_state=42,
                hidden_dim=64,
                epochs=5,
                batch_size=32
            )
            model.fit(X, t, y)
            
            # 检查显存
            monitor.check_memory(context="TransTEE Training")
            
            # 获取统计信息
            stats = monitor.get_memory_stats()
            
            # TransTEE 应该在 RTX 3090 (24GB) 上可运行
            assert stats['peak_memory_gb'] < 24.0, \
                f"TransTEE 显存使用超过 24GB: {stats['peak_memory_gb']:.2f} GB"
            
            print(f"TransTEE 峰值显存使用: {stats['peak_memory_gb']:.2f} GB")


class TestGPUMemoryInfo:
    """测试 GPU 显存信息获取功能"""
    
    def test_get_gpu_memory_info(self):
        """测试获取 GPU 显存信息"""
        info = get_gpu_memory_info()
        
        # 验证返回的信息
        assert 'allocated_gb' in info
        assert 'reserved_gb' in info
        assert 'total_gb' in info
        assert 'free_gb' in info
        assert 'utilization' in info
        
        # 验证数值范围
        assert info['allocated_gb'] >= 0
        assert info['free_gb'] >= 0
        assert info['utilization'] >= 0
        assert info['utilization'] <= 1
        
        # 如果 CUDA 可用，验证总显存
        if torch.cuda.is_available():
            assert info['total_gb'] > 0
            print(f"\nGPU 显存信息:")
            print(f"  总显存: {info['total_gb']:.2f} GB")
            print(f"  已使用: {info['allocated_gb']:.2f} GB")
            print(f"  可用: {info['free_gb']:.2f} GB")
            print(f"  利用率: {info['utilization'] * 100:.1f}%")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])

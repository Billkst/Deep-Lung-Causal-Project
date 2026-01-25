"""
DLCNet 单元测试

测试 DLCNet 主模型的所有功能。
"""

import pytest
import torch
import numpy as np
from src.dlc.dlc_net import DLCNet
from src.baselines.base_model import BaseModel


class TestDLCNetInitialization:
    """测试 DLCNet 初始化"""
    
    def test_dlcnet_initialization(self):
        """
        测试 DLCNet 初始化
        
        验收标准:
            - 模型成功初始化
            - 所有子模块存在
            - 超参数正确设置
        """
        model = DLCNet(
            input_dim=23,
            d_conf=8,
            d_effect=16,
            d_hidden=64,
            num_heads=4,
            num_layers=2,
            lambda_hsic=0.1,
            lambda_pred=1.0,
            random_state=42
        )
        
        # 断言: 模型成功初始化
        assert model is not None
        
        # 断言: 子模块存在
        assert hasattr(model, 'causal_vae')
        assert hasattr(model, 'hypergraph_nn')
        assert hasattr(model, 'outcome_head_0')
        assert hasattr(model, 'outcome_head_1')
        
        # 断言: 超参数正确设置
        assert model.input_dim == 23
        assert model.d_conf == 8
        assert model.d_effect == 16
        assert model.d_hidden == 64
        assert model.lambda_hsic == 0.1
        assert model.lambda_pred == 1.0
        assert model.random_state == 42
    
    def test_dlcnet_inherits_basemodel(self):
        """
        测试 DLCNet 继承 BaseModel
        
        验收标准:
            - DLCNet 是 BaseModel 的子类
            - 实现所有抽象方法
        """
        model = DLCNet()
        
        # 断言: 继承自 BaseModel
        assert isinstance(model, BaseModel)
        
        # 断言: 实现所有抽象方法
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'evaluate')
        
        # 断言: 方法是可调用的
        assert callable(model.fit)
        assert callable(model.predict)
        assert callable(model.predict_proba)
        assert callable(model.evaluate)


class TestDLCNetForward:
    """测试 DLCNet 前向传播"""
    
    def test_forward_output_keys(self):
        """
        测试 forward 返回字典的键
        
        验收标准:
            - 返回字典包含所有必需的键
        """
        model = DLCNet()
        model.eval()
        
        # 生成随机输入
        X = torch.randn(32, 23)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(X)
        
        # 断言: 返回字典包含所有必需的键
        required_keys = ['Y_0', 'Y_1', 'ITE', 'Z_conf', 'Z_effect', 'X_recon']
        for key in required_keys:
            assert key in outputs, f"Missing key: {key}"
    
    def test_forward_output_shapes(self):
        """
        测试所有输出张量的形状
        
        验收标准:
            - 所有输出张量形状正确
        """
        model = DLCNet(
            input_dim=23,
            d_conf=8,
            d_effect=16,
            d_hidden=64
        )
        model.eval()
        
        batch_size = 32
        X = torch.randn(batch_size, 23)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(X)
        
        # 断言: 输出形状正确
        assert outputs['Y_0'].shape == (batch_size, 1), f"Y_0 shape: {outputs['Y_0'].shape}"
        assert outputs['Y_1'].shape == (batch_size, 1), f"Y_1 shape: {outputs['Y_1'].shape}"
        assert outputs['ITE'].shape == (batch_size, 1), f"ITE shape: {outputs['ITE'].shape}"
        assert outputs['Z_conf'].shape == (batch_size, 8), f"Z_conf shape: {outputs['Z_conf'].shape}"
        assert outputs['Z_effect'].shape == (batch_size, 16), f"Z_effect shape: {outputs['Z_effect'].shape}"
        assert outputs['X_recon'].shape == (batch_size, 23), f"X_recon shape: {outputs['X_recon'].shape}"
    
    def test_forward_output_ranges(self):
        """
        测试输出值的范围
        
        验收标准:
            - Y_0 和 Y_1 在 [0, 1] 范围内（sigmoid 输出）
            - ITE 在 [-1, 1] 范围内
        """
        model = DLCNet()
        model.eval()
        
        X = torch.randn(32, 23)
        
        with torch.no_grad():
            outputs = model(X)
        
        # 断言: Y_0 和 Y_1 在 [0, 1] 范围内
        assert (outputs['Y_0'] >= 0).all() and (outputs['Y_0'] <= 1).all(), \
            f"Y_0 out of range: min={outputs['Y_0'].min()}, max={outputs['Y_0'].max()}"
        assert (outputs['Y_1'] >= 0).all() and (outputs['Y_1'] <= 1).all(), \
            f"Y_1 out of range: min={outputs['Y_1'].min()}, max={outputs['Y_1'].max()}"
        
        # 断言: ITE 在 [-1, 1] 范围内
        assert (outputs['ITE'] >= -1).all() and (outputs['ITE'] <= 1).all(), \
            f"ITE out of range: min={outputs['ITE'].min()}, max={outputs['ITE'].max()}"


class TestDLCNetLoss:
    """测试 DLCNet 损失计算"""
    
    def test_compute_loss_keys(self):
        """
        测试 compute_loss 返回的键
        
        验收标准:
            - 返回字典包含所有损失项
        """
        model = DLCNet()
        model.eval()
        
        X = torch.randn(32, 23)
        y = torch.randint(0, 2, (32,))
        
        with torch.no_grad():
            outputs = model(X)
            losses = model.compute_loss(X, y, outputs)
        
        # 断言: 返回字典包含所有损失项
        required_keys = ['loss_total', 'loss_recon', 'loss_hsic', 'loss_pred']
        for key in required_keys:
            assert key in losses, f"Missing loss key: {key}"
    
    def test_compute_loss_values(self):
        """
        测试损失值的合理性
        
        验收标准:
            - 所有损失值为非负数
            - 总损失等于各项损失的加权和
        """
        model = DLCNet(lambda_hsic=0.1, lambda_pred=1.0)
        model.eval()
        
        X = torch.randn(32, 23)
        y = torch.randint(0, 2, (32,))
        
        with torch.no_grad():
            outputs = model(X)
            losses = model.compute_loss(X, y, outputs)
        
        # 断言: 所有损失值为非负数
        assert losses['loss_recon'] >= 0, f"loss_recon < 0: {losses['loss_recon']}"
        assert losses['loss_hsic'] >= 0, f"loss_hsic < 0: {losses['loss_hsic']}"
        assert losses['loss_pred'] >= 0, f"loss_pred < 0: {losses['loss_pred']}"
        assert losses['loss_total'] >= 0, f"loss_total < 0: {losses['loss_total']}"
        
        # 断言: 总损失等于各项损失的加权和（允许小误差）
        expected_total = (
            losses['loss_recon'] + 
            model.lambda_hsic * losses['loss_hsic'] + 
            model.lambda_pred * losses['loss_pred']
        )
        assert torch.isclose(losses['loss_total'], expected_total, rtol=1e-5), \
            f"loss_total mismatch: {losses['loss_total']} vs {expected_total}"


class TestDLCNetTraining:
    """测试 DLCNet 训练方法"""
    
    def test_fit_method_exists(self):
        """
        测试 fit 方法存在
        
        验收标准:
            - fit 方法存在且可调用
        """
        model = DLCNet()
        
        assert hasattr(model, 'fit')
        assert callable(model.fit)
    
    def test_fit_basic_functionality(self):
        """
        测试 fit 方法的基本功能
        
        验收标准:
            - fit 方法可以在小数据集上运行
            - 返回 self（支持链式调用）
        """
        model = DLCNet()
        
        # 生成小数据集
        X_train = np.random.randn(100, 23)
        y_train = np.random.randint(0, 2, 100)
        
        # 训练模型（使用很少的 epochs）
        result = model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=False)
        
        # 断言: 返回 self
        assert result is model


class TestDLCNetPrediction:
    """测试 DLCNet 预测方法"""
    
    def test_predict_method_output(self):
        """
        测试 predict 输出形状和类型
        
        验收标准:
            - 输出形状正确
            - 输出值在 {0, 1}
        """
        model = DLCNet()
        
        # 生成测试数据
        X_test = np.random.randn(50, 23)
        
        # 预测（不需要训练，只测试接口）
        y_pred = model.predict(X_test)
        
        # 断言: 输出形状正确
        assert y_pred.shape == (50,), f"Unexpected shape: {y_pred.shape}"
        
        # 断言: 输出类型为 numpy array
        assert isinstance(y_pred, np.ndarray)
        
        # 断言: 输出值在 {0, 1}
        assert set(y_pred) <= {0, 1}, f"Invalid predictions: {set(y_pred)}"
    
    def test_predict_proba_method_output(self):
        """
        测试 predict_proba 输出
        
        验收标准:
            - 输出形状为 [N, 2]
            - 每行和为 1
            - 所有值在 [0, 1]
        """
        model = DLCNet()
        
        X_test = np.random.randn(50, 23)
        
        # 预测概率
        y_proba = model.predict_proba(X_test)
        
        # 断言: 输出形状正确
        assert y_proba.shape == (50, 2), f"Unexpected shape: {y_proba.shape}"
        
        # 断言: 输出类型为 numpy array
        assert isinstance(y_proba, np.ndarray)
        
        # 断言: 所有值在 [0, 1]
        assert (y_proba >= 0).all() and (y_proba <= 1).all(), \
            f"Probabilities out of range: min={y_proba.min()}, max={y_proba.max()}"
        
        # 断言: 每行和为 1（允许小误差）
        row_sums = y_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, rtol=1e-5), \
            f"Row sums not equal to 1: {row_sums}"


class TestDLCNetEvaluation:
    """测试 DLCNet 评估方法"""
    
    def test_evaluate_method_output(self):
        """
        测试 evaluate 返回指标字典
        
        验收标准:
            - 返回字典包含所有必需的指标
            - 所有指标值在合理范围内
        """
        model = DLCNet()
        
        # 生成测试数据
        X_test = np.random.randn(100, 23)
        y_test = np.random.randint(0, 2, 100)
        
        # 评估
        metrics = model.evaluate(X_test, y_test)
        
        # 断言: 返回字典
        assert isinstance(metrics, dict)
        
        # 断言: 包含所有必需的指标
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # 断言: 所有指标值在 [0, 1] 范围内
        for metric, value in metrics.items():
            assert 0 <= value <= 1, f"{metric} out of range: {value}"


class TestDLCNetIntegration:
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """
        测试完整的训练-预测-评估流程
        
        验收标准:
            - 完整流程无错误
            - 各阶段输出合理
        """
        model = DLCNet()
        
        # 生成合成数据
        np.random.seed(42)
        X_train = np.random.randn(200, 23)
        y_train = np.random.randint(0, 2, 200)
        X_test = np.random.randn(50, 23)
        y_test = np.random.randint(0, 2, 50)
        
        # 训练
        model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=False)
        
        # 预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # 评估
        metrics = model.evaluate(X_test, y_test)
        
        # 断言: 所有步骤成功完成
        assert y_pred is not None
        assert y_proba is not None
        assert metrics is not None
        assert 'accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

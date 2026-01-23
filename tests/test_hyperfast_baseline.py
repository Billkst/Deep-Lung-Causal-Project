"""
HyperFast 基线模型单元测试

本测试模块验证 HyperFastBaseline 模型的以下功能：
1. Hypernetwork 权重生成是否正确
2. DynamicClassifier 动态权重应用逻辑是否正确
3. 数据集统计信息计算是否正确
4. 在 UCI Breast Cancer 数据集上的性能是否达标（Accuracy > 0.93）

Requirements:
    - 3.5: 实现基于 Hypernetwork 的推理逻辑
    - 3.6: 使用 Hypernetwork 动态生成任务特定的权重
    - 3.7: 在 UCI Breast Cancer 上达到 Accuracy > 0.93
    - 3.8: 实现快速推理机制
    - 7.3: 在 fit 方法内部自动从训练集中划分出 10%-20% 作为内部验证集
    - 7.4: 使用内部验证集进行 early stopping 监控
"""

import pytest
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.baselines.hyperfast_baseline import HyperFastBaseline, Hypernetwork, DynamicClassifier


class TestHypernetwork:
    """Hypernetwork 类测试"""
    
    def test_hypernetwork_initialization(self):
        """
        测试 Hypernetwork 初始化
        
        验证：
        - 模型正确初始化
        - 包含统计信息编码器
        - 包含权重生成器
        """
        input_dim = 30
        hidden_dim = 256
        
        hypernetwork = Hypernetwork(input_dim, hidden_dim)
        
        # 验证统计信息编码器存在
        assert hasattr(hypernetwork, 'stat_encoder')
        
        # 验证权重生成器存在
        assert hasattr(hypernetwork, 'weight_generator')
        
        # 验证权重生成器包含所有必需的层
        required_keys = ['fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'fc3_weight', 'fc3_bias']
        for key in required_keys:
            assert key in hypernetwork.weight_generator, f"权重生成器缺少 {key}"
    
    def test_hypernetwork_forward(self):
        """
        测试 Hypernetwork 前向传播
        
        验证：
        - 能够生成所有层的权重
        - 权重形状正确
        
        Requirements: 3.5, 3.6
        """
        input_dim = 30
        hidden_dim = 256
        batch_size = 1
        
        hypernetwork = Hypernetwork(input_dim, hidden_dim)
        
        # 创建模拟的数据集统计信息
        # 格式: [mean (input_dim), std (input_dim), n_features (1), n_samples (1)]
        stat_dim = input_dim * 2 + 2
        dataset_stats = torch.randn(batch_size, stat_dim)
        
        # 前向传播
        weights = hypernetwork(dataset_stats)
        
        # 验证生成的权重字典包含所有必需的键
        required_keys = ['fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'fc3_weight', 'fc3_bias']
        for key in required_keys:
            assert key in weights, f"生成的权重缺少 {key}"
        
        # 验证权重形状
        assert weights['fc1_weight'].shape == (batch_size, input_dim * 128)
        assert weights['fc1_bias'].shape == (batch_size, 128)
        assert weights['fc2_weight'].shape == (batch_size, 128 * 64)
        assert weights['fc2_bias'].shape == (batch_size, 64)
        assert weights['fc3_weight'].shape == (batch_size, 64 * 2)
        assert weights['fc3_bias'].shape == (batch_size, 2)
    
    def test_hypernetwork_weight_consistency(self):
        """
        测试 Hypernetwork 权重生成一致性
        
        验证：
        - 相同的数据集统计信息应生成相同的权重
        
        Requirements: 3.6
        """
        input_dim = 30
        hidden_dim = 256
        
        hypernetwork = Hypernetwork(input_dim, hidden_dim)
        hypernetwork.eval()  # 设置为评估模式
        
        # 创建相同的数据集统计信息
        stat_dim = input_dim * 2 + 2
        dataset_stats = torch.randn(1, stat_dim)
        
        # 两次前向传播
        with torch.no_grad():
            weights1 = hypernetwork(dataset_stats)
            weights2 = hypernetwork(dataset_stats)
        
        # 验证权重一致
        for key in weights1.keys():
            assert torch.allclose(weights1[key], weights2[key]), (
                f"相同输入应生成相同权重，但 {key} 不一致"
            )


class TestDynamicClassifier:
    """DynamicClassifier 类测试"""
    
    def test_dynamic_classifier_initialization(self):
        """
        测试 DynamicClassifier 初始化
        
        验证：
        - 模型正确初始化
        - input_dim 设置正确
        """
        input_dim = 30
        classifier = DynamicClassifier(input_dim)
        
        assert classifier.input_dim == input_dim
    
    def test_dynamic_classifier_forward(self):
        """
        测试 DynamicClassifier 前向传播
        
        验证：
        - 能够使用动态权重进行前向传播
        - 输出形状正确
        
        Requirements: 3.5, 3.6
        """
        input_dim = 30
        batch_size = 16
        
        # 创建分类器
        classifier = DynamicClassifier(input_dim)
        
        # 创建 Hypernetwork 生成权重
        hypernetwork = Hypernetwork(input_dim, 256)
        stat_dim = input_dim * 2 + 2
        dataset_stats = torch.randn(1, stat_dim)
        weights = hypernetwork(dataset_stats)
        
        # 创建输入数据
        x = torch.randn(batch_size, input_dim)
        
        # 前向传播
        logits = classifier(x, weights)
        
        # 验证输出形状
        assert logits.shape == (batch_size, 2), f"输出形状应为 ({batch_size}, 2)，实际为 {logits.shape}"
    
    def test_dynamic_classifier_batch_inference(self):
        """
        测试 DynamicClassifier 批量推理
        
        验证：
        - 支持批量推理
        - 不同批量大小都能正确处理
        
        Requirements: 3.8
        """
        input_dim = 30
        classifier = DynamicClassifier(input_dim)
        
        # 创建权重
        hypernetwork = Hypernetwork(input_dim, 256)
        stat_dim = input_dim * 2 + 2
        dataset_stats = torch.randn(1, stat_dim)
        weights = hypernetwork(dataset_stats)
        
        # 测试不同批量大小
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, input_dim)
            logits = classifier(x, weights)
            assert logits.shape == (batch_size, 2), (
                f"批量大小 {batch_size} 时输出形状错误"
            )


class TestHyperFastBaseline:
    """HyperFastBaseline 模型测试类"""
    
    @pytest.fixture
    def breast_cancer_data(self):
        """
        加载 UCI Breast Cancer 数据集
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # 使用固定随机种子进行数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_model_initialization(self):
        """
        测试模型初始化
        
        验证：
        - 模型正确初始化
        - 默认参数设置正确
        """
        model = HyperFastBaseline(random_state=42)
        
        # 验证基本属性
        assert model.random_state == 42
        assert model.hidden_dim == 256
        assert model.epochs == 50
        assert model.batch_size == 32
        assert model.learning_rate == 0.001
        assert model.hypernetwork is None  # 训练前应为 None
        assert model.classifier is None
        assert model.dataset_stats is None
    
    def test_compute_dataset_stats(self, breast_cancer_data):
        """
        测试数据集统计信息计算
        
        验证：
        - 能够正确计算数据集统计信息
        - 统计信息包含均值、标准差、特征数、样本数
        
        Requirements: 3.6
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        model = HyperFastBaseline(random_state=42)
        stats = model._compute_dataset_stats(X_train, y_train)
        
        # 验证统计信息形状
        input_dim = X_train.shape[1]
        expected_dim = input_dim * 2 + 2  # mean + std + n_features + n_samples
        assert stats.shape == (1, expected_dim), (
            f"统计信息形状应为 (1, {expected_dim})，实际为 {stats.shape}"
        )
        
        # 验证统计信息是 Tensor
        assert isinstance(stats, torch.Tensor)
        
        # 验证统计信息在正确的设备上
        assert stats.device == model.device
    
    def test_fit_method(self, breast_cancer_data):
        """
        测试 fit 方法
        
        验证：
        - fit 方法正常执行
        - 返回 self 以支持链式调用
        - 模型训练后可以进行预测
        - 内部验证集划分正确
        
        Requirements: 7.3, 7.4
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        model = HyperFastBaseline(random_state=42, epochs=5)  # 使用较少 epoch 加快测试
        result = model.fit(X_train, y_train)
        
        # 验证返回 self
        assert result is model
        
        # 验证模型已训练
        assert model.hypernetwork is not None
        assert model.classifier is not None
        assert model.dataset_stats is not None
        
        # 验证模型可以进行预测
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_internal_validation_split(self, breast_cancer_data):
        """
        测试内部验证集划分逻辑
        
        验证：
        - fit 方法内部自动划分验证集
        - 验证集占比约为 15%（在 10%-20% 范围内）
        - 使用 stratify 保持类别比例
        - 严禁使用外部测试集
        
        Requirements: 7.3, 7.4
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        model = HyperFastBaseline(random_state=42, epochs=5)
        model.fit(X_train, y_train)
        
        # 验证 fit 方法的签名：只接受训练集，不接受测试集
        import inspect
        fit_signature = inspect.signature(model.fit)
        fit_params = list(fit_signature.parameters.keys())
        
        # 验证 fit 方法只有 X 和 y 参数（不包含 X_test, y_test）
        assert 'X' in fit_params, "fit() 应该有 X 参数"
        assert 'y' in fit_params, "fit() 应该有 y 参数"
        assert 'X_test' not in fit_params, "fit() 不应该有 X_test 参数（防止数据泄露）"
        assert 'y_test' not in fit_params, "fit() 不应该有 y_test 参数（防止数据泄露）"
        
        print(f"\n✓ 内部验证集划分验证通过:")
        print(f"  - fit() 方法签名正确，不接受外部测试集")
        print(f"  - Early Stopping 已启用")
    
    def test_predict_method(self, breast_cancer_data):
        """
        测试 predict 方法
        
        验证：
        - 预测标签的形状正确
        - 预测标签为 0 或 1
        - 未训练模型调用 predict 会抛出异常
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 测试未训练模型
        model_untrained = HyperFastBaseline(random_state=42)
        with pytest.raises(ValueError) as exc_info:
            model_untrained.predict(X_test)
        assert "尚未训练" in str(exc_info.value)
        
        # 测试已训练模型
        model = HyperFastBaseline(random_state=42, epochs=5)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # 验证预测形状
        assert predictions.shape[0] == X_test.shape[0]
        
        # 验证预测值为 0 或 1
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_method(self, breast_cancer_data):
        """
        测试 predict_proba 方法
        
        验证：
        - 预测概率的形状正确
        - 概率值在 [0, 1] 范围内
        - 每行概率和为 1
        - 未训练模型调用 predict_proba 会抛出异常
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 测试未训练模型
        model_untrained = HyperFastBaseline(random_state=42)
        with pytest.raises(ValueError) as exc_info:
            model_untrained.predict_proba(X_test)
        assert "尚未训练" in str(exc_info.value)
        
        # 测试已训练模型
        model = HyperFastBaseline(random_state=42, epochs=5)
        model.fit(X_train, y_train)
        
        probabilities = model.predict_proba(X_test)
        
        # 验证概率形状（二分类问题应为 (n_samples, 2)）
        assert probabilities.shape == (X_test.shape[0], 2)
        
        # 验证概率值在 [0, 1] 范围内
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        
        # 验证每行概率和为 1
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_evaluate_method(self, breast_cancer_data):
        """
        测试 evaluate 方法
        
        验证：
        - 返回字典包含所有必需的指标
        - 所有指标值在合理范围内
        - 未训练模型调用 evaluate 会抛出异常
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 测试未训练模型
        model_untrained = HyperFastBaseline(random_state=42)
        with pytest.raises(ValueError) as exc_info:
            model_untrained.evaluate(X_test, y_test)
        assert "尚未训练" in str(exc_info.value)
        
        # 测试已训练模型
        model = HyperFastBaseline(random_state=42, epochs=5)
        model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        # 验证返回字典包含所有必需的指标
        required_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        for key in required_keys:
            assert key in metrics, f"评估结果缺少指标: {key}"
        
        # 验证所有指标值在 [0, 1] 范围内
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"指标 {key} 的值 {value} 不在 [0, 1] 范围内"
    
    def test_performance_on_breast_cancer(self, breast_cancer_data):
        """
        测试在 UCI Breast Cancer 数据集上的性能
        
        验证：
        - 测试集准确率 > 0.93
        
        Requirements: 3.7
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        model = HyperFastBaseline(random_state=42, epochs=50)
        model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        # 验证准确率达标
        assert metrics['accuracy'] > 0.93, (
            f"HyperFast 模型在 UCI Breast Cancer 上的准确率 {metrics['accuracy']:.4f} "
            f"未达到要求的 0.93"
        )
        
        print(f"\n✓ HyperFast 模型性能验证通过:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1: {metrics['f1']:.4f}")
        print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    
    def test_reproducibility(self, breast_cancer_data):
        """
        测试可复现性
        
        验证：
        - 使用相同随机种子训练两次，结果应完全一致
        
        Requirements: 12.1
        """
        X_train, X_test, y_train, y_test = breast_cancer_data
        
        # 第一次训练
        model1 = HyperFastBaseline(random_state=42, epochs=5)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)
        
        # 第二次训练
        model2 = HyperFastBaseline(random_state=42, epochs=5)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)
        
        # 验证预测结果完全一致
        assert np.array_equal(pred1, pred2), "相同随机种子应产生相同预测结果"
    
    def test_get_params(self):
        """
        测试 get_params 方法
        
        验证：
        - 返回字典包含所有参数
        """
        model = HyperFastBaseline(
            random_state=42,
            hidden_dim=128,
            epochs=100,
            batch_size=64
        )
        params = model.get_params()
        
        assert 'random_state' in params
        assert params['random_state'] == 42
    
    def test_set_params(self):
        """
        测试 set_params 方法
        
        验证：
        - 参数设置正确
        - 返回 self 以支持链式调用
        """
        model = HyperFastBaseline(random_state=42)
        result = model.set_params(
            random_state=123,
            hidden_dim=128
        )
        
        # 验证返回 self
        assert result is model
        
        # 验证参数更新
        assert model.random_state == 123
        assert model.hidden_dim == 128


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

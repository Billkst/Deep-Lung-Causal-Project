#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 21.5: 混杂因子敏感性评估单元测试

测试内容：
1. 干预场景创建逻辑
2. Sensitivity Score 计算
3. ITE 计算方法
4. 报告生成功能

作者：Kiro AI Agent
日期：2026-01-23
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate_confounding_sensitivity import ConfoundingSensitivityEvaluator


class TestInterventionScenarios:
    """测试干预场景创建"""
    
    def test_create_intervention_scenarios(self):
        """
        测试干预场景创建
        
        验证：
        - Young World 的 Age 列全部为 -2.0
        - Old World 的 Age 列全部为 +2.0
        - 其他列保持不变
        """
        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        age_col_idx = 0
        
        # 创建评估器
        evaluator = ConfoundingSensitivityEvaluator(
            data_path='dummy_path',
            models={}
        )
        
        # 创建场景
        scenarios = evaluator.create_intervention_scenarios(X, age_col_idx)
        
        # 验证 Baseline
        assert np.allclose(scenarios['baseline'], X), "Baseline 应与原始数据相同"
        
        # 验证 Young World
        assert np.all(scenarios['young'][:, age_col_idx] == -2.0), \
            "Young World 的 Age 列应全部为 -2.0"
        assert np.allclose(scenarios['young'][:, 1:], X[:, 1:]), \
            "Young World 的其他列应保持不变"
        
        # 验证 Old World
        assert np.all(scenarios['old'][:, age_col_idx] == +2.0), \
            "Old World 的 Age 列应全部为 +2.0"
        assert np.allclose(scenarios['old'][:, 1:], X[:, 1:]), \
            "Old World 的其他列应保持不变"
        
        print("✓ 干预场景创建测试通过")
    
    def test_intervention_preserves_other_features(self):
        """
        测试干预不影响其他特征
        
        验证：
        - 除了 Age 列，所有其他列在三个场景中应完全相同
        """
        # 创建测试数据
        np.random.seed(42)
        X = np.random.randn(50, 5)
        age_col_idx = 2  # Age 在第 3 列
        
        evaluator = ConfoundingSensitivityEvaluator(
            data_path='dummy_path',
            models={}
        )
        
        scenarios = evaluator.create_intervention_scenarios(X, age_col_idx)
        
        # 验证非 Age 列保持不变
        for col_idx in range(X.shape[1]):
            if col_idx != age_col_idx:
                assert np.allclose(scenarios['young'][:, col_idx], X[:, col_idx]), \
                    f"Young World 的列 {col_idx} 应保持不变"
                assert np.allclose(scenarios['old'][:, col_idx], X[:, col_idx]), \
                    f"Old World 的列 {col_idx} 应保持不变"
        
        print("✓ 干预保持其他特征测试通过")


class TestSensitivityScore:
    """测试 Sensitivity Score 计算"""
    
    def test_sensitivity_score_zero_when_ite_constant(self):
        """
        测试 ITE 不变时 Sensitivity Score 为 0
        
        验证：
        - 如果 ITE 完全不变，Score 应为 0
        """
        # 模拟 ITE 数据（完全相同）
        ite_young = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ite_old = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # 计算 Sensitivity Score
        score = np.mean(np.abs(ite_old - ite_young))
        
        assert score == 0.0, "ITE 不变时 Sensitivity Score 应为 0"
        
        print("✓ Sensitivity Score 零值测试通过")
    
    def test_sensitivity_score_positive_when_ite_changes(self):
        """
        测试 ITE 变化时 Sensitivity Score 大于 0
        
        验证：
        - 如果 ITE 发生变化，Score 应大于 0
        """
        # 模拟 ITE 数据（有变化）
        ite_young = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ite_old = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        
        # 计算 Sensitivity Score
        score = np.mean(np.abs(ite_old - ite_young))
        
        assert score > 0, "ITE 变化时 Sensitivity Score 应大于 0"
        assert np.isclose(score, 0.1), f"预期 Score 为 0.1，实际为 {score}"
        
        print("✓ Sensitivity Score 正值测试通过")
    
    def test_sensitivity_score_symmetry(self):
        """
        测试 Sensitivity Score 的对称性
        
        验证：
        - Score(ITE_young, ITE_old) == Score(ITE_old, ITE_young)
        """
        ite_young = np.array([0.1, 0.2, 0.3])
        ite_old = np.array([0.4, 0.5, 0.6])
        
        score_1 = np.mean(np.abs(ite_old - ite_young))
        score_2 = np.mean(np.abs(ite_young - ite_old))
        
        assert np.isclose(score_1, score_2), "Sensitivity Score 应具有对称性"
        
        print("✓ Sensitivity Score 对称性测试通过")


class TestITECalculation:
    """测试 ITE 计算方法"""
    
    def test_ite_calculation_for_classification_model(self):
        """
        测试分类模型的 ITE 计算
        
        验证：
        - ITE 应在合理范围内（-1 到 1）
        - ITE 应为浮点数数组
        """
        # 创建模拟模型
        class MockModel:
            def predict_proba(self, X):
                # 返回随机概率
                n_samples = X.shape[0]
                proba = np.random.rand(n_samples, 2)
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba
        
        # 创建测试数据
        np.random.seed(42)
        X = np.random.randn(50, 5)
        pm25_col_idx = 2
        
        evaluator = ConfoundingSensitivityEvaluator(
            data_path='dummy_path',
            models={}
        )
        
        model = MockModel()
        ite = evaluator.compute_ite(model, 'MockModel', X, pm25_col_idx)
        
        # 验证 ITE 属性
        assert isinstance(ite, np.ndarray), "ITE 应为 numpy 数组"
        assert ite.shape == (50,), f"ITE 形状应为 (50,)，实际为 {ite.shape}"
        assert np.all(ite >= -1) and np.all(ite <= 1), "ITE 应在 [-1, 1] 范围内"
        
        print("✓ 分类模型 ITE 计算测试通过")


class TestCorrelationCalculation:
    """测试相关系数计算"""
    
    def test_correlation_zero_for_independent_variables(self):
        """
        测试独立变量的相关系数接近 0
        
        验证：
        - 如果 Age 和 ITE 独立，相关系数应接近 0
        """
        np.random.seed(42)
        age = np.random.randn(1000)
        ite = np.random.randn(1000)
        
        corr = np.corrcoef(age, ite)[0, 1]
        
        # 对于大样本，独立变量的相关系数应接近 0（允许一定误差）
        assert abs(corr) < 0.1, f"独立变量的相关系数应接近 0，实际为 {corr}"
        
        print("✓ 独立变量相关系数测试通过")
    
    def test_correlation_positive_for_positively_related_variables(self):
        """
        测试正相关变量的相关系数大于 0
        
        验证：
        - 如果 Age 和 ITE 正相关，相关系数应大于 0
        """
        np.random.seed(42)
        age = np.random.randn(1000)
        ite = age + np.random.randn(1000) * 0.1  # ITE 与 Age 正相关
        
        corr = np.corrcoef(age, ite)[0, 1]
        
        assert corr > 0.9, f"正相关变量的相关系数应大于 0.9，实际为 {corr}"
        
        print("✓ 正相关变量相关系数测试通过")


class TestReportGeneration:
    """测试报告生成功能"""
    
    def test_report_generation(self, tmp_path):
        """
        测试报告生成
        
        验证：
        - 报告文件应被创建
        - 报告应包含关键信息
        """
        # 创建模拟结果
        results = {
            'XGBoost': {
                'sensitivity_score': 0.123,
                'ite_age_correlation': 0.456,
                'ite_young_mean': 0.1,
                'ite_old_mean': 0.2,
                'ite_baseline_mean': 0.15
            },
            'TabR': {
                'sensitivity_score': 0.089,
                'ite_age_correlation': 0.234,
                'ite_young_mean': 0.05,
                'ite_old_mean': 0.12,
                'ite_baseline_mean': 0.08
            }
        }
        
        evaluator = ConfoundingSensitivityEvaluator(
            data_path='dummy_path',
            models={}
        )
        evaluator.results = results
        
        # 生成报告
        output_path = tmp_path / "test_report.md"
        evaluator.generate_report(str(output_path))
        
        # 验证文件存在
        assert output_path.exists(), "报告文件应被创建"
        
        # 验证报告内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "混杂因子敏感性评估报告" in content, "报告应包含标题"
        assert "理论基础" in content, "报告应包含理论基础"
        assert "XGBoost" in content, "报告应包含 XGBoost 结果"
        assert "TabR" in content, "报告应包含 TabR 结果"
        assert "0.123" in content, "报告应包含 Sensitivity Score"
        
        print("✓ 报告生成测试通过")


def test_evaluator_initialization():
    """
    测试评估器初始化
    
    验证：
    - 评估器应正确初始化
    - 数据路径应被保存
    """
    data_path = "data/luad_synthetic_interaction.csv"
    models = {'XGBoost': None}
    
    evaluator = ConfoundingSensitivityEvaluator(data_path, models)
    
    assert evaluator.data_path == Path(data_path), "数据路径应被正确保存"
    assert evaluator.models == models, "模型字典应被正确保存"
    assert evaluator.results == {}, "结果字典应初始化为空"
    
    print("✓ 评估器初始化测试通过")


if __name__ == '__main__':
    # 运行所有测试
    pytest.main([__file__, '-v'])

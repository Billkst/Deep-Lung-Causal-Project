"""
IHDP 数据加载器测试

测试 IHDPDataset 类的功能：
1. 文件存在性检查
2. 文件缺失时的错误处理
3. 数据解析逻辑
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.baselines.transtee_baseline import IHDPDataset


class TestIHDPDataset:
    """IHDP 数据集加载器测试类"""
    
    def test_file_existence_check_missing_file(self):
        """
        测试文件存在性检查 - 文件缺失场景
        
        验证：
        - 当 ihdp_npci_1.csv 文件不存在时，应抛出 FileNotFoundError
        - 错误信息应包含完整文件路径
        - 错误信息应包含下载指引
        
        Validates: Requirements 5.2, 5.6, 5.7
        """
        # 使用不存在的目录
        non_existent_dir = "data/baselines_official/IHDP_NONEXISTENT/"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            dataset = IHDPDataset(data_dir=non_existent_dir)
        
        error_message = str(exc_info.value)
        
        # 验证错误信息包含文件路径
        assert "ihdp_npci_1.csv" in error_message, "错误信息应包含文件名"
        assert non_existent_dir in error_message, "错误信息应包含目录路径"
        
        # 验证错误信息包含下载指引
        assert "下载指引" in error_message, "错误信息应包含下载指引"
        assert "IHDP" in error_message, "错误信息应提及 IHDP 数据源"
    
    def test_file_existence_check_with_valid_file(self):
        """
        测试文件存在性检查 - 文件存在场景
        
        验证：
        - 当文件存在时，不应抛出异常
        - 数据集对象应成功创建
        
        Validates: Requirements 5.2, 5.4
        """
        # 创建临时目录和测试文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试数据文件
            test_file = Path(temp_dir) / 'ihdp_npci_1.csv'
            test_data = pd.DataFrame({
                'treatment': [0, 1, 0, 1],
                'y_factual': [1.2, 2.3, 1.5, 2.8],
                'x1': [0.5, 0.6, 0.7, 0.8],
                'x2': [1.0, 1.1, 1.2, 1.3]
            })
            test_data.to_csv(test_file, index=False)
            
            # 应该成功创建数据集对象
            dataset = IHDPDataset(data_dir=temp_dir)
            assert dataset is not None
            assert dataset.data_dir == Path(temp_dir)
    
    def test_load_data_parsing(self):
        """
        测试数据解析逻辑
        
        验证：
        - 正确解析 treatment, y_factual, covariates
        - 返回的数据类型和形状正确
        - 治疗变量是二值变量 (0/1)
        
        Validates: Requirements 4.3, 4.5, 5.4
        """
        # 创建临时目录和测试文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试数据
            n_samples = 100
            n_covariates = 25
            
            # 创建数据数组（按照 IHDP 格式：treatment, y_factual, y_cfactual, covariates）
            treatment = np.random.randint(0, 2, n_samples)
            y_factual = np.random.randn(n_samples) * 2 + 5
            y_cfactual = np.random.randn(n_samples) * 2 + 5
            covariates = np.random.randn(n_samples, n_covariates)
            
            # 合并所有列
            test_data = np.column_stack([treatment, y_factual, y_cfactual, covariates])
            
            # 保存测试文件（无列名）
            test_file = Path(temp_dir) / 'ihdp_npci_1.csv'
            np.savetxt(test_file, test_data, delimiter=',')
            
            # 加载数据
            dataset = IHDPDataset(data_dir=temp_dir)
            X, t, y = dataset.load_data()
            
            # 验证数据形状
            assert X.shape == (n_samples, n_covariates), \
                f"协变量矩阵形状应为 ({n_samples}, {n_covariates})，实际为 {X.shape}"
            assert t.shape == (n_samples,), \
                f"治疗变量形状应为 ({n_samples},)，实际为 {t.shape}"
            assert y.shape == (n_samples,), \
                f"结局变量形状应为 ({n_samples},)，实际为 {y.shape}"
            
            # 验证数据类型
            assert isinstance(X, np.ndarray), "协变量应为 numpy 数组"
            assert isinstance(t, np.ndarray), "治疗变量应为 numpy 数组"
            assert isinstance(y, np.ndarray), "结局变量应为 numpy 数组"
            
            # 验证治疗变量是二值变量
            assert np.all(np.isin(t, [0, 1])), "治疗变量必须是二值变量 (0/1)"
            
            # 验证数据值与原始数据一致
            np.testing.assert_array_equal(t, treatment)
            np.testing.assert_allclose(y, y_factual, rtol=1e-10)
    
    def test_load_data_missing_columns(self):
        """
        测试数据解析 - 缺少必需列
        
        验证：
        - 当数据文件缺少 treatment 或 y_factual 列时，应抛出 ValueError
        - 错误信息应明确指出缺少的列
        
        Validates: Requirements 5.4, 5.7
        """
        # 创建临时目录和测试文件（缺少 treatment 列）
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = pd.DataFrame({
                'y_factual': [1.2, 2.3, 1.5, 2.8],
                'x1': [0.5, 0.6, 0.7, 0.8],
                'x2': [1.0, 1.1, 1.2, 1.3]
            })
            test_file = Path(temp_dir) / 'ihdp_npci_1.csv'
            test_data.to_csv(test_file, index=False)
            
            dataset = IHDPDataset(data_dir=temp_dir)
            
            with pytest.raises(ValueError) as exc_info:
                X, t, y = dataset.load_data()
            
            error_message = str(exc_info.value)
            assert "treatment" in error_message, "错误信息应指出缺少 treatment 列"
    
    def test_load_data_invalid_treatment_values(self):
        """
        测试数据解析 - 治疗变量值无效
        
        验证：
        - 当治疗变量不是二值变量时，应抛出 ValueError
        - 错误信息应说明治疗变量必须是 0/1
        
        Validates: Requirements 4.5, 5.7
        """
        # 创建临时目录和测试文件（治疗变量包含非 0/1 值）
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = pd.DataFrame({
                'treatment': [0, 1, 2, 3],  # 包含无效值 2, 3
                'y_factual': [1.2, 2.3, 1.5, 2.8],
                'x1': [0.5, 0.6, 0.7, 0.8],
                'x2': [1.0, 1.1, 1.2, 1.3]
            })
            test_file = Path(temp_dir) / 'ihdp_npci_1.csv'
            test_data.to_csv(test_file, index=False)
            
            dataset = IHDPDataset(data_dir=temp_dir)
            
            with pytest.raises(ValueError) as exc_info:
                X, t, y = dataset.load_data()
            
            error_message = str(exc_info.value)
            assert "二值变量" in error_message or "0/1" in error_message, \
                "错误信息应说明治疗变量必须是二值变量"
    
    def test_load_data_empty_covariates(self):
        """
        测试数据解析 - 协变量为空
        
        验证：
        - 当数据文件只有 treatment 和 y_factual 列时，应抛出 ValueError
        - 错误信息应说明数据格式错误
        
        Validates: Requirements 5.7
        """
        # 创建临时目录和测试文件（只有 treatment 和 y_factual，缺少 y_cfactual）
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建只有 2 列的数据（不符合 IHDP 格式）
            test_data = np.array([
                [0, 1.2],
                [1, 2.3],
                [0, 1.5],
                [1, 2.8]
            ])
            test_file = Path(temp_dir) / 'ihdp_npci_1.csv'
            np.savetxt(test_file, test_data, delimiter=',')
            
            dataset = IHDPDataset(data_dir=temp_dir)
            
            with pytest.raises(ValueError) as exc_info:
                X, t, y = dataset.load_data()
            
            error_message = str(exc_info.value)
            # 应该提示列数不足
            assert "4 列" in error_message or "格式错误" in error_message, \
                "错误信息应说明数据格式错误"
    
    def test_load_data_with_counterfactual_column(self):
        """
        测试数据解析 - 包含反事实结局列
        
        验证：
        - 当数据包含 y_cfactual 列时，应正确解析
        - 协变量应从第 4 列开始
        
        Validates: Requirements 4.3, 5.4
        """
        # 创建临时目录和测试文件（包含 y_cfactual）
        with tempfile.TemporaryDirectory() as temp_dir:
            # 按照 IHDP 格式：treatment, y_factual, y_cfactual, x1, x2
            test_data = np.array([
                [0, 1.2, 2.0, 0.5, 1.0],
                [1, 2.3, 1.5, 0.6, 1.1],
                [0, 1.5, 2.2, 0.7, 1.2],
                [1, 2.8, 1.8, 0.8, 1.3]
            ])
            test_file = Path(temp_dir) / 'ihdp_npci_1.csv'
            np.savetxt(test_file, test_data, delimiter=',')
            
            dataset = IHDPDataset(data_dir=temp_dir)
            X, t, y = dataset.load_data()
            
            # 验证协变量只包含 x1 和 x2（第 4 和第 5 列）
            assert X.shape[1] == 2, \
                f"协变量应只包含 x1 和 x2（2 列），实际为 {X.shape[1]} 列"
            
            # 验证协变量值正确
            expected_X = test_data[:, 3:]  # 第 4 列及之后
            np.testing.assert_allclose(X, expected_X, rtol=1e-10)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])

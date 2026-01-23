"""
基础设施工具函数的基本测试
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer


def test_set_global_seed():
    """测试全局随机种子设置"""
    from src.baselines.utils import set_global_seed
    
    # 设置种子
    set_global_seed(42)
    
    # 生成随机数
    random_nums_1 = np.random.randn(10)
    
    # 重新设置相同种子
    set_global_seed(42)
    
    # 再次生成随机数
    random_nums_2 = np.random.randn(10)
    
    # 验证结果一致
    assert np.array_equal(random_nums_1, random_nums_2), "相同随机种子应产生相同的随机数"
    print("✓ 全局随机种子设置测试通过")


def test_preprocess_data_basic():
    """测试数据预处理基本功能"""
    from src.baselines.utils import preprocess_data
    
    # 加载测试数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # 验证数据划分
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], "训练集和测试集样本数应等于原始数据"
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0], "训练集和测试集标签数应等于原始数据"
    
    # 验证测试集比例
    test_ratio = X_test.shape[0] / X.shape[0]
    assert abs(test_ratio - 0.2) < 0.01, "测试集比例应接近 20%"
    
    # 验证标准化
    assert scaler is not None, "应返回 StandardScaler 对象"
    
    # 验证标准化效果（训练集均值接近 0，标准差接近 1）
    # 注意：由于是在整个数据集上拟合后划分，训练集均值不会完全为 0
    # 但应该很接近（误差 < 0.1）
    assert np.allclose(X_train.mean(axis=0), 0, atol=0.1), "标准化后训练集均值应接近 0"
    assert np.allclose(X_train.std(axis=0), 1, atol=0.1), "标准化后训练集标准差应接近 1"
    
    print("✓ 数据预处理基本功能测试通过")


def test_preprocess_data_stratified():
    """测试分层划分功能"""
    from src.baselines.utils import preprocess_data
    
    # 加载测试数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 原始类别比例
    original_ratio = y.sum() / len(y)
    
    # 预处理（分层划分）
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, stratify=True)
    
    # 训练集和测试集的类别比例
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    
    # 验证比例保持（误差 < 5%）
    assert abs(train_ratio - original_ratio) < 0.05, "训练集类别比例应与原始数据一致"
    assert abs(test_ratio - original_ratio) < 0.05, "测试集类别比例应与原始数据一致"
    
    print("✓ 分层划分功能测试通过")


def test_preprocess_data_no_scaling():
    """测试不进行标准化的情况"""
    from src.baselines.utils import preprocess_data
    
    # 加载测试数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 预处理（不标准化）
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, scale=False)
    
    # 验证没有返回 scaler
    assert scaler is None, "scale=False 时不应返回 StandardScaler"
    
    # 验证数据未被标准化（均值不为 0）
    assert not np.allclose(X_train.mean(axis=0), 0, atol=1e-5), "未标准化时均值不应为 0"
    
    print("✓ 不进行标准化测试通过")


if __name__ == "__main__":
    test_set_global_seed()
    test_preprocess_data_basic()
    test_preprocess_data_stratified()
    test_preprocess_data_no_scaling()
    print("\n所有基础设施测试通过！")

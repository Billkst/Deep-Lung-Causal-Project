"""
Task 1 完成验证脚本

验证项目结构初始化与基础设施是否正确实现
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("验证 1: 模块导入")
    print("=" * 60)
    
    try:
        from src.baselines import set_global_seed, preprocess_data
        print("✓ 成功导入 set_global_seed")
        print("✓ 成功导入 preprocess_data")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_random_seed():
    """测试随机种子设置"""
    print("\n" + "=" * 60)
    print("验证 2: 随机种子设置")
    print("=" * 60)
    
    from src.baselines import set_global_seed
    import numpy as np
    
    # 设置种子并生成随机数
    set_global_seed(42)
    nums1 = np.random.randn(5)
    
    # 重新设置相同种子
    set_global_seed(42)
    nums2 = np.random.randn(5)
    
    if np.array_equal(nums1, nums2):
        print("✓ 随机种子可复现性验证通过")
        print(f"  第一次生成: {nums1[:3]}")
        print(f"  第二次生成: {nums2[:3]}")
        return True
    else:
        print("✗ 随机种子可复现性验证失败")
        return False


def test_preprocess():
    """测试数据预处理"""
    print("\n" + "=" * 60)
    print("验证 3: 数据预处理")
    print("=" * 60)
    
    from src.baselines import preprocess_data
    from sklearn.datasets import load_breast_cancer
    import numpy as np
    
    # 加载测试数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # 验证
    checks = []
    
    # 检查 1: 数据划分
    total_samples = X_train.shape[0] + X_test.shape[0]
    if total_samples == X.shape[0]:
        print(f"✓ 数据划分正确: 训练集 {X_train.shape[0]}, 测试集 {X_test.shape[0]}")
        checks.append(True)
    else:
        print(f"✗ 数据划分错误")
        checks.append(False)
    
    # 检查 2: 测试集比例
    test_ratio = X_test.shape[0] / X.shape[0]
    if abs(test_ratio - 0.2) < 0.01:
        print(f"✓ 测试集比例正确: {test_ratio:.2%}")
        checks.append(True)
    else:
        print(f"✗ 测试集比例错误: {test_ratio:.2%}")
        checks.append(False)
    
    # 检查 3: 标准化
    if scaler is not None:
        print(f"✓ StandardScaler 已返回")
        checks.append(True)
    else:
        print(f"✗ StandardScaler 未返回")
        checks.append(False)
    
    # 检查 4: 标准化效果
    mean_close_to_zero = np.allclose(X_train.mean(axis=0), 0, atol=0.1)
    std_close_to_one = np.allclose(X_train.std(axis=0), 1, atol=0.1)
    
    if mean_close_to_zero and std_close_to_one:
        print(f"✓ 标准化效果正确: 均值≈0, 标准差≈1")
        checks.append(True)
    else:
        print(f"✗ 标准化效果不正确")
        checks.append(False)
    
    return all(checks)


def test_file_structure():
    """测试文件结构"""
    print("\n" + "=" * 60)
    print("验证 4: 文件结构")
    print("=" * 60)
    
    required_files = [
        'src/baselines/__init__.py',
        'src/baselines/utils.py',
        'tests/test_utils_basic.py'
    ]
    
    checks = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
            checks.append(True)
        else:
            print(f"✗ {file_path} 不存在")
            checks.append(False)
    
    return all(checks)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Task 1 完成验证")
    print("=" * 60)
    
    results = []
    
    # 运行所有验证
    results.append(("模块导入", test_imports()))
    results.append(("随机种子设置", test_random_seed()))
    results.append(("数据预处理", test_preprocess()))
    results.append(("文件结构", test_file_structure()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Task 1 所有验证通过！")
        print("=" * 60)
        return 0
    else:
        print("❌ Task 1 部分验证失败")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

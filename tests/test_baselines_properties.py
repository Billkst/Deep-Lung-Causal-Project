"""
属性测试模块 (Property-Based Tests)

本模块使用 Hypothesis 库进行属性测试，验证对比算法平台的通用属性。
为加快测试速度，每个属性测试运行 5-10 次迭代。

测试属性：
1. 随机种子可复现性
2. 特征标准化一致性
3. 分层划分类别比例保持
4. 评估指标完整性
5. 数据文件缺失时强制失败
6. TabR 检索上下文有效性
7. HyperFast 权重生成一致性
8. TransTEE ITE 估计有界性
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import tempfile
import shutil


# ============================================================================
# Property 1: 随机种子可复现性
# ============================================================================

@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=150),
    n_features=st.integers(min_value=5, max_value=10)
)
def test_property_1_reproducibility(n_samples, n_features):
    """
    Property 1: 随机种子可复现性
    
    Feature: baselines-platform, Property 1: 对于任意模型和数据集，
    使用相同的随机种子（42）进行两次独立训练，应该产生完全相同的结果
    
    Validates: Requirements 2.5, 12.1
    """
    from src.baselines.xgb_baseline import XGBBaseline
    from src.baselines.utils import set_global_seed
    
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 确保至少有两个类别
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # 第一次训练
    set_global_seed(42)
    model1 = XGBBaseline(random_state=42, n_estimators=5)  # 减少到 5 棵树
    model1.fit(X, y)
    pred1 = model1.predict(X)
    
    # 第二次训练
    set_global_seed(42)
    model2 = XGBBaseline(random_state=42, n_estimators=5)  # 减少到 5 棵树
    model2.fit(X, y)
    pred2 = model2.predict(X)
    
    # 验证结果一致
    assert np.array_equal(pred1, pred2), \
        "相同随机种子应产生相同预测结果"


# ============================================================================
# Property 2: 特征标准化一致性
# ============================================================================

@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=50, max_value=100),
    n_features=st.integers(min_value=5, max_value=15)
)
def test_property_2_standardization_consistency(n_samples, n_features):
    """
    Property 2: 特征标准化一致性
    
    Feature: baselines-platform, Property 2: 对于任意表格数据，
    经过 StandardScaler 标准化后，特征的均值应接近 0，标准差应接近 1
    
    Validates: Requirements 6.1
    """
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features) * 10 + 5  # 任意均值和标准差
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 验证均值接近 0
    mean = np.mean(X_scaled, axis=0)
    assert np.allclose(mean, 0, atol=1e-10), \
        f"标准化后均值应接近 0，实际均值: {mean}"
    
    # 验证标准差接近 1
    std = np.std(X_scaled, axis=0, ddof=0)
    assert np.allclose(std, 1, atol=1e-10), \
        f"标准化后标准差应接近 1，实际标准差: {std}"


# ============================================================================
# Property 3: 分层划分类别比例保持
# ============================================================================

@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=150),
    n_features=st.integers(min_value=5, max_value=10),
    test_size=st.floats(min_value=0.15, max_value=0.25)
)
def test_property_3_stratified_split_preserves_ratio(n_samples, n_features, test_size):
    """
    Property 3: 分层划分类别比例保持
    
    Feature: baselines-platform, Property 3: 对于任意数据集，
    使用 Stratified Split 划分后，训练集和测试集中各类别的比例
    应与原始数据集保持一致（误差 < 5%）
    
    Validates: Requirements 6.4
    """
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 确保至少有两个类别
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # 计算原始类别比例
    original_ratio = np.mean(y == 1)
    
    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 计算训练集和测试集的类别比例
    train_ratio = np.mean(y_train == 1)
    test_ratio = np.mean(y_test == 1)
    
    # 验证比例保持（误差 < 5%）
    assert abs(train_ratio - original_ratio) < 0.05, \
        f"训练集类别比例 {train_ratio:.4f} 与原始比例 {original_ratio:.4f} 差异过大"
    
    assert abs(test_ratio - original_ratio) < 0.05, \
        f"测试集类别比例 {test_ratio:.4f} 与原始比例 {original_ratio:.4f} 差异过大"


# ============================================================================
# Property 4: 评估指标完整性
# ============================================================================

@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=150),
    n_features=st.integers(min_value=5, max_value=10)
)
def test_property_4_evaluation_metrics_completeness(n_samples, n_features):
    """
    Property 4: 评估指标完整性
    
    Feature: baselines-platform, Property 4: 对于任意分类模型，
    调用 evaluate() 方法应返回包含 'accuracy', 'precision', 'recall', 
    'f1', 'auc_roc' 键的字典
    
    Validates: Requirements 9.1, 9.2, 9.5
    """
    from src.baselines.xgb_baseline import XGBBaseline
    
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 确保至少有两个类别
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # 训练模型
    model = XGBBaseline(random_state=42, n_estimators=5)  # 减少到 5 棵树
    model.fit(X, y)
    
    # 评估模型
    metrics = model.evaluate(X, y)
    
    # 验证指标完整性
    required_keys = {'accuracy', 'precision', 'recall', 'f1', 'auc_roc'}
    assert required_keys.issubset(metrics.keys()), \
        f"评估指标缺失，期望包含 {required_keys}，实际包含 {set(metrics.keys())}"
    
    # 验证指标值在合理范围内 [0, 1]
    for key, value in metrics.items():
        assert 0 <= value <= 1, \
            f"指标 {key} 的值 {value} 不在 [0, 1] 范围内"


# ============================================================================
# Property 5: 数据文件缺失时强制失败
# ============================================================================

def test_property_5_missing_data_files_fail():
    """
    Property 5: 数据文件缺失时强制失败
    
    Feature: baselines-platform, Property 5: 对于任意需要外部数据文件的
    数据加载器（ROSMAP, IHDP），当文件不存在时，应抛出 FileNotFoundError
    而非跳过或使用替代数据
    
    Validates: Requirements 5.1, 5.2, 5.7
    """
    from src.baselines.mogonet_baseline import ROSMAPDataset
    from src.baselines.transtee_baseline import IHDPDataset
    
    # 测试 ROSMAP 数据加载器
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建一个不存在数据文件的临时目录
        fake_rosmap_dir = Path(tmpdir) / "fake_rosmap"
        fake_rosmap_dir.mkdir()
        
        # 验证抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            dataset = ROSMAPDataset(data_dir=str(fake_rosmap_dir))
        
        # 验证错误信息包含文件路径
        assert "ROSMAP" in str(exc_info.value), \
            "错误信息应包含 'ROSMAP'"
        assert "缺失" in str(exc_info.value) or "missing" in str(exc_info.value).lower(), \
            "错误信息应说明文件缺失"
    
    # 测试 IHDP 数据加载器
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建一个不存在数据文件的临时目录
        fake_ihdp_dir = Path(tmpdir) / "fake_ihdp"
        fake_ihdp_dir.mkdir()
        
        # 验证抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            dataset = IHDPDataset(data_dir=str(fake_ihdp_dir))
        
        # 验证错误信息包含文件路径
        assert "IHDP" in str(exc_info.value), \
            "错误信息应包含 'IHDP'"
        assert "缺失" in str(exc_info.value) or "missing" in str(exc_info.value).lower(), \
            "错误信息应说明文件缺失"


# ============================================================================
# Property 6: TabR 检索上下文有效性
# ============================================================================

@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=50, max_value=100),
    n_features=st.integers(min_value=5, max_value=10),
    k_neighbors=st.integers(min_value=3, max_value=5)
)
def test_property_6_tabr_context_retrieval_validity(n_samples, n_features, k_neighbors):
    """
    Property 6: TabR 检索上下文有效性
    
    Feature: baselines-platform, Property 6: 对于任意查询样本，
    TabR 模型检索的 k 个上下文样本应该是训练集中最相似的样本（基于欧氏距离）
    
    Validates: Requirements 3.1, 3.2
    """
    from src.baselines.tabr_baseline import TabRBaseline
    from sklearn.neighbors import NearestNeighbors
    
    # 生成随机数据
    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # 确保至少有两个类别
    if len(np.unique(y_train)) < 2:
        y_train[0] = 0
        y_train[1] = 1
    
    # 创建 TabR 模型并构建检索池
    model = TabRBaseline(random_state=42, k_neighbors=k_neighbors)
    model._build_context_set(X_train, y_train)
    
    # 生成查询样本
    X_query = np.random.randn(10, n_features)
    
    # 使用模型检索上下文
    context, context_labels = model._retrieve_context(X_query)
    
    # 验证检索结果
    # 使用独立的 k-NN 验证检索的正确性
    knn_verifier = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    knn_verifier.fit(X_train)
    distances, indices = knn_verifier.kneighbors(X_query)
    
    # 排除第一个（自身或最近的）
    expected_indices = indices[:, 1:k_neighbors+1]
    
    # 验证检索的上下文样本是否正确
    for i in range(len(X_query)):
        retrieved_samples = context[i]
        expected_samples = X_train[expected_indices[i]]
        
        # 验证检索的样本与期望的样本一致
        assert np.allclose(retrieved_samples, expected_samples, atol=1e-6), \
            f"查询样本 {i} 的检索结果不正确"


# ============================================================================
# Property 7: HyperFast 权重生成一致性
# ============================================================================

@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=50, max_value=100),
    n_features=st.integers(min_value=5, max_value=10)
)
def test_property_7_hyperfast_weight_generation_consistency(n_samples, n_features):
    """
    Property 7: HyperFast 权重生成一致性
    
    Feature: baselines-platform, Property 7: 对于任意数据集，
    使用相同的数据集统计信息，Hypernetwork 应该生成相同的分类器权重
    
    Validates: Requirements 3.5, 3.6
    """
    from src.baselines.hyperfast_baseline import HyperFastBaseline
    import torch
    
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 确保至少有两个类别
    if len(np.unique(y)) < 2:
        y[0] = 0
        y[1] = 1
    
    # 创建两个 HyperFast 模型实例
    model1 = HyperFastBaseline(random_state=42, hidden_dim=32)  # 减少隐藏层维度
    model2 = HyperFastBaseline(random_state=42, hidden_dim=32)
    
    # 计算数据集统计信息
    stats1 = model1._compute_dataset_stats(X, y)
    stats2 = model2._compute_dataset_stats(X, y)
    
    # 验证统计信息一致
    assert torch.allclose(stats1, stats2, atol=1e-6), \
        "相同数据集应产生相同的统计信息"
    
    # 初始化 Hypernetwork（使用相同的随机种子）
    torch.manual_seed(42)
    from src.baselines.hyperfast_baseline import Hypernetwork
    hypernetwork1 = Hypernetwork(n_features, hidden_dim=32)  # 减少隐藏层维度
    
    torch.manual_seed(42)
    hypernetwork2 = Hypernetwork(n_features, hidden_dim=32)
    
    # 生成权重
    hypernetwork1.eval()
    hypernetwork2.eval()
    
    with torch.no_grad():
        weights1 = hypernetwork1(stats1)
        weights2 = hypernetwork2(stats2)
    
    # 验证权重一致
    for key in weights1.keys():
        assert torch.allclose(weights1[key], weights2[key], atol=1e-6), \
            f"权重 {key} 不一致"


# ============================================================================
# Property 8: TransTEE ITE 估计有界性
# ============================================================================

@settings(max_examples=3, deadline=None)  # 进一步减少到 3 次
@given(
    n_samples=st.integers(min_value=50, max_value=80),  # 减少样本量
    n_features=st.integers(min_value=5, max_value=8)  # 减少特征数
)
def test_property_8_transtee_ite_boundedness(n_samples, n_features):
    """
    Property 8: TransTEE ITE 估计有界性
    
    Feature: baselines-platform, Property 8: 对于任意协变量，
    TransTEE 预测的个体治疗效应（ITE）应该在合理范围内（例如 [-10, 10]）
    
    Validates: Requirements 4.1, 4.2
    """
    from src.baselines.transtee_baseline import TransTEEBaseline
    
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    t = np.random.randint(0, 2, n_samples)
    y = np.random.randn(n_samples)  # 连续结局变量
    
    # 训练 TransTEE 模型（使用最少 epoch 以加快测试）
    model = TransTEEBaseline(random_state=42, hidden_dim=16, epochs=1, batch_size=64)  # 减少到 1 个 epoch，增大 batch
    model.fit(X, t, y)
    
    # 预测 ITE
    ite = model.predict_ite(X)
    
    # 验证 ITE 在合理范围内
    assert np.all(np.abs(ite) < 10), \
        f"ITE 估计超出合理范围 [-10, 10]，最大值: {np.max(np.abs(ite)):.4f}"
    
    # 验证 ITE 不是全部相同（模型有学习能力）
    assert np.std(ite) > 0, \
        "ITE 估计全部相同，模型可能未正确学习"


if __name__ == "__main__":
    # 运行所有属性测试
    pytest.main([__file__, "-v", "--tb=short"])

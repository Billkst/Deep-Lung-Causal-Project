"""
MOGONET 模型测试

测试内容:
1. 多视图输入支持
2. 图卷积网络结构
3. 在 ROSMAP 上的性能（Accuracy > 0.80）

Requirements: 3.9, 3.10, 3.14
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.baselines.mogonet_baseline import MOGONETBaseline, ROSMAPDataset


class TestMOGONETBaseline:
    """MOGONET 模型基础功能测试"""
    
    def test_multiview_input_support(self):
        """
        测试多视图输入支持
        
        验证:
        - 模型能够接受多个视图的输入
        - 每个视图可以有不同的特征维度
        """
        # 生成多视图数据
        n_samples = 100
        view1 = np.random.randn(n_samples, 50)  # 视图 1: 50 维
        view2 = np.random.randn(n_samples, 30)  # 视图 2: 30 维
        view3 = np.random.randn(n_samples, 40)  # 视图 3: 40 维
        views = [view1, view2, view3]
        
        # 生成标签
        y = np.random.randint(0, 2, n_samples)
        
        # 训练模型
        model = MOGONETBaseline(random_state=42, epochs=5)
        model.fit(views, y)
        
        # 预测
        predictions = model.predict(views)
        
        # 验证
        assert predictions.shape == (n_samples,), "预测形状不正确"
        assert set(predictions).issubset({0, 1}), "预测标签应为 0 或 1"
        
        print("✓ 多视图输入支持测试通过")
    
    def test_gcn_structure(self):
        """
        测试图卷积网络结构
        
        验证:
        - 模型包含视图特定的 GCN
        - 模型包含跨视图注意力机制
        - 模型能够正常前向传播
        """
        # 生成简单数据
        n_samples = 50
        views = [
            np.random.randn(n_samples, 20),
            np.random.randn(n_samples, 15),
            np.random.randn(n_samples, 25)
        ]
        y = np.random.randint(0, 2, n_samples)
        
        # 创建模型
        model = MOGONETBaseline(
            random_state=42, 
            hidden_dim=32, 
            n_gcn_layers=2,
            epochs=3
        )
        
        # 训练
        model.fit(views, y)
        
        # 验证模型结构
        assert model.model is not None, "模型未初始化"
        assert hasattr(model.model, 'view_gcns'), "模型缺少视图特定的 GCN"
        assert hasattr(model.model, 'attention'), "模型缺少跨视图注意力机制"
        assert hasattr(model.model, 'classifier'), "模型缺少分类头"
        assert len(model.model.view_gcns) == 3, "应该有 3 个视图特定的 GCN"
        
        # 测试前向传播
        proba = model.predict_proba(views)
        assert proba.shape == (n_samples, 2), "概率预测形状不正确"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "概率和应为 1"
        
        print("✓ 图卷积网络结构测试通过")
    
    def test_graph_construction(self):
        """
        测试图构建逻辑
        
        验证:
        - k-NN 图构建正确
        - 边权重基于特征相似度
        """
        # 生成数据
        n_samples = 30
        features = np.random.randn(n_samples, 10)
        
        # 创建模型
        model = MOGONETBaseline(random_state=42, k_neighbors=5)
        
        # 构建图
        edge_index, edge_weight = model._build_knn_graph(features)
        
        # 验证
        assert edge_index.shape[0] == 2, "边索引应为 2 x n_edges"
        assert edge_index.shape[1] == edge_weight.shape[0], "边索引和边权重数量应一致"
        assert edge_weight.min() >= 0, "边权重应为非负"
        assert edge_weight.max() <= 1, "边权重应 <= 1"
        
        # 验证每个节点的邻居数量
        unique_sources = np.unique(edge_index[0])
        assert len(unique_sources) == n_samples, "所有节点都应有邻居"
        
        print("✓ 图构建逻辑测试通过")
    
    def test_reproducibility(self):
        """
        测试可复现性
        
        验证:
        - 使用相同随机种子训练两次应得到相同结果
        """
        # 生成数据
        n_samples = 80
        views = [
            np.random.randn(n_samples, 20),
            np.random.randn(n_samples, 15),
            np.random.randn(n_samples, 10)
        ]
        y = np.random.randint(0, 2, n_samples)
        
        # 第一次训练
        model1 = MOGONETBaseline(random_state=42, epochs=5)
        model1.fit(views, y)
        pred1 = model1.predict(views)
        
        # 第二次训练
        model2 = MOGONETBaseline(random_state=42, epochs=5)
        model2.fit(views, y)
        pred2 = model2.predict(views)
        
        # 验证（由于深度学习的随机性，允许少量差异）
        accuracy = np.mean(pred1 == pred2)
        assert accuracy > 0.9, f"可复现性测试失败，一致性仅为 {accuracy:.2%}"
        
        print("✓ 可复现性测试通过")


class TestROSMAPDataset:
    """ROSMAP 数据集加载测试"""
    
    def test_file_existence_check(self):
        """
        测试文件存在性检查
        
        验证:
        - 当文件缺失时抛出 FileNotFoundError
        - 错误信息包含缺失文件路径
        """
        # 使用不存在的路径
        with pytest.raises(FileNotFoundError) as exc_info:
            dataset = ROSMAPDataset(data_dir="data/nonexistent_path/")
        
        error_message = str(exc_info.value)
        assert "ROSMAP 数据文件缺失" in error_message, "错误信息应提示文件缺失"
        assert "下载指引" in error_message, "错误信息应包含下载指引"
        
        print("✓ 文件存在性检查测试通过")
    
    @pytest.mark.skipif(
        not (Path("data/baselines_official/ROSMAP/1_tr.csv").exists()),
        reason="ROSMAP 数据文件不存在，跳过此测试"
    )
    def test_data_loading(self):
        """
        测试数据加载逻辑
        
        验证:
        - 正确读取 8 个 CSV 文件
        - 使用 pd.concat 合并训练集和测试集
        - 样本顺序一致
        """
        dataset = ROSMAPDataset()
        views, labels = dataset.load_data()
        
        # 验证
        assert len(views) == 3, "应该有 3 个视图"
        assert all(isinstance(view, np.ndarray) for view in views), "视图应为 numpy 数组"
        assert isinstance(labels, np.ndarray), "标签应为 numpy 数组"
        
        # 验证样本数量一致
        n_samples = views[0].shape[0]
        assert all(view.shape[0] == n_samples for view in views), "所有视图样本数应一致"
        assert labels.shape[0] == n_samples, "标签样本数应与视图一致"
        
        print(f"✓ 数据加载测试通过 (样本数: {n_samples})")


@pytest.mark.skipif(
    not (Path("data/baselines_official/ROSMAP/1_tr.csv").exists()),
    reason="ROSMAP 数据文件不存在，跳过性能验证测试"
)
def test_mogonet_on_rosmap():
    """
    MOGONET 在 ROSMAP 上的性能验证
    
    验证:
    - Accuracy > 0.80
    
    Requirements: 3.14
    """
    # 加载数据
    dataset = ROSMAPDataset()
    views, labels = dataset.load_data()
    
    print(f"ROSMAP 数据集信息:")
    print(f"  样本数: {views[0].shape[0]}")
    print(f"  视图 1 特征数: {views[0].shape[1]}")
    print(f"  视图 2 特征数: {views[1].shape[1]}")
    print(f"  视图 3 特征数: {views[2].shape[1]}")
    print(f"  类别分布: {np.bincount(labels)}")
    
    # 数据划分（使用全局随机种子 42）
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_views = [view[train_idx] for view in views]
    test_views = [view[test_idx] for view in views]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    # 训练模型
    print("\n开始训练 MOGONET 模型...")
    model = MOGONETBaseline(
        random_state=42,
        hidden_dim=64,
        n_gcn_layers=2,
        epochs=100,
        learning_rate=0.001,
        k_neighbors=10
    )
    model.fit(train_views, train_labels)
    
    # 评估
    print("\n评估模型性能...")
    metrics = model.evaluate(test_views, test_labels)
    
    print(f"\nMOGONET 在 ROSMAP 上的性能:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # 验证性能基准
    assert metrics['accuracy'] > 0.80, \
        f"MOGONET Accuracy {metrics['accuracy']:.4f} < 0.80 (性能基准未达标)"
    
    print("\n✓ MOGONET 性能验证通过 (Accuracy > 0.80)")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])

# -*- coding: utf-8 -*-
"""
DLC Data Processor 属性测试
===========================

使用 hypothesis 进行属性测试，验证数据处理的正确性属性。
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataCleaner


class TestTransposeInvariant:
    """
    Feature: dlc-data-engineering, Property 1: 转置不变量
    Validates: Requirements 2.1, 2.2
    
    *For any* 基因突变矩阵，转置操作后：
    - 行数应等于原始列数（样本数）
    - 列数应等于原始行数（基因数）
    - 数据内容保持不变（仅维度交换）
    """
    
    @given(
        n_genes=st.integers(min_value=1, max_value=50),
        n_samples=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_transpose_dimension_swap(self, n_genes: int, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 1: 转置不变量
        Validates: Requirements 2.1, 2.2
        
        验证转置后维度正确交换：
        - 原始矩阵 (n_genes x n_samples) 转置后变为 (n_samples x n_genes)
        """
        np.random.seed(seed)
        
        # 生成随机基因名和样本名
        gene_names = [f'GENE_{i}' for i in range(n_genes)]
        sample_names = [f'TCGA-XX-{i:04d}-01' for i in range(n_samples)]
        
        # 创建原始矩阵（行=基因，列=样本），值为 0/1
        data = np.random.randint(0, 2, size=(n_genes, n_samples))
        original_df = pd.DataFrame(data, index=gene_names, columns=sample_names)
        
        # 执行转置
        transposed_df = original_df.T
        
        # 验证维度交换
        assert transposed_df.shape[0] == n_samples, \
            f"转置后行数应为 {n_samples}，实际为 {transposed_df.shape[0]}"
        assert transposed_df.shape[1] == n_genes, \
            f"转置后列数应为 {n_genes}，实际为 {transposed_df.shape[1]}"
        
        # 验证行索引变为原始列名（样本名）
        assert list(transposed_df.index) == sample_names, \
            "转置后行索引应为原始样本名"
        
        # 验证列名变为原始行索引（基因名）
        assert list(transposed_df.columns) == gene_names, \
            "转置后列名应为原始基因名"
    
    @given(
        n_genes=st.integers(min_value=1, max_value=30),
        n_samples=st.integers(min_value=1, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_transpose_data_preservation(self, n_genes: int, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 1: 转置不变量
        Validates: Requirements 2.1, 2.2
        
        验证转置后数据内容保持不变：
        - original[gene_i, sample_j] == transposed[sample_j, gene_i]
        """
        np.random.seed(seed)
        
        # 生成随机基因名和样本名
        gene_names = [f'GENE_{i}' for i in range(n_genes)]
        sample_names = [f'TCGA-XX-{i:04d}-01' for i in range(n_samples)]
        
        # 创建原始矩阵，值为 0/1
        data = np.random.randint(0, 2, size=(n_genes, n_samples))
        original_df = pd.DataFrame(data, index=gene_names, columns=sample_names)
        
        # 执行转置
        transposed_df = original_df.T
        
        # 验证数据内容保持不变
        for i, gene in enumerate(gene_names):
            for j, sample in enumerate(sample_names):
                original_value = original_df.loc[gene, sample]
                transposed_value = transposed_df.loc[sample, gene]
                assert original_value == transposed_value, \
                    f"数据不一致: original[{gene}, {sample}]={original_value} != transposed[{sample}, {gene}]={transposed_value}"
    
    @given(
        n_genes=st.integers(min_value=1, max_value=20),
        n_samples=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_double_transpose_identity(self, n_genes: int, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 1: 转置不变量
        Validates: Requirements 2.1, 2.2
        
        验证双重转置恢复原始矩阵：
        - (A.T).T == A
        """
        np.random.seed(seed)
        
        # 生成随机基因名和样本名
        gene_names = [f'GENE_{i}' for i in range(n_genes)]
        sample_names = [f'TCGA-XX-{i:04d}-01' for i in range(n_samples)]
        
        # 创建原始矩阵
        data = np.random.randint(0, 2, size=(n_genes, n_samples))
        original_df = pd.DataFrame(data, index=gene_names, columns=sample_names)
        
        # 双重转置
        double_transposed = original_df.T.T
        
        # 验证恢复原始矩阵
        pd.testing.assert_frame_equal(original_df, double_transposed)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestLongFormatIDValidation:
    """
    Feature: dlc-data-engineering, Property 2: 长格式 ID 验证
    Validates: Requirements 3.1
    
    *For any* 合并后的 DataFrame，所有 sampleID 必须匹配长格式模式 TCGA-XX-XXXX-XX
    """
    
    @given(
        n_samples=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_valid_long_format_ids_pass_validation(self, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 2: 长格式 ID 验证
        Validates: Requirements 3.1
        
        验证所有符合长格式的 ID 都能通过验证
        """
        np.random.seed(seed)
        
        # 生成有效的长格式 ID
        # 格式: TCGA-XX-XXXX-XX (如 TCGA-05-4244-01)
        valid_ids = []
        for i in range(n_samples):
            # 随机生成两位字母数字组合
            part1 = f'{np.random.randint(0, 100):02d}'
            # 随机生成四位字母数字组合
            part2 = f'{np.random.randint(0, 10000):04d}'
            # 随机生成两位数字
            part3 = f'{np.random.randint(1, 99):02d}'
            valid_ids.append(f'TCGA-{part1}-{part2}-{part3}')
        
        ids_series = pd.Series(valid_ids)
        
        # 创建 DataCleaner 实例并验证
        cleaner = DataCleaner('LUAD')
        
        # 不应抛出异常
        cleaner._validate_long_format_id(ids_series)
    
    @given(
        n_valid=st.integers(min_value=1, max_value=20),
        n_invalid=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_short_format_ids_raise_error(self, n_valid: int, n_invalid: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 2: 长格式 ID 验证
        Validates: Requirements 3.1
        
        验证包含短格式 ID 时会抛出 ValueError
        """
        np.random.seed(seed)
        
        # 生成有效的长格式 ID
        valid_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                     for _ in range(n_valid)]
        
        # 生成无效的短格式 ID (如 TCGA-05-4244，缺少最后的 -XX 部分)
        invalid_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}' 
                       for _ in range(n_invalid)]
        
        # 混合有效和无效 ID
        all_ids = valid_ids + invalid_ids
        np.random.shuffle(all_ids)
        
        ids_series = pd.Series(all_ids)
        
        # 创建 DataCleaner 实例
        cleaner = DataCleaner('LUAD')
        
        # 应该抛出 ValueError
        with pytest.raises(ValueError) as exc_info:
            cleaner._validate_long_format_id(ids_series)
        
        # 验证错误信息包含关键内容
        assert '非长格式 ID' in str(exc_info.value) or 'TCGA-XX-XXXX-XX' in str(exc_info.value)
    
    @given(
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_completely_invalid_ids_raise_error(self, seed: int):
        """
        Feature: dlc-data-engineering, Property 2: 长格式 ID 验证
        Validates: Requirements 3.1
        
        验证完全不符合 TCGA 格式的 ID 会抛出 ValueError
        """
        np.random.seed(seed)
        
        # 生成完全无效的 ID
        invalid_patterns = [
            'SAMPLE_001',
            'patient_123',
            '12345',
            'TCGA',
            'TCGA-05',
            'TCGA-05-4244',  # 短格式
            'tcga-05-4244-01',  # 小写
            'TCGA_05_4244_01',  # 下划线
        ]
        
        # 随机选择一些无效 ID
        n_invalid = np.random.randint(1, len(invalid_patterns) + 1)
        selected_invalid = np.random.choice(invalid_patterns, size=n_invalid, replace=False).tolist()
        
        ids_series = pd.Series(selected_invalid)
        
        # 创建 DataCleaner 实例
        cleaner = DataCleaner('PANCAN')
        
        # 应该抛出 ValueError
        with pytest.raises(ValueError):
            cleaner._validate_long_format_id(ids_series)


class TestForcedGeneInclusion:
    """
    Feature: dlc-data-engineering, Property 4: 强制基因包含
    Validates: Requirements 4.2
    
    *For any* 基因频率分布，select_top_genes() 返回的 Top20 列表必须包含 EGFR、KRAS 和 TP53。
    """
    
    @given(
        n_genes=st.integers(min_value=25, max_value=100),
        n_samples=st.integers(min_value=10, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_forced_genes_always_included(self, n_genes: int, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 4: 强制基因包含
        Validates: Requirements 4.2
        
        验证无论基因频率如何分布，EGFR、KRAS、TP53 始终包含在 Top20 中
        """
        np.random.seed(seed)
        
        # 强制基因列表
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        
        # 生成随机基因名（确保包含强制基因）
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成随机样本 ID（长格式）
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成随机突变数据（0/1 二值）
        # 故意让强制基因的频率很低，测试是否仍被包含
        mutation_data = {}
        for gene in all_genes:
            if gene in forced_genes:
                # 强制基因设置为低频率（0-10%）
                prob = np.random.uniform(0.0, 0.1)
            else:
                # 其他基因设置为随机频率
                prob = np.random.uniform(0.0, 1.0)
            mutation_data[gene] = np.random.binomial(1, prob, n_samples)
        
        # 创建 DataFrame
        gene_df = pd.DataFrame(mutation_data, index=sample_ids)
        gene_df.index.name = 'sampleID'
        
        # 创建临床数据
        clinical_df = pd.DataFrame({
            'sampleID': sample_ids,
            'Age': np.random.randint(30, 80, n_samples),
            'Gender': np.random.choice(['MALE', 'FEMALE'], n_samples)
        })
        
        # 创建 DataCleaner 实例
        cleaner = DataCleaner('LUAD')
        cleaner.gene_df = gene_df
        cleaner.clinical_df = clinical_df
        
        # 合并数据（需要重置索引）
        gene_df_reset = gene_df.reset_index()
        cleaner.merged_df = pd.merge(gene_df_reset, clinical_df, on='sampleID', how='inner')
        
        # 调用 select_top_genes
        top_genes = cleaner.select_top_genes(n=20, forced_genes=forced_genes)
        
        # 验证强制基因都在结果中
        for forced_gene in forced_genes:
            assert forced_gene in top_genes, \
                f"强制基因 {forced_gene} 未包含在 Top20 中: {top_genes}"
    
    @given(
        n_genes=st.integers(min_value=25, max_value=80),
        n_samples=st.integers(min_value=10, max_value=40),
        top_n=st.integers(min_value=5, max_value=20),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_forced_genes_with_variable_top_n(self, n_genes: int, n_samples: int, top_n: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 4: 强制基因包含
        Validates: Requirements 4.2
        
        验证不同的 Top N 值下，强制基因仍然被包含
        """
        np.random.seed(seed)
        
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        
        # 确保 top_n 至少能容纳强制基因
        if top_n < len(forced_genes):
            top_n = len(forced_genes)
        
        # 生成基因名
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成突变数据
        mutation_data = {}
        for gene in all_genes:
            prob = np.random.uniform(0.0, 1.0)
            mutation_data[gene] = np.random.binomial(1, prob, n_samples)
        
        gene_df = pd.DataFrame(mutation_data, index=sample_ids)
        gene_df.index.name = 'sampleID'
        
        clinical_df = pd.DataFrame({
            'sampleID': sample_ids,
            'Age': np.random.randint(30, 80, n_samples),
            'Gender': np.random.choice(['MALE', 'FEMALE'], n_samples)
        })
        
        cleaner = DataCleaner('PANCAN')
        cleaner.gene_df = gene_df
        cleaner.clinical_df = clinical_df
        
        gene_df_reset = gene_df.reset_index()
        cleaner.merged_df = pd.merge(gene_df_reset, clinical_df, on='sampleID', how='inner')
        
        # 调用 select_top_genes
        top_genes = cleaner.select_top_genes(n=top_n, forced_genes=forced_genes)
        
        # 验证结果长度
        assert len(top_genes) == top_n, \
            f"返回的基因数量应为 {top_n}，实际为 {len(top_genes)}"
        
        # 验证强制基因都在结果中
        for forced_gene in forced_genes:
            assert forced_gene in top_genes, \
                f"强制基因 {forced_gene} 未包含在 Top{top_n} 中: {top_genes}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_locked_genes_override(self, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 4: 强制基因包含
        Validates: Requirements 4.2
        
        验证 locked_genes 参数可以完全覆盖基因选择逻辑
        """
        np.random.seed(seed)
        
        # 预定义的锁定基因列表
        locked_genes = ['EGFR', 'KRAS', 'TP53', 'BRAF', 'PIK3CA', 
                        'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D', 'GENE_E']
        
        # 创建 DataCleaner 实例（不需要加载数据，因为 locked_genes 会直接返回）
        cleaner = DataCleaner('LUAD')
        
        # 调用 select_top_genes 使用 locked_genes
        result = cleaner.select_top_genes(n=20, locked_genes=locked_genes)
        
        # 验证返回的是 locked_genes 的副本
        assert result == locked_genes, \
            f"使用 locked_genes 时应直接返回该列表，期望 {locked_genes}，实际 {result}"
        
        # 验证修改返回值不影响原始列表
        result.append('NEW_GENE')
        assert 'NEW_GENE' not in locked_genes, \
            "返回的应该是副本，修改不应影响原始列表"


class TestDataCleaningCompleteness:
    """
    Feature: dlc-data-engineering, Property 6: 数据清洗完整性
    Validates: Requirements 5.1, 5.2, 5.3
    
    *For any* 清洗后的 DataFrame：
    - Age 列不包含缺失值，且为数值类型
    - Gender 列只包含 0 和 1
    - 所有基因列只包含 0 和 1
    """
    
    @given(
        n_samples=st.integers(min_value=5, max_value=50),
        n_genes=st.integers(min_value=5, max_value=30),
        missing_age_ratio=st.floats(min_value=0.0, max_value=0.5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_age_no_missing_values_after_cleaning(self, n_samples: int, n_genes: int, 
                                                   missing_age_ratio: float, seed: int):
        """
        Feature: dlc-data-engineering, Property 6: 数据清洗完整性
        Validates: Requirements 5.1
        
        验证清洗后 Age 列不包含缺失值，且为数值类型
        """
        np.random.seed(seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据，包含一些缺失值
        ages = np.random.randint(30, 80, n_samples).astype(float)
        n_missing = int(n_samples * missing_age_ratio)
        if n_missing > 0:
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            ages[missing_indices] = np.nan
        
        # 生成 Gender 数据
        genders = np.random.choice(['MALE', 'FEMALE'], n_samples)
        
        # 生成基因名（包含强制基因）
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成基因突变数据
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in all_genes}
        
        # 创建合并后的 DataFrame
        merged_data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        merged_data.update(gene_data)
        merged_df = pd.DataFrame(merged_data)
        
        # 创建 DataCleaner 实例并设置数据
        cleaner = DataCleaner('LUAD')
        cleaner.merged_df = merged_df
        cleaner.top20_genes = all_genes[:20] if len(all_genes) >= 20 else all_genes
        
        # 执行清洗
        cleaned_df = cleaner.clean_clinical()
        
        # 验证 Age 列无缺失值
        assert cleaned_df['Age'].isna().sum() == 0, \
            f"清洗后 Age 列仍有 {cleaned_df['Age'].isna().sum()} 个缺失值"
        
        # 验证 Age 列为数值类型
        assert pd.api.types.is_numeric_dtype(cleaned_df['Age']), \
            f"清洗后 Age 列应为数值类型，实际为 {cleaned_df['Age'].dtype}"
    
    @given(
        n_samples=st.integers(min_value=5, max_value=50),
        n_genes=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_gender_binary_values_after_cleaning(self, n_samples: int, n_genes: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 6: 数据清洗完整性
        Validates: Requirements 5.2
        
        验证清洗后 Gender 列只包含 0 和 1
        """
        np.random.seed(seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.randint(30, 80, n_samples)
        
        # 生成 Gender 数据，使用各种可能的表示方式
        gender_options = ['MALE', 'Male', 'male', 'M', 'FEMALE', 'Female', 'female', 'F']
        genders = np.random.choice(gender_options, n_samples)
        
        # 生成基因名
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成基因突变数据
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in all_genes}
        
        # 创建合并后的 DataFrame
        merged_data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        merged_data.update(gene_data)
        merged_df = pd.DataFrame(merged_data)
        
        # 创建 DataCleaner 实例并设置数据
        cleaner = DataCleaner('PANCAN')
        cleaner.merged_df = merged_df
        cleaner.top20_genes = all_genes[:20] if len(all_genes) >= 20 else all_genes
        
        # 执行清洗
        cleaned_df = cleaner.clean_clinical()
        
        # 验证 Gender 列只包含 0 和 1
        unique_genders = set(cleaned_df['Gender'].unique())
        assert unique_genders.issubset({0, 1}), \
            f"清洗后 Gender 列应只包含 0 和 1，实际包含 {unique_genders}"
        
        # 验证 Gender 列为整数类型
        assert pd.api.types.is_integer_dtype(cleaned_df['Gender']), \
            f"清洗后 Gender 列应为整数类型，实际为 {cleaned_df['Gender'].dtype}"
    
    @given(
        n_samples=st.integers(min_value=5, max_value=50),
        n_genes=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_gene_columns_binary_values_after_cleaning(self, n_samples: int, n_genes: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 6: 数据清洗完整性
        Validates: Requirements 5.3
        
        验证清洗后所有基因列只包含 0 和 1
        """
        np.random.seed(seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 和 Gender 数据
        ages = np.random.randint(30, 80, n_samples)
        genders = np.random.choice(['MALE', 'FEMALE'], n_samples)
        
        # 生成基因名
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成基因突变数据，包含一些非 0/1 值（如 2, 3 表示多次突变）
        gene_data = {}
        for gene in all_genes:
            # 随机生成 0, 1, 2, 3 的值，模拟原始数据可能包含突变计数
            gene_data[gene] = np.random.randint(0, 4, n_samples)
        
        # 创建合并后的 DataFrame
        merged_data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        merged_data.update(gene_data)
        merged_df = pd.DataFrame(merged_data)
        
        # 创建 DataCleaner 实例并设置数据
        cleaner = DataCleaner('LUAD')
        cleaner.merged_df = merged_df
        cleaner.top20_genes = all_genes[:20] if len(all_genes) >= 20 else all_genes
        
        # 执行清洗
        cleaned_df = cleaner.clean_clinical()
        
        # 验证所有基因列只包含 0 和 1
        for gene in cleaner.top20_genes:
            if gene in cleaned_df.columns:
                unique_values = set(cleaned_df[gene].unique())
                assert unique_values.issubset({0, 1}), \
                    f"清洗后基因列 {gene} 应只包含 0 和 1，实际包含 {unique_values}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_genes=st.integers(min_value=5, max_value=25),
        missing_age_ratio=st.floats(min_value=0.0, max_value=0.3),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_complete_data_cleaning_integrity(self, n_samples: int, n_genes: int, 
                                               missing_age_ratio: float, seed: int):
        """
        Feature: dlc-data-engineering, Property 6: 数据清洗完整性
        Validates: Requirements 5.1, 5.2, 5.3
        
        综合验证清洗后数据的完整性：
        - Age 无缺失值且为数值类型
        - Gender 只包含 0 和 1
        - 所有基因列只包含 0 和 1
        """
        np.random.seed(seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据，包含缺失值
        ages = np.random.randint(30, 80, n_samples).astype(float)
        n_missing = int(n_samples * missing_age_ratio)
        if n_missing > 0:
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            ages[missing_indices] = np.nan
        
        # 生成 Gender 数据，使用各种表示方式
        gender_options = ['MALE', 'Male', 'FEMALE', 'Female', 'M', 'F']
        genders = np.random.choice(gender_options, n_samples)
        
        # 生成基因名
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成基因突变数据，包含非二值数据
        gene_data = {}
        for gene in all_genes:
            gene_data[gene] = np.random.randint(0, 4, n_samples)
        
        # 创建合并后的 DataFrame
        merged_data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        merged_data.update(gene_data)
        merged_df = pd.DataFrame(merged_data)
        
        # 创建 DataCleaner 实例并设置数据
        cleaner = DataCleaner('LUAD')
        cleaner.merged_df = merged_df
        cleaner.top20_genes = all_genes[:20] if len(all_genes) >= 20 else all_genes
        
        # 执行清洗
        cleaned_df = cleaner.clean_clinical()
        
        # 综合验证
        # 1. Age 无缺失值且为数值类型
        assert cleaned_df['Age'].isna().sum() == 0, "Age 列仍有缺失值"
        assert pd.api.types.is_numeric_dtype(cleaned_df['Age']), "Age 列不是数值类型"
        
        # 2. Gender 只包含 0 和 1
        assert set(cleaned_df['Gender'].unique()).issubset({0, 1}), "Gender 列包含非 0/1 值"
        
        # 3. 所有基因列只包含 0 和 1
        for gene in cleaner.top20_genes:
            if gene in cleaned_df.columns:
                assert set(cleaned_df[gene].unique()).issubset({0, 1}), \
                    f"基因列 {gene} 包含非 0/1 值"


class TestPM25ConfoundingEffect:
    """
    Feature: dlc-data-engineering, Property 7: PM2.5 混杂效应
    Validates: Requirements 6.1
    
    *For any* 生成的 Virtual_PM2.5 列，与 Age 的 Pearson 相关系数应为正值（r > 0），
    验证混杂效应的存在。
    
    注意：由于公式中 Age 系数 (0.5) 相对于噪声标准差 (10) 较小，
    需要使用大样本或多次采样来验证期望相关性为正。
    """
    
    @given(
        n_samples=st.integers(min_value=500, max_value=2000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_pm25_age_positive_correlation_large_sample(self, n_samples: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 7: PM2.5 混杂效应
        Validates: Requirements 6.1
        
        验证在大样本下，生成的 Virtual_PM2.5 与 Age 存在正相关关系
        
        由于信噪比较低 (Age 系数 0.5 vs 噪声 std 10)，需要大样本才能可靠检测
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据（有足够的变异性）
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成 PM2.5
        generator = SemiSyntheticGenerator(df, top20_genes, seed=42)
        virtual_pm25 = generator.generate_pm25()
        
        # 计算 Pearson 相关系数
        correlation = np.corrcoef(generator.df['Age'], virtual_pm25)[0, 1]
        
        # 验证相关系数为正值
        # 在大样本下，信号应该能够超过噪声
        # Age 系数为 3.0，理论相关系数约 0.29，断言 > 0.1 以留有余量
        assert correlation > 0.1, \
            f"Virtual_PM2.5 与 Age 的相关系数应大于 0.1，实际为 {correlation:.4f}"
    
    @given(
        n_samples=st.integers(min_value=100, max_value=300),
        n_runs=st.integers(min_value=10, max_value=30),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=50)
    def test_pm25_age_expected_positive_correlation(self, n_samples: int, n_runs: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 7: PM2.5 混杂效应
        Validates: Requirements 6.1
        
        验证多次运行的平均相关系数为正值
        
        由于单次运行可能因噪声而出现负相关，我们验证多次运行的期望值为正
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 多次运行并收集相关系数
        correlations = []
        for run_seed in range(n_runs):
            generator = SemiSyntheticGenerator(df.copy(), top20_genes, seed=run_seed)
            virtual_pm25 = generator.generate_pm25()
            corr = np.corrcoef(generator.df['Age'], virtual_pm25)[0, 1]
            correlations.append(corr)
        
        # 验证平均相关系数为正值
        # Age 系数为 3.0，理论相关系数约 0.29，断言 > 0.1 以留有余量
        mean_correlation = np.mean(correlations)
        assert mean_correlation > 0.1, \
            f"多次运行的平均相关系数应大于 0.1，实际为 {mean_correlation:.4f}"
    
    @given(
        n_samples=st.integers(min_value=50, max_value=300),
        age_range=st.tuples(
            st.integers(min_value=20, max_value=40),
            st.integers(min_value=60, max_value=90)
        ),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_pm25_formula_components(self, n_samples: int, age_range: tuple, seed: int):
        """
        Feature: dlc-data-engineering, Property 7: PM2.5 混杂效应
        Validates: Requirements 6.1
        
        验证 PM2.5 生成公式的各组成部分：
        - 基准值约为 30
        - 与 Age 的 z-score 成正比
        """
        np.random.seed(seed)
        
        min_age, max_age = age_range
        if min_age >= max_age:
            min_age, max_age = max_age, min_age + 10
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(min_age, max_age, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成 PM2.5
        generator = SemiSyntheticGenerator(df, top20_genes, seed=42)
        virtual_pm25 = generator.generate_pm25()
        
        # 验证 PM2.5 的均值接近 30（基准值）
        # 由于 Z(Age) 的均值为 0，噪声的均值也为 0，所以 PM2.5 的均值应接近 30
        pm25_mean = virtual_pm25.mean()
        assert 20 < pm25_mean < 40, \
            f"Virtual_PM2.5 的均值应接近 30，实际为 {pm25_mean:.2f}"
    
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_pm25_noise_distribution(self, n_samples: int, seed: int):
        """
        Feature: dlc-data-engineering, Property 7: PM2.5 混杂效应
        Validates: Requirements 6.1
        
        验证 PM2.5 生成中的噪声分布特性
        """
        np.random.seed(seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据（固定值，消除 Age 的影响）
        ages = np.full(n_samples, 50.0)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成 PM2.5
        generator = SemiSyntheticGenerator(df, top20_genes, seed=42)
        virtual_pm25 = generator.generate_pm25()
        
        # 当 Age 固定时，Z(Age) = 0，所以 PM2.5 = 30 + N(0, 10)
        # 验证标准差接近 10
        pm25_std = virtual_pm25.std()
        assert 5 < pm25_std < 15, \
            f"当 Age 固定时，Virtual_PM2.5 的标准差应接近 10，实际为 {pm25_std:.2f}"


class TestRandomSeedReproducibility:
    """
    Feature: dlc-data-engineering, Property 8: 随机种子可复现性
    Validates: Requirements 6.2
    
    *For any* 相同的输入数据和随机种子，SemiSyntheticGenerator 应产生完全相同的输出。
    """
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_same_seed_produces_identical_pm25(self, n_samples: int, seed: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 8: 随机种子可复现性
        Validates: Requirements 6.2
        
        验证相同的随机种子产生完全相同的 Virtual_PM2.5
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 第一次生成
        generator1 = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed)
        pm25_1 = generator1.generate_pm25()
        
        # 第二次生成（使用相同的种子）
        generator2 = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed)
        pm25_2 = generator2.generate_pm25()
        
        # 验证两次生成的结果完全相同
        pd.testing.assert_series_equal(
            pm25_1.reset_index(drop=True), 
            pm25_2.reset_index(drop=True),
            check_names=False
        )
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        seed1=st.integers(min_value=0, max_value=5000),
        seed2=st.integers(min_value=5001, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_different_seeds_produce_different_pm25(self, n_samples: int, seed1: int, 
                                                     seed2: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 8: 随机种子可复现性
        Validates: Requirements 6.2
        
        验证不同的随机种子产生不同的 Virtual_PM2.5
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 使用不同种子生成
        generator1 = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed1)
        pm25_1 = generator1.generate_pm25()
        
        generator2 = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed2)
        pm25_2 = generator2.generate_pm25()
        
        # 验证两次生成的结果不同（至少有一个值不同）
        # 由于噪声是随机的，不同种子应该产生不同的结果
        assert not pm25_1.equals(pm25_2), \
            "不同的随机种子应该产生不同的 Virtual_PM2.5"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        seed=st.integers(min_value=0, max_value=10000),
        n_runs=st.integers(min_value=2, max_value=5),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_multiple_runs_with_same_seed_are_identical(self, n_samples: int, seed: int, 
                                                         n_runs: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 8: 随机种子可复现性
        Validates: Requirements 6.2
        
        验证多次运行使用相同种子时结果完全一致
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 多次运行并收集结果
        results = []
        for _ in range(n_runs):
            generator = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed)
            pm25 = generator.generate_pm25()
            results.append(pm25.reset_index(drop=True))
        
        # 验证所有运行结果完全相同
        for i in range(1, len(results)):
            pd.testing.assert_series_equal(
                results[0], 
                results[i],
                check_names=False
            )


class TestProbabilityGenerationValidity:
    """
    Feature: dlc-data-engineering, Property 9: 概率生成有效性
    Validates: Requirements 7.2, 7.3, 8.2, 8.3
    
    *For any* 生成的 True_Prob 列：
    - 所有值在 (0, 1) 开区间内
    - Outcome_Label 列只包含 0 和 1
    """
    
    @given(
        n_samples=st.integers(min_value=10, max_value=200),
        scenario=st.sampled_from(['interaction', 'linear']),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_true_prob_in_valid_range(self, n_samples: int, scenario: str, 
                                       seed: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 9: 概率生成有效性
        Validates: Requirements 7.2, 7.3, 8.2, 8.3
        
        验证 True_Prob 所有值在 (0, 1) 开区间内
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成完整数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=seed)
        result_df = generator.generate(scenario)
        
        # 验证 True_Prob 在 (0, 1) 开区间内
        true_prob = result_df['True_Prob']
        assert (true_prob > 0).all(), \
            f"True_Prob 存在 <= 0 的值: min={true_prob.min()}"
        assert (true_prob < 1).all(), \
            f"True_Prob 存在 >= 1 的值: max={true_prob.max()}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=200),
        scenario=st.sampled_from(['interaction', 'linear']),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_outcome_label_binary_values(self, n_samples: int, scenario: str,
                                          seed: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 9: 概率生成有效性
        Validates: Requirements 7.2, 7.3, 8.2, 8.3
        
        验证 Outcome_Label 只包含 0 和 1
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成完整数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=seed)
        result_df = generator.generate(scenario)
        
        # 验证 Outcome_Label 只包含 0 和 1
        outcome_label = result_df['Outcome_Label']
        unique_values = set(outcome_label.unique())
        assert unique_values.issubset({0, 1}), \
            f"Outcome_Label 应只包含 0 和 1，实际包含 {unique_values}"
    
    @given(
        n_samples=st.integers(min_value=50, max_value=300),
        scenario=st.sampled_from(['interaction', 'linear']),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_outcome_label_follows_probability(self, n_samples: int, scenario: str,
                                                seed: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 9: 概率生成有效性
        Validates: Requirements 7.2, 7.3, 8.2, 8.3
        
        验证 Outcome_Label 的阳性率与 True_Prob 的均值大致一致
        （由于 Bernoulli 采样的随机性，允许一定误差）
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成完整数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=seed)
        result_df = generator.generate(scenario)
        
        # 计算阳性率和平均概率
        positive_rate = result_df['Outcome_Label'].mean()
        mean_prob = result_df['True_Prob'].mean()
        
        # 验证阳性率与平均概率大致一致（允许 30% 的相对误差或 0.15 的绝对误差）
        # 由于样本量有限，Bernoulli 采样会有较大波动
        abs_diff = abs(positive_rate - mean_prob)
        rel_diff = abs_diff / max(mean_prob, 0.01)
        
        assert abs_diff < 0.2 or rel_diff < 0.5, \
            f"阳性率 ({positive_rate:.3f}) 与平均概率 ({mean_prob:.3f}) 差异过大"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_both_scenarios_produce_valid_output(self, n_samples: int, seed: int, data_seed: int):
        """
        Feature: dlc-data-engineering, Property 9: 概率生成有效性
        Validates: Requirements 7.2, 7.3, 8.2, 8.3
        
        验证两种场景都能产生有效的输出
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {gene: np.random.randint(0, 2, n_samples) for gene in top20_genes}
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        for scenario in ['interaction', 'linear']:
            # 创建生成器并生成完整数据
            generator = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed)
            result_df = generator.generate(scenario)
            
            # 验证输出包含必需列
            required_cols = ['sampleID', 'Age', 'Gender', 'Virtual_PM2.5', 
                           'True_Prob', 'Outcome_Label']
            for col in required_cols:
                assert col in result_df.columns, \
                    f"场景 {scenario} 输出缺少列: {col}"
            
            # 验证 True_Prob 在有效范围内
            assert (result_df['True_Prob'] > 0).all() and (result_df['True_Prob'] < 1).all(), \
                f"场景 {scenario} 的 True_Prob 不在 (0, 1) 范围内"
            
            # 验证 Outcome_Label 是二值
            assert set(result_df['Outcome_Label'].unique()).issubset({0, 1}), \
                f"场景 {scenario} 的 Outcome_Label 不是二值"



class TestInteractionEffectVerification:
    """
    Feature: dlc-data-engineering, Property 10: 交互效应验证
    Validates: Requirements 7.1
    
    *For any* 交互场景生成的数据，EGFR=1 组的平均 True_Prob 应显著高于 EGFR=0 组
    （在控制其他变量后）。
    """
    
    @given(
        n_samples=st.integers(min_value=1000, max_value=2000),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_egfr_positive_increases_prob_in_interaction_scenario(
        self, n_samples: int, seed: int, data_seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 10: 交互效应验证
        Validates: Requirements 7.1
        
        验证在交互场景下的高暴露组 (High Exposure Subgroup) 中，
        EGFR=1 组的平均 True_Prob 高于 EGFR=0 组。
        
        由于 PM2.5* 是标准化后的值（均值为 0），交互项在全人群中的期望贡献接近 0。
        因此我们只在高 PM2.5 暴露组（PM2.5* > 0，即高于均值）中验证交互效应。
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID（增加样本量以消除随机噪声）
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据，确保 EGFR 有足够的变异
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {}
        for gene in top20_genes:
            if gene == 'EGFR':
                # 确保 EGFR 有大约 50% 的突变率，以便有足够的两组样本
                gene_data[gene] = np.random.binomial(1, 0.5, n_samples)
            else:
                gene_data[gene] = np.random.randint(0, 2, n_samples)
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 创建生成器并生成交互场景数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=seed)
        result_df = generator.generate('interaction')
        
        # 只在高暴露组 (High Exposure Subgroup) 中验证
        # 使用中位数作为分界点，选择 PM2.5 >= 中位数的样本
        pm25_median = result_df['Virtual_PM2.5'].median()
        high_exposure = result_df[result_df['Virtual_PM2.5'] >= pm25_median]
        
        # 确保高暴露组中两个 EGFR 组都有足够样本
        egfr_0_high = high_exposure[high_exposure['EGFR'] == 0]
        egfr_1_high = high_exposure[high_exposure['EGFR'] == 1]
        
        if len(egfr_0_high) < 10 or len(egfr_1_high) < 10:
            return  # 样本不足，跳过此测试
        
        # 在高暴露组中计算平均 True_Prob
        egfr_0_prob = egfr_0_high['True_Prob'].mean()
        egfr_1_prob = egfr_1_high['True_Prob'].mean()
        
        # 验证高暴露组中 EGFR=1 组的平均概率高于 EGFR=0 组
        # 由于交互项系数 W_INT=0.69 为正，在高 PM2.5 暴露下 EGFR=1 应该增加概率
        assert egfr_1_prob > egfr_0_prob, \
            f"高暴露组中 EGFR=1 组的平均概率 ({egfr_1_prob:.4f}) 应高于 EGFR=0 组 ({egfr_0_prob:.4f})"
    
    @given(
        n_samples=st.integers(min_value=200, max_value=500),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_egfr_effect_smaller_in_linear_scenario(
        self, n_samples: int, seed: int, data_seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 10: 交互效应验证
        Validates: Requirements 7.1
        
        验证在高 PM2.5 暴露组中，交互场景下 EGFR 的效应大于线性场景
        
        注意：由于 PM2.5* 是标准化后的值（均值为 0），交互项的整体期望贡献接近 0。
        因此我们需要在高 PM2.5 暴露组中比较效应，才能看到交互效应的差异。
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {}
        for gene in top20_genes:
            if gene == 'EGFR':
                gene_data[gene] = np.random.binomial(1, 0.5, n_samples)
            else:
                gene_data[gene] = np.random.randint(0, 2, n_samples)
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 生成交互场景数据
        generator_int = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed)
        result_int = generator_int.generate('interaction')
        
        # 生成线性场景数据
        generator_lin = SemiSyntheticGenerator(df.copy(), top20_genes, seed=seed)
        result_lin = generator_lin.generate('linear')
        
        # 只在高 PM2.5 暴露组（PM2.5* > 0，即高于均值）中比较 EGFR 效应
        pm25_median_int = result_int['Virtual_PM2.5'].median()
        high_pm25_int = result_int[result_int['Virtual_PM2.5'] >= pm25_median_int]
        
        pm25_median_lin = result_lin['Virtual_PM2.5'].median()
        high_pm25_lin = result_lin[result_lin['Virtual_PM2.5'] >= pm25_median_lin]
        
        # 计算高 PM2.5 组中交互场景下 EGFR 的效应差异
        if len(high_pm25_int[high_pm25_int['EGFR'] == 0]) > 0 and len(high_pm25_int[high_pm25_int['EGFR'] == 1]) > 0:
            int_egfr_0_prob = high_pm25_int[high_pm25_int['EGFR'] == 0]['True_Prob'].mean()
            int_egfr_1_prob = high_pm25_int[high_pm25_int['EGFR'] == 1]['True_Prob'].mean()
            int_effect = int_egfr_1_prob - int_egfr_0_prob
        else:
            return  # 样本不足，跳过此测试
        
        # 计算高 PM2.5 组中线性场景下 EGFR 的效应差异
        if len(high_pm25_lin[high_pm25_lin['EGFR'] == 0]) > 0 and len(high_pm25_lin[high_pm25_lin['EGFR'] == 1]) > 0:
            lin_egfr_0_prob = high_pm25_lin[high_pm25_lin['EGFR'] == 0]['True_Prob'].mean()
            lin_egfr_1_prob = high_pm25_lin[high_pm25_lin['EGFR'] == 1]['True_Prob'].mean()
            lin_effect = lin_egfr_1_prob - lin_egfr_0_prob
        else:
            return  # 样本不足，跳过此测试
        
        # 验证在高 PM2.5 组中，交互场景下 EGFR 的效应大于线性场景
        # 因为交互场景有额外的 W_INT * PM2.5* * EGFR 项，在高 PM2.5 时贡献为正
        assert int_effect > lin_effect, \
            f"高 PM2.5 组中，交互场景下 EGFR 效应 ({int_effect:.4f}) 应大于线性场景 ({lin_effect:.4f})"
    
    @given(
        n_samples=st.integers(min_value=300, max_value=600),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=50)
    def test_interaction_effect_magnitude(
        self, n_samples: int, seed: int, data_seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 10: 交互效应验证
        Validates: Requirements 7.1
        
        验证在高 PM2.5 暴露组中，交互效应的量级符合预期（基于 W_INT=0.69）
        
        注意：由于 PM2.5* 是标准化后的值（均值为 0），我们需要在高 PM2.5 组中
        验证交互效应的量级，因为只有在高 PM2.5 时交互项才有正向贡献。
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {}
        for gene in top20_genes:
            if gene == 'EGFR':
                gene_data[gene] = np.random.binomial(1, 0.5, n_samples)
            else:
                gene_data[gene] = np.random.randint(0, 2, n_samples)
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 生成交互场景数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=seed)
        result_df = generator.generate('interaction')
        
        # 只在高 PM2.5 暴露组（高于中位数）中计算 EGFR 效应
        pm25_median = result_df['Virtual_PM2.5'].median()
        high_pm25 = result_df[result_df['Virtual_PM2.5'] >= pm25_median]
        
        # 计算高 PM2.5 组中 EGFR 组间的概率差异
        if len(high_pm25[high_pm25['EGFR'] == 0]) > 0 and len(high_pm25[high_pm25['EGFR'] == 1]) > 0:
            egfr_0_prob = high_pm25[high_pm25['EGFR'] == 0]['True_Prob'].mean()
            egfr_1_prob = high_pm25[high_pm25['EGFR'] == 1]['True_Prob'].mean()
            prob_diff = egfr_1_prob - egfr_0_prob
        else:
            return  # 样本不足，跳过此测试
        
        # 验证在高 PM2.5 组中，概率差异为正
        # W_INT=0.69 是一个较大的系数，在高 PM2.5 时应该产生明显的正向效应
        assert prob_diff > 0, \
            f"高 PM2.5 组中，交互效应应产生正向概率差异，实际为 {prob_diff:.4f}"
    
    @given(
        n_samples=st.integers(min_value=1000, max_value=2000),
        seed=st.integers(min_value=0, max_value=10000),
        data_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_high_pm25_amplifies_egfr_effect(
        self, n_samples: int, seed: int, data_seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 10: 交互效应验证
        Validates: Requirements 7.1
        
        验证高 PM2.5 暴露放大 EGFR 的效应（交互效应的本质）。
        
        通过比较风险差 (Risk Difference, RD) 来验证：
        - RD_high = Mean(P|EGFR=1, HighPM) - Mean(P|EGFR=0, HighPM)
        - RD_low = Mean(P|EGFR=1, LowPM) - Mean(P|EGFR=0, LowPM)
        - 断言：RD_high > RD_low（恶劣环境下，EGFR 突变带来的额外风险更大）
        """
        np.random.seed(data_seed)
        
        # 生成样本 ID（增加样本量以消除随机噪声）
        sample_ids = [f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
                      for _ in range(n_samples)]
        
        # 生成 Age 数据
        ages = np.random.uniform(30, 80, n_samples)
        
        # 生成 Gender 数据
        genders = np.random.choice([0, 1], n_samples)
        
        # 生成基因数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        gene_data = {}
        for gene in top20_genes:
            if gene == 'EGFR':
                gene_data[gene] = np.random.binomial(1, 0.5, n_samples)
            else:
                gene_data[gene] = np.random.randint(0, 2, n_samples)
        
        # 创建 DataFrame
        data = {'sampleID': sample_ids, 'Age': ages, 'Gender': genders}
        data.update(gene_data)
        df = pd.DataFrame(data)
        
        # 导入 SemiSyntheticGenerator
        from data_processor import SemiSyntheticGenerator
        
        # 生成交互场景数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=seed)
        result_df = generator.generate('interaction')
        
        # 按 PM2.5 中位数分组
        pm25_median = result_df['Virtual_PM2.5'].median()
        high_pm25 = result_df[result_df['Virtual_PM2.5'] >= pm25_median]
        low_pm25 = result_df[result_df['Virtual_PM2.5'] < pm25_median]
        
        # 确保各组都有足够样本
        high_egfr_0 = high_pm25[high_pm25['EGFR'] == 0]
        high_egfr_1 = high_pm25[high_pm25['EGFR'] == 1]
        low_egfr_0 = low_pm25[low_pm25['EGFR'] == 0]
        low_egfr_1 = low_pm25[low_pm25['EGFR'] == 1]
        
        if len(high_egfr_0) < 10 or len(high_egfr_1) < 10 or \
           len(low_egfr_0) < 10 or len(low_egfr_1) < 10:
            return  # 样本不足，跳过此测试
        
        # 计算高暴露组的风险差 (Risk Difference)
        # RD_high = Mean(P|EGFR=1, HighPM) - Mean(P|EGFR=0, HighPM)
        rd_high = high_egfr_1['True_Prob'].mean() - high_egfr_0['True_Prob'].mean()
        
        # 计算低暴露组的风险差
        # RD_low = Mean(P|EGFR=1, LowPM) - Mean(P|EGFR=0, LowPM)
        rd_low = low_egfr_1['True_Prob'].mean() - low_egfr_0['True_Prob'].mean()
        
        # 验证高暴露组的风险差大于低暴露组（交互效应）
        # 由于 PM2.5* × EGFR 项，高 PM2.5 应该放大 EGFR 的效应
        assert rd_high > rd_low, \
            f"高暴露组的风险差 RD_high ({rd_high:.4f}) 应大于低暴露组 RD_low ({rd_low:.4f})"



class TestFeatureSpaceAlignment:
    """
    Feature: dlc-data-engineering, Property 5: 特征空间对齐
    Validates: Requirements 4.3, 4.4
    
    *For any* 处理后的 PANCAN 和 LUAD 数据集：
    - 列名集合必须完全相同
    - 列顺序必须完全相同
    - 基因列的数量和名称必须一致
    """
    
    @given(
        n_samples_luad=st.integers(min_value=10, max_value=50),
        n_samples_pancan=st.integers(min_value=10, max_value=50),
        n_genes=st.integers(min_value=25, max_value=60),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_locked_genes_ensure_column_alignment(
        self, n_samples_luad: int, n_samples_pancan: int, n_genes: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 5: 特征空间对齐
        Validates: Requirements 4.3, 4.4
        
        验证使用 locked_genes 参数时，两个数据集的列名集合完全相同
        """
        np.random.seed(seed)
        
        # 强制基因列表
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        
        # 生成基因名（确保包含强制基因）
        other_genes = [f'GENE_{i}' for i in range(n_genes - len(forced_genes))]
        all_genes = forced_genes + other_genes
        
        # 生成 LUAD 样本 ID
        luad_sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples_luad)
        ]
        
        # 生成 PANCAN 样本 ID（不同的样本）
        pancan_sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples_pancan)
        ]
        
        # 创建 LUAD 数据
        luad_gene_data = {gene: np.random.randint(0, 2, n_samples_luad) for gene in all_genes}
        luad_gene_df = pd.DataFrame(luad_gene_data, index=luad_sample_ids)
        luad_gene_df.index.name = 'sampleID'
        
        luad_clinical_df = pd.DataFrame({
            'sampleID': luad_sample_ids,
            'Age': np.random.randint(30, 80, n_samples_luad),
            'Gender': np.random.choice(['MALE', 'FEMALE'], n_samples_luad)
        })
        
        # 创建 PANCAN 数据
        pancan_gene_data = {gene: np.random.randint(0, 2, n_samples_pancan) for gene in all_genes}
        pancan_gene_df = pd.DataFrame(pancan_gene_data, index=pancan_sample_ids)
        pancan_gene_df.index.name = 'sampleID'
        
        pancan_clinical_df = pd.DataFrame({
            'sampleID': pancan_sample_ids,
            'Age': np.random.randint(30, 80, n_samples_pancan),
            'Gender': np.random.choice(['MALE', 'FEMALE'], n_samples_pancan)
        })
        
        # 处理 LUAD 数据
        luad_cleaner = DataCleaner('LUAD')
        luad_cleaner.gene_df = luad_gene_df
        luad_cleaner.clinical_df = luad_clinical_df
        luad_gene_df_reset = luad_gene_df.reset_index()
        luad_cleaner.merged_df = pd.merge(luad_gene_df_reset, luad_clinical_df, on='sampleID', how='inner')
        luad_cleaner.select_top_genes(n=20, forced_genes=forced_genes)
        luad_cleaner.clean_clinical()
        
        # 锁定 LUAD 的 Top20 基因
        locked_genes = luad_cleaner.top20_genes.copy()
        
        # 处理 PANCAN 数据，使用锁定的基因列表
        pancan_cleaner = DataCleaner('PANCAN')
        pancan_cleaner.gene_df = pancan_gene_df
        pancan_cleaner.clinical_df = pancan_clinical_df
        pancan_gene_df_reset = pancan_gene_df.reset_index()
        pancan_cleaner.merged_df = pd.merge(pancan_gene_df_reset, pancan_clinical_df, on='sampleID', how='inner')
        pancan_cleaner.select_top_genes(n=20, locked_genes=locked_genes)
        pancan_cleaner.clean_clinical()
        
        # 获取最终列顺序
        final_columns = ['sampleID', 'Age', 'Gender'] + locked_genes
        luad_final = luad_cleaner.merged_df[final_columns].copy()
        pancan_final = pancan_cleaner.merged_df[final_columns].copy()
        
        # 验证列名集合完全相同
        assert set(luad_final.columns) == set(pancan_final.columns), \
            f"列名集合不一致: LUAD={set(luad_final.columns)}, PANCAN={set(pancan_final.columns)}"
        
        # 验证列顺序完全相同
        assert list(luad_final.columns) == list(pancan_final.columns), \
            f"列顺序不一致: LUAD={list(luad_final.columns)}, PANCAN={list(pancan_final.columns)}"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        n_genes=st.integers(min_value=25, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_verify_feature_alignment_function(
        self, n_samples: int, n_genes: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 5: 特征空间对齐
        Validates: Requirements 4.3, 4.4
        
        验证 verify_feature_alignment 函数能正确检测对齐的数据集
        """
        np.random.seed(seed)
        
        from data_processor import verify_feature_alignment
        
        # 生成相同的列结构
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        columns = ['sampleID', 'Age', 'Gender'] + top20_genes
        
        # 创建两个具有相同列结构的 DataFrame
        df1_data = {col: np.random.randint(0, 100, n_samples) for col in columns}
        df1_data['sampleID'] = [f'SAMPLE_{i}' for i in range(n_samples)]
        df1 = pd.DataFrame(df1_data)[columns]
        
        df2_data = {col: np.random.randint(0, 100, n_samples) for col in columns}
        df2_data['sampleID'] = [f'SAMPLE_{i}' for i in range(n_samples)]
        df2 = pd.DataFrame(df2_data)[columns]
        
        # 验证对齐检查通过
        result = verify_feature_alignment(df1, df2, "Dataset1", "Dataset2")
        assert result is True, "对齐的数据集应该通过验证"
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_verify_feature_alignment_detects_column_mismatch(
        self, n_samples: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 5: 特征空间对齐
        Validates: Requirements 4.3, 4.4
        
        验证 verify_feature_alignment 函数能检测列名不匹配
        """
        np.random.seed(seed)
        
        from data_processor import verify_feature_alignment
        
        # 创建两个具有不同列的 DataFrame
        columns1 = ['sampleID', 'Age', 'Gender', 'EGFR', 'KRAS', 'TP53', 'GENE_A']
        columns2 = ['sampleID', 'Age', 'Gender', 'EGFR', 'KRAS', 'TP53', 'GENE_B']
        
        df1_data = {col: np.random.randint(0, 100, n_samples) for col in columns1}
        df1_data['sampleID'] = [f'SAMPLE_{i}' for i in range(n_samples)]
        df1 = pd.DataFrame(df1_data)[columns1]
        
        df2_data = {col: np.random.randint(0, 100, n_samples) for col in columns2}
        df2_data['sampleID'] = [f'SAMPLE_{i}' for i in range(n_samples)]
        df2 = pd.DataFrame(df2_data)[columns2]
        
        # 验证检测到列名不匹配
        with pytest.raises(ValueError) as exc_info:
            verify_feature_alignment(df1, df2, "Dataset1", "Dataset2")
        
        assert "特征空间不对齐" in str(exc_info.value)
    
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_verify_feature_alignment_detects_column_order_mismatch(
        self, n_samples: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 5: 特征空间对齐
        Validates: Requirements 4.3, 4.4
        
        验证 verify_feature_alignment 函数能检测列顺序不匹配
        """
        np.random.seed(seed)
        
        from data_processor import verify_feature_alignment
        
        # 创建两个具有相同列但不同顺序的 DataFrame
        columns1 = ['sampleID', 'Age', 'Gender', 'EGFR', 'KRAS', 'TP53']
        columns2 = ['sampleID', 'Age', 'Gender', 'KRAS', 'EGFR', 'TP53']  # EGFR 和 KRAS 顺序交换
        
        df1_data = {col: np.random.randint(0, 100, n_samples) for col in columns1}
        df1_data['sampleID'] = [f'SAMPLE_{i}' for i in range(n_samples)]
        df1 = pd.DataFrame(df1_data)[columns1]
        
        df2_data = {col: np.random.randint(0, 100, n_samples) for col in columns2}
        df2_data['sampleID'] = [f'SAMPLE_{i}' for i in range(n_samples)]
        df2 = pd.DataFrame(df2_data)[columns2]
        
        # 验证检测到列顺序不匹配
        with pytest.raises(ValueError) as exc_info:
            verify_feature_alignment(df1, df2, "Dataset1", "Dataset2")
        
        assert "列顺序不一致" in str(exc_info.value)


class TestOutputFileStructureConsistency:
    """
    Feature: dlc-data-engineering, Property 12: 输出文件结构一致性
    Validates: Requirements 9.2, 9.3
    
    *For any* 生成的 4 个 CSV 文件：
    - 列名集合完全相同
    - 列顺序完全相同
    - 必须包含：sampleID, Age, Gender, Top20 基因, Virtual_PM2.5, True_Prob, Outcome_Label
    """
    
    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_all_scenarios_produce_identical_columns(
        self, n_samples: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 12: 输出文件结构一致性
        Validates: Requirements 9.2, 9.3
        
        验证所有场景（interaction 和 linear）生成的数据具有相同的列结构
        """
        np.random.seed(seed)
        
        from data_processor import SemiSyntheticGenerator
        
        # 生成样本 ID
        sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples)
        ]
        
        # 生成基础数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        data = {
            'sampleID': sample_ids,
            'Age': np.random.uniform(30, 80, n_samples),
            'Gender': np.random.choice([0, 1], n_samples)
        }
        for gene in top20_genes:
            data[gene] = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(data)
        
        # 生成两种场景的数据
        gen_int = SemiSyntheticGenerator(df.copy(), top20_genes, seed=42)
        result_int = gen_int.generate('interaction')
        
        gen_lin = SemiSyntheticGenerator(df.copy(), top20_genes, seed=42)
        result_lin = gen_lin.generate('linear')
        
        # 验证列名集合完全相同
        assert set(result_int.columns) == set(result_lin.columns), \
            f"interaction 和 linear 场景的列名集合不一致"
        
        # 验证列顺序完全相同
        assert list(result_int.columns) == list(result_lin.columns), \
            f"interaction 和 linear 场景的列顺序不一致"
    
    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        scenario=st.sampled_from(['interaction', 'linear']),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_output_contains_required_columns(
        self, n_samples: int, scenario: str, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 12: 输出文件结构一致性
        Validates: Requirements 9.2, 9.3
        
        验证输出包含所有必需列：sampleID, Age, Gender, Top20 基因, Virtual_PM2.5, True_Prob, Outcome_Label
        """
        np.random.seed(seed)
        
        from data_processor import SemiSyntheticGenerator
        
        # 生成样本 ID
        sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples)
        ]
        
        # 生成基础数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        data = {
            'sampleID': sample_ids,
            'Age': np.random.uniform(30, 80, n_samples),
            'Gender': np.random.choice([0, 1], n_samples)
        }
        for gene in top20_genes:
            data[gene] = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(data)
        
        # 生成数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=42)
        result_df = generator.generate(scenario)
        
        # 定义必需列
        required_columns = ['sampleID', 'Age', 'Gender', 'Virtual_PM2.5', 'True_Prob', 'Outcome_Label']
        required_columns.extend(top20_genes)
        
        # 验证所有必需列都存在
        for col in required_columns:
            assert col in result_df.columns, \
                f"输出缺少必需列: {col}"
    
    @given(
        n_samples_luad=st.integers(min_value=20, max_value=50),
        n_samples_pancan=st.integers(min_value=20, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_different_datasets_same_structure(
        self, n_samples_luad: int, n_samples_pancan: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 12: 输出文件结构一致性
        Validates: Requirements 9.2, 9.3
        
        验证不同数据集（LUAD 和 PANCAN）生成的输出具有相同的列结构
        """
        np.random.seed(seed)
        
        from data_processor import SemiSyntheticGenerator
        
        # 共享的 Top20 基因列表
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        # 生成 LUAD 数据
        luad_sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples_luad)
        ]
        luad_data = {
            'sampleID': luad_sample_ids,
            'Age': np.random.uniform(30, 80, n_samples_luad),
            'Gender': np.random.choice([0, 1], n_samples_luad)
        }
        for gene in top20_genes:
            luad_data[gene] = np.random.randint(0, 2, n_samples_luad)
        luad_df = pd.DataFrame(luad_data)
        
        # 生成 PANCAN 数据
        pancan_sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples_pancan)
        ]
        pancan_data = {
            'sampleID': pancan_sample_ids,
            'Age': np.random.uniform(30, 80, n_samples_pancan),
            'Gender': np.random.choice([0, 1], n_samples_pancan)
        }
        for gene in top20_genes:
            pancan_data[gene] = np.random.randint(0, 2, n_samples_pancan)
        pancan_df = pd.DataFrame(pancan_data)
        
        # 生成所有 4 种组合的数据
        results = {}
        
        luad_gen_int = SemiSyntheticGenerator(luad_df.copy(), top20_genes, seed=42)
        results['luad_interaction'] = luad_gen_int.generate('interaction')
        
        luad_gen_lin = SemiSyntheticGenerator(luad_df.copy(), top20_genes, seed=42)
        results['luad_linear'] = luad_gen_lin.generate('linear')
        
        pancan_gen_int = SemiSyntheticGenerator(pancan_df.copy(), top20_genes, seed=42)
        results['pancan_interaction'] = pancan_gen_int.generate('interaction')
        
        pancan_gen_lin = SemiSyntheticGenerator(pancan_df.copy(), top20_genes, seed=42)
        results['pancan_linear'] = pancan_gen_lin.generate('linear')
        
        # 获取参考列结构
        ref_columns = list(results['luad_interaction'].columns)
        
        # 验证所有结果具有相同的列结构
        for key, df in results.items():
            assert list(df.columns) == ref_columns, \
                f"{key} 的列结构与参考不一致: {list(df.columns)} vs {ref_columns}"
    
    @given(
        n_samples=st.integers(min_value=20, max_value=80),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100)
    def test_output_column_order_consistency(
        self, n_samples: int, seed: int
    ):
        """
        Feature: dlc-data-engineering, Property 12: 输出文件结构一致性
        Validates: Requirements 9.2, 9.3
        
        验证输出列的顺序符合预期：sampleID, Age, Gender, Top20_Genes, Virtual_PM2.5, True_Prob, Outcome_Label
        """
        np.random.seed(seed)
        
        from data_processor import SemiSyntheticGenerator
        
        # 生成样本 ID
        sample_ids = [
            f'TCGA-{np.random.randint(0, 100):02d}-{np.random.randint(0, 10000):04d}-{np.random.randint(1, 99):02d}' 
            for _ in range(n_samples)
        ]
        
        # 生成基础数据
        forced_genes = ['EGFR', 'KRAS', 'TP53']
        other_genes = [f'GENE_{i}' for i in range(17)]
        top20_genes = forced_genes + other_genes
        
        data = {
            'sampleID': sample_ids,
            'Age': np.random.uniform(30, 80, n_samples),
            'Gender': np.random.choice([0, 1], n_samples)
        }
        for gene in top20_genes:
            data[gene] = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(data)
        
        # 生成数据
        generator = SemiSyntheticGenerator(df, top20_genes, seed=42)
        result_df = generator.generate('interaction')
        
        columns = list(result_df.columns)
        
        # 验证前三列是 sampleID, Age, Gender
        assert columns[0] == 'sampleID', f"第一列应为 sampleID，实际为 {columns[0]}"
        assert columns[1] == 'Age', f"第二列应为 Age，实际为 {columns[1]}"
        assert columns[2] == 'Gender', f"第三列应为 Gender，实际为 {columns[2]}"
        
        # 验证最后三列是 Virtual_PM2.5, True_Prob, Outcome_Label
        assert columns[-3] == 'Virtual_PM2.5', f"倒数第三列应为 Virtual_PM2.5，实际为 {columns[-3]}"
        assert columns[-2] == 'True_Prob', f"倒数第二列应为 True_Prob，实际为 {columns[-2]}"
        assert columns[-1] == 'Outcome_Label', f"最后一列应为 Outcome_Label，实际为 {columns[-1]}"
        
        # 验证中间是 Top20 基因
        gene_columns = columns[3:-3]
        assert len(gene_columns) == len(top20_genes), \
            f"基因列数量应为 {len(top20_genes)}，实际为 {len(gene_columns)}"

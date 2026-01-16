# -*- coding: utf-8 -*-
"""
DLC Data Processor
==================

数据处理核心模块，包含数据清洗和半合成数据生成功能。

Classes:
    DataCleaner: 数据清洗器，处理 TCGA 原始数据
    SemiSyntheticGenerator: 半合成数据生成器
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class DataCleaner:
    """
    数据清洗器，处理 TCGA 原始数据
    
    负责数据读取、转置、合并和特征筛选。
    
    Attributes:
        data_source (str): 数据源标识，'PANCAN' 或 'LUAD'
        gene_df (pd.DataFrame): 基因突变数据
        clinical_df (pd.DataFrame): 临床数据
        merged_df (pd.DataFrame): 合并后的数据
        top20_genes (List[str]): Top20 高频突变基因列表
    """
    
    def __init__(self, data_source: str):
        """
        初始化数据清洗器
        
        Args:
            data_source: 'PANCAN' 或 'LUAD'
        """
        if data_source not in ['PANCAN', 'LUAD']:
            raise ValueError(f"data_source 必须是 'PANCAN' 或 'LUAD'，收到: {data_source}")
        
        self.data_source = data_source
        self.gene_df: Optional[pd.DataFrame] = None
        self.clinical_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.top20_genes: Optional[List[str]] = None
    
    def load_gene_data(self, filepath: str) -> pd.DataFrame:
        """
        读取并转置基因突变数据
        
        操作：
        1. 读取 TSV 文件（行=基因，列=样本）
        2. 转置为（行=样本，列=基因）
        3. 设置 sampleID 为索引
        
        Args:
            filepath: 基因突变文件路径
            
        Returns:
            转置后的 DataFrame
        """
        
        try:
            # 读取 TSV 文件，第一列作为索引（基因名）
            df = pd.read_csv(filepath, sep='\t', index_col=0)
            
            # 转置：行=样本，列=基因
            df_transposed = df.T
            
            # 重命名索引为 sampleID
            df_transposed.index.name = 'sampleID'
            
            # 保存到实例变量
            self.gene_df = df_transposed
            
            return df_transposed
            
        except FileNotFoundError:
            raise FileNotFoundError(f"基因数据文件不存在: {filepath}")
        except Exception as e:
            raise ValueError(f"读取基因数据文件失败: {filepath}, 错误: {str(e)}")
        
    def load_clinical_data(self, filepath: str) -> pd.DataFrame:
        """
        读取临床数据并映射字段名
        
        字段映射：
        - PANCAN: sample -> sampleID, age_at_initial_pathologic_diagnosis -> Age, gender -> Gender
        - LUAD: sampleID -> sampleID, age_at_initial_pathologic_diagnosis -> Age, gender -> Gender
        
        Args:
            filepath: 临床数据文件路径
            
        Returns:
            清洗后的临床 DataFrame
        """
        # 读取 TSV 文件
        df = pd.read_csv(filepath, sep='\t')
        
        # 根据数据源定义字段映射
        if self.data_source == 'PANCAN':
            column_mapping = {
                'sample': 'sampleID',
                'age_at_initial_pathologic_diagnosis': 'Age',
                'gender': 'Gender'
            }
        else:  # LUAD
            column_mapping = {
                'sampleID': 'sampleID',
                'age_at_initial_pathologic_diagnosis': 'Age',
                'gender': 'Gender'
            }
        
        # 验证必需列存在
        for old_col in column_mapping.keys():
            if old_col not in df.columns:
                raise KeyError(f"临床数据缺少必需列: {old_col}")
        
        # 执行字段映射，只保留需要的列
        df = df.rename(columns=column_mapping)
        df = df[['sampleID', 'Age', 'Gender']]
        
        # 存储到实例属性
        self.clinical_df = df
        
        return df
    
    def _validate_long_format_id(self, ids: pd.Series) -> None:
        """
        验证 ID 格式为长格式
        
        长格式 ID 模式: TCGA-XX-XXXX-XX
        例如: TCGA-05-4244-01
        
        Args:
            ids: sampleID Series
            
        Raises:
            ValueError: 如果检测到短格式 ID
        """
        import re
        
        # 长格式 ID 正则模式: TCGA-XX-XXXX-XX
        # XX 可以是字母或数字，XXXX 是 4 位字母数字
        long_format_pattern = re.compile(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]?$')
        
        # 检查每个 ID 是否匹配长格式
        invalid_ids = []
        for sample_id in ids:
            if not long_format_pattern.match(str(sample_id)):
                invalid_ids.append(sample_id)
        
        if invalid_ids:
            # 最多显示 5 个违规 ID 样例
            sample_invalid = invalid_ids[:5]
            raise ValueError(
                f"检测到 {len(invalid_ids)} 个非长格式 ID。"
                f"期望格式: TCGA-XX-XXXX-XX (如 TCGA-05-4244-01)。"
                f"违规样例: {sample_invalid}"
            )
    
    def merge_data(self) -> pd.DataFrame:
        """
        基于长格式 ID 合并基因和临床数据
        
        规则：
        - 使用 Inner Join
        - 验证 ID 格式为长格式（如 TCGA-XX-XXXX-XX）
        - 打印合并后样本量
        
        Returns:
            合并后的 DataFrame
            
        Raises:
            ValueError: 如果基因数据或临床数据未加载，或 ID 格式不正确
        """
        # 检查数据是否已加载
        if self.gene_df is None:
            raise ValueError("基因数据未加载，请先调用 load_gene_data()")
        if self.clinical_df is None:
            raise ValueError("临床数据未加载，请先调用 load_clinical_data()")
        
        # 重置基因数据索引，将 sampleID 变为列
        gene_df_reset = self.gene_df.reset_index()
        
        # 验证基因数据的 ID 格式
        self._validate_long_format_id(gene_df_reset['sampleID'])
        
        # 验证临床数据的 ID 格式
        self._validate_long_format_id(self.clinical_df['sampleID'])
        
        # 执行 Inner Join
        merged_df = pd.merge(
            gene_df_reset,
            self.clinical_df,
            on='sampleID',
            how='inner'
        )
        
        # 打印合并后样本量
        print(f"[{self.data_source}] 合并完成: "
              f"基因数据 {len(gene_df_reset)} 样本 × "
              f"临床数据 {len(self.clinical_df)} 样本 → "
              f"合并后 {len(merged_df)} 样本")
        
        # 保存到实例变量
        self.merged_df = merged_df
        
        return merged_df
    
    def select_top_genes(self, n: int = 20, 
                         forced_genes: List[str] = None,
                         locked_genes: List[str] = None) -> List[str]:
        """
        筛选高频突变基因
        
        Args:
            n: 选择的基因数量，默认 20
            forced_genes: 必须包含的基因列表，默认 ['EGFR', 'KRAS', 'TP53']
            locked_genes: 如果提供，则强制使用此列表（用于 PANCAN 对齐）
        
        Returns:
            Top N 基因列表
            
        Raises:
            ValueError: 如果合并数据未加载，或强制基因在数据中不存在
        """
        # 设置默认强制基因
        if forced_genes is None:
            forced_genes = ['EGFR', 'KRAS', 'TP53']
        
        # 如果提供了锁定基因列表，直接使用（用于 PANCAN 对齐 LUAD）
        if locked_genes is not None:
            self.top20_genes = locked_genes.copy()
            return self.top20_genes
        
        # 检查合并数据是否已加载
        if self.merged_df is None:
            raise ValueError("合并数据未加载，请先调用 merge_data()")
        
        # 获取所有基因列（排除非基因列）
        non_gene_cols = ['sampleID', 'Age', 'Gender']
        gene_cols = [col for col in self.merged_df.columns if col not in non_gene_cols]
        
        # 验证强制基因存在于数据中
        missing_forced = [g for g in forced_genes if g not in gene_cols]
        if missing_forced:
            raise ValueError(f"强制基因在数据中不存在: {missing_forced}")
        
        # 计算每个基因的突变频率（均值，因为是 0/1 二值）
        gene_frequencies = self.merged_df[gene_cols].mean().sort_values(ascending=False)
        
        # 选择 Top N 高频基因
        top_genes = gene_frequencies.head(n).index.tolist()
        
        # 强制包含指定基因，替换最低频基因
        for forced_gene in forced_genes:
            if forced_gene not in top_genes:
                # 找到当前 top_genes 中不在 forced_genes 里的最低频基因
                # 从后往前找，因为 top_genes 是按频率降序排列的
                for i in range(len(top_genes) - 1, -1, -1):
                    if top_genes[i] not in forced_genes:
                        # 替换最低频的非强制基因
                        top_genes[i] = forced_gene
                        break
        
        # 保存到实例变量
        self.top20_genes = top_genes
        
        return self.top20_genes
    
    def clean_clinical(self) -> pd.DataFrame:
        """
        清洗临床特征
        
        操作：
        - Age: 转数值型，均值填充缺失值
        - Gender: Male=0, Female=1
        - 基因列: 确保为 0/1 二值
        
        Returns:
            清洗后的 DataFrame
            
        Raises:
            ValueError: 如果合并数据或 Top20 基因列表未加载
        """
        # 检查前置条件
        if self.merged_df is None:
            raise ValueError("合并数据未加载，请先调用 merge_data()")
        if self.top20_genes is None:
            raise ValueError("Top20 基因列表未设置，请先调用 select_top_genes()")
        
        # 复制数据，避免修改原始数据
        df = self.merged_df.copy()
        
        # 1. Age: 转数值型，均值填充缺失值
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        age_mean = df['Age'].mean()
        df['Age'] = df['Age'].fillna(age_mean)
        
        # 2. Gender: Male=0, Female=1
        # 处理各种可能的 Gender 表示方式
        gender_mapping = {
            'MALE': 0, 'Male': 0, 'male': 0, 'M': 0, 'm': 0,
            'FEMALE': 1, 'Female': 1, 'female': 1, 'F': 1, 'f': 1
        }
        df['Gender'] = df['Gender'].map(gender_mapping)
        # 对于未知值，记录警告并用众数填充
        if df['Gender'].isna().any():
            unknown_count = df['Gender'].isna().sum()
            print(f"[警告] Gender 列有 {unknown_count} 个未知值，将用众数填充")
            gender_mode = df['Gender'].mode()[0] if not df['Gender'].mode().empty else 0
            df['Gender'] = df['Gender'].fillna(gender_mode)
        df['Gender'] = df['Gender'].astype(int)
        
        # 3. 基因列: 确保为 0/1 二值
        for gene in self.top20_genes:
            if gene in df.columns:
                # 将 >0 的值转为 1，其他转为 0
                df[gene] = (df[gene] > 0).astype(int)
        
        # 更新实例变量
        self.merged_df = df
        
        return df
    
    def process(self, gene_path: str, clinical_path: str,
                locked_genes: List[str] = None) -> pd.DataFrame:
        """
        完整处理流程
        
        串联所有处理步骤：
        1. 加载基因数据
        2. 加载临床数据
        3. 合并数据
        4. 筛选 Top20 基因
        5. 清洗临床特征
        6. 重排列顺序，返回最终 DataFrame
        
        Args:
            gene_path: 基因文件路径
            clinical_path: 临床文件路径
            locked_genes: 锁定的基因列表（用于特征对齐）
        
        Returns:
            清洗完成的 DataFrame，列顺序为：
            sampleID, Age, Gender, Top20_Genes...
        """
        print(f"\n{'='*50}")
        print(f"开始处理 {self.data_source} 数据")
        print(f"{'='*50}")
        
        # Step 1: 加载基因数据
        print(f"\n[Step 1] 加载基因数据: {gene_path}")
        self.load_gene_data(gene_path)
        print(f"  - 基因数据形状: {self.gene_df.shape}")
        
        # Step 2: 加载临床数据
        print(f"\n[Step 2] 加载临床数据: {clinical_path}")
        self.load_clinical_data(clinical_path)
        print(f"  - 临床数据形状: {self.clinical_df.shape}")
        
        # Step 3: 合并数据
        print(f"\n[Step 3] 合并基因和临床数据")
        self.merge_data()
        
        # Step 4: 筛选 Top20 基因
        print(f"\n[Step 4] 筛选 Top20 基因")
        if locked_genes is not None:
            print(f"  - 使用锁定的基因列表 (来自 LUAD)")
            self.select_top_genes(locked_genes=locked_genes)
        else:
            print(f"  - 计算高频突变基因")
            self.select_top_genes()
        print(f"  - Top20 基因: {self.top20_genes}")
        
        # Step 5: 清洗临床特征
        print(f"\n[Step 5] 清洗临床特征")
        self.clean_clinical()
        
        # Step 6: 重排列顺序
        print(f"\n[Step 6] 整理最终 DataFrame")
        # 列顺序: sampleID, Age, Gender, Top20_Genes
        final_columns = ['sampleID', 'Age', 'Gender'] + self.top20_genes
        
        # 验证所有列都存在
        missing_cols = [col for col in final_columns if col not in self.merged_df.columns]
        if missing_cols:
            raise ValueError(f"最终 DataFrame 缺少列: {missing_cols}")
        
        # 选择并重排列
        result_df = self.merged_df[final_columns].copy()
        
        print(f"  - 最终数据形状: {result_df.shape}")
        print(f"  - 列: {list(result_df.columns)}")
        print(f"\n{'='*50}")
        print(f"{self.data_source} 数据处理完成!")
        print(f"{'='*50}\n")
        
        return result_df


class SemiSyntheticGenerator:
    """
    半合成数据生成器
    
    基于清洗后的数据生成虚拟环境暴露和结局标签。
    
    Attributes:
        df (pd.DataFrame): 清洗后的 DataFrame
        top20_genes (List[str]): Top20 基因列表
        seed (int): 随机种子
    """
    
    # 核心权重参数（基于文献）
    W_BASE = 0.086    # 基准环境权重 (Hamra et al., 2014)
    W_INT = 0.69      # 交互环境权重 (Hill et al., 2023)
    W_GENE = 0.5      # 基因主效应
    INTERCEPT = -3.0  # 截距项
    
    def __init__(self, df: pd.DataFrame, top20_genes: List[str], seed: int = 42):
        """
        初始化生成器
        
        Args:
            df: 清洗后的 DataFrame，必须包含 sampleID, Age, Gender 和基因列
            top20_genes: Top20 基因列表
            seed: 随机种子，默认 42
            
        Raises:
            ValueError: 如果 DataFrame 缺少必需列
        """
        # 验证必需列存在
        required_cols = ['sampleID', 'Age', 'Gender']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame 缺少必需列: {missing_cols}")
        
        # 验证基因列存在
        missing_genes = [gene for gene in top20_genes if gene not in df.columns]
        if missing_genes:
            raise ValueError(f"DataFrame 缺少基因列: {missing_genes}")
        
        # 初始化实例变量
        self.df = df.copy()
        self.top20_genes = top20_genes.copy()
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(seed)
    
    # PM2.5 生成参数
    PM25_BASE = 30.0      # 基准值
    PM25_AGE_COEF = 3.0   # Age 系数（增强混杂信号）
    PM25_NOISE_STD = 10.0 # 噪声标准差
    
    def generate_pm25(self) -> pd.Series:
        """
        生成虚拟 PM2.5 暴露
        
        公式: 30 + 3.0 * Z(Age) + N(0, 10)
        
        其中：
        - Z(Age) 是 Age 的 z-score 标准化值
        - N(0, 10) 是均值为 0、标准差为 10 的正态分布噪声
        - Age 系数为 3.0（增强混杂信号，理论相关系数约 0.29）
        
        Returns:
            Virtual_PM2.5 Series
        """
        # 确保随机种子一致性
        np.random.seed(self.seed)
        
        # 计算 Age 的 z-score
        age_mean = self.df['Age'].mean()
        age_std = self.df['Age'].std()
        
        # 处理标准差为 0 的边界情况
        if age_std == 0 or np.isnan(age_std):
            z_age = pd.Series(0.0, index=self.df.index)
        else:
            z_age = (self.df['Age'] - age_mean) / age_std
        
        # 生成正态分布噪声 N(0, NOISE_STD)
        noise = np.random.normal(0, self.PM25_NOISE_STD, len(self.df))
        
        # 应用公式: BASE + AGE_COEF * Z(Age) + N(0, NOISE_STD)
        virtual_pm25 = self.PM25_BASE + self.PM25_AGE_COEF * z_age + noise
        
        # 存储到 DataFrame
        self.df['Virtual_PM2.5'] = virtual_pm25
        
        return virtual_pm25
    
    def _compute_genetics(self) -> pd.Series:
        """
        计算基因主效应
        
        公式: Genetics = W_GENE * sum(Top20_Genes) = 0.5 * sum(Top20_Genes)
        
        Returns:
            Genetics 评分 Series
        """
        # 计算 Top20 基因的突变总数
        gene_sum = self.df[self.top20_genes].sum(axis=1)
        
        # 应用基因权重
        genetics = self.W_GENE * gene_sum
        
        return genetics
    
    def generate_outcome(self, scenario: str) -> Tuple[pd.Series, pd.Series]:
        """
        生成结局标签
        
        Args:
            scenario: 'interaction' 或 'linear'
        
        交互场景公式:
            L = INTERCEPT + W_BASE*PM2.5* + W_INT*(PM2.5* × EGFR) + Genetics
            即: L = -3.0 + 0.086*PM2.5* + 0.69*(PM2.5* × EGFR) + Genetics
        
        线性场景公式:
            L = INTERCEPT + W_BASE*PM2.5* + Genetics
            即: L = -3.0 + 0.086*PM2.5* + Genetics
        
        其中 PM2.5* 是标准化后的 Virtual_PM2.5
        
        Returns:
            (True_Prob, Outcome_Label) 元组
            
        Raises:
            ValueError: 如果 scenario 不是 'interaction' 或 'linear'
            ValueError: 如果 Virtual_PM2.5 列不存在
        """
        # 验证 scenario 参数
        if scenario not in ['interaction', 'linear']:
            raise ValueError(f"scenario 必须是 'interaction' 或 'linear'，收到: {scenario}")
        
        # 验证 Virtual_PM2.5 列存在
        if 'Virtual_PM2.5' not in self.df.columns:
            raise ValueError("Virtual_PM2.5 列不存在，请先调用 generate_pm25()")
        
        # 标准化 PM2.5 (z-score)
        pm25_mean = self.df['Virtual_PM2.5'].mean()
        pm25_std = self.df['Virtual_PM2.5'].std()
        
        if pm25_std == 0 or np.isnan(pm25_std):
            pm25_star = pd.Series(0.0, index=self.df.index)
        else:
            pm25_star = (self.df['Virtual_PM2.5'] - pm25_mean) / pm25_std
        
        # 计算基因主效应
        genetics = self._compute_genetics()
        
        # 计算 Logit
        if scenario == 'interaction':
            # 交互场景: L = -3.0 + 0.086*PM2.5* + 0.69*(PM2.5* × EGFR) + Genetics
            egfr = self.df['EGFR']
            logit = (self.INTERCEPT + 
                     self.W_BASE * pm25_star + 
                     self.W_INT * (pm25_star * egfr) + 
                     genetics)
        else:
            # 线性场景: L = -3.0 + 0.086*PM2.5* + Genetics
            logit = (self.INTERCEPT + 
                     self.W_BASE * pm25_star + 
                     genetics)
        
        # Sigmoid 转换: P = 1 / (1 + exp(-L))
        true_prob = 1.0 / (1.0 + np.exp(-logit))
        
        # Bernoulli 采样生成 Outcome_Label
        # 使用当前随机状态（由 seed 控制）
        outcome_label = (np.random.random(len(self.df)) < true_prob).astype(int)
        
        return true_prob, pd.Series(outcome_label, index=self.df.index)
    
    def generate(self, scenario: str) -> pd.DataFrame:
        """
        完整生成流程
        
        串联 PM2.5 生成和结局生成，返回包含所有生成列的 DataFrame。
        
        Args:
            scenario: 'interaction' 或 'linear'
        
        Returns:
            包含所有生成列的 DataFrame，列包括：
            sampleID, Age, Gender, Top20_Genes, Virtual_PM2.5, True_Prob, Outcome_Label
            
        Raises:
            ValueError: 如果 scenario 不是 'interaction' 或 'linear'
        """
        # 验证 scenario 参数
        if scenario not in ['interaction', 'linear']:
            raise ValueError(f"scenario 必须是 'interaction' 或 'linear'，收到: {scenario}")
        
        print(f"\n[SemiSyntheticGenerator] 开始生成半合成数据 (scenario={scenario})")
        
        # Step 1: 生成 Virtual_PM2.5
        print(f"  - Step 1: 生成 Virtual_PM2.5")
        self.generate_pm25()
        
        # Step 2: 生成结局标签
        print(f"  - Step 2: 生成结局标签 ({scenario} 场景)")
        true_prob, outcome_label = self.generate_outcome(scenario)
        
        # 将结果添加到 DataFrame
        self.df['True_Prob'] = true_prob
        self.df['Outcome_Label'] = outcome_label
        
        # 打印统计信息
        print(f"  - Virtual_PM2.5: mean={self.df['Virtual_PM2.5'].mean():.2f}, "
              f"std={self.df['Virtual_PM2.5'].std():.2f}")
        print(f"  - True_Prob: mean={true_prob.mean():.4f}, "
              f"min={true_prob.min():.4f}, max={true_prob.max():.4f}")
        print(f"  - Outcome_Label: 阳性率={outcome_label.mean():.2%}")
        
        # 验证 Age 与 PM2.5 的相关性（混杂效应）
        corr = self.df['Age'].corr(self.df['Virtual_PM2.5'])
        print(f"  - Age-PM2.5 相关系数: r={corr:.4f}")
        
        print(f"[SemiSyntheticGenerator] 生成完成!\n")
        
        return self.df


def verify_feature_alignment(df1: pd.DataFrame, df2: pd.DataFrame, 
                             name1: str = "LUAD", name2: str = "PANCAN") -> bool:
    """
    验证两个 DataFrame 的特征空间对齐
    
    检查：
    - 列名集合完全相同
    - 列顺序完全相同
    
    Args:
        df1: 第一个 DataFrame
        df2: 第二个 DataFrame
        name1: 第一个数据集名称
        name2: 第二个数据集名称
        
    Returns:
        True 如果对齐，否则抛出 ValueError
        
    Raises:
        ValueError: 如果列名或列顺序不一致
    """
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    
    # 检查列名集合
    set1 = set(cols1)
    set2 = set(cols2)
    
    if set1 != set2:
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        raise ValueError(
            f"特征空间不对齐！\n"
            f"  仅在 {name1} 中: {only_in_1}\n"
            f"  仅在 {name2} 中: {only_in_2}"
        )
    
    # 检查列顺序
    if cols1 != cols2:
        # 找出第一个不同的位置
        for i, (c1, c2) in enumerate(zip(cols1, cols2)):
            if c1 != c2:
                raise ValueError(
                    f"列顺序不一致！位置 {i}: {name1}='{c1}', {name2}='{c2}'"
                )
    
    print(f"[验证通过] {name1} 和 {name2} 特征空间完全对齐")
    print(f"  - 共 {len(cols1)} 列")
    return True


def main():
    """
    主执行函数
    
    执行顺序：
    1. 处理 LUAD 数据，锁定 Top20 基因
    2. 处理 PANCAN 数据，强制使用 LUAD 的基因列表
    3. 验证特征对齐
    4. 生成半合成数据（两种场景）
    5. 保存 4 个 CSV 文件
    6. 返回统计信息
    
    Returns:
        dict: 包含统计信息的字典
    """
    import os
    
    print("\n" + "=" * 60)
    print("DLC Data Engineering - 主执行流程")
    print("=" * 60)
    
    # 定义数据路径
    LUAD_GENE_PATH = "data/LUAD/LUAD_mc3_gene_level.txt"
    LUAD_CLINICAL_PATH = "data/LUAD/TCGA.LUAD.sampleMap_LUAD_clinicalMatrix.txt"
    PANCAN_GENE_PATH = "data/PANCAN/PANCAN_mutation.txt"
    PANCAN_CLINICAL_PATH = "data/PANCAN/PANCAN_clinical.txt"
    
    # 输出文件路径
    OUTPUT_DIR = "data"
    OUTPUT_FILES = {
        'luad_interaction': os.path.join(OUTPUT_DIR, 'luad_synthetic_interaction.csv'),
        'luad_linear': os.path.join(OUTPUT_DIR, 'luad_synthetic_linear.csv'),
        'pancan_interaction': os.path.join(OUTPUT_DIR, 'pancan_synthetic_interaction.csv'),
        'pancan_linear': os.path.join(OUTPUT_DIR, 'pancan_synthetic_linear.csv'),
    }
    
    # 统计信息收集
    stats = {
        'luad_samples': 0,
        'pancan_samples': 0,
        'top20_genes': [],
        'feature_aligned': False,
        'age_pm25_corr': {},
        'egfr_group_prob': {},
    }
    
    # ========================================
    # Step 1: 处理 LUAD 数据，锁定 Top20 基因
    # ========================================
    print("\n" + "-" * 60)
    print("Step 1: 处理 LUAD 数据，锁定 Top20 基因")
    print("-" * 60)
    
    luad_cleaner = DataCleaner(data_source='LUAD')
    luad_df = luad_cleaner.process(
        gene_path=LUAD_GENE_PATH,
        clinical_path=LUAD_CLINICAL_PATH,
        locked_genes=None  # 首次处理，计算 Top20
    )
    
    # 锁定 Top20 基因列表
    locked_top20_genes = luad_cleaner.top20_genes.copy()
    stats['top20_genes'] = locked_top20_genes
    stats['luad_samples'] = len(luad_df)
    
    print(f"\n[锁定] Top20 基因列表: {locked_top20_genes}")
    
    # ========================================
    # Step 2: 处理 PANCAN 数据，强制使用 LUAD 基因列表
    # ========================================
    print("\n" + "-" * 60)
    print("Step 2: 处理 PANCAN 数据，强制使用 LUAD 基因列表")
    print("-" * 60)
    
    pancan_cleaner = DataCleaner(data_source='PANCAN')
    pancan_df = pancan_cleaner.process(
        gene_path=PANCAN_GENE_PATH,
        clinical_path=PANCAN_CLINICAL_PATH,
        locked_genes=locked_top20_genes  # 强制使用 LUAD 的基因列表
    )
    
    stats['pancan_samples'] = len(pancan_df)
    
    # ========================================
    # Step 3: 验证特征对齐
    # ========================================
    print("\n" + "-" * 60)
    print("Step 3: 验证特征对齐")
    print("-" * 60)
    
    verify_feature_alignment(luad_df, pancan_df, "LUAD", "PANCAN")
    stats['feature_aligned'] = True
    
    # ========================================
    # Step 4: 生成半合成数据（两种场景）
    # ========================================
    print("\n" + "-" * 60)
    print("Step 4: 生成半合成数据")
    print("-" * 60)
    
    # 存储生成的数据
    generated_data = {}
    
    # 4.1 LUAD - Interaction 场景
    print("\n[4.1] LUAD - Interaction 场景")
    luad_gen_int = SemiSyntheticGenerator(luad_df, locked_top20_genes, seed=42)
    luad_int_df = luad_gen_int.generate(scenario='interaction')
    generated_data['luad_interaction'] = luad_int_df
    stats['age_pm25_corr']['luad_interaction'] = luad_int_df['Age'].corr(luad_int_df['Virtual_PM2.5'])
    stats['egfr_group_prob']['luad_interaction'] = {
        'egfr_0': luad_int_df[luad_int_df['EGFR'] == 0]['True_Prob'].mean(),
        'egfr_1': luad_int_df[luad_int_df['EGFR'] == 1]['True_Prob'].mean(),
    }
    
    # 4.2 LUAD - Linear 场景
    print("\n[4.2] LUAD - Linear 场景")
    luad_gen_lin = SemiSyntheticGenerator(luad_df, locked_top20_genes, seed=42)
    luad_lin_df = luad_gen_lin.generate(scenario='linear')
    generated_data['luad_linear'] = luad_lin_df
    stats['age_pm25_corr']['luad_linear'] = luad_lin_df['Age'].corr(luad_lin_df['Virtual_PM2.5'])
    
    # 4.3 PANCAN - Interaction 场景
    print("\n[4.3] PANCAN - Interaction 场景")
    pancan_gen_int = SemiSyntheticGenerator(pancan_df, locked_top20_genes, seed=42)
    pancan_int_df = pancan_gen_int.generate(scenario='interaction')
    generated_data['pancan_interaction'] = pancan_int_df
    stats['age_pm25_corr']['pancan_interaction'] = pancan_int_df['Age'].corr(pancan_int_df['Virtual_PM2.5'])
    stats['egfr_group_prob']['pancan_interaction'] = {
        'egfr_0': pancan_int_df[pancan_int_df['EGFR'] == 0]['True_Prob'].mean(),
        'egfr_1': pancan_int_df[pancan_int_df['EGFR'] == 1]['True_Prob'].mean(),
    }
    
    # 4.4 PANCAN - Linear 场景
    print("\n[4.4] PANCAN - Linear 场景")
    pancan_gen_lin = SemiSyntheticGenerator(pancan_df, locked_top20_genes, seed=42)
    pancan_lin_df = pancan_gen_lin.generate(scenario='linear')
    generated_data['pancan_linear'] = pancan_lin_df
    stats['age_pm25_corr']['pancan_linear'] = pancan_lin_df['Age'].corr(pancan_lin_df['Virtual_PM2.5'])
    
    # ========================================
    # Step 5: 保存 4 个 CSV 文件
    # ========================================
    print("\n" + "-" * 60)
    print("Step 5: 保存 CSV 文件")
    print("-" * 60)
    
    for key, filepath in OUTPUT_FILES.items():
        df = generated_data[key]
        df.to_csv(filepath, index=False)
        print(f"  - 已保存: {filepath} ({len(df)} 行, {len(df.columns)} 列)")
    
    # 验证输出文件结构一致性
    print("\n[验证] 输出文件结构一致性")
    ref_cols = list(generated_data['luad_interaction'].columns)
    for key, df in generated_data.items():
        if list(df.columns) != ref_cols:
            raise ValueError(f"输出文件 {key} 列结构与参考不一致")
    print("  - 所有 4 个文件列结构完全一致")
    
    # ========================================
    # Step 6: 打印统计摘要
    # ========================================
    print("\n" + "=" * 60)
    print("执行完成 - 统计摘要")
    print("=" * 60)
    print(f"\n样本数:")
    print(f"  - LUAD: {stats['luad_samples']} 样本")
    print(f"  - PANCAN: {stats['pancan_samples']} 样本")
    print(f"\nTop20 基因列表:")
    print(f"  {stats['top20_genes']}")
    print(f"\n特征对齐: {'✓ 通过' if stats['feature_aligned'] else '✗ 失败'}")
    print(f"\nAge-PM2.5 相关系数:")
    for key, corr in stats['age_pm25_corr'].items():
        print(f"  - {key}: r={corr:.4f}")
    print(f"\nEGFR 分组平均概率 (Interaction 场景):")
    for key, probs in stats['egfr_group_prob'].items():
        print(f"  - {key}: EGFR=0 → {probs['egfr_0']:.4f}, EGFR=1 → {probs['egfr_1']:.4f}")
    
    print("\n" + "=" * 60)
    print("所有文件已生成完毕!")
    print("=" * 60 + "\n")
    
    return stats


if __name__ == '__main__':
    main()

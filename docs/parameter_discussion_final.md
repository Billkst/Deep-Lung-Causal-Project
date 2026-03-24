# 参数讨论与敏感性分析 (Parameter Discussion & Sensitivity Analysis)

本文档详细阐述 Deep-Lung-Causal (DLC) 模型优化过程中的参数选择逻辑、实验设计及最终结果。

---

## 1. 实验设计与数据解读 (Experimental Design)

为了验证模型在因果推断(Causal Inference)与预测性能(Prediction Performance)之间的平衡,我们采用了控制变量法(Control Variable Strategy)对关键超参数进行了独立的扫描实验。

### 1.1 最终SOTA配置

**架构参数**:
- `d_hidden = 128`
- `num_layers = 3`

**损失函数权重**:
- `lambda_cate = 5.0` (因果效应监督权重，数据驱动最优值)
- `lambda_hsic = 0.1` (去混杂强度权重)

### 1.2 评估指标定义

- **Delta CATE** (关键指标): $\Delta \tau = \mathbb{E}[\tau | G=Mut] - \mathbb{E}[\tau | G=WT]$。正值表示EGFR突变组获益更高,符合临床预期。
- **PEHE** (Precision in Estimation of Heterogeneous Effect): $\sqrt{\frac{1}{N}\sum (\hat{\tau}_i - \tau_i)^2}$。衡量个体治疗效应预测的误差,**越低越好**。
- **AUC**: 衡量模型对生存结果(Outcome)的分类预测能力,**越高越好**。

---

## 2. 参数敏感性分析结果

### 2.1 Lambda CATE (因果监督权重)

**实验设置**: 固定 d_hidden=128, num_layers=3, lambda_hsic=0.1,密集扫描 lambda_cate ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0}

**数据摘要** (关键点):

| lambda_cate | AUC    | PEHE   | Delta CATE |
|-------------|--------|--------|------------|
| 0.0         | 0.7879 | 0.1319 | 0.0644     |
| 1.0         | 0.7885 | 0.1214 | 0.1443     |
| 2.0         | 0.7888 | 0.1203 | 0.1697     |
| 5.0         | 0.7938 | 0.1125 | 0.1975     |
| 7.0         | 0.7925 | 0.1122 | 0.2018     |

**关键发现**:

1. **强敏感参数**: lambda_cate 对模型性能有显著影响
2. **单调趋势**: 
   - AUC: 在 [0, 5.0] 单调递增,5.0达到峰值 (0.7938)
   - Delta CATE: 在整个 [0, 7.0] 区间单调递增,7.0达到峰值 (0.2018)
   - PEHE: 在整个 [0, 7.0] 区间单调递减,7.0达到最优 (0.1122)
3. **lambda_cate=2.0 的定位**: 
   - **不是任何单一指标的最优点**
   - 这是一个**历史默认值**,性能处于中等水平
   - 所有指标在更高的lambda_cate值处都有更好表现
4. **推荐值**: 
   - **lambda_cate=5.0** (综合平衡最佳): AUC最优,Delta CATE和PEHE优秀
   - lambda_cate=7.0 (激进选择): Delta CATE和PEHE最优,AUC略低

**可视化**: 见 `fig_lambda_cate_tradeoff.png` 和 `fig_lambda_cate_auc.png`

---

### 2.2 Lambda HSIC (去混杂权重)

**实验设置**: 固定 d_hidden=128, num_layers=3, lambda_cate=2.0,扫描 lambda_hsic ∈ {0.0, 0.01, 0.1, 1.0, 10.0}

**数据摘要**:

| lambda_hsic | AUC    | PEHE   | Delta CATE |
|-------------|--------|--------|------------|
| 0.0         | 0.7868 | 0.1177 | 0.1723     |
| 0.01        | 0.7864 | 0.1176 | 0.1724     |
| 0.1         | 0.7868 | 0.1179 | 0.1721     |
| 1.0         | 0.7867 | 0.1177 | 0.1724     |
| 10.0        | 0.7869 | 0.1176 | 0.1717     |

**关键发现**:

1. **极弱敏感参数**: lambda_hsic 在4个数量级范围内影响极小
2. **数值分析**:
   - AUC 变化范围: [0.7864, 0.7869],相对变化仅 **0.06%**
   - PEHE 变化范围: [0.1176, 0.1179],相对变化仅 **0.28%**
   - Delta CATE 变化范围: [0.1717, 0.1724],相对变化仅 **0.43%**
   - 所有变化均小于标准误差范围
3. **统计意义**: lambda_hsic 的影响**不具有统计显著性**,无明显最优点
4. **当前选择 (lambda_hsic=0.1)**:
   - **不是因为它是最优点**(因为不存在明显最优点)
   - **而是因为模型对该参数不敏感**,任何合理值都可以
   - 0.1是中等值,避免极端

**可视化**: 见 `fig_lambda_hsic_tradeoff.png` 和 `fig_lambda_hsic_auc.png`

---

### 2.3 网络架构 (d_hidden × num_layers)

**实验设置**: 固定 lambda_cate=2.0, lambda_hsic=0.1,扫描不同的 (d_hidden, num_layers) 组合

**关键发现**:

1. **128×3 是多目标平衡点**:
   - 在预测性能(AUC)与因果推断(Delta CATE)之间取得良好平衡
   - PEHE误差处于较低水平
2. **过小架构** (如64×2): 容量不足,难以捕捉复杂的因果关系
3. **过大架构** (如256×4): 容易过拟合,且计算成本显著增加

**可视化**: 见 `architecture_tradeoff_scatter.png` 和相关热力图

---

## 3. 结论与建议

### 3.1 最终SOTA配置

```python
d_hidden = 128
num_layers = 3
lambda_cate = 5.0  # 数据驱动的最优值
lambda_hsic = 0.1
```

### 3.2 参数选择原则

1. **lambda_cate** (强敏感参数): 
   - 对所有指标有显著影响
   - **推荐值: 5.0** (综合平衡最佳)
   - 当前值2.0是历史默认值,非数据驱动的最优选择
   - 更高的值(7.0)可获得最强的因果效应,但AUC略有下降
   
2. **lambda_hsic** (极弱敏感参数): 
   - 在4个数量级范围内(0.01-10.0)影响<0.5%
   - **保持默认值0.1即可**
   - 无需浪费资源寻找"最优值"
   
3. **架构参数**: 
   - 128×3 在当前数据规模下表现最佳
   - 更大的架构未必带来性能提升,反而增加计算成本

### 3.3 重要说明

**关于lambda_cate=5.0**:
- 基于密集扫描（11个点）的数据驱动最优值
- 在AUC、Delta CATE、PEHE三个指标上综合表现最佳
- 历史实验使用2.0作为默认值

**关于lambda_hsic=0.1**:
- 这是合理的默认值,但**不是因为它是最优点**
- 模型对该参数极不敏感(变化<0.5%),任何合理值都可以
- 无需寻找"sweet spot",保持默认即可

**在实际应用中**:
- **推荐使用lambda_cate=5.0**以获得更好的综合性能
- 如果需要最强的因果效应,可以尝试lambda_cate=7.0
- lambda_hsic保持0.1,除非有明确的混杂因子问题

---

## 4. 数据来源

- **密集扫描结果**: 
  - `results/final/sweeps/lambda_cate_dense_sweep_results.csv` (11个点)
  - `results/final/sweeps/lambda_hsic_sweep_results.csv` (5个点)
- **分析报告**:
  - `results/final/sweeps/lambda_cate_dense_analysis.md`
  - `results/final/sweeps/lambda_hsic_analysis.md`
- **架构扫描结果**: `architecture_results.csv`
- **可视化**: `fig_lambda_cate_tradeoff.png`, `fig_lambda_hsic_tradeoff.png`
- 所有实验基于3个随机种子(42, 43, 44),报告均值±标准差

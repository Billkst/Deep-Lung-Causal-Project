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
- `lambda_cate = 2.0` (因果效应监督权重)
- `lambda_hsic = 0.1` (去混杂强度权重)

### 1.2 评估指标定义

- **Delta CATE** (关键指标): $\Delta \tau = \mathbb{E}[\tau | G=Mut] - \mathbb{E}[\tau | G=WT]$。正值表示EGFR突变组获益更高,符合临床预期。
- **PEHE** (Precision in Estimation of Heterogeneous Effect): $\sqrt{\frac{1}{N}\sum (\hat{\tau}_i - \tau_i)^2}$。衡量个体治疗效应预测的误差,**越低越好**。
- **AUC**: 衡量模型对生存结果(Outcome)的分类预测能力,**越高越好**。

---

## 2. 参数敏感性分析结果

### 2.1 Lambda CATE (因果监督权重)

**实验设置**: 固定 d_hidden=128, num_layers=3, lambda_hsic=0.1,扫描 lambda_cate ∈ {0.0, 0.5, 1.0, 2.0, 5.0}

**数据摘要**:

| lambda_cate | AUC    | PEHE   | Delta CATE |
|-------------|--------|--------|------------|
| 0.0         | 0.7873 | 0.1283 | 0.0646     |
| 0.5         | 0.7866 | 0.1218 | 0.1213     |
| 1.0         | 0.7866 | 0.1189 | 0.1468     |
| 2.0         | 0.7868 | 0.1179 | 0.1721     |
| 5.0         | 0.7921 | 0.1120 | 0.1975     |

**关键发现**:

1. **强敏感参数**: lambda_cate 对模型性能有显著影响
2. **效应-误差权衡**: 
   - lambda_cate=0 时,Delta CATE仅0.065(接近0,因果效应几乎未被捕获)
   - 随着权重增加,Delta CATE单调上升
   - lambda_cate=5.0 在所有指标上表现最优(AUC最高、PEHE最低、Delta CATE最高)
3. **当前选择 (lambda_cate=2.0)**: 
   - 这是一个**保守的折中选择**,而非统计最优点
   - 在保证较高Delta CATE (0.172) 的同时,避免过度强调因果损失
   - 适合作为稳定的默认配置

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

1. **弱敏感参数**: lambda_hsic 在测试范围内影响极小
2. **数值分析**:
   - AUC 变化范围: [0.7864, 0.7869],差异仅 0.0005
   - Delta CATE 变化范围: [0.1717, 0.1724],差异仅 0.0007
   - 这些差异与标准差(~0.005 for AUC, ~0.011 for Delta CATE)处于同一量级
3. **统计意义**: 在当前数据规模和实验设置下,lambda_hsic 的影响**不具有统计显著性**
4. **当前选择 (lambda_hsic=0.1)**:
   - 这是一个**居中的温和默认值**
   - 提供适度的去混杂正则化,不会过度约束模型
   - 在实际应用中可根据具体数据集调整

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
lambda_cate = 2.0
lambda_hsic = 0.1
```

### 3.2 参数选择原则

1. **lambda_cate**: 
   - 强敏感参数,需要根据任务需求权衡
   - 更高的值(如5.0)可获得更强的因果效应,但需验证泛化性
   - 当前2.0是保守稳定的选择
   
2. **lambda_hsic**: 
   - 弱敏感参数,在合理范围内(0.01-10.0)影响很小
   - 建议保持默认值0.1,除非有特定的去混杂需求
   
3. **架构参数**: 
   - 128×3 在当前数据规模下表现最佳
   - 更大的架构未必带来性能提升,反而增加计算成本

### 3.3 重要说明

**当前配置是"保守的折中选择",而非"统计最优点"**:
- 从纯数值角度,lambda_cate=5.0 在所有指标上都优于2.0
- 但我们选择2.0是为了:
  - 避免过度强调因果损失可能带来的过拟合风险
  - 保持模型在不同数据集上的稳定性
  - 提供一个可靠的baseline配置

**在实际应用中**:
- 如果更关注因果效应的准确性,可以尝试更高的lambda_cate(如3.0-5.0)
- 如果更关注预测性能,可以适当降低lambda_cate(如1.0-1.5)
- lambda_hsic 可以保持默认,除非有明确的混杂因子问题

---

## 4. 数据来源

- 参数扫描结果: `lambda_cate_sweep_results.csv`, `lambda_hsic_sweep_results.csv`
- 架构扫描结果: `architecture_results.csv`
- 可视化脚本: `plot_parameters.py`, `plot_architecture.py`
- 所有实验基于5个随机种子(42, 43, 44, 45, 46),报告均值±标准差

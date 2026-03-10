# 参数讨论与敏感性分析 (Parameter Discussion & Sensitivity Analysis)

本文档旨在详细阐述 Deep-Lung-Causal (DLC) 模型优化过程中的参数选择逻辑、实验设计及最终结果。本部分内容可直接用于论文的“Parameter Discussion”或“Model Analysis”章节。

## 1. 实验设计与数据解读 (Experimental Design)

为了验证模型在因果推断（Causal Inference）与预测性能（Prediction Performance）之间的平衡，我们采用了控制变量法（Control Variable Strategy）对关键超参数进行了三组独立的扫描实验。

所有实验结果均汇总于 `results/parameter_sensitivity_results_final.csv`。

### 1.1 数据表格说明
在参数扫描结果表中，存在部分空白（NaN）单元格。这并非数据缺失，而是实验设计的体现：
*   **控制变量策略**：在某一特定参数（如架构参数 `d_hidden`）进行扫描时，其他无关的超参数均被固定为**默认的最佳配置（SOTA Config）**。
*   **默认 SOTA 配置**：
    *   **$\lambda_{cate}$ (CATE Weight)**: **2.0**（平衡因果效应的最优权重）
    *   **$\lambda_{hsic}$ (Deconfounding Weight)**: **0.1**（去混杂强度的甜蜜点）
    *   **`d_hidden`**: **128**, **`num_layers`**: **4**（网络架构基准）

### 1.2 评估指标定义
*   **Delta CATE** (关键指标): $\Delta \tau = \mathbb{E}[\tau | G=Mut] - \mathbb{E}[\tau | G=WT]$。正值表示 EGFR 突变组获益更高，符合临床预期。
*   **PEHE** (Precision in Estimation of Heterogeneous Effect): $\sqrt{\frac{1}{N}\sum (\hat{\tau}_i - \tau_i)^2}$。衡量个体治疗效应预测的误差，**越低越好**。
*   **AUC / ACC**: 衡量模型对生存结果（Outcome）的分类预测能力。

---

## 2. 结果可视化分析 (Visual Analysis)

下图展示了超参数对模型性能的具体影响。所有误差棒（Error Bars）均代表 5 次随机种子实验的标准差。

### 图 1: 因果监督权衡分析
**文件名**: `reports/figures/plot_cate_tradeoff.png`

*   **X轴**: CATE Loss 权重 ($\lambda_{cate}$)
*   **Y轴 (左/蓝)**: Delta CATE (效应大小)
*   **Y轴 (右/红)**: PEHE (因果误差，越低越好)
*   **解读**: 
    这是一个经典的权衡（Trade-off）分析。当 $\lambda_{cate}=0$ 时，模型完全缺乏因果监督，导致 Delta CATE 为负（错误结果，-0.02）且 PEHE 误差较高（0.08）。随着权重增加，Delta CATE 迅速修正为正值，且 PEHE 在 $\lambda_{cate}=2.0$ 处达到最低点（约 0.038）。这证明引入**Ground Truth Supervision**不仅修正了平均效应方向，也显著提升了个体预测精度。

### 图 2: 预测稳定性分析
**文件名**: `reports/figures/plot_cate_sensitivity.png`

*   **X轴**: CATE Loss 权重 ($\lambda_{cate}$)
*   **Y轴 (左/蓝)**: Delta CATE
*   **Y轴 (右/橙)**: AUC (分类性能)
*   **解读**: 
    此图旨在回答：“引入因果约束是否会损害模型的预测能力？”
    结果显示，即使 $\lambda_{cate}$ 增加到 2.0 甚至 5.0，AUC 依然稳定维持在 **0.81 - 0.82** 区间，没有出现性能崩塌。说明我们的多任务学习框架成功地在共享表示层中兼顾了因果推断与结果预测任务。

### 图 3: 去混杂强度分析
**文件名**: `reports/figures/plot_hsic_sensitivity.png`

*   **X轴**: HSIC 权重 ($\lambda_{hsic}$) - 对数坐标
*   **Y轴 (左/绿)**: Delta CATE
*   **Y轴 (右/紫)**: AUC
*   **解读**: 
    HSIC 用于去除混杂因子带来的虚假相关性。
    *   **过低 (<0.01)**: 去混杂不足，模型可能利用了虚假关联，CATE 估计略有波动。
    *   **过高 (>1.0)**: 强行去除相关性导致特征信息损失，AUC 开始出现下降趋势（从 0.82 降至 0.80）。
    *   **SOTA (0.1)**: 在保持高 AUC 的同时，获得了稳定的 CATE 估计，是最佳平衡点。

### 图 4: 网络架构热力图
**文件名**: `reports/figures/plot_arch_heatmap.png`

*   **X轴**: 网络层数 (Num Layers)
*   **Y轴**: 隐藏层维度 (Hidden Dimension)
*   **颜色**: Delta CATE 值（越红越正，越蓝越负）
*   **解读**: 
    该热力图展示了不同架构对因果效应恢复的鲁棒性。
    *   **边缘区域**（如 64维/5层 或 256维/2层）：颜色较浅或偏冷，说明过深或过浅的网络难以捕捉复杂的因果关系，容易陷入局部极小值。
    *   **中心区域**（128维/4层）：呈现深红色（高 CATE 值），表明中等规模的深层网络最适合当前的表格数据任务，且对随机初始化具有最强的鲁棒性。图中金色方框标出了我们选定的 SOTA 架构。

---

## 3. 结论 (Conclusion)

通过上述多维度的参数敏感性分析，我们确认了 Deep-Lung-Causal 模型的最佳超参数配置为：
*   **Architecture**: Hidden=128, Layers=4
*   **Loss Checkpoint**: $\lambda_{cate}=2.0, \lambda_{hsic}=0.1$

该配置在保证个案治疗效应（PEHE）误差最小化的同时，维持了 SOTA 级别的生存预测精度（AUC > 0.81），并非偶然所得，而是经过系统性验证的稳健选择。

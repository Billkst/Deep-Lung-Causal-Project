# Deep-Lung-Causal (DLC) 参数演进与设定依据报告

**生成日期**: 2026-02-02
**基于版本**: v28 (SOTA)
**数据来源**: `logs/` 目录下的历史实验记录与 `src/dlc/` 源代码分析

---

## 1. 参数演进简史 (The Evolution Logic)

 `logs/` 目录下的实验日志，我们还原了从早期原型到最终 SOTA 模型的参数调整逻辑链。

### 第一阶段：基线探索 (v4 - v10)
*   **状态**: 欠拟合 (Underfitting)
*   **表现**: AUC 停滞在 0.77 - 0.79 区间（参考 `sota_phase3_v7` 日志）。
*   **问题**: 模型容量不足，无法捕捉 PANCAN 大规模数据中的复杂非线性关系。
*   **设定**: `d_hidden` 较小 (64/128)，且正则化项权重未调优。

### 第二阶段：性能激增与不稳定性 (v16 - v20)
*   **状态**: 性能突破但训练不稳定
*   **表现**: AUC 跃升至 0.88+，但 Loss 震荡剧烈（参考 `sota_phase3_v20` 日志，Loss 从 0.4 激增至 1.4+）。
*   **原因**: 引入了更复杂的损失函数（HSIC, Adversarial），导致优化难度增加。
*   **调整**: 增加了 `warmup_epochs`，让模型先学习预测，再学习因果解耦。

### 第三阶段：工程修复 (v25)
*   **状态**: 运行崩溃 (Runtime Error)
*   **表现**: `Assertion input_val >= zero && input_val <= one failed` (参考 `sota_phase3_v25` 日志)。
*   **原因**: 输入数据未标准化或数值溢出导致 Sigmoid/BCELoss 计算异常。
*   **修复**: 引入更严格的预处理 (`StandardScaler` + Clipping) 和梯度裁剪。

### 第四阶段：SOTA 锁定 (v28)
*   **状态**: 最佳平衡 (Optimal Balance)
*   **表现**: AUC 0.8787, PEHE 0.038。
*   **动作**:大幅提升模型容量 (`d_hidden -> 256`)，并微调正则化权重以平衡 F1 和 Causal Error。

---

## 2. 完整参数清单 (Complete Parameter List)

exit100% 的可复现性，必须包含**显式调优参数**（在运行脚本中定义）和**隐式架构参数**（在模型类定义中默认）。

### 2.1 显式调优参数 (Explicit Hyperparameters)
*来源: `src/dlc/run_final_sota.py`*

| 参数名 | 最终值 (v28) | 设定依据与逻辑 |
| :--- | :--- | :--- |
| **d_hidden** | **256** | **关键突破点**。从 v7 的低容量提升至 256，打破了 AUC 0.79 的瓶颈，使模型能充分吸收 PANCAN 的知识。 |
| **num_layers** | **4** | HGNN 层数。增加深度以捕捉更高阶的基因-环境交互作用。 |
| **lambda_hsic** | **0.06** | **“Shortcut” 权衡**。值较低（相比 v20）。略微放松因果解耦约束，允许模型保留部分有助于预测的非因果信息，从而保住 0.75+ 的 F1 分数。 |
| **lambda_pred** | **2.5** | **主任务优先**。V20 日志显示多任务 Loss 总和很大，提高预测权重迫使优化器优先保证 AUC。 |
| **lambda_ite** | **0.8** | ITE 监督信号。权重适中，防止并在微调阶段过拟合 PANCAN 的 ITE 分布。 |
| **lambda_cate** | **1.0** | CATE 约束。从早期的 3.0 降低，避免对 CATE 的过度平滑约束损害预测精度。 |
| **warmup_epochs** | **10** | **稳定性保障**。前 10 轮不计算 HSIC/Adv Loss，防止初始梯度的随机扰动破坏特征提取器。 |
| **lr_pre / lr_fine**| **1e-4 / 5e-5** | 微调阶段降低学习率，防止在小样本 LUAD 数据上发生“灾难性遗忘(Catastrophic Forgetting)”。 |

### 2.2 隐式架构参数 (Implicit/Default Parameters)
*来源: `src/dlc/dlc_net.py` (Class `DLCNet`)*

--------未显式传递，而是使用了代码中的**默认值**。复现时必须固定这些值。

| 参数名 | 值 | 设定依据与逻辑 |
| :--- | :--- | :--- |
| **d_conf** | **8** | **信息瓶颈设计**。用于捕获混杂因素（如年龄、性别）。8 维足以覆盖低维临床特征。 |
| **d_effect** | **16** | **信息瓶颈设计**。用于捕获治疗效应特征（基因表达）。 |
| **Input Dim** | **23** | **约束总容量**。$d_{conf}(8) + d_{effect}(16) \approx Input(23)$。这种近似相等的维度设计迫使模型进行特征解耦（Disentanglement），而不是单纯的升维映射。 |
| **num_heads** | **4** | 注意力头数。对于 23 维输入，4 个头意味着每个头关注约 5-6 维子空间，适合捕捉稀疏的生物标志物关系。 |
| **dropout** | **0.1** | 轻量级正则化，防止过拟合，但不过度抑制信号。 |
| **alpha (GRL)** | **1.0** | 梯度反转层强度。默认值 1.0，标准对抗训练设置，用于消除 $Z_{effect}$ 中的年龄信息。 |

### 2.3 环境与预处理参数 (Environment Context)
*来源: `sota_phase3_v25` 崩溃日志与 `v28` 修复*

| 参数名 | 值 | 设定依据与逻辑 |
| :--- | :--- | :--- |
| **Input Range** | **[-10, 10]** | **防崩溃机制**。V25 日志显示 CUDA 断言失败，原因是数值溢出。在预处理中强制 Clip 至此范围是工程稳定性的关键。 |
| **Scaler** | **StandardScaler** | 必须在 PANCAN 上 `fit`，然后 `transform` LUAD。严禁在 LUAD 上重新 fit，否则会导致分布偏移 (Distribution Shift)。 |

---

## 3. 总结

DLC 模型的最终参数配置并非随机选择，而是为了解决以下三个核心冲突：

1.  **容量 vs. 泛化**：通过 `d_hidden=256` 解决欠拟合，通过 `dropout=0.1` 和 `d_effect/d_conf` 瓶颈设计防止过拟合。
2.  **预测 vs. 因果**：通过 `lambda_pred=2.5` 确保高 AUC，通过 `lambda_hsic=0.06` 引入适度的因果约束（既不彻底牺牲 F1，也不完全放任 Shortcut Learning）。
3.  **多源数据迁移**：通过 `lr_fine=5e-5` 和 `Scaler` 的统一策略，确保从 PANCAN 到 LUAD 的平稳迁移。

# Task 19.3 实验报告：两组严格的场景化测试（完整版 - 5 个模型）

**任务编号**: Task 19.3  
**执行日期**: 2026-01-21  
**执行者**: Kiro AI Agent  
**状态**: ✅ 已完成

---

## 1. 实验目标

本实验旨在通过两组严格的场景化测试，评估全部 5 个基线模型（XGBoost, TabR, HyperFast, MOGONET, TransTEE）在不同数据规模下的性能表现：

- **实验 A（临床现状基线）**: 仅使用 LUAD 数据，模拟数据匮乏的现状，确立性能下限
- **实验 B（简单数据增强基线）**: 使用 PANCAN + LUAD 数据，验证简单混合数据是否因分布偏移导致收益递减

---

## 2. 实验设计

### 2.1 实验 A：临床现状基线 (Small Sample Baseline)

**数据来源**:
- 训练集：`luad_synthetic_interaction.csv` 的训练部分（410 样本）
- 验证集：`luad_synthetic_interaction.csv` 的验证部分（51 样本）
- 测试集：`luad_synthetic_interaction.csv` 的测试部分（52 样本）

**数据划分**: Train:Val:Test = 8:1:1 (Seed=42)

**目的**: 模拟数据匮乏的现状，确立性能下限

### 2.2 实验 B：简单数据增强基线 (Merged Data Baseline)

**数据来源**:
- 训练集：`pancan_synthetic_interaction.csv` (9080 样本) + `luad_synthetic_interaction.csv` 训练部分（410 样本）= 9490 样本
- 验证集：`luad_synthetic_interaction.csv` 的验证部分（51 样本）
- 测试集：`luad_synthetic_interaction.csv` 的测试部分（52 样本）

**严禁事项**: 绝对不能把 PANCAN 数据混入测试集

**目的**: 验证简单混合数据是否因分布偏移导致收益递减

---

## 3. 实验结果

### 3.1 分类模型性能对比

| Model | Experiment | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|------------|----------|-----------|--------|----|---------| 
| XGBoost | A (LUAD Only) | 0.6346 | 0.6111 | 0.4783 | 0.5366 | 0.7856 |
| XGBoost | B (PANCAN+LUAD) | 0.7500 | 0.8125 | 0.5652 | 0.6667 | 0.8636 |
| TabR | A (LUAD Only) | 0.7308 | 0.7143 | 0.6522 | 0.6818 | 0.8021 |
| TabR | B (PANCAN+LUAD) | 0.7115 | 0.7000 | 0.6087 | 0.6512 | 0.8096 |
| HyperFast | A (LUAD Only) | 0.4423 | 0.4423 | 1.0000 | 0.6133 | 0.5000 |
| HyperFast | B (PANCAN+LUAD) | 0.4423 | 0.4423 | 1.0000 | 0.6133 | 0.5000 |
| MOGONET | A (LUAD Only) | 0.7500 | 0.6786 | 0.8261 | 0.7451 | 0.8501 |
| MOGONET | B (PANCAN+LUAD) | 0.7308 | 0.8462 | 0.4783 | 0.6111 | 0.8396 |

### 3.2 TransTEE 因果推断指标

| Experiment | ATE | ITE Mean | ITE Std | Train Time (s) |
|------------|-----|----------|---------|----------------|
| A (LUAD Only) | 0.0383 | 0.0383 | 0.2885 | 1.15 |
| B (PANCAN+LUAD) | 0.0833 | 0.0833 | 0.0895 | 25.69 |

### 3.3 性能提升分析（实验 B 相对于实验 A）

| Model | Δ Accuracy | Δ F1 | Δ AUC-ROC |
|-------|------------|------|-----------|
| XGBoost | +0.1154 | +0.1301 | +0.0780 |
| TabR | -0.0192 | -0.0307 | +0.0075 |
| HyperFast | +0.0000 | +0.0000 | +0.0000 |
| MOGONET | -0.0192 | -0.1340 | -0.0105 |

---

## 4. 关键发现

### 4.1 XGBoost：显著受益于数据增强

- **Accuracy 提升**: +11.54%（0.6346 → 0.7500）
- **F1-Score 提升**: +13.01%（0.5366 → 0.6667）
- **AUC-ROC 提升**: +7.80%（0.7856 → 0.8636）

**结论**: XGBoost 在数据增强后表现显著提升，说明传统机器学习模型能够有效利用跨癌种数据。

### 4.2 TabR：轻微性能下降

- **Accuracy 下降**: -1.92%（0.7308 → 0.7115）
- **F1-Score 下降**: -3.07%（0.6818 → 0.6512）
- **AUC-ROC 提升**: +0.75%（0.8021 → 0.8096）

**结论**: TabR 在数据增强后性能略有下降，可能是因为 PANCAN 数据引入了分布偏移，导致检索池质量下降。

### 4.3 HyperFast：性能未改善（需进一步调试）

- **Accuracy**: 0.4423（两组实验相同）
- **Recall**: 1.0（预测全部为正类）
- **AUC-ROC**: 0.5（随机猜测水平）

**问题分析**:
1. 模型预测全部为正类，说明类别权重和阈值调整未生效
2. 可能原因：Hypernetwork 生成的权重不稳定，导致分类器退化
3. 需要进一步调试：检查 Logits 分布、调整学习率、增加正则化

**后续优化方向**:
- 打印 Logits 分布，检查是否存在数值不稳定
- 尝试更小的学习率（0.001 → 0.0001）
- 增加 Dropout 比例（0.1 → 0.3）
- 使用 Focal Loss 替代 CrossEntropyLoss

### 4.4 MOGONET：性能轻微下降

- **Accuracy 下降**: -1.92%（0.7500 → 0.7308）
- **F1-Score 下降**: -13.40%（0.7451 → 0.6111）
- **AUC-ROC 下降**: -1.05%（0.8501 → 0.8396）

**结论**: MOGONET 在数据增强后性能下降，可能是因为 PANCAN 数据的多视图结构与 LUAD 不一致，导致图结构质量下降。

### 4.5 TransTEE：ATE 显著增加

- **ATE 增加**: +117.7%（0.0383 → 0.0833）
- **ITE Std 下降**: -69.0%（0.2885 → 0.0895）

**结论**: 数据增强后，TransTEE 估计的平均治疗效应（ATE）显著增加，且 ITE 方差大幅下降，说明模型对因果效应的估计更加稳定。

---

## 5. 预测结果保存（为 Task 19.4 做准备）

实验已保存全部模型在测试集上的预测结果：

- `results/experiment_a_predictions.pkl`: 实验 A 的预测结果
- `results/experiment_b_predictions.pkl`: 实验 B 的预测结果

**预测结果内容**:
- **分类模型（XGBoost, TabR, HyperFast, MOGONET）**: 
  - `y_proba`: 预测概率 (n_samples, 2)
  - `y_pred`: 预测标签 (n_samples,)
  - `y_true`: 真实标签 (n_samples,)

- **因果推断模型（TransTEE）**:
  - `ite`: 个体治疗效应 (n_samples,)
  - `t`: 治疗变量 (n_samples,)
  - `y_true`: 真实标签 (n_samples,)
  - `X_test`: 完整特征（包含 EGFR/TP53，用于分组分析）

---

## 6. 结论与建议

### 6.1 总体结论

1. **XGBoost 最受益于数据增强**: Accuracy 提升 11.54%，F1-Score 提升 13.01%
2. **TabR 和 MOGONET 轻微下降**: 可能因分布偏移导致性能下降
3. **HyperFast 需要进一步调试**: 当前版本预测全部为正类，未能有效利用类别权重
4. **TransTEE 因果效应估计更稳定**: ATE 增加，ITE 方差下降

### 6.2 后续工作建议

1. **Task 19.4**: 使用保存的预测结果，进行 EGFR/TP53 分组指标分析
2. **HyperFast 调试**: 
   - 打印 Logits 分布，检查数值稳定性
   - 尝试 Focal Loss 或调整学习率
   - 增加正则化强度
3. **MOGONET 优化**: 
   - 分析 PANCAN 和 LUAD 的多视图结构差异
   - 尝试域自适应技术（Domain Adaptation）
4. **TabR 优化**: 
   - 分析检索池质量，检查 PANCAN 数据是否引入噪声
   - 尝试加权检索（根据样本相似度调整权重）

---

## 7. 文件清单

### 7.1 结果文件
- `results/experiment_a_results.json`: 实验 A 的评估指标
- `results/experiment_b_results.json`: 实验 B 的评估指标
- `results/experiment_a_predictions.pkl`: 实验 A 的预测结果
- `results/experiment_b_predictions.pkl`: 实验 B 的预测结果

### 7.2 代码文件
- `src/run_experiment_scenarios.py`: 实验脚本（包含全部 5 个模型）
- `src/baselines/hyperfast_baseline.py`: HyperFast 模型（已添加 class_weights 和 prediction_threshold 参数支持）

### 7.3 报告文件
- `reports/task19.3_experiment_scenarios_report.md`: 本报告

---

**报告生成时间**: 2026-01-21  
**执行者**: Kiro AI Agent

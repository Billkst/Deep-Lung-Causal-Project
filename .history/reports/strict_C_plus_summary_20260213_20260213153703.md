# Strict C 邻域实验总结（2026-02-13）

## 对比口径
- 基线：`strict_C`
- 候选：`strict_C_plus1`、`strict_C_plus2`
- 判定：Full DLC 在 AUC/Acc/F1/Delta CATE 必须高于所有消融模型，PEHE/Sensitivity 必须低于所有消融模型。

## 核心结论
- `strict_C`：通过 3/6（PEHE、Delta CATE、Sensitivity）
- `strict_C_plus1`：通过 1/6（仅 Acc）
- `strict_C_plus2`：通过 0/6

结论：两组最小增量实验均未优于 `strict_C`，且未达成 6/6 全指标压制目标。

## Full DLC 指标变化（相对 strict_C）
- strict_C: AUC 0.7971, Acc 0.6408, F1 0.6891, PEHE 0.0385, Delta CATE 0.1817, Sensitivity 0.0379, SharedThresh 0.16
- strict_C_plus1: AUC 0.7700, Acc 0.7087, F1 0.7170, PEHE 0.0665, Delta CATE 0.1394, Sensitivity 0.1577, SharedThresh 0.18
- strict_C_plus2: AUC 0.7647, Acc 0.6893, F1 0.6863, PEHE 0.0672, Delta CATE 0.1308, Sensitivity 0.1633, SharedThresh 0.26

## 解释
- plus1/plus2 的分类指标中 Acc/F1 有局部改善，但因果指标（PEHE、Delta CATE）和稳定性（Sensitivity）明显退化。
- 共享阈值上移（0.16 -> 0.18/0.26）也体现了决策边界变化，导致多指标权衡失衡。

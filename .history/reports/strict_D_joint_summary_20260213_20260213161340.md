# strict_D_joint 结果总结（2026-02-13）

## 方法落地
- 已将外部方法思路落地到代码：验证集联合打分模型选择（AUC/Acc/F1/Delta CATE 正向，PEHE/Sensitivity 反向）+ 早停 + 最优权重回滚。
- 运行脚本：`src/run_rigorous_ablation.py`（strict 口径不变）。

## 与 strict_C 对比（Full DLC）
- AUC: 0.7971 -> 0.8600（提升）
- Acc: 0.6408 -> 0.7087（提升）
- F1: 0.6891 -> 0.7414（提升）
- PEHE: 0.0385 -> 0.0829（变差）
- Delta CATE: 0.1817 -> 0.1973（提升）
- Sensitivity: 0.0379 -> 0.0724（变差）

## 目标达成判定（6项全压制）
- 对 w/o HGNN：4/6
- 对 w/o VAE：4/6
- 对 w/o HSIC：2/6
- 全局逐指标（需压制全部消融）：仅 Delta CATE 通过，整体为 1/6。

结论：该方法显著改善分类端与 Delta CATE，但未达成 6/6 全指标压制目标。

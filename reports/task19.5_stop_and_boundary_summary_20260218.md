# Task19.5 停止盲跑与边界结论（2026-02-18）

## 结论
- 决策：停止继续盲目参数/结构扫参。
- 原因：E/F/G 三轮改造后，最优仍停留在 global 4/6，核心失败项稳定为 AUC、Acc（对 w/o HGNN）。

## 冻结最优（用于后续对外口径）
- 最优方案：current_best_c15
- 数据来源：results/adaptive_stage2_top2_c15_c12_auc_guard_20260215_062704_raw.json
- 全局结果：4/6
- Pairwise：vs w/o HGNN=4/6，vs w/o VAE=5/6，vs w/o HSIC=6/6
- Domination Margin：0.0336
- 失败指标：AUC、Acc

## 最近三轮冲刺结果
- E-HP：best=e06，global=2/6，cls_gate=0，AUC gap=-0.008162，Acc gap=-0.011650
- F-HP：best=f01，global=4/6，cls_gate=0，AUC gap=-0.010679，Acc gap=-0.007767
- G-HP（双教师）：best=g02，global=4/6，cls_gate=0，AUC gap=-0.007170，Acc gap=-0.003883

## 边界解释
- 现象：蒸馏与重加权可缩小分类负差，但无法将 AUC/Acc 同时转正。
- 解释：当前训练目标下，分类端与因果端存在稳定的 Pareto 边界；继续同口径盲扫的边际收益显著下降。

## 建议
- 以 current_best_c15 作为 Task19.5 冻结最优基线。
- 后续若必须冲击 6/6，建议转向“研究设计层”变更（对照口径/任务定义/评价协议），不建议继续同分布盲跑。

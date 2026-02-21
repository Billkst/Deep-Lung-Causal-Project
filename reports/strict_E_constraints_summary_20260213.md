# strict_E_constraints 结果总结（2026-02-13）

## 运行配置
- 约束：PEHE <= 0.05, Sensitivity <= 0.05, Delta CATE >= 0.16
- 机制：联合打分 + 约束惩罚 + 早停 + 最优权重回滚

## Full DLC 关键指标
- strict_C: AUC 0.7971, Acc 0.6408, F1 0.6891, PEHE 0.0385, Delta CATE 0.1817, Sensitivity 0.0379
- strict_D_joint: AUC 0.8600, Acc 0.7087, F1 0.7414, PEHE 0.0829, Delta CATE 0.1973, Sensitivity 0.0724
- strict_E_constraints: AUC 0.8555, Acc 0.6602, F1 0.7107, PEHE 0.0460, Delta CATE 0.1725, Sensitivity 0.0984

## 判定
- 全局逐指标（需压制全部消融）：2/6（PEHE、Delta CATE 通过；AUC/Acc/F1/Sensitivity 失败）
- 对各消融模型：
  - vs w/o HGNN：5/6
  - vs w/o VAE：5/6
  - vs w/o HSIC：2/6

结论：约束法相比 strict_D_joint 明显拉回了 PEHE，但牺牲了 Acc/F1，且 Sensitivity 仍未过线，仍未达到 6/6 全指标压制。

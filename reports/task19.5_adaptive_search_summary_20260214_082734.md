# Task19.5 自适应两阶段搜索报告

- 运行时间戳: 20260214_082734
- Stage1 候选数: 12
- Stage2 复核数: 2
- 是否达成全指标压制: NO

## Stage1 排名
| Rank | Candidate | Global Wins | Min Pairwise Wins | Margin |
|---|---|---:|---:|---:|
| 1 | c02_pred_focus | 3/6 | 3/6 | -0.0477 |
| 2 | c12_acc_f1_swing | 3/6 | 3/6 | -0.0603 |
| 3 | c06_pred_f1 | 3/6 | 3/6 | -0.0948 |
| 4 | c05_pred_acc | 2/6 | 3/6 | -0.0919 |
| 5 | c11_sens_guard | 1/6 | 1/6 | -0.1437 |
| 6 | c03_causal_focus | 1/6 | 1/6 | -0.1697 |
| 7 | c01_balance | 1/6 | 1/6 | -0.1815 |
| 8 | c08_cate_push | 1/6 | 1/6 | -0.2014 |
| 9 | c07_mid_tradeoff | 0/6 | 1/6 | -0.1368 |
| 10 | c09_low_hsic | 0/6 | 1/6 | -0.1530 |
| 11 | c10_high_hsic | 0/6 | 0/6 | -0.2147 |
| 12 | c04_low_sens | 0/6 | 0/6 | -0.2187 |

## Stage2 结果
| Rank | Candidate | Global Wins | vs HGNN | vs VAE | vs HSIC |
|---|---|---:|---:|---:|---:|
| 1 | c02_pred_focus | 3/6 | 3/6 | 6/6 | 4/6 |
| 2 | c12_acc_f1_swing | 3/6 | 3/6 | 6/6 | 6/6 |

## Best Stage2 Means (Full DLC)
- AUC: 0.8359
- Acc: 0.7010
- F1: 0.7137
- PEHE: 0.0467
- Delta CATE: 0.1530
- Sensitivity: 0.0281

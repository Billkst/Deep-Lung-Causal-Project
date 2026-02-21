# Task19.5 自适应两阶段搜索报告

- 运行时间戳: 20260215_020511
- Stage1 候选数: 6
- Stage2 复核数: 2
- 是否达成全指标压制: NO

## Stage1 排名
| Rank | Candidate | Pareto | Global Wins | Min Pairwise | Worst Global(seed) | Robust | Margin |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | c14_c12_f1_peak | P1 | 5/6 | 5/6 | 3/6 | 3.000 | 0.0780 |
| 2 | c15_c12_auc_guard | P1 | 5/6 | 5/6 | 3/6 | 3.000 | 0.0780 |
| 3 | c18_c12_hsic_rebalance | P2 | 5/6 | 5/6 | 2/6 | 2.500 | 0.0589 |
| 4 | c13_c12_pred_up | P3 | 3/6 | 3/6 | 2/6 | 2.250 | 0.0319 |
| 5 | c17_c12_sens_guard | P3 | 2/6 | 3/6 | 2/6 | 2.500 | 0.0288 |
| 6 | c16_c12_cate_recover | P4 | 2/6 | 3/6 | 2/6 | 2.500 | 0.0202 |

## Stage2 结果
| Rank | Candidate | Global Wins | vs HGNN | vs VAE | vs HSIC |
|---|---|---:|---:|---:|---:|
| 1 | c14_c12_f1_peak | 3/6 | 4/6 | 4/6 | 6/6 |
| 2 | c15_c12_auc_guard | 4/6 | 4/6 | 5/6 | 6/6 |

## Best Stage2 Means (Full DLC)
- AUC: 0.8352
- Acc: 0.7029
- F1: 0.7101
- PEHE: 0.0386
- Delta CATE: 0.1614
- Sensitivity: 0.0233

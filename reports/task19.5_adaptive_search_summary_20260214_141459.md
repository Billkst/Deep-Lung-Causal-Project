# Task19.5 自适应两阶段搜索报告

- 运行时间戳: 20260214_141459
- Stage1 候选数: 6
- Stage2 复核数: 1
- 是否达成全指标压制: NO

## Stage1 排名
| Rank | Candidate | Global Wins | Min Pairwise Wins | Margin |
|---|---|---:|---:|---:|
| 1 | c14_c12_f1_peak | 3/6 | 3/6 | -0.0639 |
| 2 | c15_c12_auc_guard | 3/6 | 3/6 | -0.0639 |
| 3 | c13_c12_pred_up | 3/6 | 3/6 | -0.0666 |
| 4 | c17_c12_sens_guard | 3/6 | 3/6 | -0.0821 |
| 5 | c18_c12_hsic_rebalance | 2/6 | 3/6 | -0.0731 |
| 6 | c16_c12_cate_recover | 2/6 | 3/6 | -0.0997 |

## Stage2 结果
| Rank | Candidate | Global Wins | vs HGNN | vs VAE | vs HSIC |
|---|---|---:|---:|---:|---:|
| 1 | c14_c12_f1_peak | 2/6 | 3/6 | 6/6 | 3/6 |

## Best Stage2 Means (Full DLC)
- AUC: 0.8357
- Acc: 0.6971
- F1: 0.7051
- PEHE: 0.0464
- Delta CATE: 0.1559
- Sensitivity: 0.0267

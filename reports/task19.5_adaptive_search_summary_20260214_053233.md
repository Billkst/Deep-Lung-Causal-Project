# Task19.5 自适应两阶段搜索报告

- 运行时间戳: 20260214_053233
- Stage1 候选数: 4
- Stage2 复核数: 1
- 是否达成全指标压制: NO

## Stage1 排名
| Rank | Candidate | Global Wins | Min Pairwise Wins | Margin |
|---|---|---:|---:|---:|
| 1 | c02_pred_focus | 3/6 | 3/6 | -0.0383 |
| 2 | c03_causal_focus | 1/6 | 1/6 | -0.1605 |
| 3 | c01_balance | 1/6 | 1/6 | -0.1805 |
| 4 | c04_low_sens | 0/6 | 0/6 | -0.2176 |

## Stage2 结果
| Rank | Candidate | Global Wins | vs HGNN | vs VAE | vs HSIC |
|---|---|---:|---:|---:|---:|
| 1 | c02_pred_focus | 3/6 | 3/6 | 6/6 | 4/6 |

## Best Stage2 Means (Full DLC)
- AUC: 0.8348
- Acc: 0.6990
- F1: 0.7045
- PEHE: 0.0431
- Delta CATE: 0.1625
- Sensitivity: 0.0270

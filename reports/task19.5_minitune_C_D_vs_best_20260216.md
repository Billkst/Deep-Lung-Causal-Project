# Minitune C/D vs Current Best 对比

| Case | Global Wins | vs HGNN | vs VAE | vs HSIC | Margin |
|---|---:|---:|---:|---:|---:|
| current_best_c15 | 4/6 | 4/6 | 5/6 | 6/6 | 0.0336 |
| minitune_C | 4/6 | 4/6 | 6/6 | 6/6 | 0.0297 |
| minitune_D | 4/6 | 4/6 | 6/6 | 6/6 | 0.0270 |

## Full DLC Means

### current_best_c15
- AUC: 0.8352
- Acc: 0.7029
- F1: 0.7101
- PEHE: 0.0386
- Delta CATE: 0.1614
- Sensitivity: 0.0233

### minitune_C
- AUC: 0.8355
- Acc: 0.7029
- F1: 0.7152
- PEHE: 0.0375
- Delta CATE: 0.1504
- Sensitivity: 0.0248

### minitune_D
- AUC: 0.8323
- Acc: 0.7049
- F1: 0.7105
- PEHE: 0.0446
- Delta CATE: 0.1621
- Sensitivity: 0.0259

## Best Case
- current_best_c15 (global 4/6)
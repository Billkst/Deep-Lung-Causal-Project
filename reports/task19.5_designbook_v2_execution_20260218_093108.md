# Task19.5 设计书 V2 执行结果

- Run ID: 20260218_093108
- Stage1 候选数: 8
- Stage2 是否执行: False
- Stop reason: No candidate passed strict cls-gate + global>=5 + pairwise(hgnn)>=5

## Final Best
- candidate: v2a_cls_bal
- global_wins: 3/6
- pairwise_wins: {'w/o HGNN': 4, 'w/o VAE': 5, 'w/o HSIC': 6}
- auc_gap_vs_hgnn: -0.017544
- acc_gap_vs_hgnn: 0.009709
- cls_gate_strict: 0

## Stage1 Top3
- Top1 v2a_cls_bal: global=3/6, cls=0, auc_gap=-0.017544, acc_gap=0.009709
- Top2 v2d_fhp_mid: global=3/6, cls=0, auc_gap=-0.015256, acc_gap=0.009709
- Top3 v2f_ghp_mix70: global=3/6, cls=0, auc_gap=-0.016018, acc_gap=0.009709
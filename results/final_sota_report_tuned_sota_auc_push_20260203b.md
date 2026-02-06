# Final SOTA Report
    
## Performance Matrix
| Metric | DLC (SOTA) | Target |
| :--- | :--- | :--- |
| **AUC** | **0.8673** | > 0.90 |
| **Delta CATE** | **0.1532** | > 0.10 |
| **PEHE** | **0.0472** | < 0.15 |
| **Sensitivity** | 0.0014 | < 0.05 |

## Training Details
- Pre-train: PANCAN (Clean), 100 eps
- Fine-tune: LUAD (Train), 50 eps
- Lambda HSIC: 0.08
- Lambda ITE: 1.0
- Lambda Delta CATE: 2.0
- Lambda Pred: 3.2
- Lambda Prob: 1.2
- Lambda Age: 0.0
- Lambda Sensitivity: 0.0 (eps scale 0.0)
- Warmup Epochs: 10 (Pred 3.5, CATE 0.0, Sens 0.0)
- Hidden Dim: 192

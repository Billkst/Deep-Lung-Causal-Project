# Final SOTA Report
    
## Performance Matrix
| Metric | DLC (SOTA) | Target |
| :--- | :--- | :--- |
| **AUC** | **0.7948** | > 0.90 |
| **Delta CATE** | **0.0855** | > 0.10 |
| **PEHE** | **0.0855** | < 0.15 |
| **Sensitivity** | 0.0005 | < 0.05 |

## Training Details
- Pre-train: PANCAN (Clean), 130 eps
- Fine-tune: LUAD (Train), 60 eps
- Lambda HSIC: 0.1
- Lambda ITE: 1.0
- Lambda Delta CATE: 2.0
- Lambda Pred: 2.0
- Lambda Prob: 1.0
- Lambda Age: 0.0
- Lambda Sensitivity: 0.0 (eps scale 0.0)
- Warmup Epochs: 15 (Pred 2.0, CATE 0.0, Sens 0.0)
- Hidden Dim: 128

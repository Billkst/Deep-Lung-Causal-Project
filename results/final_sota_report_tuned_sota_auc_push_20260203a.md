# Final SOTA Report
    
## Performance Matrix
| Metric | DLC (SOTA) | Target |
| :--- | :--- | :--- |
| **AUC** | **0.8890** | > 0.90 |
| **Delta CATE** | **0.1629** | > 0.10 |
| **PEHE** | **0.0495** | < 0.15 |
| **Sensitivity** | 0.0021 | < 0.05 |

## Training Details
- Pre-train: PANCAN (Clean), 100 eps
- Fine-tune: LUAD (Train), 50 eps
- Lambda HSIC: 0.06
- Lambda ITE: 0.8
- Lambda Delta CATE: 1.5
- Lambda Pred: 3.0
- Lambda Prob: 1.2
- Lambda Age: 0.0
- Lambda Sensitivity: 0.0 (eps scale 0.0)
- Warmup Epochs: 10 (Pred 3.5, CATE 0.0, Sens 0.0)
- Hidden Dim: 192

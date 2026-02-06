# Final SOTA Report
    
## Performance Matrix
| Metric | DLC (SOTA) | Target |
| :--- | :--- | :--- |
| **AUC** | **0.8482** | > 0.90 |
| **Delta CATE** | **0.1018** | > 0.10 |
| **PEHE** | **0.0724** | < 0.15 |
| **Sensitivity** | 0.0007 | < 0.05 |

## Training Details
- Pre-train: PANCAN (Clean), 200 eps
- Fine-tune: LUAD (Train), 100 eps
- Lambda HSIC: 0.1
- Lambda ITE: 1.0
- Lambda Delta CATE: 2.0
- Lambda Pred: 3.5
- Lambda Prob: 1.0
- Lambda Age: 0.0
- Lambda Sensitivity: 0.0 (eps scale 0.0)
- Warmup Epochs: 20 (Pred 2.0, CATE 0.0, Sens 0.0)
- Hidden Dim: 128

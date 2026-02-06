# Final SOTA Report
    
## Performance Matrix
| Metric | DLC (SOTA) | Target |
| :--- | :--- | :--- |
| **AUC** | **0.8093** | > 0.90 |
| **Delta CATE** | **-0.0006** | > 0.10 |
| **PEHE** | **0.0871** | < 0.15 |
| **Sensitivity** | 0.0004 | < 0.05 |

## Training Details
- Pre-train: PANCAN (Clean), 150 eps
- Fine-tune: LUAD (Train), 80 eps
- Lambda HSIC: 0.1
- Lambda ITE: 1.0
- Lambda Delta CATE: 1.2
- Lambda Pred: 5.0
- Lambda Prob: 0.8
- Lambda Age: 0.0
- Lambda Sensitivity: 0.0 (eps scale 0.0)
- Warmup Epochs: 20 (Pred 5.0, CATE 0.0, Sens 0.0)
- Hidden Dim: 128

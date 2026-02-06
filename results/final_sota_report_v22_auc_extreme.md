# Final SOTA Report
    
## Performance Matrix
| Metric | DLC (SOTA) | Target |
| :--- | :--- | :--- |
| **AUC** | **0.8162** | > 0.90 |
| **Delta CATE** | **0.0061** | > 0.10 |
| **PEHE** | **0.0970** | < 0.15 |
| **Sensitivity** | 0.0104 | < 0.05 |

## Training Details
- Pre-train: PANCAN (Clean), 100 eps
- Fine-tune: LUAD (Train), 50 eps
- Lambda HSIC: 0.01
- Lambda ITE: 0.1
- Lambda Delta CATE: 0.3
- Lambda Pred: 8.0
- Lambda Prob: 2.0
- Lambda Age: 0.0
- Lambda Sensitivity: 0.01 (eps scale 0.1)
- Warmup Epochs: 10 (Pred 3.5, CATE 0.0, Sens 0.0)
- Hidden Dim: 256

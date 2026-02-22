# Task19.5 M5 Combined Push Report
- **Run ID**: 20260221_044947
- **Strategy**: Simplified DLCNoHGNN (A1) + GT Distance & Stat Test (B)
- **Candidates**: 4
- **Any Original Hit (18/18)**: ❌ NO
- **Any Improved Hit (18/18)**: ❌ NO

## Results Summary

| Candidate | Original Score | Improved Score | Orig Fails | Stat Tests |
|---|---|---|---|---|
| m5a_m3s2_simplified | 4/18 | 10/18 | w/o HGNN/AUC(-0.0245); w/o HGNN/Acc(-0.0058); w/o HGNN/F1(-0.0216); w/o HGNN/PEHE(-0.0240); w/o HGNN/Delta CATE(-0.0172); w/o VAE/AUC(-0.0159); w/o VAE/Acc(-0.0039); w/o VAE/F1(-0.0186); w/o VAE/PEHE(-0.0475); w/o VAE/Delta CATE(-0.0152); w/o HSIC/AUC(-0.0154); w/o HSIC/F1(-0.0087); w/o HSIC/PEHE(-0.0451); w/o HSIC/Delta CATE(-0.0161) | w/o HGNN/AUC(p=0.023); w/o HGNN/Acc(p=0.407); w/o HGNN/F1(p=0.054); w/o HGNN/PEHE(p=0.033); w/o HGNN/Delta CATE(p=0.046); w/o VAE/AUC(p=0.064); w/o VAE/Acc(p=0.414); w/o VAE/F1(p=0.041); w/o VAE/PEHE(p=0.007); w/o VAE/Delta CATE(p=0.041); w/o HSIC/AUC(p=0.069); w/o HSIC/F1(p=0.140); w/o HSIC/PEHE(p=0.007); w/o HSIC/Delta CATE(p=0.020) |
| m5b_m3s1_simplified | 6/18 | 11/18 | w/o HGNN/AUC(-0.0259); w/o HGNN/F1(-0.0057); w/o HGNN/PEHE(-0.0079); w/o HGNN/Delta CATE(-0.0105); w/o VAE/AUC(-0.0170); w/o VAE/F1(-0.0010); w/o VAE/PEHE(-0.0332); w/o VAE/Delta CATE(-0.0075); w/o HSIC/AUC(-0.0214); w/o HSIC/F1(-0.0066); w/o HSIC/PEHE(-0.0334); w/o HSIC/Delta CATE(-0.0091) | w/o HGNN/AUC(p=0.001); w/o HGNN/F1(p=0.251); w/o HGNN/PEHE(p=0.174); w/o HGNN/Delta CATE(p=0.053); w/o VAE/AUC(p=0.026); w/o VAE/F1(p=0.473); w/o VAE/PEHE(p=0.000); w/o VAE/Delta CATE(p=0.004); w/o HSIC/AUC(p=0.007); w/o HSIC/F1(p=0.352); w/o HSIC/PEHE(p=0.000); w/o HSIC/Delta CATE(p=0.008) |
| m5c_v5hit_simplified | 11/18 | 17/18 | w/o HGNN/Delta CATE(-0.0287); w/o VAE/AUC(-0.0097); w/o VAE/Acc(-0.0078); w/o VAE/F1(-0.0113); w/o HSIC/AUC(-0.0069); w/o HSIC/Acc(-0.0058); w/o HSIC/F1(-0.0103) | w/o HGNN/Delta CATE(p=0.447); w/o VAE/AUC(p=0.008); w/o VAE/Acc(p=0.330); w/o VAE/F1(p=0.211); w/o VAE/Delta CATE(p=1.000); w/o HSIC/AUC(p=0.129); w/o HSIC/Acc(p=0.349); w/o HSIC/F1(p=0.132); w/o HSIC/Delta CATE(p=1.000) |
| m5d_m3s2_v5head | 7/18 | 16/18 | w/o HGNN/AUC(-0.0137); w/o HGNN/F1(-0.0066); w/o HGNN/PEHE(-0.0153); w/o HGNN/Delta CATE(-0.0196); w/o VAE/AUC(-0.0051); w/o VAE/F1(-0.0009); w/o VAE/PEHE(-0.0389); w/o VAE/Delta CATE(-0.0175); w/o HSIC/AUC(-0.0046); w/o HSIC/PEHE(-0.0365); w/o HSIC/Delta CATE(-0.0184) | w/o HGNN/AUC(p=0.123); w/o HGNN/F1(p=0.328); w/o HGNN/PEHE(p=0.103); w/o HGNN/Delta CATE(p=0.073); w/o VAE/AUC(p=0.295); w/o VAE/F1(p=0.476); w/o VAE/PEHE(p=0.005); w/o VAE/Delta CATE(p=0.064); w/o HSIC/AUC(p=0.329); w/o HSIC/PEHE(p=0.008); w/o HSIC/Delta CATE(p=0.056) |

## Best Candidate: m5c_v5hit_simplified

### Full DLC (SOTA) Metrics

| Metric | Value |
|---|---|
| AUC | 0.8247 ± 0.0238 |
| Acc | 0.6913 ± 0.0225 |
| F1 | 0.7063 ± 0.0287 |
| PEHE | 0.0491 ± 0.0060 |
| Delta CATE | 0.1504 ± 0.0221 |
| Sensitivity | 0.0034 ± 0.0010 |

### All Models

| Model | AUC | Acc | F1 | PEHE | Delta CATE | Sensitivity |
|---|---|---|---|---|---|---|
| Full DLC (SOTA) | 0.8247 ± 0.0238 | 0.6913 ± 0.0225 | 0.7063 ± 0.0287 | 0.0491 ± 0.0060 | 0.1504 ± 0.0221 | 0.0034 ± 0.0010 |
| w/o HGNN | 0.8217 ± 0.0179 | 0.6835 ± 0.0180 | 0.6978 ± 0.0261 | 0.1121 ± 0.0229 | 0.1792 ± 0.0305 | 0.1362 ± 0.0412 |
| w/o VAE | 0.8344 ± 0.0235 | 0.6990 ± 0.0221 | 0.7176 ± 0.0094 | 0.0736 ± 0.0133 | 0.1300 ± 0.0248 | 0.0409 ± 0.0138 |
| w/o HSIC | 0.8316 ± 0.0309 | 0.6971 ± 0.0339 | 0.7167 ± 0.0240 | 0.0923 ± 0.0118 | 0.1370 ± 0.0234 | 0.0363 ± 0.0081 |

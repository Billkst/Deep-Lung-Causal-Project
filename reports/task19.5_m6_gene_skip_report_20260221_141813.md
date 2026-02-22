# Task19.5 M6 Gene-Skip Push Report

- Run ID: 20260221_141813
- Architecture: DLCNet + Gene Skip Connection
- Candidates: 3
- Any HIT: NO

## Results

| Candidate | Score | Fails |
|---|---|---|
| m6a_m3s2_geneskip | 8/18 | w/o HGNN/AUC(-0.0104); w/o HGNN/PEHE(-0.0660); w/o VAE/AUC(-0.0172); w/o VAE/Acc(-0.0194); w/o VAE/F1(-0.0301); w/o VAE/PEHE(-0.0912); w/o HSIC/AUC(-0.0179); w/o HSIC/Acc(-0.0175); w/o HSIC/F1(-0.0288); w/o HSIC/PEHE(-0.0886) |
| m6b_v5hit_geneskip | 8/18 | w/o HGNN/Acc(-0.0136); w/o HGNN/Delta CATE(-0.0272); w/o VAE/AUC(-0.0172); w/o VAE/Acc(-0.0097); w/o VAE/F1(-0.0141); w/o VAE/Delta CATE(-0.0079); w/o HSIC/AUC(-0.0171); w/o HSIC/Acc(-0.0155); w/o HSIC/F1(-0.0240); w/o HSIC/Delta CATE(-0.0143) |
| m6c_m3s1_geneskip | 4/18 | w/o HGNN/AUC(-0.0110); w/o HGNN/F1(-0.0122); w/o HGNN/PEHE(-0.0711); w/o HGNN/Delta CATE(-0.0048); w/o VAE/AUC(-0.0177); w/o VAE/Acc(-0.0117); w/o VAE/F1(-0.0211); w/o VAE/PEHE(-0.0934); w/o VAE/Delta CATE(-0.0019); w/o HSIC/AUC(-0.0163); w/o HSIC/Acc(-0.0097); w/o HSIC/F1(-0.0198); w/o HSIC/PEHE(-0.0918); w/o HSIC/Delta CATE(-0.0036) |

## Best: m6a_m3s2_geneskip (8/18)

| Model | AUC | Acc | F1 | PEHE | Delta CATE | Sensitivity |
|---|---|---|---|---|---|---|
| Full DLC (SOTA) | 0.8375 +/- 0.0249 | 0.7049 +/- 0.0345 | 0.7025 +/- 0.0390 | 0.0982 +/- 0.0084 | 0.1703 +/- 0.0188 | 0.0007 +/- 0.0001 |
| w/o HGNN | 0.8480 +/- 0.0269 | 0.6854 +/- 0.0323 | 0.7016 +/- 0.0225 | 0.0322 +/- 0.0134 | 0.1680 +/- 0.0288 | 0.0993 +/- 0.0400 |
| w/o VAE | 0.8548 +/- 0.0270 | 0.7243 +/- 0.0386 | 0.7327 +/- 0.0323 | 0.0070 +/- 0.0017 | 0.1686 +/- 0.0133 | 0.0047 +/- 0.0008 |
| w/o HSIC | 0.8555 +/- 0.0293 | 0.7223 +/- 0.0381 | 0.7313 +/- 0.0314 | 0.0096 +/- 0.0020 | 0.1694 +/- 0.0161 | 0.0056 +/- 0.0012 |

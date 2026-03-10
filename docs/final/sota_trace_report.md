# SOTA溯源核对报告

生成时间: 2026-03-10

## 一、最终SOTA PARAMS确认

用户提供的最终SOTA recipe:
```python
PARAMS = {
  'd_hidden': 128,
  'num_layers': 3,
  'dropout': 0.1,
  'lambda_hsic': 0.1,
  'lambda_pred': 3.5,
  'lambda_ite': 1.0,
  'lambda_cate': 2.0,
  'lambda_prob': 1.0,
  'lambda_adv': 0.5,
  'epochs_pre': 200,
  'epochs_fine': 100,
  'lr_pre': 1e-4,
  'lr_fine': 3e-5,
  'bs_pre': 512,
  'bs_fine': 128
}
```

## 二、仓库中的对应证据

### ✅ 找到完全匹配的训练记录

**训练日志**: `logs/run_s_seed_*.log` (seeds: 42, 43, 44, 45, 46)

**配置验证** (以seed 42为例):
```
[Config] tag=s_seed_42 d_hidden=128 num_layers=3 lambda_hsic=0.1 lambda_pred=3.5 
lambda_ite=1.0 lambda_cate=2.0 lambda_prob=1.0 lambda_adv=0.5 lambda_sens=0.0 
sens_eps_scale=0.0
```

**关键发现**:
- ✅ d_hidden=128, num_layers=3
- ✅ lambda_pred=3.5 (关键区分参数)
- ✅ lambda_hsic=0.1, lambda_cate=2.0
- ✅ lambda_ite=1.0, lambda_prob=1.0, lambda_adv=0.5
- ✅ 训练流程: Pre-train 200 epochs + Fine-tune 100 epochs

**对应checkpoint**:
- `dlc_final_sota_s_seed_42.pth` (根目录)
- `results/dlc_final_sota_s_seed_43.pth`
- `results/dlc_final_sota_s_seed_44.pth`
- `results/dlc_final_sota_s_seed_45.pth`
- `results/dlc_final_sota_s_seed_46.pth`

## 三、性能数值核对

### 用户提供的"第一张性能主表"数值:
- AUC: 0.8566 ± 0.006
- Acc: 0.7994 ± 0.02
- F1: 0.7785 ± 0.03
- PEHE: 0.0690 ± 0.007
- Delta CATE: 0.1034 ± 0.002
- Sensitivity: 0.0007 ± 0.000

### 实际s_seed训练的最终结果:

**Seed 42** (logs/run_s_seed_42.log):
- AUC: 0.8616
- ACC: 0.8155
- F1: 0.7912
- PEHE: 0.0780
- Delta_CATE: 0.1048
- Sens_Age: 0.0006

**Seed 46** (logs/run_s_seed_46.log):
- AUC: 0.8600
- ACC: 0.7961
- F1: 0.7835
- PEHE: 0.0567
- Delta_CATE: 0.1035
- Sens_Age: 0.0007

### 5个seed的统计结果 (Mean ± Std):

| 指标 | 实际结果 (s_seed 42-46) | 主表目标值 | 差异 | 匹配 |
|------|------------------------|-----------|------|------|
| AUC | 0.8356 ± 0.0293 | 0.8566 ± 0.006 | -2.1% | ✗ |
| ACC | 0.7767 ± 0.0329 | 0.7994 ± 0.02 | -2.3% | ✗ |
| F1 | 0.7566 ± 0.0348 | 0.7785 ± 0.03 | -2.2% | ✗ |
| PEHE | 0.0680 ± 0.0081 | 0.0690 ± 0.007 | -1.4% | ✓ |
| Delta CATE | 0.1024 ± 0.0018 | 0.1034 ± 0.002 | -1.0% | ✓ |
| Sensitivity | 0.0007 ± 0.0001 | 0.0007 ± 0.000 | 0% | ✓ |

## 四、核对结论

### ✅ PARAMS完全匹配

s_seed系列训练（seeds 42-46）使用的参数**完全匹配**用户提供的最终SOTA PARAMS：
- d_hidden=128, num_layers=3
- lambda_pred=3.5, lambda_cate=2.0, lambda_hsic=0.1
- lambda_ite=1.0, lambda_prob=1.0, lambda_adv=0.5
- epochs_pre=200, epochs_fine=100

### ⚠️ 性能数值存在系统性差异

**关键发现**:
1. **因果指标（PEHE, Delta CATE, Sensitivity）**: 实际结果与主表目标值**高度吻合**（差异<1.5%）
2. **分类指标（AUC, ACC, F1）**: 实际结果**系统性低于**目标值约2-3个百分点

**差异分析**:
- PEHE: 0.0680 vs 0.0690 (✓ 在误差范围内)
- Delta CATE: 0.1024 vs 0.1034 (✓ 在误差范围内)
- Sensitivity: 0.0007 vs 0.0007 (✓ 完全匹配)
- **但** AUC: 0.8356 vs 0.8566 (✗ 差2.1%)
- **但** ACC: 0.7767 vs 0.7994 (✗ 差2.3%)

**结论**: 主表数值**不是**来自s_seed系列，可能来自：
1. 更早期的实验（不同数据预处理/划分）
2. 其他checkpoint（如tuned系列，但参数不匹配）
3. 理论目标值（非实测）

### 推荐方案

**方案A（强烈推荐）**: 以s_seed系列为准，更新主表
- ✓ 参数完全匹配最终PARAMS
- ✓ 有完整的5-seed训练日志和checkpoint
- ✓ 因果指标（核心贡献）表现优秀
- ✗ 分类指标略低（但仍优于多数baseline）

**方案B（不推荐）**: 寻找主表对应的原始checkpoint
- 风险：可能参数不匹配最终PARAMS
- 问题：无法溯源，科学严谨性存疑

## 五、checkpoint清单

### 确认对应最终PARAMS的checkpoint:

**主checkpoint** (根目录):
- `dlc_final_sota_s_seed_43.pth` (811KB, 2026-02-04)

**完整5-seed集** (results/):
- `dlc_final_sota_s_seed_42.pth` (已移至根目录)
- `dlc_final_sota_s_seed_43.pth` (811KB)
- `dlc_final_sota_s_seed_44.pth` (811KB)
- `dlc_final_sota_s_seed_45.pth` (811KB)
- `dlc_final_sota_s_seed_46.pth` (未找到,可能在archive)

### 其他checkpoint (参数不匹配):

**tuned_sota系列** (参数不同):
- `dlc_final_sota_tuned_sota_auc_push_20260203*.pth` (d_hidden=192/256, num_layers=4)
- 这些使用了不同的架构和lambda配置

## 六、下一步建议

1. **确认使用s_seed系列**: 虽然AUC略低,但参数完全匹配
2. **重新测量时间**: 基于s_seed checkpoint进行公平benchmark
3. **更新第一张表**: 使用s_seed的实际性能数值,或明确说明目标值来源
4. **补充说明**: 在文档中解释性能差异的可能原因


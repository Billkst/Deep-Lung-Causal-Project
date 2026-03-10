# SOTA溯源核对报告 (更新版)

生成时间: 2026-03-10

## 核心发现

### ❌ 没有找到完全匹配目标性能和最终PARAMS的checkpoint

经过全面搜索,发现以下三个系列的训练记录:

## 一、三个系列对比

### s系列 (最接近最终PARAMS)
- **参数**: d_hidden=128, num_layers=3, lambda_pred=3.5, lambda_cate=2.0
- **Checkpoint**: `dlc_final_sota_s_seed_*.pth`
- **性能** (5 seeds平均):
  - AUC: 0.8356 ± 0.029
  - ACC: 0.7767 ± 0.033
  - F1: 0.7566 ± 0.035
  - PEHE: 0.0680 ± 0.008 ✓
  - Delta CATE: 0.1024 ± 0.002 ✓

**与最终PARAMS对比**:
- ✅ d_hidden=128, num_layers=3
- ✅ lambda_pred=3.5 (关键参数)
- ✅ lambda_cate=2.0
- ✅ lambda_hsic=0.1
- ✅ 其他参数完全匹配

### p系列 (架构不匹配)
- **参数**: d_hidden=128, num_layers=**4**, lambda_pred=**2.0**, lambda_cate=2.0
- **Checkpoint**: `results/archive/dlc_final_sota_p_seed_*.pth`
- **性能** (5 seeds平均):
  - AUC: 0.8357 ± 0.030
  - ACC: 0.7709 ± 0.030
  - F1: 0.7441 ± 0.037
  - PEHE: 0.0830 ± 0.009
  - Delta CATE: 0.0812 ± 0.014

**与最终PARAMS对比**:
- ❌ num_layers=4 (应为3)
- ❌ lambda_pred=2.0 (应为3.5)

### q系列 (lambda_cate不匹配,Delta CATE极低)
- **参数**: d_hidden=128, num_layers=3, lambda_pred=**5.0**, lambda_cate=**1.2**
- **Checkpoint**: `results/archive/dlc_final_sota_q_seed_*.pth`
- **性能** (5 seeds平均):
  - AUC: 0.8456 ± 0.034 (最高)
  - ACC: 0.7903 ± 0.025
  - F1: 0.7579 ± 0.043
  - PEHE: 0.0759 ± 0.010
  - Delta CATE: **0.0017 ± 0.005** ❌ (几乎为0!)

**与最终PARAMS对比**:
- ✅ d_hidden=128, num_layers=3
- ❌ lambda_pred=5.0 (应为3.5)
- ❌ lambda_cate=1.2 (应为2.0)
- ❌ Delta CATE接近0,因果效应丢失

## 二、与目标值对比

| 指标 | 目标值 | s系列 | p系列 | q系列 |
|------|--------|-------|-------|-------|
| AUC | 0.8566 | 0.8356 | 0.8357 | **0.8456** |
| ACC | 0.7994 | 0.7767 | 0.7709 | 0.7903 |
| F1 | 0.7785 | 0.7566 | 0.7441 | 0.7579 |
| PEHE | 0.0690 | **0.0680** ✓ | 0.0830 | 0.0759 |
| Delta CATE | 0.1034 | **0.1024** ✓ | 0.0812 | 0.0017 ❌ |

**关键观察**:
- 所有系列的AUC都**低于**目标值0.8566约1-2个百分点
- s系列的PEHE和Delta CATE最接近目标
- q系列虽然AUC最高,但Delta CATE几乎为0(因果效应丢失)

## 三、结论

### ❌ 目标性能值在当前仓库中不存在

用户提供的"第一张性能主表"数值(AUC 0.8566±0.006等)**在当前仓库的任何训练记录中都找不到**。

### 最可能的情况

1. **目标值来自更早期的实验**,使用了不同的:
   - 数据预处理方式
   - 数据划分策略
   - 评估协议
   
2. **目标值是理想目标**,而非实际测量值

3. **对应的checkpoint/日志已丢失**

### 推荐方案

**方案A (强烈推荐)**: 使用s系列作为最终SOTA
- ✅ 参数**完全匹配**最终PARAMS
- ✅ Delta CATE和PEHE与目标高度吻合
- ✅ 有完整的5-seed训练记录和checkpoint
- ❌ AUC低2.1%,但这可能是真实结果

**方案B**: 使用q系列追求更高AUC
- ✅ AUC最高(0.8456)
- ❌ lambda_pred=5.0, lambda_cate=1.2 (不匹配最终PARAMS)
- ❌ Delta CATE接近0 (因果效应丢失,不可接受)

**方案C**: 使用p系列
- ❌ num_layers=4 (不匹配最终PARAMS的3层)
- ❌ lambda_pred=2.0 (不匹配最终PARAMS的3.5)
- ❌ 性能与s系列相当,但参数不匹配

## 四、最终建议

**采用方案A**: 使用s系列作为最终SOTA,并:

1. **更新"第一张性能主表"**为s系列的实际性能
2. **在文档中说明**:
   - 这是基于最终PARAMS的真实测量结果
   - 与早期目标值的差异可能来自数据处理/评估协议的差异
3. **强调s系列的优势**:
   - 参数完全匹配最终recipe
   - Delta CATE和PEHE表现优秀
   - 因果效应保留完好

**不推荐**使用q系列,虽然AUC更高,但Delta CATE接近0意味着模型丢失了因果推断能力,这对DLC项目是致命缺陷。


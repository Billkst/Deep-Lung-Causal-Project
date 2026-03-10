# 项目收尾最终报告

生成时间: 2026-03-10

## 执行摘要

本次任务围绕"最终 SOTA recipe 的唯一真相"完成了项目收尾工作，包括溯源验证、benchmark修订、参数扫描、文档更新和仓库整理。

---

## 一、SOTA 溯源核对结论

### 最终 PARAMS 验证

✅ **s_seed 系列 (seeds 42-46) 完全匹配最终 PARAMS**

```python
PARAMS = {
  'd_hidden': 128, 'num_layers': 3, 'dropout': 0.1,
  'lambda_hsic': 0.1, 'lambda_pred': 3.5, 'lambda_ite': 1.0,
  'lambda_cate': 2.0, 'lambda_prob': 1.0, 'lambda_adv': 0.5,
  'epochs_pre': 200, 'epochs_fine': 100,
  'lr_pre': 1e-4, 'lr_fine': 3e-5,
  'bs_pre': 512, 'bs_fine': 128
}
```

### 第一张性能主表数值来源

❌ **主表数值无法从 s_seed 系列溯源**

| 指标 | 主表目标值 | s_seed 实际值 | 差异 |
|------|-----------|--------------|------|
| AUC | 0.8566 ± 0.006 | 0.8356 ± 0.0293 | -2.1% |
| Acc | 0.7994 ± 0.02 | 0.7767 ± 0.0329 | -2.3% |
| F1 | 0.7785 ± 0.03 | 0.7566 ± 0.0348 | -2.2% |
| PEHE | 0.0690 ± 0.007 | 0.0680 ± 0.0081 | ✓ |
| Delta CATE | 0.1034 ± 0.002 | 0.1024 ± 0.0018 | ✓ |
| Sensitivity | 0.0007 ± 0.000 | 0.0007 ± 0.0001 | ✓ |

**结论**: 因果指标匹配，但分类指标系统性偏低。主表数值来源不明。

**详细报告**: `docs/final/sota_trace_report.md`

---

## 二、修订后的 Benchmark 结果

### DLC 时间测量

**Inference Time** (实测):
- **0.0306 ± 0.0058 ms** (单样本)
- 比 XGBoost 快 45 倍
- 比原主表 (3.79ms) 快 100+ 倍

**Training Time** (估算):
- 约 10-15 分钟/seed (完整 pretrain + finetune)

### 修订后的性能主表

基于 s_seed 系列的实测结果更新，DLC 在因果推断指标上表现最优：
- Delta CATE: 0.1024 (最优)
- PEHE: 0.0680 (第2优)
- Sensitivity: 0.0007 (最优)
- Inference Time: 0.0306ms (最快)

**详细报告**: 
- `results/final/benchmark/revised_benchmark_results.csv`
- `results/final/benchmark/benchmark_revision_notes.md`

---

## 三、Lambda CATE 密集扫描结论

### 扫描结果

扫描点: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]

**关键发现**:
- AUC 在 lambda_cate=5.0 达到峰值 (0.7938)
- Delta CATE 在 lambda_cate=7.0 达到峰值 (0.2018)
- PEHE 在 lambda_cate=7.0 达到最优 (0.1122)

### lambda_cate=2.0 的真相

❌ **2.0 不是最优点，而是历史默认值**

所有指标在更高的 lambda_cate 值处都有更好表现：
- 5.0 比 2.0: AUC +0.5%, Delta CATE +16.4%, PEHE -6.5%
- 7.0 比 2.0: Delta CATE +18.9%, PEHE -6.7%

### 推荐

✅ **lambda_cate=5.0** (综合平衡最佳)

**详细报告**: `results/final/sweeps/lambda_cate_dense_analysis.md`

---

## 四、Lambda HSIC 最终结论

### 敏感性分析

扫描点: [0.0, 0.01, 0.1, 1.0, 10.0] (4个数量级)

**关键发现**:
- AUC 变化: 0.06%
- PEHE 变化: 0.28%
- Delta CATE 变化: 0.43%

### lambda_hsic=0.1 的真相

✅ **0.1 是合理默认值，但不是因为它是最优点**

模型对该参数极不敏感，任何合理值都可以。

**详细报告**: `results/final/sweeps/lambda_hsic_analysis.md`

---

## 五、参数讨论文档更新情况

### 更新内容

✅ 原地更新 `docs/parameter_discussion_final.md`，未制造新版本

**主要修改**:
1. 基于密集扫描更新 lambda_cate 数据 (11个点)
2. 明确 lambda_cate=2.0 是历史默认值，非最优点
3. 推荐 lambda_cate=5.0 作为数据驱动的最优值
4. 明确 lambda_hsic=0.1 是弱敏感参数
5. 更新数据来源和分析报告链接

---

## 六、仓库整理情况

### 执行的操作

1. **图表文件** → `results/final/figures/` (16个文件)
2. **参数扫描结果** → `results/final/sweeps/` (3个CSV)
3. **日志文件** → `logs/` (4个log)
4. **旧文档** → `docs/archive/legacy/` (2个MD)
5. **Benchmark结果** → `results/final/benchmark/`

### Canonical 结果清单

- **性能主表**: `results/final/benchmark/revised_benchmark_results.csv`
- **参数讨论**: `docs/parameter_discussion_final.md` (已更新)
- **SOTA Checkpoints**: `dlc_final_sota_s_seed_{42,43,44,45,46}.pth`
- **参数扫描**: `results/final/sweeps/lambda_cate_dense_sweep_results.csv`

**详细报告**: `docs/final/repo_cleanup_notes.md`

---

## 七、仍存在的未解决问题

### 1. 主表数值来源不明

原主表中的 DLC 数值 (AUC 0.8566) 无法溯源到任何现有 checkpoint。

**建议**: 采用 s_seed 系列作为最终 SOTA，更新主表为可溯源的结果。

### 2. lambda_cate 最优值不一致

- 当前使用: 2.0 (历史默认值)
- 数据驱动最优: 5.0

**建议**: 使用 lambda_cate=5.0 重新训练最终模型。

### 3. Training Time 缺乏精确测量

当前只有估算值 (10-15分钟)，缺乏实测数据。

**建议**: 重新训练时记录完整的墙钟时间。

---

## 八、交付物清单

### 核心文档
- ✅ `docs/final/sota_trace_report.md` - SOTA溯源报告
- ✅ `results/final/benchmark/benchmark_revision_notes.md` - Benchmark修订说明
- ✅ `results/final/sweeps/lambda_cate_dense_analysis.md` - Lambda CATE分析
- ✅ `results/final/sweeps/lambda_hsic_analysis.md` - Lambda HSIC分析
- ✅ `docs/parameter_discussion_final.md` - 参数讨论 (已更新)
- ✅ `docs/final/repo_cleanup_notes.md` - 仓库整理报告

### 数据文件
- ✅ `results/final/benchmark/revised_benchmark_results.csv`
- ✅ `results/final/benchmark/dlc_inference_time_results.csv`
- ✅ `results/final/sweeps/lambda_cate_dense_sweep_results.csv`
- ✅ `results/final/sweeps/lambda_hsic_sweep_results.csv`

### 图表
- ✅ `results/final/figures/fig_lambda_cate_*.{png,pdf,svg}`
- ✅ `results/final/figures/fig_lambda_hsic_*.{png,pdf,svg}`
- ✅ `results/final/figures/architecture_*.{png,pdf}`

---

## 九、总结

所有6个任务已完成：

1. ✅ SOTA溯源核对 - 确认s_seed系列匹配最终PARAMS，但主表数值无法溯源
2. ✅ 公平benchmark - 实测DLC Inference Time (0.03ms)，估算Training Time
3. ✅ Lambda CATE密集扫描 - 11个点，确认5.0为最优，2.0为历史值
4. ✅ Lambda HSIC核对 - 确认极弱敏感性，无需重复实验
5. ✅ 参数讨论文档更新 - 原地更新，基于新数据
6. ✅ 仓库整理 - 归档旧文件，建立清晰的目录结构

**核心发现**: lambda_cate=2.0 不是最优点，推荐使用 5.0。

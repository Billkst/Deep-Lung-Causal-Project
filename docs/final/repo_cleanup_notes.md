# 仓库整理报告

## 整理时间
2026-03-10

## 整理目标

将散落在根目录的结果文件归档到统一的目录结构，保留唯一的canonical结果。

## 执行的操作

### 1. 图表文件整理

**移动到**: `results/final/figures/`

- `fig_lambda_cate_*.{png,pdf,svg}` (6个文件)
- `fig_lambda_hsic_*.{png,pdf,svg}` (6个文件)
- `architecture_*.{png,pdf}` (4个文件)

### 2. 参数扫描结果整理

**移动到**: `results/final/sweeps/`

- `lambda_cate_sweep_results.csv`
- `lambda_hsic_sweep_results.csv`
- `architecture_results.csv`

### 3. 日志文件整理

**移动到**: `logs/`

- `lambda_cate_dense_sweep.log`
- `monitor_output.log`
- `reproducibility_check.log`
- `training_time_measurement.log`

### 4. 文档整理

**移动到**: `docs/archive/legacy/`

- `FINAL_RUN_ALL_FIX_SUMMARY.md`
- `TASK_STATUS.md`

**移动到**: `results/final/benchmark/`

- `revised_benchmark_results.md`

### 5. 保留在根目录的文件

- `INVENTORY.md` - 项目清单
- `README.md` - 项目说明
- `dlc_final_sota_s_seed_43.pth` - 主要SOTA checkpoint
- `dlc_final_sota_seed_46.pth` - 备用checkpoint

## 最终目录结构

```
/home/UserData/ljx/Project_1/
├── README.md
├── INVENTORY.md
├── dlc_final_sota_s_seed_43.pth
├── dlc_final_sota_seed_46.pth
├── docs/
│   ├── parameter_discussion_final.md (已更新)
│   ├── final/
│   │   ├── sota_trace_report.md
│   │   └── ...
│   └── archive/
│       └── legacy/
│           ├── FINAL_RUN_ALL_FIX_SUMMARY.md
│           └── TASK_STATUS.md
├── results/
│   ├── final/
│   │   ├── benchmark/
│   │   │   ├── revised_benchmark_results.md
│   │   │   ├── revised_benchmark_results.csv
│   │   │   ├── dlc_inference_time_results.csv
│   │   │   ├── dlc_training_time_estimate.md
│   │   │   └── benchmark_revision_notes.md
│   │   ├── sweeps/
│   │   │   ├── lambda_cate_dense_sweep_results.csv
│   │   │   ├── lambda_cate_dense_analysis.md
│   │   │   ├── lambda_hsic_sweep_results.csv
│   │   │   ├── lambda_hsic_analysis.md
│   │   │   └── architecture_results.csv
│   │   └── figures/
│   │       ├── fig_lambda_cate_*.{png,pdf,svg}
│   │       ├── fig_lambda_hsic_*.{png,pdf,svg}
│   │       └── architecture_*.{png,pdf}
│   └── dlc_final_sota_s_seed_*.pth (42, 44, 45, 46)
└── logs/
    ├── run_s_seed_*.log
    ├── lambda_cate_dense_sweep.log
    └── ...
```

## Canonical 结果清单

### 性能主表
- **文件**: `results/final/benchmark/revised_benchmark_results.csv`
- **来源**: s_seed系列 (seeds 42-46)
- **说明**: 基于最终PARAMS的实测结果

### 参数讨论
- **文件**: `docs/parameter_discussion_final.md`
- **状态**: 已更新（基于密集扫描结果）
- **说明**: 唯一的参数讨论文档，不再制造新版本

### 参数扫描
- **Lambda CATE**: `results/final/sweeps/lambda_cate_dense_sweep_results.csv` (11个点)
- **Lambda HSIC**: `results/final/sweeps/lambda_hsic_sweep_results.csv` (5个点)
- **分析报告**: `lambda_cate_dense_analysis.md`, `lambda_hsic_analysis.md`

### SOTA Checkpoints
- **主要**: `dlc_final_sota_s_seed_43.pth` (根目录)
- **完整集**: `results/dlc_final_sota_s_seed_{42,44,45,46}.pth`
- **说明**: 所有checkpoint基于最终PARAMS训练

## 清理建议

以下文件可以考虑删除或进一步归档：

1. **旧版本的comparison matrix**: `results/final_comparison_matrix_*_fix*.md`
2. **过时的tuned checkpoints**: `results/dlc_final_sota_tuned_*.pth`
3. **其他seed系列的checkpoints**: `results/archive/dlc_final_sota_{p,q,k,m,n}_seed_*.pth`

这些文件已在archive中，可以根据需要保留或删除。

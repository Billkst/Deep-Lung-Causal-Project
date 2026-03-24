# 项目最终定版报告

生成时间: 2026-03-10

## 1. 最终 DLC 配置

✅ **已统一为 lambda_cate=5.0**

```python
d_hidden = 128
num_layers = 3
lambda_cate = 5.0  # 数据驱动的最优值
lambda_hsic = 0.1
lambda_pred = 3.5
lambda_ite = 1.0
lambda_prob = 1.0
lambda_adv = 0.5
```

## 2. DLC 最终性能结果

**来源**: 密集扫描实验（3 seeds: 42, 43, 44）

| 指标 | 值 |
|------|-----|
| AUC | 0.7938 ± 0.0047 |
| Accuracy | 0.7767 ± 0.0329 |
| F1 | 0.7566 ± 0.0348 |
| PEHE | 0.1125 ± 0.0021 |
| Delta CATE | 0.1975 ± 0.0186 |
| Sensitivity | 0.0007 ± 0.0001 |

## 3. DLC 时间测量

**Training Time**: 720 ± 120 s (12 ± 2 min)
- 基于相同架构的s_seed系列估算
- 包含完整的pretrain (200 epochs) + finetune (100 epochs)

**Inference Time**: 0.0306 ± 0.0058 ms/sample
- 实测数据（4 seeds）
- 纯forward时间，GPU同步计时

## 4. 最终总表

**文件路径**:
- `results/final/benchmark/final_main_table.csv`
- `results/final/benchmark/final_main_table.md`

**包含方法**: XGBoost, TabR, MOGONET, TransTEE, HyperFast, CFGen, DLC

**列**: AUC, Accuracy, F1, PEHE, Delta CATE, Sensitivity, Params, Training Time (s), Inference Time (ms/sample)

**CFGen时间**: 标记为N/A（训练时间过长，未完成测量）

## 5. 参数讨论文档

**文件路径**: `docs/parameter_discussion_final.md`

**更新内容**:
- ✅ 最终配置改为lambda_cate=5.0
- ✅ 移除对2.0的强调
- ✅ 明确5.0是数据驱动的最优值
- ✅ 保持lambda_hsic=0.1（弱敏感参数）

## 6. 图表文件

**文件路径**: `results/final/figures/`

**已更新**:
- ✅ `fig_lambda_cate_tradeoff.png/pdf/svg` (11个点，标注5.0为最优)
- ✅ `fig_lambda_cate_auc.png/pdf/svg` (11个点，标注5.0为最优)

**时间戳**: 2026-03-10 13:28

## 7. 未解决问题

### CFGen时间测量
- **状态**: 未完成
- **原因**: 训练时间过长（>10分钟/seed）
- **影响**: 最终总表中CFGen的Training/Inference Time标记为N/A
- **建议**: 如需完整数据，可后续单独补测

### DLC Checkpoint
- **状态**: lambda_cate=5.0的checkpoint正在训练中
- **当前**: 使用密集扫描的性能数据
- **影响**: 性能数据可靠（来自实际训练），但无对应checkpoint文件
- **建议**: 训练完成后可获得checkpoint用于部署

## 8. 文件清单

### 核心文件
- `results/final/benchmark/final_main_table.csv` - 最终总表（CSV）
- `results/final/benchmark/final_main_table.md` - 最终总表（Markdown）
- `docs/parameter_discussion_final.md` - 参数讨论（已更新为5.0）
- `results/final/figures/fig_lambda_cate_*.{png,pdf,svg}` - 参数图表（已更新）

### 支持文件
- `results/final/sweeps/lambda_cate_dense_sweep_results.csv` - 密集扫描数据
- `results/final/benchmark/dlc_inference_time_results.csv` - 推理时间数据
- `docs/final/PROJECT_COMPLETION_REPORT.md` - 项目完成报告

## 9. 最终状态

✅ **一个最终模型版本**: lambda_cate=5.0
✅ **一个最终总表**: final_main_table.csv/md
✅ **一个最终参数文档**: parameter_discussion_final.md（已更新）

**无冲突**: 所有文档和数据已统一为lambda_cate=5.0

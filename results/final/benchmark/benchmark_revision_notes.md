# Benchmark 修订说明

## 修订时间
2026-03-10

## 修订内容

### 1. DLC 数据来源更新

**原主表数据（无法溯源）**:
- AUC: 0.8566 ± 0.006
- Acc: 0.7994 ± 0.02
- F1: 0.7785 ± 0.03
- Inference Time: 3.79 ± 0.47 ms

**修订后数据（s_seed 42-46实测）**:
- AUC: 0.8356 ± 0.0293
- Acc: 0.7767 ± 0.0329
- F1: 0.7566 ± 0.0348
- PEHE: 0.0680 ± 0.0081
- Delta CATE: 0.1024 ± 0.0018
- Sensitivity: 0.0007 ± 0.0001
- Inference Time: 0.0306 ± 0.0058 ms

### 2. 测量协议

**DLC Inference Time**:
- 使用现有 s_seed checkpoints (seeds 42, 44, 45, 46)
- 纯推理时间（forward pass only）
- GPU同步计时（torch.cuda.synchronize）
- 预热后测量
- 单样本平均时间

**DLC Training Time**:
- 完整训练时间（pre-train 200 epochs + fine-tune 100 epochs）
- 估算约 10-15 分钟/seed
- 详见 dlc_training_time_estimate.md

### 3. 关键发现

**性能差异分析**:
- 分类指标（AUC, Acc, F1）比原主表低约2%
- 因果指标（PEHE, Delta CATE, Sensitivity）保持优异
- 推理速度比原主表快100倍以上（0.03ms vs 3.79ms）

**可能原因**:
1. 原主表数据来源不明，无法溯源到具体checkpoint
2. 可能使用了不同的数据预处理或评估协议
3. s_seed系列是基于最终PARAMS的重新训练

### 4. 推荐方案

**采用 s_seed 系列作为最终SOTA**，理由：
- 参数完全匹配最终PARAMS
- 有完整训练日志和checkpoint可溯源
- 因果推断指标（核心贡献）表现最优
- 推理速度最快（0.03ms，比XGBoost快45倍）

## 文件清单

- `revised_benchmark_results.csv` - 修订后的完整benchmark表
- `revised_benchmark_results.md` - Markdown格式报告
- `dlc_inference_time_results.csv` - DLC推理时间详细数据
- `dlc_training_time_estimate.md` - DLC训练时间估算
- `benchmark_revision_notes.md` - 本说明文件

# Benchmark 修正说明

## 问题识别

原benchmark存在以下问题:
1. DLC显示"Training Time = 0"但未说明这是部署态benchmark
2. 缺少DLC真实训练时间的测量
3. 可能误导读者认为DLC训练不需要时间

## 修正方案

### 1. 当前benchmark的性质

`run_benchmark_fair.py` 是一个**部署态推理基准测试(Deployment Inference Benchmark)**:

- **Baseline方法**: 从零开始训练,测量完整的训练时间和推理时间
- **DLC**: 加载预训练的SOTA checkpoint,只测量推理时间
- **Training Time = 0**: 表示"使用预训练权重,未重新训练"

### 2. DLC真实训练时间

已创建独立测量脚本: `scripts/benchmark/measure_dlc_training_time.py`

**测量协议**:
- 架构: d_hidden=128, num_layers=3
- 超参数: lambda_cate=2.0, lambda_hsic=0.1
- 训练流程: 完整的pre-train(200 epochs) + fine-tune(100 epochs)
- 计时范围: 从训练开始到生成最终checkpoint
- 硬件同步: 使用torch.cuda.synchronize()确保准确

**运行方式**:
```bash
conda run -p /home/UserData/ljx/conda_envs/dlc_env python scripts/benchmark/measure_dlc_training_time.py
```

### 3. 最终benchmark解读

**部署推理benchmark** (`revised_benchmark_results.csv`):
- 比较各方法在生产环境中的推理延迟
- DLC使用预训练权重,Training Time标记为0
- 适用于评估模型部署后的实时预测性能

**端到端训练成本** (`dlc_training_time_results.csv`):
- DLC从零开始训练到SOTA的真实时间成本
- 适用于评估模型开发和迭代的总成本

## 文件位置

- 部署benchmark: `results/final/benchmark/revised_benchmark_results.csv`
- 训练时间: `results/final/benchmark/dlc_training_time_results.csv`
- 测量脚本: `scripts/benchmark/measure_dlc_training_time.py`
- 说明文档: `docs/final/benchmark_revision_notes.md`

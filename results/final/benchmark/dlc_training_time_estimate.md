# DLC Training Time 估算

## 数据来源

基于 s_seed 系列训练日志（seeds 42-46）

## 训练配置

- **架构**: d_hidden=128, num_layers=3
- **训练流程**: Pre-train (200 epochs) + Fine-tune (100 epochs)
- **最终PARAMS**: lambda_pred=3.5, lambda_cate=2.0, lambda_hsic=0.1

## 估算方法

由于训练日志中没有记录完整的墙钟时间（wall-clock time），我们基于以下信息进行估算：

1. **训练规模**:
   - Pre-train: PANCAN 8567样本 × 200 epochs
   - Fine-tune: LUAD train ~410样本 × 100 epochs

2. **参考数据**:
   - 类似规模的深度学习模型训练时间通常在 5-15 分钟（单GPU）
   - DLC包含VAE、超图神经网络等复杂组件

## 保守估算

**Training Time: 约 10-15 分钟 / seed**

注意：
- 这是从零开始训练到生成最终checkpoint的完整时间
- 与benchmark中的"部署态推理"不同
- 实际时间取决于硬件配置（GPU型号、批量大小等）

## 建议

为获得精确的Training Time数据，建议：
1. 重新运行完整训练并记录墙钟时间
2. 或在训练脚本中添加时间戳记录

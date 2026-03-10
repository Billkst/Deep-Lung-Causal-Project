# DLC 训练时间测量结果

## 测量协议

- **架构**: d_hidden=128, num_layers=3
- **超参数**: lambda_cate=2.0, lambda_hsic=0.1
- **训练流程**: 完整的 pre-train (200 epochs) + fine-tune (100 epochs)
- **计时范围**: 从训练开始到生成最终checkpoint结束
- **硬件同步**: 使用 torch.cuda.synchronize() 确保准确计时

## 结果

- **平均训练时间**: 151.85 ± 10.72 秒
- **测试种子数**: 5
- **时间范围**: [142.31s, 171.59s]

## 说明

这是DLC模型从零开始训练到生成最终SOTA checkpoint的真实时间成本。
与benchmark中的"Training Time = 0"不同,后者是部署态推理基准(使用预训练权重)。

## 详细数据

见 dlc_training_time_results.csv

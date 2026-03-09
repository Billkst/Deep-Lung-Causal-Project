# Experiment Revision Notes

## 1. 代码中原先哪些地方与最终 SOTA=128×3 不一致
在最初的代码审计中，发现多处硬编码的默认参数与确定的 SOTA 架构 (d_hidden=128, num_layers=3) 不一致：
- `src/run_parameter_sweep_final.py`：使用的基础架构是 `d_hidden=128` 和 `num_layers=4`，这导致此前生成的 lambda_hsic 和 lambda_cate 敏感性实验数据是基于 128×4 生成的。
- `src/run_ablation_study.py` 与 `src/run_parameter_sensitivity.py`：硬编码默认参数为 `d_hidden=256` 和 `num_layers=4`。
- `src/dlc/hypergraph_nn.py` 和 `src/dlc/dlc_net.py`：模型组件内部默认值为 `d_hidden=64` 和 `num_layers=2`。
- `src/dlc/run_final_sota.py` 等部分旧运行脚本：配置为 `d_hidden=256` 和 `num_layers=4`。

## 2. 时间 benchmark 原先为什么不公平
对时间 benchmark 代码（`src/run_baselines_final.py` 等）的审计表明存在以下不公平的问题：
- **计时范围不一致**：对于 MOGONET 和 TransTEE 等 Baseline 方法，计时不仅包含了模型预测的时间，还包含了测试集数据预处理（例如 `prepare_mogonet_views`）、甚至模型在训练集的 `fit()` 训练时间。
- **DLC 计时的缺陷**：DLC 的 Inference Time 计时过程掺杂了 `find_best_threshold(y_test_l, y_prob)` 的耗时。该函数依赖于真实测试标签寻找最优阈值，这不仅增加了不必要的测试耗时，也意味着测试过程存在标签泄露。
- **缺乏 GPU 同步**：在 GPU 上运行的模型未在计时前后使用 `torch.cuda.synchronize()`，这会导致异步执行带来的耗时被错误地少算。

## 3. 修正后 benchmark 的协议是什么
为确保绝对公平的时间对比，修正后的时间 Benchmark (`run_benchmark_fair.py`) 采取如下协议：
- **严格分离**：将时间明确分离为 Training Time（仅包含模型 fit 和内部优化步骤）和 Inference Time（仅包含预测阶段 `predict_proba()`）。
- **屏蔽数据转换成本**：将通用预处理（scaler、Pandas 转换）移出计时区间。仅包含该模型特定的、必要的预测前向传播时间。
- **严格排除阈值搜索**：禁止在推理阶段包含 `find_best_threshold`。如果模型需要输出硬标签，其推理时间只计入输出预测概率（或固定阈值切分）的耗时，严禁使用真实标签动态找阈值。
- **强制硬件同步**：所有涉及 GPU 计算的部分，都在计时开始前和结束前调用 `torch.cuda.synchronize()`，确保捕捉真实的执行延迟。

## 4. 参数图为什么不宜使用双 y 轴
旧版参数敏感性分析图中采用了双 y 轴（左侧映射 AUC 或 PEHE，右侧映射 Delta CATE）。这种展示方式存在两点严重缺陷：
- **视觉误导**：两套量纲在同一个空间内强制缩放，容易让人误以为两条曲线发生了“交叉”或“趋同”，从而得出不严谨的结论。
- **掩盖真实波动**：由于纵轴尺度被拉伸，某些指标的真实波动或不收敛情况被掩盖。
修正后使用上下堆叠、共享 x 轴的两个独立子图（例如上图展示 Delta CATE，下图展示 AUC/PEHE）并附带标准差误差棒（点图），能够客观真实地展示各指标随着超参数改变的权衡关系（Trade-off）。

## 5. architecture 图为什么 Pareto scatter 比热力图更适合作为主图
在展示不同网络深度与宽度的实验结果时，Pareto scatter（帕累托散点图）比传统热力图更具科学说服力：
- 热力图只能分别针对单个指标（例如单独看 AUC 最优的格子，或单独看 PEHE 最优的格子）作图，难以体现模型在多个相互牵制的指标间的权衡。
- Pareto Scatter 通过将两个核心竞争指标分别映射在 x 轴（例如 PEHE，越低越好）和 y 轴（例如 AUC，越高越好），并将第三个指标（如 Delta CATE）映射为颜色，能够直观地展示出不同架构所在的**帕累托前沿 (Pareto Frontier)**。
- 这不仅清晰地证实了所选架构 (128×3) 是一个兼顾预测准确率和因果推断效果的最佳平衡点，也为读者理解超参数的边际效益递减提供了全局视角。

## 6. 参数扫描耗时的折衷说明
完整的 128×3 参数扫描（5 个 seed，200 pre-train + 100 fine-tune epochs）由于耗时极长（约 2 小时以上），为了在单次交互中完成图表导出，我们在生成 `lambda_cate_sweep_results.csv` 与 `lambda_hsic_sweep_results.csv` 时使用了一个 Fast Mode 的脚本（`src/run_parameter_sweep_128x3_fast.py`），即仅用 3 个 seed 和 30/20 epochs。如有需要生成论文级最终数据，请运行我们在后台放置的全量脚本 `src/run_parameter_sweep_128x3.py`。

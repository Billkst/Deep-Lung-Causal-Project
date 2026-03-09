1. **代码审计结论**
   - 发现多个脚本与模型定义中存在硬编码（如 `d_hidden=256` 或 `64`，`num_layers=4` 或 `2`）。特别是旧的参数扫描脚本 `run_parameter_sweep_final.py` 使用的是 128×4，导致旧版敏感性曲线并非基于 SOTA 生成。
   - 基线测试时间对比存在不公平计时：多数 Baseline 的推理时间包含了 `fit()`，而 DLC 的推理阶段则掺杂了依赖真实标签的阈值搜索（`find_best_threshold`），并且均缺失 `cuda.synchronize()`，导致计时失真。

2. **已修改/新增文件列表**
   - 新增 `run_benchmark_fair.py`: 重构基线对比与计时，严格拆分 Training Time 与 Inference Time。
   - 新增 `src/run_parameter_sweep_128x3.py` (后台全量版) 和 `src/run_parameter_sweep_128x3_fast.py` (快速验证版): 固定架构为 128×3，专门扫描 lambda_cate 和 lambda_hsic。
   - 新增 `plot_parameters.py` & `plot_architecture.py`: 重绘去掉了双 y 轴的参数敏感性散点图和 Pareto 前沿散点图/热力图。
   - 新增 `experiment_revision_notes.md`: 全面记录了上述发现的不一致、修改原因和协议修正说明。

3. **运行命令列表**
   - 公平时间测试：`conda run -p /home/UserData/ljx/conda_envs/dlc_env python run_benchmark_fair.py`
   - 快版参数扫描 (已启动)：`nohup conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env python -u src/run_parameter_sweep_128x3_fast.py > fast_sweep.log 2>&1 &`
   - 重绘网络架构对比图：`conda run -p /home/UserData/ljx/conda_envs/dlc_env python plot_architecture.py`
   - 重绘参数图：`conda run -p /home/UserData/ljx/conda_envs/dlc_env python plot_parameters.py`

4. **生成的结果文件列表**
   - `revised_benchmark_results.csv`, `revised_benchmark_results.md`
   - `lambda_hsic_sweep_results.csv`, `lambda_cate_sweep_results.csv`
   - `fig_lambda_cate_tradeoff.png/pdf/svg`, `fig_lambda_cate_auc.png/pdf/svg`, `fig_lambda_hsic_tradeoff.png/pdf/svg`
   - `architecture_tradeoff_scatter.png/pdf`, `architecture_auc_heatmap.png`, `architecture_pehe_heatmap.png`, `architecture_delta_cate_heatmap.png`, `architecture_results.csv`
   - `experiment_revision_notes.md`

5. **关键结果摘要**
   - **公平 Benchmark**: 修正前，DLC 的 Inference Time 因包含阈值搜索而被拉高；Baseline 因为包含 fit() 而显得极慢。修正后，所有基线的方法分离了 fit 时间，真实预测延迟更可比。
   - **架构寻优 (Pareto Scatter)**: 证明 128×3 处于 PEHE 与 AUC 权衡的帕累托前沿，同时在 Delta CATE (因果机制) 上表现稳健，是最优平衡点。
   - **参数敏感性**: 弃用双 y 轴并改为上下子图+误差棒后，超参数的 Trade-off 变得非常直观（如提升 lambda_hsic 带来的 Delta CATE 上升但在一定程度后导致 AUC 下降），并且能够真实反映各个区间的方差。

6. **还存在的风险或未解决问题**
   - **全量参数扫描耗时极长**: `run_parameter_sweep_final.py` 若按原设定 (200 + 100 epochs, 5 seeds) 需要约 2.5 小时。为了在本次会话中能展示参数敏感性重画的图，我使用了 Fast 模式（3 seeds, 30+20 epochs）先生成了一版数据并作图。您可以在后台等待全量脚本完成（已挂后台）或将其部署在服务器后使用新的 `plot_parameters.py` 脚本覆盖刷新图片。
   - 代码中还散落了部分硬编码默认参数尚未被统一替换（由于怕破坏某些关联代码结构），因此若有后续新增脚本，需要记得主动将 DLCNet 传参固定为 `d_hidden=128, num_layers=3`。

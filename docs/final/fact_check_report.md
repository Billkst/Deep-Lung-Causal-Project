# 事实核对报告 (Fact Check Report)

生成时间: 2026-03-09

## A. 真正的最终 tuned SOTA checkpoint 是哪个?

### 核对结果:

**✅ 当前使用的 checkpoint 是正确的最终 SOTA**

- **Checkpoint 文件**: `dlc_final_sota_s_seed_43.pth` (位于项目根目录)
- **创建时间**: 2026-02-04 12:50
- **架构验证**: 
  - `d_hidden = 128` ✅
  - `num_layers = 3` ✅ (通过检查 hypergraph_nn.conv_layers 确认有3层)
- **对应的训练脚本**: `src/run_parameter_sweep_128x3.py` 和 `src/run_parameter_sweep_128x3_fast.py`
- **关键超参数**: 
  - `lambda_cate = 2.0`
  - `lambda_hsic = 0.1`

### 其他可用的 SOTA checkpoints (同一批次训练):

- `results/dlc_final_sota_s_seed_42.pth` (2026-02-04 12:47)
- `results/dlc_final_sota_s_seed_44.pth` (2026-02-04 12:52)
- `results/dlc_final_sota_s_seed_45.pth` (2026-02-04 12:56)
- `results/dlc_final_sota_s_seed_46.pth` (2026-02-04 12:59)

### 与 benchmark 的一致性:

**✅ `run_benchmark_fair.py` 默认使用正确的 checkpoint**

- 第48行: `parser.add_argument("--dlc-weights", type=str, default="dlc_final_sota_s_seed_43.pth")`
- 第49-50行: `parser.add_argument("--dlc-d-hidden", type=int, default=128)` 和 `parser.add_argument("--dlc-num-layers", type=int, default=3)`
- 架构参数与 checkpoint 完全匹配

### 结论:

✅ **当前 benchmark 使用的是正确的最终 tuned SOTA checkpoint (128×3)**

---

## B. 当前 `run_benchmark_fair.py` 到底在做什么?

### 核对结果:

**这是一个 Pre-trained Deployment Benchmark (部署态基准测试)**

### DLC 的处理方式:

1. **不重新训练**: 第155-172行直接加载预训练权重
2. **Training Time = 0.0**: 第188行硬编码 `"Training Time (s)": 0.0, # Pre-trained`
3. **只测推理时间**: 第174-176行只计时 `predict_proba()` 调用
4. **不做阈值搜索**: 第180行使用固定阈值0.5,注释明确说明 "Threshold should ideally be fixed at 0.5 or searched on train, but not on test"

### Baseline 的处理方式:

1. **完整训练**: 第92-132行对每个 baseline 执行 `fit()` 并计时
2. **分离计时**: 
   - Training Time: 包含 `fit()` 的完整时间
   - Inference Time: 只包含 `predict_proba()` 的时间
3. **GPU 同步**: 第39-42行定义了 `sync_time()` 函数确保 CUDA 同步

### 问题识别:

**❌ DLC 的 "Training Time = 0" 是误导性的**

- 这不是"真的为0",而是"没有测量"
- 当前 benchmark 是 **deployment inference benchmark** (部署推理基准)
- 但表格中 "Training Time = 0" 会让读者误以为 DLC 训练不需要时间
- **必须补充**: DLC 真实训练时间的单独测量

### 结论:

✅ 当前 benchmark 是 **pre-trained deployment benchmark**  
❌ 缺少 DLC 真实训练时间的测量  
✅ 推理时间测量协议是公平的

---

## C. 当前参数讨论的三张图是否与最新 csv 一致?

### 核对结果:

**✅ 图表数据与 CSV 完全一致**

### 图1: `fig_lambda_cate_tradeoff.png/pdf/svg`

- **数据源**: `lambda_cate_sweep_results.csv` (最新生成于 2026-03-09 08:19)
- **数据点**: 5个 (lambda_cate = 0.0, 0.5, 1.0, 2.0, 5.0)
- **红色高亮点**: lambda_cate = 2.0 (当前默认值)
- **图表生成时间**: 2026-03-09 08:47
- **一致性**: ✅ 完全一致

### 图2: `fig_lambda_hsic_tradeoff.png/pdf/svg`

- **数据源**: `lambda_hsic_sweep_results.csv` (最新生成于 2026-03-09 07:11)
- **数据点**: 5个 (lambda_hsic = 0.0, 0.01, 0.1, 1.0, 10.0)
- **红色高亮点**: lambda_hsic = 0.1 (当前默认值)
- **图表生成时间**: 2026-03-09 08:47
- **一致性**: ✅ 完全一致

### 图3: `fig_lambda_hsic_auc.png/pdf/svg`

- **数据源**: `lambda_hsic_sweep_results.csv` (同上)
- **红色高亮点**: lambda_hsic = 0.1
- **图表生成时间**: 2026-03-09 08:47
- **一致性**: ✅ 完全一致

### 红色高亮点的语义:

**⚠️ 需要澄清**: 红色点表示 "当前选择的默认值" (chosen default),而非 "统计最优点" (statistical best)

- 在 `plot_parameters.py` 第23-26行和第36-37行,红色点被标注但没有明确说明其含义
- 图注中应该明确写成 "Selected Default" 或 "Chosen Configuration"

### 数据分析:

**Lambda CATE (强敏感)**:
- 从 CSV 数据看: lambda_cate=5.0 在 Delta CATE (0.197) 和 AUC (0.792) 上都优于 2.0
- 但 PEHE 略高 (0.112 vs 0.118)
- **结论**: 2.0 是保守的折中选择,不是绝对最优

**Lambda HSIC (弱敏感)**:
- 从 CSV 数据看: 所有测试点的 AUC 在 0.786-0.787 范围内,差异 < 0.001
- Delta CATE 在 0.171-0.172 范围内,差异 < 0.001
- 标准差与均值差异同量级
- **结论**: lambda_hsic 在测试范围内影响极弱,0.1 是居中的温和默认值

### 结论:

✅ 图表数据与 CSV 完全一致  
⚠️ 需要修正: 红色点的语义说明  
⚠️ 需要修正: 文字结论不能过度解读 (尤其是 lambda_hsic)

---

## D. 当前仓库里哪些 md / png / csv / 脚本已经过时?

### 核对结果:

### 1. 过时的参数讨论文档 (128×4 口径):

**❌ `docs/parameter_discussion_material.md`**
- 第17行: 明确写着 `d_hidden=128, num_layers=4`
- 第70行: "中心区域(128维/4层)"
- 第77行: "Architecture: Hidden=128, Layers=4"
- **状态**: 完全过时,基于旧架构

**✅ `docs/parameter_discussion_material_revised.md`**
- 第71行: 已更新为 `d_hidden=128, num_layers=3`
- **状态**: 这是较新版本,但仍需进一步修正结论

### 2. 根目录散落的文件 (需要整理):

**结果文件 (应移入 results/final/)**:
- `lambda_cate_sweep_results.csv`
- `lambda_hsic_sweep_results.csv`
- `revised_benchmark_results.csv`
- `revised_benchmark_results.md`
- `architecture_results.csv`

**图表文件 (应移入 results/final/figures/)**:
- `fig_lambda_cate_tradeoff.png/pdf/svg`
- `fig_lambda_cate_auc.png/pdf/svg`
- `fig_lambda_hsic_tradeoff.png/pdf/svg`
- `fig_lambda_hsic_auc.png/pdf/svg`
- `architecture_tradeoff_scatter.png/pdf`
- `architecture_*_heatmap.png`

**说明文档 (应移入 docs/final/)**:
- `summary.md`
- `experiment_revision_notes.md`

### 3. 重复/临时文件 (可删除或归档):

**日志文件**:
- `benchmark_debug.log/pid`
- `benchmark_final.log/pid`
- `benchmark_run.log`
- `fast_sweep.log/pid`
- `run_sweep.log/pid`
- `wait_plot.log`
- `exp*.log` (多个实验日志)

**临时脚本**:
- `check_baseline_params.py`
- `check_params.py`
- `count_xgb.py`
- `debug_ite_gen.py`
- `inspect_checkpoint_layers.py`
- `inspect_weights.py`
- `extract_architecture_results.py`
- `analyze_results_quick.py`

### 4. 旧版 checkpoint (应归档到 results/archive/):

- `results/dlc_final_sota_tuned_sota_*.pth` (多个旧版本)
- `results/dlc_final_sota_k_seed_*.pth`
- `results/dlc_final_sota_m_seed_*.pth`
- `results/dlc_final_sota_n_seed_*.pth`
- `results/dlc_final_sota_p_seed_*.pth`
- `results/dlc_final_sota_q_seed_*.pth`

### 5. .history 目录 (可完全删除):

- `.history/` 包含大量历史版本,占用空间且无实际用途

### 结论:

❌ 过时文档: `docs/parameter_discussion_material.md` (128×4)  
✅ 较新文档: `docs/parameter_discussion_material_revised.md` (128×3,但需修正结论)  
⚠️ 根目录混乱: 大量结果文件、图表、日志散落  
⚠️ 旧版 checkpoint: results/ 下有大量旧版本需归档

---

## 总结

### 核心发现:

1. ✅ **Checkpoint 正确**: 当前使用的 `dlc_final_sota_s_seed_43.pth` 确实是 128×3 架构
2. ✅ **Benchmark 协议清晰**: 这是 deployment benchmark,不是 full training benchmark
3. ❌ **缺少训练时间**: 必须补充 DLC 真实训练时间的测量
4. ✅ **图表数据一致**: 三张参数图与最新 CSV 完全匹配
5. ⚠️ **结论需修正**: 不能过度解读 lambda_hsic 的影响,不能把 "default" 说成 "best"
6. ❌ **仓库混乱**: 根目录散落大量文件,需要系统性整理

### 下一步行动:

1. **补充 DLC 训练时间测量** (最高优先级)
2. **修正参数讨论文档** (原地更新 canonical 版本)
3. **清理仓库结构** (归档旧文件,整理目录)
4. **生成最终交付报告**

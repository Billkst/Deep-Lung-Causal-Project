
# Copilot 指南 — Deep-Lung-Causal (DLC)

## 项目概览
- 流水线：原始 TCGA 数据 -> 清洗/合并 -> 半合成 CSV -> 基线/DLC 模型 -> 结果/报告。
- 数据清洗与半合成生成在 [src/data_processor.py](src/data_processor.py)（`DataCleaner`、`SemiSyntheticGenerator`）。输出： [data/luad_synthetic_interaction.csv](data/luad_synthetic_interaction.csv)、[data/luad_synthetic_linear.csv](data/luad_synthetic_linear.csv)、[data/pancan_synthetic_interaction.csv](data/pancan_synthetic_interaction.csv)、[data/pancan_synthetic_linear.csv](data/pancan_synthetic_linear.csv)。

## 核心架构
- 基线模型在 [src/baselines/](src/baselines/)；所有模型实现 [src/baselines/base_model.py](src/baselines/base_model.py) 的 `BaseModel`（`fit`、`predict`、`predict_proba`、`evaluate`、`get_params`、`set_params`）。
- 通用工具（随机种子、预处理、GPU 显存监控、梯度累积、自动批量大小）在 [src/baselines/utils.py](src/baselines/utils.py)。建议使用 `set_global_seed(42)` 和 `preprocess_data` 的分割+标准化流程。
- DLC 核心模型在 [src/dlc/](src/dlc/)（`causal_vae.py`、`hypergraph_nn.py`、`dlc_net.py`）。

## 数据约定与预处理
- CSV 包含 `sampleID`、`Outcome_Label`，有时还有 `True_Prob`；建模前会移除 `sampleID`/`True_Prob`（见 [src/run_baselines_dlc.py](src/run_baselines_dlc.py)）。
- TransTEE：`Virtual_PM2.5` 以中位数二值化为 treatment `t`，并从协变量中移除；见 [src/run_baselines_dlc.py](src/run_baselines_dlc.py) 的 `DLCDataPreprocessor.prepare_transtee_data`。
- MOGONET：多视图拆分将 `Age`、`Gender`、`Virtual_PM2.5` 作为临床视图，其余为组学特征（同上）。
- 默认分割为分层 8:1:1（train/val/test），随机种子 42；见 [src/run_baselines_dlc.py](src/run_baselines_dlc.py) 的 `DLCDataLoader.split_data`。

## 关键可运行脚本（入口）
- 数据生成：运行 [src/data_processor.py](src/data_processor.py) 的 main 重建 [data/](data/) 下 CSV。
- DLC 数据集基线评估： [src/run_baselines_dlc.py](src/run_baselines_dlc.py)（输出 [results/baseline_metrics.json](results/baseline_metrics.json)）。
- 场景实验与预测： [src/run_experiment_scenarios.py](src/run_experiment_scenarios.py)（输出 [results/experiment_a_results.json](results/experiment_a_results.json)、[results/experiment_b_results.json](results/experiment_b_results.json) 与 PKL 预测）。
- 三层指标： [src/evaluate_three_tier_metrics.py](src/evaluate_three_tier_metrics.py) 与修正版 [src/evaluate_three_tier_metrics_fix.py](src/evaluate_three_tier_metrics_fix.py)（输出 [results/three_tier_metrics.json](results/three_tier_metrics.json)）。
- 混杂敏感性： [src/run_confounding_sensitivity.py](src/run_confounding_sensitivity.py) 与评估器 [src/evaluate_confounding_sensitivity.py](src/evaluate_confounding_sensitivity.py)（输出 [results/confounding_sensitivity_report.md](results/confounding_sensitivity_report.md)）。

## 测试与验证
- 测试在 [tests/](tests/)；覆盖数据处理、基线、DLC 模块和硬件适配。GPU 相关测试在 [tests/test_hardware_adaptation.py](tests/test_hardware_adaptation.py)，无 CUDA 时可跳过。
- 属性测试（可复现性、标准化、分层划分）见 [tests/test_baselines_properties.py](tests/test_baselines_properties.py) 与 [tests/test_data_processor_properties.py](tests/test_data_processor_properties.py)。

## 编辑指引
- 新增基线模型需完整实现 `BaseModel` 方法，并复用 [src/baselines/utils.py](src/baselines/utils.py) 的预处理与随机种子流程。
- 输出保持兼容 [reports/](reports/) 与 [results/](results/) 的既有 JSON/报告格式。

# Workflow & Documentation Rules

**核心原则：代码与文档必须同步。**
任何涉及代码变更、文件生成或任务完成的操作，只有在更新了相关文档后，才视为“完成 (DONE)”。

## 1. 强制文档更新流程 (Mandatory Documentation Update)

在执行完任何 Task 或 Sub-task 后，必须**自动**执行以下两个动作，无需用户再次提醒：

### A. 更新工作日志 ([docs/工作过程.md](docs/工作过程.md))
* **动作：** 以 Markdown 追加模式写入日志。
* **时机：** 代码执行成功或测试通过后立即执行。
* **内容要求：**
	 * 记录操作时间、操作者 (Kiro)。
	 * 记录关键变更（新增了什么类、修改了什么逻辑）。
	 * 记录关键数据统计（如样本量、相关系数、测试通过率）。
	 * **严禁**只写“任务完成”，必须包含具体的执行结果数据。

### B. 更新项目结构索引 ([docs/项目结构说明.md](docs/项目结构说明.md))
* **动作：** 检查当前文件树。
* **时机：** 当新创建了文件（.py, .ipynb, .csv, .txt）或新建了文件夹时。
* **内容要求：**
	 * 将新文件添加到目录树结构中。
	 * 简要说明新文件的作用。
	 * 保持文档中的文件树与实际文件系统实时一致。

## 2. 完成标准 (Definition of Done)
* **Check 1:** 代码已通过测试。
* **Check 2:** [docs/工作过程.md](docs/工作过程.md) 已追加最新记录。
* **Check 3:** [docs/项目结构说明.md](docs/项目结构说明.md) 已反映最新文件结构。
* **只有当上述 3 点全部满足时，才可向用户报告“任务已完成”。**

# 环境配置规则 (Environment Rules)

本项目依赖一个指定路径的 Conda 环境，路径为：`/home/UserData/ljx/conda_envs/dlc_env`

## ⚠️ 核心执行指令
AI Agent 在执行任何 Python 脚本、安装依赖或运行命令时，**必须严格遵守**以下规则：

1. **路径参数**：
	由于环境是指定路径的，在使用 conda 命令时必须使用 `-p` 参数，**不能**使用 `-n`。

2. **代码执行格式**（任选其一，推荐第一种）：
	- **推荐**：`conda run -p /home/UserData/ljx/conda_envs/dlc_env python <你的脚本.py>`
	- **备选**：`source activate /home/UserData/ljx/conda_envs/dlc_env && python <你的脚本.py>`

3. **禁止行为**：
	- ❌ **严禁**直接输入 `python script.py`（这会调用系统 Python）。
	- ❌ **严禁**直接输入 `pip install`（这会将包安装到错误的地方）。
	- 必须使用 `conda run -p /home/UserData/ljx/conda_envs/dlc_env pip install ...`。

4. **环境验证**：
	- 如果你怀疑环境是否正确，请先执行 `which python` 或在代码中打印 `sys.executable`，确保其输出包含 `/home/UserData/ljx/conda_envs/dlc_env`。

# 语言与沟通规范 (Language Rules)

## 1. 语言要求
- **必须始终使用简体中文**回复用户。
- 无论用户输入的是英文代码、报错信息还是英文问题，你的解释和对话都必须翻译成**简体中文**。
- **例外情况**：专有名词、技术术语（如 `PyTorch`, `Tensor`, `DataFrame`, `forward propagation`）以及代码片段，请保留英文原文，不要强行翻译，以免产生歧义。

## 2. 回复风格
- 简洁明了，逻辑清晰。
- 作为专业深度学习开发伙伴，解释问题时直击重点。

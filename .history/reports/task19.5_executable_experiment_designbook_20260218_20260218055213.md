# Task19.5 可执行实验设计书（V1，2026-02-18）

## 1. 目标与范围
- 目标：在现有严格评估口径下，验证是否能将 Full DLC 从 `global 4/6` 推进到 `>=5/6`，并优先检查分类短板（AUC/Acc 相对 `w/o HGNN` 是否转正）。
- 评估口径：沿用 `src/run_rigorous_ablation.py` 的 strict ablation + joint selection。
- 数据与分割：沿用脚本内置流程（PANCAN+LUAD 训练池，LUAD 测试，seed 驱动分割）。
- 非目标：本轮不变更数据构造、不新增模型结构，不修改指标定义。

## 2. 实验矩阵与变量表

| 变量类别 | 变量名 | 取值/设置 | 作用 | 备注 |
|---|---|---|---|---|
| 自变量（候选） | Candidate-A | `c15`（当前冻结最优配置） | 对照最优上限 | 无蒸馏 |
| 自变量（候选） | Candidate-B | `g02`（双教师蒸馏） | 验证蒸馏对分类短板修复能力 | `fhp+ghp` 开启 |
| 固定变量 | Seeds-Stage1 | `[42]` | 快速筛查 | 低成本 |
| 固定变量 | Seeds-Stage2 | `[42,43,44,45,46]` | 稳健复核 | 仅 Stage1 触发后执行 |
| 固定变量 | strict_ablation | `True` | 统一公平口径 | 与既往一致 |
| 固定变量 | joint_selection | `True` | 多目标联合选择 | 与既往一致 |
| 固定变量 | 指标集 | AUC/Acc/F1/PEHE/Delta CATE/Sensitivity | 判定依据 | 6 指标 |

### 候选参数明细

#### Candidate-A（`c15_baseline`）
- `joint_w_auc=1.05`, `joint_w_acc=1.40`, `joint_w_f1=1.75`
- `joint_w_cate=0.82`, `joint_w_pehe=0.92`, `joint_w_sens=0.88`
- `constraint_pehe_max=0.11`, `constraint_sens_max=0.11`, `constraint_cate_min=0.11`, `constraint_penalty=205`
- `sota_lambda_pred=5.5`, `sota_lambda_hsic=0.009`, `sota_lambda_cate=5.0`, `sota_lambda_ite=9.0`, `sota_lambda_sens=0.009`

#### Candidate-B（`g02_dual_teacher`）
- 在 Candidate-A 基础上启用：
- `--fhp-enable-distill`, `--fhp-distill-weight=0.55`, `--fhp-teacher-epochs=35`
- `--ghp-enable-dual-teacher`, `--ghp-teacher-mix=0.60`, `--ghp-teacher2-epochs=35`

## 3. 对照组设计
- 主实验组：`Full DLC (SOTA)`。
- 必备对照组（脚本自动训练）：`w/o HGNN`、`w/o VAE`、`w/o HSIC`。
- 对照比较规则：
  - `pairwise_wins`：Full 对每个消融模型 6 指标逐项胜出计数；
  - `global_wins`：Full 是否同时优于三个消融模型（按指标逐项）。

## 4. 停机准则（Stop Criteria）

### Stage1（快筛）停机
- 若任一候选满足：
  - `AUC gap vs w/o HGNN > 0` 且 `Acc gap vs w/o HGNN > 0`，并且
  - `global_wins >= 5`，
  则进入 Stage2（5-seed 复核）。

- 若所有候选均不满足，则立即停止本轮并输出结论：
  - `stop_reason = "No candidate passed cls-gate + global>=5 under current protocol"`。

### Stage2（稳健复核）终止
- 仅复核 Stage1 排名第一候选。
- 若 Stage2 `global_wins < 6`，则判定“未达成终极目标（6/6）”，结束本轮。
- 若 Stage2 `global_wins = 6`，则判定“达成”。

## 5. 日志模板（执行与复盘）

### 5.1 结构化日志字段（JSONL）
每条记录包含：
- `timestamp`
- `run_id`
- `stage`
- `candidate`
- `seeds`
- `status`（`started/success/failed/stopped`）
- `global_wins`
- `pairwise_wins`
- `auc_gap_vs_hgnn`
- `acc_gap_vs_hgnn`
- `cls_gate`
- `stop_reason`
- `raw_path` / `report_path` / `log_path`

### 5.2 人工速记模板（Markdown）
| 时间 | Stage | Candidate | 结果 | 关键指标 | 结论 |
|---|---|---|---|---|---|
| YYYY-MM-DD HH:MM | stage1 | c15_baseline | success | global=4/6, cls_gate=0 | 未触发晋级 |

## 6. 执行命令
```bash
conda run -p /home/UserData/ljx/conda_envs/dlc_env python src/run_task19_5_designbook_execution.py
```

## 7. 产物定义
- 结构化结果：`results/task19.5_designbook_execution_<timestamp>.json`
- 过程日志：`logs/task19.5_designbook_execution_<timestamp>.jsonl`
- 摘要报告：`reports/task19.5_designbook_execution_<timestamp>.md`

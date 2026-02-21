# Task19.5 可执行实验设计书 V3（AUC 过零优先，2026-02-18）

## 1. 目标
- 第一目标：先让 `AUC gap vs w/o HGNN` 过零（`> 0`）。
- 第二目标：在不牺牲过多全局表现前提下，争取 `global_wins >= 4`。
- 第三目标：若出现可行候选，再考虑下一轮扩展验证。

## 2. 设计原则
- 仅保留 V2 Top3 邻域（`v2a/v2d/v2f`）进行小范围重构，避免盲目扩搜。
- 明确从“6/6 终局目标”拆成“先修 AUC 短板”的阶段目标。
- 运行采用快速参数（短 epoch）保证一轮可收口。

## 3. 变量表

| 类别 | 变量 | 取值 |
|---|---|---|
| 自变量 | 候选配置 | 4 组（v3a~v3d） |
| 固定变量 | seeds | `[42]`（快速轮） |
| 固定变量 | 协议 | strict ablation + joint selection |
| 固定变量 | 指标 | AUC/Acc/F1/PEHE/Delta CATE/Sensitivity |

### 候选定义
- `v3a_auc_push_base`：AUC 权重显著提升、约束适度放松（无蒸馏）
- `v3b_auc_push_fhp`：在 v3a 基础上启用 FHP（轻蒸馏）
- `v3c_auc_push_ghp`：在 v3a 基础上启用 GHP（双教师）
- `v3d_auc_extreme`：更强 AUC 导向（无蒸馏）

## 4. 对照组
- 主组：`Full DLC (SOTA)`
- 对照：`w/o HGNN`, `w/o VAE`, `w/o HSIC`

## 5. 停机准则（V3）
- `auc_gate`: `AUC gap vs w/o HGNN > 0`
- `acc_soft_gate`: `Acc gap vs w/o HGNN >= -0.005`
- 命中候选定义：`auc_gate=1` 且 `acc_soft_gate=1` 且 `global_wins >= 4`
- 若无命中候选：本轮停机并输出“未修复 AUC 短板”。

## 6. 日志模板（JSONL）
字段：
- `timestamp`, `run_id`, `candidate`, `status`
- `global_wins`, `pairwise_wins`
- `auc_gap_vs_hgnn`, `acc_gap_vs_hgnn`
- `auc_gate`, `acc_soft_gate`, `hit`
- `raw_path`, `report_path`, `log_path`

## 7. 执行命令（screen）
```bash
screen -L -Logfile logs/task19.5_designbook_v3_screen_<ts>.log -dmS task195_designbook_v3 \
  conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
  python src/run_task19_5_designbook_execution_v3.py
```

## 8. 产物
- `results/task19.5_designbook_v3_execution_<timestamp>.json`
- `reports/task19.5_designbook_v3_execution_<timestamp>.md`
- `logs/task19.5_designbook_v3_execution_<timestamp>.jsonl`

# Task19.5 可执行实验设计书 V2（2026-02-18）

## 1. 目标
- 在保持严格口径（strict ablation + joint selection）不变前提下，执行“分类优先”V2 冲刺。
- 目标优先级：
  1) 修复 `AUC/Acc vs w/o HGNN` 双负差；
  2) `global_wins >= 5`；
  3) 若可行，冲击 `6/6`。

## 2. 范围与非目标
- 范围：仅调整损失权重、联合选择权重、约束阈值、蒸馏开关与混合比。
- 非目标：不改数据构建、不改模型结构、不改指标定义。

## 3. 变量表

| 类别 | 变量 | 取值 |
|---|---|---|
| 自变量 | 候选配置 | 8 组（V2-A~V2-H） |
| 固定变量 | Stage1 seeds | `[42]` |
| 固定变量 | Stage2 seeds | `[42,43,44,45,46]`（仅晋级后执行） |
| 固定变量 | 评估指标 | AUC, Acc, F1, PEHE, Delta CATE, Sensitivity |
| 固定变量 | 训练协议 | strict ablation + joint selection |

### 候选组（Stage1）
- V2-A (`v2a_cls_bal`)：分类增强基线（无蒸馏）
- V2-B (`v2b_cls_hard`)：更强分类权重（无蒸馏）
- V2-C (`v2c_fhp_light`)：单教师蒸馏（轻）
- V2-D (`v2d_fhp_mid`)：单教师蒸馏（中）
- V2-E (`v2e_ghp_mix55`)：双教师蒸馏（mix=0.55）
- V2-F (`v2f_ghp_mix70`)：双教师蒸馏（mix=0.70）
- V2-G (`v2g_acc_max`)：Acc 极限导向
- V2-H (`v2h_auc_guard`)：AUC 保护导向

## 4. 对照组设计
- 主实验组：`Full DLC (SOTA)`。
- 对照组：`w/o HGNN`、`w/o VAE`、`w/o HSIC`（由主脚本统一训练评估）。

## 5. 停机准则（V2 强化）

### 5.1 分类硬门槛（Strict cls-gate）
满足以下同时成立才算通过：
- `AUC gap vs w/o HGNN >= 0.003`
- `Acc gap vs w/o HGNN >= 0.003`

### 5.2 Stage1 晋级条件
候选需同时满足：
- Strict cls-gate 通过；
- `global_wins >= 5`；
- `pairwise_wins['w/o HGNN'] >= 5`。

若无候选满足，立即停机并输出结论。

### 5.3 Stage2 终审条件
- 对 Stage1 Top1 进行 5-seed 复核。
- 若 `global_wins == 6`：判定达成。
- 否则：判定本轮未达成并停机。

## 6. 日志模板

### 6.1 结构化 JSONL 字段
- `timestamp`, `run_id`, `stage`, `candidate`, `seeds`, `status`
- `global_wins`, `pairwise_wins`, `auc_gap_vs_hgnn`, `acc_gap_vs_hgnn`
- `cls_gate_strict`, `stop_reason`
- `raw_path`, `report_path`, `log_path`

### 6.2 人工摘要模板（Markdown）
| 时间 | Stage | Candidate | global | cls_gate_strict | AUC/Acc gap(HGNN) | 结论 |
|---|---|---:|---:|---:|---|---|

## 7. 执行命令（screen）
```bash
screen -L -Logfile logs/task19.5_designbook_v2_screen_<ts>.log -dmS task195_designbook_v2 \
  conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
  python src/run_task19_5_designbook_execution_v2.py
```

## 8. 预期产物
- `results/task19.5_designbook_v2_execution_<timestamp>.json`
- `reports/task19.5_designbook_v2_execution_<timestamp>.md`
- `logs/task19.5_designbook_v2_execution_<timestamp>.jsonl`

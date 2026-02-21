# Task19.5 可执行实验设计书 V4（最小机制改造版，2026-02-18）

## 1. 目标
- 在保持主流程不重构的前提下，验证“机制级”最小改造是否修复分类短板：
  - `AUC gap vs w/o HGNN > 0`
  - `Acc gap vs w/o HGNN >= 0`

## 2. 机制改造（最小集）
- 新增 checkpoint 选择约束：
  - `constraint_auc_min`（验证集 AUC 下限）
  - `constraint_acc_min`（验证集 Acc 下限）
- 新增训练后期分类强化：
  - `pred_boost_start_epoch`
  - `pred_boost_factor`

## 3. 实验变量表
| 类别 | 变量 | 取值 |
|---|---|---|
| 自变量 | 候选配置 | 3 组（v4a/v4b/v4c） |
| 固定变量 | seeds | `[42]` |
| 固定变量 | 协议 | strict ablation + joint selection |
| 固定变量 | 基础指标 | AUC/Acc/F1/PEHE/Delta CATE/Sensitivity |

### 候选
- `v4a_mech_base`：机制改造基础版（中等 pred boost）
- `v4b_mech_boost`：更强 pred boost
- `v4c_mech_distill`：机制改造 + 单教师蒸馏

## 4. 对照组
- 主组：`Full DLC (SOTA)`
- 对照：`w/o HGNN`, `w/o VAE`, `w/o HSIC`

## 5. 停机准则
- 命中条件：
  - `auc_gap_vs_hgnn > 0`
  - `acc_gap_vs_hgnn >= 0`
  - `global_wins >= 4`
- 无命中则本轮停机并输出“机制改造未越过分类边界”。

## 6. 执行命令（screen）
```bash
screen -L -Logfile logs/task19.5_designbook_v4_screen_<ts>.log -dmS task195_designbook_v4 \
  conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
  python src/run_task19_5_designbook_execution_v4.py
```

## 7. 预期产物
- `results/task19.5_designbook_v4_execution_<timestamp>.json`
- `reports/task19.5_designbook_v4_execution_<timestamp>.md`
- `logs/task19.5_designbook_v4_execution_<timestamp>.jsonl`

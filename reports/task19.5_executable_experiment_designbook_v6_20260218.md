# Task19.5 可执行实验设计书 V6（自适应 alpha，2026-02-18）

## 目标
- 在 V5 双头解耦基础上，通过验证集自适应选择融合系数 alpha，优先保障 Acc，再推动 AUC 过零。

## 机制
- 启用 `--v6-enable-adaptive-alpha`。
- 在每次验证中从 `v6_alpha_grid` 搜索最优 alpha；可设置 `v6_acc_floor` 作为筛选约束。
- 最优 alpha 随最佳 checkpoint 一并固化并用于推理。

## 候选
- `v6a_acc_guard`: `acc_floor=0.82`, `alpha_grid=0.0,0.1,0.2,0.3,0.4`
- `v6b_balanced`: `acc_floor=0.80`, `alpha_grid=0.0,0.2,0.4,0.6`

## 命中条件
- `auc_gap_vs_hgnn > 0`
- `acc_gap_vs_hgnn >= 0`
- `global_wins >= 4`

## 执行
```bash
screen -L -Logfile logs/task19.5_designbook_v6_screen_<ts>.log -dmS task195_designbook_v6 \
  conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
  python src/run_task19_5_designbook_execution_v6.py
```

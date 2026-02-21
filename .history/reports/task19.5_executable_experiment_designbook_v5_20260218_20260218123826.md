# Task19.5 可执行实验设计书 V5（双头解耦最小实现，2026-02-18）

## 1. 目标
- 通过“分类辅助头 + 因果主头”解耦，验证是否能突破 AUC/Acc 对 `w/o HGNN` 的负差边界。

## 2. 机制实现
- 在 `Full DLC` 训练中新增可选分类辅助头（输入 `Z_effect`，输出分类 logit）。
- 训练时引入辅助头 BCE 损失（权重 `v5_head_weight`）。
- 预测时将辅助头概率与主头概率加权融合（系数 `v5_head_alpha`）。

## 3. 候选
- `v5a_detach_bal`: detach=True, alpha=0.35, weight=0.6
- `v5b_detach_strong`: detach=True, alpha=0.50, weight=0.8
- `v5c_nodetach`: detach=False, alpha=0.40, weight=0.6

## 4. 对照
- 主组：`Full DLC (SOTA)`
- 对照：`w/o HGNN`, `w/o VAE`, `w/o HSIC`

## 5. 停机准则
- 命中条件：
  - `auc_gap_vs_hgnn > 0`
  - `acc_gap_vs_hgnn >= 0`
  - `global_wins >= 4`
- 未命中则停机并输出边界未突破。

## 6. 执行命令
```bash
screen -L -Logfile logs/task19.5_designbook_v5_screen_<ts>.log -dmS task195_designbook_v5 \
  conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
  python src/run_task19_5_designbook_execution_v5.py
```

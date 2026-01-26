# Phase 7: 性能冲刺与科学验证任务清单

**协作模式：** 本地最强模型编写 → 远程 Kiro 审计 → 用户执行  
**Kiro 职责：** 规划任务、审查代码、验收结果  
**用户职责：** 使用本地模型编写代码、执行测试

---

## 任务概览

| 任务编号 | 任务名称 | 状态 | 负责方 | 验收标准 |
|---------|---------|------|--------|---------|
| 7.1 | 高级因果指标模块 | [✓] | 用户编写 | 代码已就绪，待执行 |
| 7.2 | 贝叶斯超参数搜索 | [✓] | 用户编写 | 代码已就绪，待执行 |
| 7.3 | 全量数据 SOTA 训练 | [✓] | 用户编写 | 代码已就绪，待执行 |

---

## Task 7.1: 实现高级因果指标模块

### 目标
创建 `src/dlc/metrics.py`，实现三个核心因果推断评估函数。

### 功能需求

#### 1. `compute_pehe(y_true_ite, y_pred_ite) -> float`
- **输入：**
  - `y_true_ite`: 真实 ITE (Individual Treatment Effect)，形状 `(n_samples,)`
  - `y_pred_ite`: 预测 ITE，形状 `(n_samples,)`
- **输出：** PEHE (Precision in Estimation of Heterogeneous Effect) 标量值
- **公式：** `sqrt(mean((y_true_ite - y_pred_ite)^2))`
- **约束：**
  - 必须支持 NumPy 数组和 PyTorch Tensor
  - 必须处理 NaN 值（跳过或报错）

#### 2. `compute_cate(model, X, treatment_col_idx) -> np.ndarray`
- **输入：**
  - `model`: 训练好的 DLC 模型实例
  - `X`: 特征矩阵，形状 `(n_samples, n_features)`
  - `treatment_col_idx`: 治疗变量在 X 中的列索引
- **输出：** CATE (Conditional Average Treatment Effect)，形状 `(n_samples,)`
- **方法：** 反事实干预法 (Counterfactual Intervention)
  1. 计算 `y_pred_1 = model.predict(X with treatment=1)`
  2. 计算 `y_pred_0 = model.predict(X with treatment=0)`
  3. 返回 `y_pred_1 - y_pred_0`
- **约束：**
  - 必须保持其他特征不变
  - 必须支持批量计算（避免循环）

#### 3. `compute_sensitivity_score(model, X, confounder_idx, epsilon=0.1) -> float`
- **输入：**
  - `model`: 训练好的 DLC 模型
  - `X`: 特征矩阵
  - `confounder_idx`: 混杂变量的列索引（如年龄）
  - `epsilon`: 扰动幅度（默认 0.1，即 10% 变化）
- **输出：** 敏感度分数（标量，越小越好）
- **方法：**
  1. 计算原始预测 `y_pred_original = model.predict(X)`
  2. 扰动混杂变量：`X_perturbed = X.copy(); X_perturbed[:, confounder_idx] *= (1 + epsilon)`
  3. 计算扰动后预测 `y_pred_perturbed = model.predict(X_perturbed)`
  4. 返回 `mean(abs(y_pred_perturbed - y_pred_original))`
- **约束：**
  - 必须支持多个混杂变量（传入列表）
  - 必须归一化到 [0, 1] 范围

### 测试需求（TDD）

#### 单元测试 (`tests/test_metrics.py`)
- `test_pehe_perfect_prediction()`: 完美预测时 PEHE = 0
- `test_pehe_constant_error()`: 恒定误差时 PEHE = 误差值
- `test_cate_binary_treatment()`: 二元治疗变量的 CATE 计算
- `test_sensitivity_zero_perturbation()`: 零扰动时敏感度 = 0

#### 属性测试 (`tests/test_metrics_properties.py`)
- **Property 1 (非负性):** `PEHE >= 0` 对所有输入成立
- **Property 2 (对称性):** `PEHE(a, b) == PEHE(b, a)`
- **Property 3 (单调性):** 扰动幅度增大时，敏感度分数单调递增

### 验收标准
- [ ] 所有单元测试通过
- [ ] 所有属性测试通过（至少 100 个随机样本）
- [ ] 代码通过 `mypy` 类型检查
- [ ] 函数文档字符串完整（包含公式和示例）

---

## Task 7.2: 实现贝叶斯超参数搜索

### 目标
创建 `src/dlc/tune.py`，使用 Optuna 框架实现自动超参数优化。

### 功能需求

#### 核心函数：`tune_dlc_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50) -> dict`

**搜索空间定义：**
```python
{
    'd_hidden': IntUniformDistribution(32, 256),
    'num_layers': IntUniformDistribution(1, 4),
    'lambda_hsic': LogUniformDistribution(0.1, 10.0),
    'lr': LogUniformDistribution(1e-5, 1e-2),
    'batch_size': CategoricalDistribution([32, 64, 128]),
    'dropout': UniformDistribution(0.0, 0.5)
}
```

**目标函数：**
- **主目标：** 最大化验证集 AUC
- **硬约束：** HSIC 独立性损失 < 0.05（违反则返回 -inf）
- **软约束：** 训练时间 < 10 分钟（超时则提前停止）

**输出格式：**
```python
{
    'best_params': {...},
    'best_auc': 0.85,
    'best_hsic': 0.03,
    'optimization_history': [...],
    'total_time': 3600.0
}
```

### 实现要点
1. **Early Stopping:** 使用 Optuna 的 `MedianPruner`，在验证 AUC 连续 5 轮不提升时剪枝
2. **并行化:** 支持 `n_jobs` 参数（默认 -1，使用所有 CPU 核心）
3. **可视化:** 自动保存优化历史图（`results/optuna_history.png`）
4. **断点续传:** 支持从 SQLite 数据库恢复之前的搜索状态

### 测试需求

#### 集成测试 (`tests/test_tune.py`)
- `test_tune_runs_successfully()`: 在小数据集上运行 5 个 trial
- `test_tune_respects_hsic_constraint()`: 验证所有最佳参数满足 HSIC < 0.05
- `test_tune_saves_results()`: 验证结果文件正确保存

#### 性能测试
- 在 LUAD 数据集（~500 样本）上，50 个 trial 应在 30 分钟内完成

### 验收标准
- [ ] 成功运行 50 个 trial 并输出最佳参数
- [ ] 最佳 AUC 比默认参数提升至少 3%
- [ ] 生成优化历史可视化图
- [ ] 代码支持断点续传（手动中断后可恢复）

---

## Task 7.3: 全量数据 SOTA 训练

### 目标
创建 `src/run_final_sota.py`，使用 Task 7.2 的最佳参数在 PANCAN 全量数据上训练最终模型。

### 功能需求

#### 训练流程
1. **数据加载：** 使用 `data/PANCAN/` 全量数据（~10,000 样本）
2. **参数加载：** 从 `results/best_hyperparameters.json` 读取最佳参数
3. **模型训练：** 使用最佳参数训练 DLC 模型（最多 200 epochs）
4. **评估指标：**
   - 标准指标：AUC, Accuracy, F1-Score
   - 因果指标：PEHE, CATE 分布统计, 敏感度分数
5. **对比基线：** 与 XGBoost, TransTEE, MOGONET 的结果对比

#### 输出文件
1. **模型权重：** `results/dlc_final_sota.pth`
2. **评估报告：** `results/final_sota_report.md`（Markdown 格式）
3. **可视化：**
   - `results/cate_distribution.png`: CATE 分布直方图
   - `results/sensitivity_heatmap.png`: 各混杂变量的敏感度热图
   - `results/roc_curve_comparison.png`: 与基线的 ROC 曲线对比

### 报告内容要求

#### `results/final_sota_report.md` 必须包含：
1. **执行摘要：** 最佳 AUC, PEHE, 训练时间
2. **超参数表格：** 最佳参数的完整列表
3. **性能对比表：**
   ```markdown
   | 模型 | AUC | PEHE | 敏感度分数 |
   |------|-----|------|-----------|
   | DLC (Ours) | 0.87 | 0.12 | 0.05 |
   | XGBoost | 0.82 | 0.18 | 0.12 |
   | TransTEE | 0.84 | 0.15 | 0.08 |
   ```
4. **因果解释：** CATE 分布的医学意义解读
5. **局限性分析：** 模型在哪些子群体上表现不佳

### 测试需求

#### 冒烟测试 (`tests/test_final_sota.py`)
- `test_final_sota_runs_on_small_data()`: 在 100 样本子集上快速验证流程
- `test_final_sota_generates_all_outputs()`: 验证所有输出文件存在

### 验收标准
- [ ] 模型在 PANCAN 全量数据上训练成功
- [ ] PEHE < 0.15（低于所有基线）
- [ ] AUC > 0.85（高于 XGBoost 基线）
- [ ] 敏感度分数 < 0.1（表明混杂控制有效）
- [ ] 生成完整的 Markdown 报告和所有可视化图
- [ ] 报告已追加到 `docs/工作过程.md`

---

## 协作流程说明

### 用户侧工作流
1. **编写代码：** 使用本地最强模型（如 Claude Opus）编写上述模块
2. **提交审计：** 将代码文件路径发送给 Kiro
3. **执行测试：** 根据 Kiro 的审计意见修改后，在本地执行测试
4. **报告结果：** 将测试输出和生成的报告发送给 Kiro 验收

### Kiro 审计检查清单
- [ ] 代码符合 PEP 8 规范
- [ ] 函数签名与任务文档一致
- [ ] 包含完整的类型注解
- [ ] 包含详细的文档字符串（含公式和示例）
- [ ] 测试覆盖率 > 80%
- [ ] 无明显的性能瓶颈（如不必要的循环）
- [ ] 错误处理完善（边界情况、异常输入）

---

## 时间估算

| 任务 | 预计时间 | 依赖关系 |
|------|---------|---------|
| 7.1 | 2-3 小时 | 无 |
| 7.2 | 3-4 小时 | 依赖 7.1 |
| 7.3 | 4-6 小时 | 依赖 7.2 |
| **总计** | **9-13 小时** | 顺序执行 |

---

## 成功标准

Phase 7 完成的标志：
1. ✅ 所有测试通过（单元测试 + 属性测试 + 集成测试）
2. ✅ `results/final_sota_report.md` 生成并包含完整数据
3. ✅ PEHE 指标优于所有基线模型
4. ✅ `docs/工作过程.md` 已更新最新进展
5. ✅ `docs/项目结构说明.md` 已反映新增文件

---

**下一步：** 请用户使用本地模型编写 `src/dlc/metrics.py`，完成后提交给 Kiro 审计。

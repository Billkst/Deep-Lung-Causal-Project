# 训练可视化规则

## 核心要求

所有训练脚本必须同时满足以下三点：

### 1. 进度条可视化
- 使用 `tqdm` 显示 epoch 进度
- 进度条输出到 `stderr`：`tqdm(..., file=sys.stderr)`
- 显示格式：`desc="Training", ncols=80`

### 2. 实时日志输出
- 日志文件路径：`logs/{script_name}.log`
- 使用行缓冲模式：`open(log_file, 'w', buffering=1)`
- 每次写入后立即 `flush()`
- 同时输出到终端和文件

### 3. 关键指标打印
- 每 N 个 epoch 打印一次（N=10 或 20）
- 必须包含：Epoch 编号、Loss 值
- 可选：学习率、验证指标

## 标准实现模板

```python
import sys
from tqdm import tqdm

# 日志设置
log_file = open('logs/training.log', 'w', buffering=1)

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

# 训练循环
for epoch in tqdm(range(epochs), desc="Training", ncols=80, file=sys.stderr):
    # 训练代码
    loss = train_one_epoch()
    
    if (epoch + 1) % 20 == 0:
        log(f"Epoch {epoch+1}: Loss={loss:.4f}")

log_file.close()
```

## 禁止行为

❌ 不使用 `python -u` 参数运行
❌ 依赖 Python logging 模块（缓冲问题）
❌ 只输出到终端或只输出到文件
❌ 训练完成后才写入日志
❌ 使用 `print()` 不加 `flush=True`

## 验证方法

训练过程中执行：
```bash
tail -f logs/training.log
```

应该能实时看到日志更新。

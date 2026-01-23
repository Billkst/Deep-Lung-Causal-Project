"""
基础设施工具函数

提供全局随机种子设置、通用数据预处理功能和 GPU 显存监控
"""

import random
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    设置全局随机种子，确保可复现性
    
    该函数会设置以下库的随机种子：
    - Python 内置 random 模块
    - NumPy
    - PyTorch (如果已安装)
    
    Args:
        seed: 随机种子，默认 42
        
    Example:
        >>> set_global_seed(42)
        >>> # 所有后续的随机操作都将是可复现的
    
    Requirements:
        - 12.1: 在所有随机操作中使用全局随机种子 42
        - 12.2: 固定 NumPy 随机种子为 42
        - 12.3: 固定 PyTorch 随机种子为 42（如果使用）
        - 12.4: 固定 Python 内置 random 模块种子为 42
    """
    # 设置 Python 内置 random 模块种子
    random.seed(seed)
    
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    
    # 尝试设置 PyTorch 随机种子（如果已安装）
    try:
        import torch
        torch.manual_seed(seed)
        
        # 如果有 CUDA，也设置 CUDA 随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保 CUDA 操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch 未安装，跳过
        pass


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    数据预处理标准流程
    
    执行以下操作：
    1. 特征标准化 (StandardScaler) - 可选
    2. 分层划分训练集/测试集 (Stratified Split)
    
    Args:
        X: 特征矩阵，shape (n_samples, n_features)
        y: 标签向量，shape (n_samples,)
        test_size: 测试集比例，默认 0.2 (20%)
        random_state: 随机种子，默认 42
        stratify: 是否使用分层划分，默认 True
        scale: 是否进行特征标准化，默认 True
        
    Returns:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
        scaler: StandardScaler 对象（如果 scale=True），否则为 None
        
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        >>> print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
    Requirements:
        - 6.1: 使用 StandardScaler 进行特征标准化
        - 6.4: 使用 Stratified Split 进行训练集和测试集划分
        - 6.5: 确保数据划分的随机种子为 42
    """
    # 特征标准化
    scaler = None
    X_processed = X
    
    if scale:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    
    # 分层划分训练集和测试集
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test, scaler



class GPUMemoryMonitor:
    """
    GPU 显存监控工具
    
    用于在训练过程中监控 GPU 显存使用情况，当显存超过阈值时发出警告。
    适配单卡 RTX 3090 (24GB) 环境。
    
    功能：
    1. 实时监控 GPU 显存使用
    2. 当显存超过 20GB 时发出警告
    3. 提供显存使用统计信息
    
    使用方法：
    ```python
    monitor = GPUMemoryMonitor(warning_threshold_gb=20.0)
    
    # 在训练循环中
    for epoch in range(epochs):
        # 训练代码...
        monitor.check_memory()  # 检查显存使用
    
    # 获取显存统计信息
    stats = monitor.get_memory_stats()
    print(f"峰值显存使用: {stats['peak_memory_gb']:.2f} GB")
    ```
    
    Requirements:
        - 13.1: 确保所有模型在单卡 RTX 3090 (24GB) 上可运行
        - 13.3: 在模型训练时监控 GPU 显存使用情况
    """
    
    def __init__(self, warning_threshold_gb: float = 20.0, device_id: int = 0):
        """
        初始化 GPU 显存监控器
        
        Args:
            warning_threshold_gb: 显存警告阈值（GB），默认 20.0 GB
            device_id: GPU 设备 ID，默认 0
        """
        self.warning_threshold_gb = warning_threshold_gb
        self.device_id = device_id
        self.warning_threshold_bytes = warning_threshold_gb * 1024**3
        
        # 检查 PyTorch 和 CUDA 是否可用
        try:
            import torch
            self.torch = torch
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.device_name = torch.cuda.get_device_name(device_id)
                self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
                logger.info(f"GPU 显存监控已启用：{self.device_name} "
                          f"(总显存: {self.total_memory / 1024**3:.2f} GB)")
            else:
                logger.warning("CUDA 不可用，GPU 显存监控已禁用")
        except ImportError:
            self.torch = None
            self.cuda_available = False
            logger.warning("PyTorch 未安装，GPU 显存监控已禁用")
        
        # 显存使用统计
        self.peak_memory_bytes = 0
        self.warning_count = 0
    
    def check_memory(self, context: str = "") -> Dict[str, float]:
        """
        检查当前 GPU 显存使用情况
        
        如果显存使用超过阈值，发出警告并记录。
        
        Args:
            context: 上下文信息（如 "Epoch 10"），用于日志记录
            
        Returns:
            memory_info: 显存使用信息字典
                - allocated_gb: 已分配显存（GB）
                - reserved_gb: 已保留显存（GB）
                - total_gb: 总显存（GB）
                - utilization: 显存利用率（0-1）
        """
        if not self.cuda_available or self.torch is None:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'utilization': 0.0
            }
        
        # 获取显存使用情况
        allocated_bytes = self.torch.cuda.memory_allocated(self.device_id)
        reserved_bytes = self.torch.cuda.memory_reserved(self.device_id)
        
        allocated_gb = allocated_bytes / 1024**3
        reserved_gb = reserved_bytes / 1024**3
        total_gb = self.total_memory / 1024**3
        utilization = allocated_bytes / self.total_memory
        
        # 更新峰值显存
        if allocated_bytes > self.peak_memory_bytes:
            self.peak_memory_bytes = allocated_bytes
        
        # 检查是否超过警告阈值
        if allocated_bytes > self.warning_threshold_bytes:
            self.warning_count += 1
            context_str = f" ({context})" if context else ""
            logger.warning(
                f"⚠️  GPU 显存使用超过阈值{context_str}！\n"
                f"   当前使用: {allocated_gb:.2f} GB / {total_gb:.2f} GB "
                f"({utilization * 100:.1f}%)\n"
                f"   警告阈值: {self.warning_threshold_gb:.2f} GB\n"
                f"   建议：\n"
                f"   1. 减小批量大小 (batch_size)\n"
                f"   2. 启用梯度累积 (gradient accumulation)\n"
                f"   3. 使用混合精度训练 (FP16)"
            )
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'total_gb': total_gb,
            'utilization': utilization
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取显存使用统计信息
        
        Returns:
            stats: 统计信息字典
                - peak_memory_gb: 峰值显存使用（GB）
                - current_memory_gb: 当前显存使用（GB）
                - total_memory_gb: 总显存（GB）
                - warning_count: 警告次数
        """
        if not self.cuda_available or self.torch is None:
            return {
                'peak_memory_gb': 0.0,
                'current_memory_gb': 0.0,
                'total_memory_gb': 0.0,
                'warning_count': 0
            }
        
        current_bytes = self.torch.cuda.memory_allocated(self.device_id)
        
        return {
            'peak_memory_gb': self.peak_memory_bytes / 1024**3,
            'current_memory_gb': current_bytes / 1024**3,
            'total_memory_gb': self.total_memory / 1024**3,
            'warning_count': self.warning_count
        }
    
    def reset_peak_memory(self):
        """
        重置峰值显存统计
        
        用于在新的训练阶段开始时重置统计信息。
        """
        if self.cuda_available and self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats(self.device_id)
        self.peak_memory_bytes = 0
        self.warning_count = 0
        logger.info("GPU 显存统计已重置")
    
    def clear_cache(self):
        """
        清空 GPU 缓存
        
        释放未使用的缓存显存，可能有助于缓解显存不足问题。
        注意：这会导致性能下降，仅在必要时使用。
        """
        if self.cuda_available and self.torch is not None:
            self.torch.cuda.empty_cache()
            logger.info("GPU 缓存已清空")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.reset_peak_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        stats = self.get_memory_stats()
        logger.info(
            f"GPU 显存监控结束 - "
            f"峰值使用: {stats['peak_memory_gb']:.2f} GB, "
            f"警告次数: {stats['warning_count']}"
        )
        return False


def get_gpu_memory_info() -> Dict[str, float]:
    """
    获取当前 GPU 显存使用信息（便捷函数）
    
    Returns:
        memory_info: 显存信息字典
            - allocated_gb: 已分配显存（GB）
            - reserved_gb: 已保留显存（GB）
            - total_gb: 总显存（GB）
            - free_gb: 可用显存（GB）
            - utilization: 显存利用率（0-1）
    
    Example:
        >>> info = get_gpu_memory_info()
        >>> print(f"GPU 显存使用: {info['allocated_gb']:.2f} GB / {info['total_gb']:.2f} GB")
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'free_gb': 0.0,
                'utilization': 0.0
            }
        
        device_id = 0
        allocated_bytes = torch.cuda.memory_allocated(device_id)
        reserved_bytes = torch.cuda.memory_reserved(device_id)
        total_bytes = torch.cuda.get_device_properties(device_id).total_memory
        
        allocated_gb = allocated_bytes / 1024**3
        reserved_gb = reserved_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        free_gb = (total_bytes - allocated_bytes) / 1024**3
        utilization = allocated_bytes / total_bytes
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'total_gb': total_gb,
            'free_gb': free_gb,
            'utilization': utilization
        }
    except ImportError:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'total_gb': 0.0,
            'free_gb': 0.0,
            'utilization': 0.0
        }



class GradientAccumulationTrainer:
    """
    梯度累积训练器
    
    当 GPU 显存不足时，通过梯度累积技术实现大批量训练的效果。
    梯度累积允许在多个小批量上累积梯度，然后一次性更新参数，
    从而在显存受限的情况下模拟大批量训练。
    
    工作原理：
    1. 将大批量拆分为多个小批量（accumulation_steps）
    2. 在每个小批量上计算梯度但不更新参数
    3. 累积 accumulation_steps 个小批量的梯度后，一次性更新参数
    
    等效关系：
    - 有效批量大小 = batch_size × accumulation_steps
    - 例如：batch_size=16, accumulation_steps=4 等效于 batch_size=64
    
    使用方法：
    ```python
    trainer = GradientAccumulationTrainer(
        model=model,
        optimizer=optimizer,
        accumulation_steps=4
    )
    
    for epoch in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            loss = criterion(model(X_batch), y_batch)
            trainer.backward_step(loss, batch_idx)
    ```
    
    Requirements:
        - 13.2: 为大模型实现梯度累积机制，支持在显存不足时自动启用
    """
    
    def __init__(self, model, optimizer, accumulation_steps: int = 1, 
                 max_grad_norm: Optional[float] = None):
        """
        初始化梯度累积训练器
        
        Args:
            model: PyTorch 模型
            optimizer: PyTorch 优化器
            accumulation_steps: 梯度累积步数，默认 1（不累积）
            max_grad_norm: 梯度裁剪的最大范数，None 表示不裁剪
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # 统计信息
        self.total_steps = 0
        self.update_count = 0
        
        logger.info(
            f"梯度累积训练器已初始化 - "
            f"累积步数: {accumulation_steps}, "
            f"梯度裁剪: {max_grad_norm if max_grad_norm else '禁用'}"
        )
    
    def backward_step(self, loss, step_idx: int):
        """
        执行反向传播步骤（带梯度累积）
        
        Args:
            loss: 当前批次的损失值
            step_idx: 当前步骤索引（用于判断是否需要更新参数）
        """
        # 缩放损失（平均到累积步数上）
        scaled_loss = loss / self.accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 判断是否需要更新参数
        if (step_idx + 1) % self.accumulation_steps == 0:
            # 梯度裁剪（可选）
            if self.max_grad_norm is not None:
                try:
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                except ImportError:
                    pass
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.update_count += 1
        
        self.total_steps += 1
    
    def finalize_epoch(self):
        """
        完成当前 epoch 的训练
        
        如果还有未更新的累积梯度，执行最后一次参数更新。
        在每个 epoch 结束时调用此方法。
        """
        # 如果还有未更新的梯度，执行更新
        if self.total_steps % self.accumulation_steps != 0:
            # 梯度裁剪（可选）
            if self.max_grad_norm is not None:
                try:
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                except ImportError:
                    pass
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.update_count += 1
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取训练统计信息
        
        Returns:
            stats: 统计信息字典
                - total_steps: 总步数
                - update_count: 参数更新次数
                - accumulation_steps: 梯度累积步数
        """
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'accumulation_steps': self.accumulation_steps
        }


def auto_adjust_batch_size(
    model,
    sample_input,
    initial_batch_size: int = 64,
    memory_threshold_gb: float = 20.0,
    min_batch_size: int = 8
) -> Tuple[int, int]:
    """
    自动调整批量大小和梯度累积步数
    
    根据 GPU 显存使用情况，自动调整批量大小。如果显存不足，
    减小批量大小并增加梯度累积步数，以保持有效批量大小不变。
    
    Args:
        model: PyTorch 模型
        sample_input: 样本输入（用于测试显存使用）
        initial_batch_size: 初始批量大小
        memory_threshold_gb: 显存阈值（GB）
        min_batch_size: 最小批量大小
        
    Returns:
        adjusted_batch_size: 调整后的批量大小
        accumulation_steps: 梯度累积步数
        
    Example:
        >>> model = TabRNet(input_dim=30, hidden_dim=128)
        >>> sample_input = (torch.randn(1, 30), torch.randn(1, 5, 30))
        >>> batch_size, accum_steps = auto_adjust_batch_size(
        ...     model, sample_input, initial_batch_size=64
        ... )
        >>> print(f"调整后批量大小: {batch_size}, 累积步数: {accum_steps}")
    
    Requirements:
        - 13.2: 支持在显存不足时自动启用梯度累积
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.info("CUDA 不可用，使用默认批量大小")
            return initial_batch_size, 1
        
        device = torch.device('cuda')
        model = model.to(device)
        model.eval()
        
        # 测试不同批量大小的显存使用
        current_batch_size = initial_batch_size
        accumulation_steps = 1
        
        while current_batch_size >= min_batch_size:
            try:
                # 清空缓存
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 创建测试批次
                if isinstance(sample_input, tuple):
                    # 多输入情况（如 TabR）
                    test_batch = tuple(
                        inp.repeat(current_batch_size, *([1] * (inp.dim() - 1))).to(device)
                        if inp.dim() > 1 else inp.repeat(current_batch_size).to(device)
                        for inp in sample_input
                    )
                else:
                    # 单输入情况
                    test_batch = sample_input.repeat(
                        current_batch_size, *([1] * (sample_input.dim() - 1))
                    ).to(device)
                
                # 前向传播测试
                with torch.no_grad():
                    if isinstance(test_batch, tuple):
                        _ = model(*test_batch)
                    else:
                        _ = model(test_batch)
                
                # 检查显存使用
                memory_used_gb = torch.cuda.max_memory_allocated() / 1024**3
                
                if memory_used_gb < memory_threshold_gb:
                    # 显存充足，使用当前批量大小
                    logger.info(
                        f"批量大小调整完成 - "
                        f"批量大小: {current_batch_size}, "
                        f"累积步数: {accumulation_steps}, "
                        f"显存使用: {memory_used_gb:.2f} GB"
                    )
                    return current_batch_size, accumulation_steps
                else:
                    # 显存不足，减小批量大小
                    logger.warning(
                        f"批量大小 {current_batch_size} 显存不足 "
                        f"({memory_used_gb:.2f} GB > {memory_threshold_gb:.2f} GB)，"
                        f"尝试减小批量大小..."
                    )
                    current_batch_size = current_batch_size // 2
                    accumulation_steps *= 2
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(
                        f"批量大小 {current_batch_size} 导致 OOM，"
                        f"尝试减小批量大小..."
                    )
                    torch.cuda.empty_cache()
                    current_batch_size = current_batch_size // 2
                    accumulation_steps *= 2
                else:
                    raise
        
        # 如果所有批量大小都失败，使用最小批量大小
        logger.warning(
            f"无法找到合适的批量大小，使用最小批量大小 {min_batch_size}"
        )
        accumulation_steps = initial_batch_size // min_batch_size
        return min_batch_size, accumulation_steps
        
    except ImportError:
        logger.warning("PyTorch 未安装，使用默认批量大小")
        return initial_batch_size, 1
    except Exception as e:
        logger.error(f"批量大小调整失败: {str(e)}，使用默认批量大小")
        return initial_batch_size, 1

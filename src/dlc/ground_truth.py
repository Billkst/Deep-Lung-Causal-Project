"""Ground Truth ITE 生成器。

基于半合成数据生成逻辑，提供“上帝视角”的个体治疗效应 (ITE) 计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


ArrayLike = Union[np.ndarray, torch.Tensor, pd.DataFrame]


@dataclass(frozen=True)
class GroundTruthConfig:
    """与半合成数据生成一致的固定参数。"""

    W_INT: float = 0.69
    W_BASE: float = 0.086
    W_GENE: float = 0.5
    INTERCEPT: float = -3.0


class GroundTruthGenerator:
    """Ground Truth ITE 生成器。"""

    def __init__(
        self,
        feature_names: Optional[Iterable[str]] = None,
        pm25_idx: int = 2,
        egfr_idx: int = 3,
        config: Optional[GroundTruthConfig] = None,
    ) -> None:
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.pm25_idx = pm25_idx
        self.egfr_idx = egfr_idx
        self.config = config or GroundTruthConfig()

    def compute_true_ite(self, X_features: ArrayLike) -> np.ndarray:
        """
        计算真实 ITE。

        Args:
            X_features: 特征矩阵 (包含 Age, Gender, PM2.5, Genes...)

        Returns:
            np.ndarray: True ITE [N]
        """
        X_np, columns = self._to_numpy(X_features)
        pm25_idx, egfr_idx, gene_indices = self._resolve_indices(X_np, columns)

        gene_sum = X_np[:, gene_indices].sum(axis=1) if gene_indices else 0.0
        genetics = self.config.W_GENE * gene_sum
        egfr = X_np[:, egfr_idx]

        logit_treat = (
            self.config.INTERCEPT
            + self.config.W_BASE * 1.0
            + self.config.W_INT * (1.0 * egfr)
            + genetics
        )
        logit_control = (
            self.config.INTERCEPT
            + self.config.W_BASE * (-1.0)
            + self.config.W_INT * ((-1.0) * egfr)
            + genetics
        )

        ite = self._sigmoid(logit_treat) - self._sigmoid(logit_control)
        return ite.astype(np.float64)

    def _resolve_indices(
        self,
        X_np: np.ndarray,
        columns: Optional[Iterable[str]],
    ) -> Tuple[int, int, Iterable[int]]:
        if columns is not None:
            columns = list(columns)
            pm25_idx = self._find_pm25_idx(columns)
            egfr_idx = self._find_egfr_idx(columns)

            exclude = {pm25_idx}
            gene_indices = [i for i in range(len(columns)) if i not in exclude]
            for name in ["Age", "Gender", "Outcome_Label", "True_Prob", "sampleID"]:
                if name in columns:
                    idx = columns.index(name)
                    if idx in gene_indices:
                        gene_indices.remove(idx)
            return pm25_idx, egfr_idx, gene_indices

        n_features = X_np.shape[1]
        pm25_idx = self.pm25_idx
        egfr_idx = self.egfr_idx
        if pm25_idx >= n_features:
            raise ValueError("pm25_idx 超出特征范围")
        if egfr_idx >= n_features:
            raise ValueError("egfr_idx 超出特征范围")
        gene_indices = [i for i in range(n_features) if i not in {0, 1, pm25_idx}]
        return pm25_idx, egfr_idx, gene_indices

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0 or np.isnan(std):
            return np.zeros_like(values, dtype=np.float64)
        return (values - mean) / std

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits))

    @staticmethod
    def _to_numpy(
        X_features: ArrayLike,
    ) -> Tuple[np.ndarray, Optional[Iterable[str]]]:
        if isinstance(X_features, pd.DataFrame):
            return X_features.values.astype(np.float64), X_features.columns
        if isinstance(X_features, torch.Tensor):
            return X_features.detach().cpu().numpy().astype(np.float64), None
        return np.asarray(X_features, dtype=np.float64), None

    @staticmethod
    def _find_pm25_idx(columns: Iterable[str]) -> int:
        candidates = [
            "Virtual_PM2.5",
            "PM2.5",
            "PM25",
            "pm25",
            "pm2.5",
        ]
        for name in candidates:
            if name in columns:
                return columns.index(name)
        raise ValueError("无法在列名中找到 PM2.5 列")

    @staticmethod
    def _find_egfr_idx(columns: Iterable[str]) -> int:
        if "EGFR" in columns:
            return columns.index("EGFR")
        raise ValueError("无法在列名中找到 EGFR 列")

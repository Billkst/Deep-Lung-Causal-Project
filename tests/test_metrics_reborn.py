# -*- coding: utf-8 -*-

import numpy as np
import torch

from src.dlc.ground_truth import GroundTruthGenerator
from src.dlc.metrics import compute_sensitivity_score


def test_ground_truth_ite_nonzero():
    X = np.zeros((8, 5), dtype=np.float64)
    generator = GroundTruthGenerator(pm25_idx=2, egfr_idx=3)
    ite = generator.compute_true_ite(X)
    assert np.any(np.abs(ite) > 1e-6)


def test_sensitivity_score_positive():
    torch.manual_seed(0)

    class DummyModel(torch.nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, 1)
            with torch.no_grad():
                weights = torch.tensor([[0.5, 0.2, 1.0, 0.3, 0.1]])
                self.linear.weight.copy_(weights)
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = DummyModel(input_dim=5)
    X = torch.randn(64, 5)

    sens = compute_sensitivity_score(
        model,
        X,
        confounder_idx=0,
        epsilon=0.5,
        treatment_col_idx=2,
    )
    print(f"Sensitivity Score: {sens:.6f}")
    assert sens > 0

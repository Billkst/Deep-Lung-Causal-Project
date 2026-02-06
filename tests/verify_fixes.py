
import sys
import os
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

sys.path.append(os.getcwd())

from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.baselines.transtee_baseline import TransTEEBaseline
from src.dlc.dlc_net import DLCNet

def test_baselines():
    print("Testing Baselines Params...")
    xgb = XGBBaseline()
    print("XGB Params:", xgb.count_parameters())
    
    hf = HyperFastBaseline()
    print("HyperFast Params:", hf.count_parameters())
    
    tt = TransTEEBaseline()
    print("TransTEE Params:", tt.count_parameters())

def test_dlc_logic():
    print("\nTesting DLCNet Logic...")
    model = DLCNet(input_dim=23)
    
    # Dummy Input: [B, 23]
    X = torch.randn(4, 23)
    # PM2.5 is last column.
    
    # Forward
    out = model(X)
    print("Forward successful.")
    print("ITE shape:", out['ITE'].shape)
    
    # Loss
    y = torch.tensor([0, 1, 0, 1]).float()
    t = torch.tensor([0, 1, 0, 1]).float()
    
    losses = model.compute_loss(X, y, out, t=t)
    print("Loss with T:", losses['loss_total'].item())
    
    # Loss without T (inference logic)
    losses_auto = model.compute_loss(X, y, out)
    print("Loss auto T:", losses_auto['loss_total'].item())

if __name__ == "__main__":
    test_baselines()
    test_dlc_logic()

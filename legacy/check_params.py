
import torch
import sys
import os
sys.path.append(os.getcwd())
from src.dlc.dlc_net import DLCNet

model = DLCNet(input_dim=23, d_hidden=256, num_layers=4, num_heads=4)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

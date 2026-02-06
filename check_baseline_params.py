import sys
import os
import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

# Import baselines
try:
    from src.baselines.tabr_baseline import TabRBaseline
except ImportError as e:
    print(f"TabR Import Error: {e}")

try:
    from src.baselines.mogonet_baseline import MOGONETBaseline
except ImportError as e:
    print(f"MOGONET Import Error: {e}")

try:
    from src.baselines.transtee_baseline import TransTEEBaseline
except ImportError as e:
    print(f"TransTEE Import Error: {e}")

try:
    from src.baselines.hyperfast_baseline import HyperFastBaseline
except ImportError as e:
    print(f"HyperFast Import Error: {e}")

try:
    from src.baselines.xgb_baseline import XGBBaseline
except ImportError as e:
    print(f"XGB Import Error: {e}")

try:
    from src.dlc.dlc_net import DLCNet
except ImportError as e:
    print(f"DLCNet Import Error: {e}")


def count_torch_params(model):
    try:
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
             return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        elif isinstance(model, torch.nn.Module):
             return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
             return "N/A"
    except:
        return "Error"

print("=== Counting Parameters for Baselines ===")

# Dummy Data
N = 50
D_GEN = 23
X_dummy = np.random.randn(N, D_GEN)
y_dummy = np.random.randint(0, 2, N)

# 1. XGBoost
print("\n--- XGBoost ---")
try:
    xgb = XGBBaseline()
    xgb.fit(X_dummy, y_dummy)
    print(f"XGBoost Params: N/A (Tree)")
except Exception as e:
    print(f"XGBoost Error: {e}")

# 2. TabR
print("\n--- TabR ---")
try:
    print("Initializing TabR...")
    tabr = TabRBaseline(epochs=1, batch_size=16)
    print("Fitting TabR...")
    tabr.fit(X_dummy, y_dummy)
    p_tabr = count_torch_params(tabr)
    print(f"TabR Params: {p_tabr}")
except Exception as e:
    print(f"TabR Error: {e}")

# 3. HyperFast
print("\n--- HyperFast ---")
try:
    print("Initializing HyperFast...")
    hf = HyperFastBaseline(epochs=1, batch_size=16)
    print("Fitting HyperFast...")
    hf.fit(X_dummy, y_dummy)
    if hasattr(hf, 'hypernetwork') and hf.hypernetwork is not None:
         # Params in hypernetwork
         p_hf = sum(p.numel() for p in hf.hypernetwork.parameters() if p.requires_grad)
    else:
         p_hf = count_torch_params(hf)
    print(f"HyperFast Params: {p_hf}")
except Exception as e:
    print(f"HyperFast Error: {e}")

# 4. TransTEE
print("\n--- TransTEE ---")
try:
    # Needs X, t, y.
    X_tt = np.random.randn(N, 22)
    t_tt = np.random.randint(0, 2, N)
    y_tt = y_dummy
    
    print("Initializing TransTEE...")
    tt = TransTEEBaseline(epochs=1, batch_size=16)
    print("Fitting TransTEE...")
    tt.fit(X_tt, t_tt, y_tt)
    p_tt = count_torch_params(tt)
    print(f"TransTEE Params: {p_tt}")
except Exception as e:
    print(f"TransTEE Error: {e}")

# 5. MOGONET
print("\n--- MOGONET ---")
try:
    # Mogonet takes list of views.
    # Clinical=3, Omics=20
    X_v1 = np.random.randn(N, 3)
    X_v2 = np.random.randn(N, 20)
    
    print("Initializing MOGONET...")
    mogo = MOGONETBaseline(epochs=1, batch_size=16)
    print("Fitting MOGONET...")
    mogo.fit([X_v1, X_v2], y_dummy)
    p_mogo = count_torch_params(mogo)
    print(f"MOGONET Params: {p_mogo}")
except Exception as e:
    print(f"MOGONET Error: {e}")

# 6. DLC Config M (Large)
print("\n--- DLC (Config M - Large 256) ---")
try:
    dlc = DLCNet(input_dim=23, d_hidden=256, num_layers=4)
    p_dlc = count_torch_params(dlc)
    print(f"DLC (Config M) Params: {p_dlc}")
except Exception as e:
    print(f"DLC Error: {e}")

# 7. DLC Config N (Compact 128)
print("\n--- DLC (Config N - Compact 128) ---")
try:
    dlc = DLCNet(input_dim=23, d_hidden=128, num_layers=3)
    p_dlc = count_torch_params(dlc)
    print(f"DLC (Config N - 128) Params: {p_dlc}")
except Exception as e:
    print(f"DLC Error: {e}")

# 8. DLC Config S (Small 96)
print("\n--- DLC (Config S - Small 96) ---")
try:
    dlc = DLCNet(input_dim=23, d_hidden=96, num_layers=3)
    p_dlc = count_torch_params(dlc)
    print(f"DLC (Config S - 96) Params: {p_dlc}")
except Exception as e:
    print(f"DLC Error: {e}")

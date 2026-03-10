import torch
import sys

def inspect(path):
    print(f"--- Inspecting {path} ---")
    try:
        payload = torch.load(path, map_location='cpu')
        if isinstance(payload, dict):
            print(f"Keys: {list(payload.keys())[:5]} ...")
            if 'model_state_dict' in payload:
                print("Found 'model_state_dict'")
            if 'scaler_mean' in payload:
                print("Found 'scaler_mean'")
        else:
            print("Loaded state dict directly.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect("results/dlc_final_sota_s_seed_42.pth")

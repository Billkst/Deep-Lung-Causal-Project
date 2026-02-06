import torch
import sys
sys.path.append('src')

def inspect_checkpoint(path):
    print(f"--- Inspecting {path} ---")
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print("Keys:", list(state_dict.keys())[:5])
        else:
            state_dict = ckpt
            print("Loaded state dict directly.")
        
        # Count params
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Total Params in State Dict: {total_params}")
        
        # Infer dimensions
        for k, v in state_dict.items():
            if 'encoder.0.weight' in k: # First layer
                print(f"Input Layer: {k} -> {v.shape}")
            if 'prediction_head.0.weight' in k:
                print(f"Head Layer: {k} -> {v.shape}")
                
    except Exception as e:
        print(f"Error: {e}")

inspect_checkpoint("results/dlc_final_sota.pth")
inspect_checkpoint("results/dlc_golden_final.pth")

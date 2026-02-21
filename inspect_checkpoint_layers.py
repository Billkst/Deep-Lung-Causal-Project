import torch
import os
import sys

# Add src to path just in case, though we primarily need torch load
sys.path.append('src')

checkpoint_path = 'results/dlc_final_sota_s_seed_46.pth'

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

try:
    # Load checkpoint
    # Note: If the model class is needed for pickling, this might fail if not imported.
    # But usually 'state_dict' is just a dict.
    # If the formatting is specific, we might need to verify.
    # Often pth contains {'model_state_dict': ..., 'hyperparameters': ...}
    
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    print("Keys in checkpoint:", chkpt.keys())
    
    if 'model_state_dict' in chkpt:
        state_dict = chkpt['model_state_dict']
        # Count layers by looking at 'encoder.shared_layers' or similar keys
        keys = list(state_dict.keys())
        # print("Layer keys:", keys[:10]) # Print first few
        
        # Try to infer depth
        layer_indices = set()
        for k in keys:
            if 'shared_layers' in k and 'weight' in k:
                # format expectation: encoder.shared_layers.0.weight
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part == 'shared_layers':
                        if i + 1 < len(parts) and parts[i+1].isdigit():
                            layer_indices.add(int(parts[i+1]))
        
        print(f"Detected Layer Indices: {sorted(list(layer_indices))}")
        print(f"Estimated Depth: {len(layer_indices)}")

    if 'hyperparameters' in chkpt:
        print("Hyperparameters:", chkpt['hyperparameters'])

except Exception as e:
    print(f"Error loading checkpoint: {e}")


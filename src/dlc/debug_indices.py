
import numpy as np
import pandas as pd
import os

def diagnose():
    csv_path = "data/pancan_synthetic_interaction.csv"
    npy_path = "data/PANCAN/clean_pancan_indices.npy"
    
    if not os.path.exists(csv_path) or not os.path.exists(npy_path):
        print("Files not found.")
        return

    df = pd.read_csv(csv_path)
    indices = np.load(npy_path)
    
    print(f"CSV Shape: {df.shape}")
    print(f"Indices Count: {len(indices)}")
    print(f"Indices Max: {indices.max()}")
    print(f"Indices Min: {indices.min()}")
    
    if indices.max() >= len(df):
        print("FAIL: Index out of bounds!")
        print(f"Overflow amount: {indices.max() - len(df) + 1}")
    else:
        print("PASS: Indices are valid.")

if __name__ == "__main__":
    diagnose()

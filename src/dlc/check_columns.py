
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

def check():
    print("Checking columns...")
    
    # Load raw CSV
    try:
        df = pd.read_csv("data/luad_synthetic_interaction.csv")
        print("\nCSV Columns:", df.columns.tolist())
        print("Sample row:", df.iloc[0].values)
        
        # Check indices map
        feature_cols = [c for c in df.columns if c not in ['sampleID', 'Virtual_PM2.5', 'True_Prob', 'Outcome_Label']]
        print("\nFeatures (excluding PM2.5):", feature_cols)
        print("Feature count:", len(feature_cols))
        
        # Check where PM2.5 is
        if 'Virtual_PM2.5' in df.columns:
            print("Virtual_PM2.5 is present.")
            
        print("Age index in features:", feature_cols.index('Age'))
        print("Gender index in features:", feature_cols.index('Gender'))
    except Exception as e:
        print("Error reading CSV:", e)

if __name__ == "__main__":
    check()

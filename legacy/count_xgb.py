
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from src.baselines.xgb_baseline import XGBBaseline
from src.data_processor import DataCleaner, SemiSyntheticGenerator

# Ensure we can import src
sys.path.append(os.getcwd())

def count_xgb_params():
    print("Loading data...")
    # Load a small sample or full data to train
    data_path = 'data/luad_synthetic_linear.csv' # Use one of the synthetic datasets
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['sampleID', 'True_Prob', 'Outcome_Label']).values
    y = df['Outcome_Label'].values
    
    # Initialize XGBBaseline
    # We use default params from the class to match the reported results
    model_wrapper = XGBBaseline(random_state=42)
    
    print("Training XGBoost...")
    model_wrapper.fit(X, y)
    
    # Access the underlying xgb.XGBClassifier
    booster = model_wrapper.model.get_booster()
    
    # Get all trees as a dataframe
    trees_df = booster.trees_to_dataframe()
    
    # Number of nodes (split nodes + leaf nodes)
    num_nodes = trees_df.shape[0]
    
    # Number of trees
    num_trees = trees_df['Tree'].max() + 1
    
    print(f"XGBoost Stats:")
    print(f"  Num Trees: {num_trees}")
    print(f"  Total Nodes (Params): {num_nodes}")
    
    # Alternative: Size of the model binary?
    # model_wrapper.model.save_model("temp.json")
    # print(f"  Model File Size: {os.path.getsize('temp.json')} bytes")

if __name__ == "__main__":
    count_xgb_params()

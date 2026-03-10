
import pandas as pd
import numpy as np

# Load CSV
try:
    df = pd.read_csv("results/parameter_sensitivity_results_final.csv")
    print(f"Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit(1)

# 1. Architecture
print("\n--- Architecture Analysis ---")
da = df[df['type']=='arch'].copy()
if not da.empty:
    da['config'] = da['d_hidden'].astype(str) + "/" + da['num_layers'].astype(str)
    best_auc = da.loc[da['AUC'].idxmax()]
    print(f"Best AUC:    {best_auc['AUC']:.4f} (Config: {best_auc['config']})")
    print("\nDetailed Stats:")
    print(da[['config','AUC','AUC_Std','CATE']].to_string(index=False))

# 2. HSIC
print("\n--- HSIC Analysis ---")
dh = df[df['type']=='hsic'].copy()
if not dh.empty:
    best_auc = dh.loc[dh['AUC'].idxmax()]
    print(f"Best AUC:    {best_auc['AUC']:.4f} (Lambda: {best_auc['lambda_hsic']})")
    print("\nDetailed Stats:")
    print(dh[['lambda_hsic','AUC','AUC_Std','CATE']].to_string(index=False))

# 3. CATE Weight
print("\n--- CATE Weight Analysis ---")
dc = df[df['type']=='cate'].copy()
if not dc.empty:
    best_auc = dc.loc[dc['AUC'].idxmax()]
    print(f"Best AUC:    {best_auc['AUC']:.4f} (Lambda: {best_auc['lambda_cate']})")
    print("\nDetailed Stats:")
    print(dc[['lambda_cate','AUC','AUC_Std','CATE']].to_string(index=False))

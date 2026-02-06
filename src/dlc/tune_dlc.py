# -*- coding: utf-8 -*-
"""
Bayesian Hyperparameter Tuning for Deep-Lung-Causal (DLC)
=========================================================

Goal: Optimize Delta CATE signal while maintaining AUC > 0.85 and Sensitivity < 0.05.
Search Space:
- d_hidden: [64, 128, 256]
- num_layers: [1, 2, 3]
- lambda_hsic: 1e-4 ~ 1.0 (LogUniform)
- lr_fine_tune: 1e-5 ~ 1e-3 (LogUniform)
- dropout: 0.0 ~ 0.5 (Float)

Strategy:
- Objective: AUC_val + 0.5 * I(Delta CATE > 0.01)
- Constraint: Sensitivity <= 0.05 (Soft pruning via score penalty)
- Trials: 20
"""

import sys
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Project Imports
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_cate, compute_sensitivity_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Classes & Functions (Ported from run_golden_reproducibility.py) ---

class FastTensorDataLoader:
    def __init__(self, x_tensor, y_tensor, batch_size=4096, shuffle=True):
        assert x_tensor.device == y_tensor.device
        self.x = x_tensor
        self.y = y_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = x_tensor.size(0)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.x.device)
        else:
            indices = torch.arange(self.n_samples, device=self.x.device)

        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return self.n_batches

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def load_synthetic_csv(data_source: str, scenario: str = "interaction", enforce_feature_count: int = 23):
    data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / f"{data_source.lower()}_synthetic_{scenario}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    exclude_cols = ["sampleID", "Outcome_Label", "True_Prob"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if enforce_feature_count is not None and len(feature_cols) > enforce_feature_count:
        feature_cols = feature_cols[:enforce_feature_count]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Outcome_Label"].values.astype(np.int64)
    return df, X, y, feature_cols

def split_train_val(X, y, random_state=42, val_size=0.1):
    return train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)

def evaluate_auc(model, X_tensor, y_prob_np, device):
    # This assumes input is a tensor
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        if isinstance(outputs, dict):
            y_pred = outputs["Y_1"].squeeze().cpu().numpy()
        else:
            y_pred = outputs.squeeze().cpu().numpy()
    return float(roc_auc_score(y_prob_np, y_pred))

# Training Loops (Stripped down version for optimization speed, less printing)
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        outputs = model(X_batch)
        losses = model.compute_loss(X_batch, y_batch, outputs)
        loss = losses["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

def train_epoch_heads(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        outputs = model(X_batch)
        y_pred = outputs["Y_1"].squeeze()
        loss = criterion(y_pred, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

# --- Global Data Loading (To avoid reloading in every trial) ---
print("Loading Data Globally for Optimization...")

# 1. PANCAN Clean
pancan_df, _, _, pancan_features = load_synthetic_csv("pancan", "interaction")
indices_path = PROJECT_ROOT / "data" / "PANCAN" / "clean_pancan_indices.npy"
clinical_path = PROJECT_ROOT / "data" / "PANCAN" / "PANCAN_clinical.txt"
if indices_path.exists() and clinical_path.exists():
    clean_indices = np.load(indices_path)
    clinical_df = pd.read_csv(clinical_path, sep="\t", dtype=str)
    sample_col = clinical_df.columns[0]
    clinical_ids = clinical_df[sample_col].astype(str).str.strip()
    pancan_aligned = pancan_df.set_index("sampleID").reindex(clinical_ids).reset_index()
    pancan_clean = pancan_aligned.iloc[clean_indices].dropna(subset=pancan_features)
    pancan_df = pancan_clean.reset_index(drop=True)
else:
    print("Warning: Clean PANCAN indices not found, using full PANCAN.")

X_pancan = pancan_df[pancan_features].values.astype(np.float32)
y_pancan = pancan_df["Outcome_Label"].values.astype(np.int64)

# 2. LUAD Split
luad_df, X_luad, y_luad, luad_features = load_synthetic_csv("luad", "interaction", enforce_feature_count=23)
all_indices = np.arange(len(X_luad))
train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42, stratify=y_luad)
X_luad_train_raw = X_luad[train_idx]
y_luad_train = y_luad[train_idx]
X_luad_test_raw = X_luad[test_idx]
y_luad_test = y_luad[test_idx]

# Prepare Scaled versions
# PANCAN Split
X_p_train, X_p_val, y_p_train, y_p_val = split_train_val(X_pancan, y_pancan, random_state=42, val_size=0.1)
pancan_scaler = StandardScaler()
X_p_train_scaled = pancan_scaler.fit_transform(X_p_train)
X_p_val_scaled = pancan_scaler.transform(X_p_val)

# LUAD Split for Training
X_l_train, X_l_val, y_l_train, y_l_val = split_train_val(X_luad_train_raw, y_luad_train, random_state=42, val_size=0.1)
luad_scaler = StandardScaler()
X_l_train_scaled = luad_scaler.fit_transform(X_l_train)
X_l_val_scaled = luad_scaler.transform(X_l_val)
X_luad_test_scaled = luad_scaler.transform(X_luad_test_raw) # For metric eval if needed

# Move to GPU if available (Create Tensors)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# PANCAN Tensors
X_p_train_tensor = torch.as_tensor(X_p_train_scaled, dtype=torch.float32, device=device)
y_p_train_tensor = torch.as_tensor(y_p_train, dtype=torch.float32, device=device)
X_p_val_tensor = torch.as_tensor(X_p_val_scaled, dtype=torch.float32, device=device)

# LUAD Tensors
X_l_train_tensor = torch.as_tensor(X_l_train_scaled, dtype=torch.float32, device=device)
y_l_train_tensor = torch.as_tensor(y_l_train, dtype=torch.float32, device=device)
X_l_val_tensor = torch.as_tensor(X_l_val_scaled, dtype=torch.float32, device=device)
X_luad_test_scaled_tensor = torch.as_tensor(X_luad_test_scaled, dtype=torch.float32, device=device)
# Raw LUAD Test Tensor for CATE
X_luad_test_raw_tensor = torch.as_tensor(X_luad_test_raw, dtype=torch.float32, device=device)

def objective(trial):
    # 1. Hyperparameters
    d_hidden = trial.suggest_categorical("d_hidden", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lambda_hsic = trial.suggest_float("lambda_hsic", 1e-4, 1.0, log=True)
    lr_fine_tune = trial.suggest_float("lr_fine_tune", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # 2. Init Model
    set_seed(42) # Ensure model init is reproducible-ish for same params
    model = DLCNet(
        input_dim=23,
        d_hidden=d_hidden,
        num_layers=num_layers,
        lambda_hsic=lambda_hsic,
        dropout=dropout,
        random_state=42
    ).to(device)

    # 3. Stage 1: PANCAN Pretrain (50 epochs, lr=1e-3)
    # Using slightly reduced epochs for tuning speed? Or sticky to Golden? 
    # Optuna recommends fewer epochs for bad trials (Pruning), but we use fixed epochs here.
    # To save time, we might reduce it, but for accuracy we keep it. 
    # Let's keep 50 epochs as it's FastDataLoader on GPU.
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader_pre = FastTensorDataLoader(X_p_train_tensor, y_p_train_tensor, batch_size=4096)
    
    for epoch in range(50):
        # We don't prune inside pretraining usually, but could check val loss?
        train_epoch(model, loader_pre, optimizer_pre)
        # Check pruning every 10 epochs? Nah, let's just train.

    # 4. Stage 2: LUAD Head Tuning (20 epochs, lr=1e-3)
    for param in model.parameters(): param.requires_grad = False
    for param in model.outcome_head_0.parameters(): param.requires_grad = True
    for param in model.outcome_head_1.parameters(): param.requires_grad = True
    
    optimizer_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.BCELoss()
    loader_luad = FastTensorDataLoader(X_l_train_tensor, y_l_train_tensor, batch_size=4096)
    
    for epoch in range(20):
        train_epoch_heads(model, loader_luad, optimizer_head, criterion)

    # 5. Stage 3: LUAD Fine Tuning (30 epochs, lr=lr_fine_tune)
    for param in model.parameters(): param.requires_grad = True
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=lr_fine_tune)
    
    # We will track validation AUC here
    for epoch in range(30):
        train_epoch(model, loader_luad, optimizer_ft)
    
    # 6. Evaluation on LUAD Validation Set
    val_auc = evaluate_auc(model, X_l_val_tensor, y_l_val, device)
    
    # Compute Sensitivity on Validation Set (to avoid test leakage)
    model.scaler = luad_scaler # Important for metrics
    sens_score = float(compute_sensitivity_score(
        model, 
        X_l_val_tensor, # Scaled input
        confounder_idx=0, # Age
        treatment_col_idx=2
    ))
    
    # Pruning Constraint
    if sens_score > 0.05:
        # Penalize heavily
        return -1.0 # Or very low score

    # Compute Delta CATE
    # Using Validation set for Delta CATE estimation too? Or Test? 
    # Tuning on Test is cheating. We use Validation.
    # We need Raw Validation Data for CATE? 
    # We only have X_l_val (Unscaled) which generated X_l_val_scaled.
    # We need to reconstruct raw tensor or keep it.
    X_l_val_raw = X_l_val # This is numpy
    X_l_val_raw_tensor = torch.as_tensor(X_l_val_raw, dtype=torch.float32, device=device)
    
    cate = compute_cate(model, X_l_val_raw_tensor, treatment_col_idx=2)
    # EGFR Index: luad_features.index("EGFR") -> usually 3 (0=Age, 1=Gender, 2=PM2.5, 3=EGFR)
    egfr_idx = luad_features.index("EGFR") if "EGFR" in luad_features else 3
    egfr_mask = X_l_val_raw[:, egfr_idx] > 0
    
    if np.sum(egfr_mask) == 0 or np.sum(~egfr_mask) == 0:
        delta_cate = 0.0
    else:
        delta_cate = float(np.mean(cate[egfr_mask]) - np.mean(cate[~egfr_mask]))
        
    # Composite Score
    # Score = AUC + 0.5 * I(Delta > 0.01)
    bonus = 0.5 if delta_cate > 0.01 else 0.0
    score = val_auc + bonus
    
    # Reporting
    trial.set_user_attr("auc", val_auc)
    trial.set_user_attr("sensitivity", sens_score)
    trial.set_user_attr("delta_cate", delta_cate)
    
    return score

def main():
    print("Start Bayesian Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\noptimization Finished!")
    print("Best Params:", study.best_params)
    print("Best Value:", study.best_value)
    
    best_trial = study.best_trial
    print("\nBest Trial Details:")
    print(f"  AUC: {best_trial.user_attrs.get('auc', 'N/A')}")
    print(f"  Sensitivity: {best_trial.user_attrs.get('sensitivity', 'N/A')}")
    print(f"  Delta CATE: {best_trial.user_attrs.get('delta_cate', 'N/A')}")
    
    # Save results
    results = {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "best_trial_metrics": best_trial.user_attrs
    }
    
    out_path = PROJECT_ROOT / "results" / "best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()

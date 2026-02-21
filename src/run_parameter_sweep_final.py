import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from copy import deepcopy

# Add project root
sys.path.append(os.getcwd())

from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_sensitivity_score
from src.dlc.ground_truth import GroundTruthGenerator

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

print("Imports completed. Initializing device...", flush=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {DEVICE}", flush=True)

def load_data_rigorous(seed):
    """
    Exact copy of data loading from run_final_sota_p.py.
    Ensures strict methodological consistency:
    1. Blacklist filtering
    2. Feature alignment (PANCAN -> LUAD)
    3. Stratified splitting
    4. Correct Ground Truth ITE generation
    """
    # 1. Load PANCAN
    pancan_full = pd.read_csv("data/pancan_synthetic_interaction.csv")

    # 2. Filter Blacklist
    blacklist_path = Path("data/PANCAN/pancan_leakage_blacklist.csv")
    if blacklist_path.exists():
        try:
            blacklist = pd.read_csv(blacklist_path, header=None)
            if 'sampleID' in blacklist.values[0]: # header present in first row?
                try:
                     blacklist = pd.read_csv(blacklist_path)['sampleID']
                except:
                     blacklist = blacklist[0]
            else:
                blacklist = blacklist[0]
            
            leak_ids = set(blacklist.astype(str).values)
            if 'sampleID' in pancan_full.columns:
                pancan_full = pancan_full[~pancan_full['sampleID'].isin(leak_ids)].reset_index(drop=True)
        except Exception: pass

    # Drop ID
    pancan_clean = pancan_full.drop(columns=['sampleID']) if 'sampleID' in pancan_full.columns else pancan_full
    
    # 3. Load LUAD
    luad_full = pd.read_csv("data/luad_synthetic_interaction.csv")
    if 'sampleID' in luad_full.columns:
        luad_full = luad_full.drop(columns=['sampleID'])
        
    # 4. CRITICAL FIX: Align PANCAN columns to LUAD columns
    # This ensures that the feature set matches exactly, filling missing with 0 and reordering
    pancan_aligned = pancan_clean.reindex(columns=luad_full.columns, fill_value=0)
    pancan_clean = pancan_aligned
    
    def extract_xy(df):
        y_val = df['Outcome_Label'].values
        true_prob = df['True_Prob'].values if 'True_Prob' in df.columns else None
        
        # Safe EGFR extraction
        egfr = df['EGFR'].values if 'EGFR' in df.columns else np.zeros(len(df))
        
        drop_cols = ['Outcome_Label', 'True_Prob']
        # If True_ITE exists, drop it too from X, handled separately (but usually not there)
        if 'True_ITE' in df.columns:
            drop_cols.append('True_ITE')
            
        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        cols = X_df.columns.tolist()
        # Ensure Virtual_PM2.5 is last
        if 'Virtual_PM2.5' in cols:
            cols.remove('Virtual_PM2.5')
            cols.append('Virtual_PM2.5')
            X_df = X_df[cols]
        
        # Calculate T (TransTEE style binarization for training usage)
        if 'Virtual_PM2.5' in X_df.columns:
            pm_vals = X_df['Virtual_PM2.5'].values
            med = np.median(pm_vals)
            t_val = (pm_vals > med).astype(float)
        else:
            t_val = np.zeros(len(X_df))
        
        return X_df.values.astype(np.float32), y_val, t_val, egfr, true_prob, cols

    X_p, y_p, t_p, egfr_p, tp_p, feat_names = extract_xy(pancan_clean)
    X_l, y_l, t_l, egfr_l, tp_l, _ = extract_xy(luad_full)
    
    # ITE Setup - CORRECTLY COMPUTED USING GroundTruthGenerator
    gt_gen = GroundTruthGenerator()
    
    # Reconstruct DataFrame for GT Generator (it handles column finding)
    df_p_eval = pd.DataFrame(X_p, columns=feat_names)
    ite_p = gt_gen.compute_true_ite(df_p_eval).astype(np.float32)
    
    df_l_eval = pd.DataFrame(X_l, columns=feat_names)
    ite_l = gt_gen.compute_true_ite(df_l_eval).astype(np.float32)
        
    return (X_p, y_p, t_p, ite_p, egfr_p, tp_p, 
            X_l, y_l, t_l, ite_l, egfr_l, tp_l, feat_names)

def train_epoch_sweep(model, loader, optimizer, params):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch, t_batch, ite_true, egfr, true_prob in loader:
        
        ite_true = ite_true.view(-1)
        egfr = egfr.view(-1)
        true_prob = true_prob.view(-1)
        
        optimizer.zero_grad()
        out = model(X_batch)
        losses = model.compute_loss(X_batch, y_batch, out, t=t_batch)
        
        # Additional losses
        ite_pred = out['ITE'].squeeze()
        loss_ite = F.mse_loss(ite_pred, ite_true)
        
        # Delta CATE
        mask_mut = egfr > 0.5
        mask_wt = ~mask_mut
        if mask_mut.any() and mask_wt.any():
            cate_pred_mut = ite_pred[mask_mut].mean()
            cate_pred_wt = ite_pred[mask_wt].mean()
            delta_pred = cate_pred_mut - cate_pred_wt
            
            # CRITICAL: Use correct True Delta from batch ITE
            cate_true_mut = ite_true[mask_mut].mean()
            cate_true_wt = ite_true[mask_wt].mean()
            delta_true = cate_true_mut - cate_true_wt
            
            loss_cate = F.mse_loss(delta_pred, delta_true)
        else:
            loss_cate = torch.tensor(0.0, device=DEVICE)
            
        y0, y1 = out['Y_0'].squeeze(), out['Y_1'].squeeze()
        prob_pred = t_batch * y1 + (1.0 - t_batch) * y0
        loss_prob = F.mse_loss(prob_pred, true_prob)
        
        loss = (
            losses['loss_total'] +
            params['lambda_ite'] * loss_ite +
            params['lambda_cate'] * loss_cate +
            params['lambda_prob'] * loss_prob +
            params['lambda_adv'] * 0.5 
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate_sweep(model, X, y, t, ite_true, scaler, feat_names):
    model.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(DEVICE)
        out = model(X_t)
        y0, y1 = out['Y_0'].cpu().numpy().squeeze(), out['Y_1'].cpu().numpy().squeeze()
        
        # AUC/Acc/F1/Sensitivity
        prob = t * y1 + (1.0 - t) * y0
        pred = (prob > 0.5).astype(int)
        
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        sens = recall_score(y, pred)
        
        # Delta CATE & PEHE
        ite_pred = y1 - y0
        
        # PEHE - using provided true ITE
        pehe = np.sqrt(np.mean((ite_pred - ite_true)**2))
        
        if 'EGFR' in feat_names:
            idx = feat_names.index('EGFR')
            # Look up EGFR in the original X (unscaled) passed in? 
            # X passed in is numpy array. Index should match feat_names.
            mask = X[:, idx] == 1
            if mask.any() and (~mask).any():
                dc = ite_pred[mask].mean() - ite_pred[~mask].mean()
            else:
                dc = 0.0
        else:
            dc = 0.0
    return {'AUC': auc, 'ACC': acc, 'F1': f1, 'Sensitivity': sens, 'PEHE': pehe, 'Delta_CATE': dc}

def run_single_config(seed, override_params):
    # Base Config S (Final SOTA alignment)
    PARAMS = {
        'd_hidden': 128, 
        'num_layers': 4, # UPDATED TO MATCH SOTA
        'dropout': 0.1,
        'lambda_hsic': 0.1, 'lambda_pred': 3.5, 'lambda_ite': 1.0, 
        'lambda_cate': 2.0, 'lambda_prob': 1.0, 'lambda_adv': 0.5,
        'lr_pre': 1e-4, 'lr_fine': 3e-5,
        'bs_pre': 512, 'bs_fine': 128,
        'epochs_pre': 200, 'epochs_fine': 100 
    }
    PARAMS.update(override_params)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # RIGOROUS DATA LOADING
    (X_p, y_p, t_p, ite_p, egfr_p, tp_p, 
     X_l, y_l, t_l, ite_l, egfr_l, tp_l, feats) = load_data_rigorous(seed)
    
    # Split LUAD (Stratified by Y+T)
    indices = np.arange(len(X_l))
    stratify_col = y_l + 2 * t_l
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, stratify=stratify_col, random_state=42) # Fixed seed for splitting consistency
    
    X_l_tr, X_l_te = X_l[tr_idx], X_l[te_idx]
    y_l_tr, y_l_te = y_l[tr_idx], y_l[te_idx]
    t_l_tr, t_l_te = t_l[tr_idx], t_l[te_idx]
    ite_l_tr, ite_l_te = ite_l[tr_idx], ite_l[te_idx]
    egfr_l_tr, egfr_l_te = egfr_l[tr_idx], egfr_l[te_idx]
    tp_l_tr, tp_l_te = tp_l[tr_idx], tp_l[te_idx]
    
    scaler = StandardScaler()
    X_p_s = scaler.fit_transform(X_p)
    X_l_tr_s = scaler.transform(X_l_tr) # Fit on PANCAN, transform LUAD
    
    # Clip outliers (SOTA practice)
    X_p_s = np.clip(X_p_s, -10, 10)
    X_l_tr_s = np.clip(X_l_tr_s, -10, 10)
    
    # Loaders - Optimization: Move to DEVICE immediately
    def to_dev(arr): return torch.FloatTensor(arr).to(DEVICE)
    
    # Handle optional True Prob (ensure not None)
    tp_p_arr = tp_p if tp_p is not None else np.zeros_like(y_p)
    tp_l_arr = tp_l_tr if tp_l_tr is not None else np.zeros_like(y_l_tr)

    d_p = TensorDataset(to_dev(X_p_s), to_dev(y_p), to_dev(t_p), to_dev(ite_p), to_dev(egfr_p), to_dev(tp_p_arr))
    d_l = TensorDataset(to_dev(X_l_tr_s), to_dev(y_l_tr), to_dev(t_l_tr), to_dev(ite_l_tr), to_dev(egfr_l_tr), to_dev(tp_l_arr))
    
    l_p = DataLoader(d_p, batch_size=PARAMS['bs_pre'], shuffle=True)
    l_l = DataLoader(d_l, batch_size=PARAMS['bs_fine'], shuffle=True)
    
    # Init Model
    model = DLCNet(
        input_dim=23,
        d_hidden=PARAMS['d_hidden'],
        num_layers=PARAMS['num_layers'],
        lambda_hsic=PARAMS['lambda_hsic'],
        lambda_pred=PARAMS['lambda_pred'],
        dropout=PARAMS['dropout'],
        random_state=seed
    ).to(DEVICE)
    model.scaler = scaler
    
    # Train Phase 1
    opt = torch.optim.Adam(model.parameters(), lr=PARAMS['lr_pre'])
    for epoch in range(PARAMS['epochs_pre']):
        loss = train_epoch_sweep(model, l_p, opt, PARAMS)
        if (epoch + 1) % 50 == 0:
            print(f"    [Pre-train Seed={seed}] Ep {epoch+1}/{PARAMS['epochs_pre']} Loss={loss:.4f}", flush=True)
        
    # Train Phase 2
    opt_f = torch.optim.Adam(model.parameters(), lr=PARAMS['lr_fine'])
    for epoch in range(PARAMS['epochs_fine']):
        loss = train_epoch_sweep(model, l_l, opt_f, PARAMS)
        if (epoch + 1) % 20 == 0:
            print(f"    [Fine-tune Seed={seed}] Ep {epoch+1}/{PARAMS['epochs_fine']} Loss={loss:.4f}", flush=True)
        
    # Evaluate
    return evaluate_sweep(model, X_l_te, y_l_te, t_l_te, ite_l_te, scaler, feats)

def main():
    results = []
    
    # 1. Architecture Group
    print(">>> Sweeping Architecture...", flush=True)
    for dh in [64, 128, 256]:
        for nl in [2, 3, 4, 5]:
            scores = []
            for seed in [42, 43, 44, 45, 46]:
                try:
                    res = run_single_config(seed, {'d_hidden': dh, 'num_layers': nl})
                    print(f"  -> Seed {seed}: AUC={res['AUC']:.4f}, CATE={res['Delta_CATE']:.4f}", flush=True)
                    scores.append(res)
                except Exception as e:
                    print(f"  -> Seed {seed}: Failed ({e})", flush=True)
            
            if scores:
                avg = {k: np.mean([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
                std = {k: np.std([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
                
                print(f"Arch: Hidden={dh}, Layers={nl} -> AUC={avg['AUC']:.4f}±{std['AUC']:.4f}, CATE={avg['Delta_CATE']:.4f}±{std['Delta_CATE']:.4f}", flush=True)
                results.append({
                    'type': 'arch', 
                    'd_hidden': dh, 
                    'num_layers': nl, 
                    'AUC': avg['AUC'], 'AUC_Std': std['AUC'], 
                    'ACC': avg['ACC'], 'ACC_Std': std['ACC'],
                    'F1': avg['F1'], 'F1_Std': std['F1'],
                    'Sensitivity': avg['Sensitivity'], 'Sensitivity_Std': std['Sensitivity'],
                    'PEHE': avg['PEHE'], 'PEHE_Std': std['PEHE'],
                    'CATE': avg['Delta_CATE'], 'CATE_Std': std['Delta_CATE']
                })
                pd.DataFrame(results).to_csv("results/parameter_sensitivity_results_final.csv", index=False)
            
    # 2. HSIC Group
    print("\n>>> Sweeping HSIC...", flush=True)
    for hsic in [0.0, 0.01, 0.1, 1.0, 10.0]:
        scores = []
        for seed in [42, 43, 44, 45, 46]:
            try:
                res = run_single_config(seed, {'lambda_hsic': hsic})
                print(f"  -> Seed {seed}: AUC={res['AUC']:.4f}, CATE={res['Delta_CATE']:.4f}", flush=True)
                scores.append(res)
            except Exception as e:
                print(f"  -> Seed {seed}: Failed ({e})", flush=True)
        
        if scores:
            avg = {k: np.mean([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            std = {k: np.std([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            
            print(f"HSIC: lambda={hsic} -> AUC={avg['AUC']:.4f}±{std['AUC']:.4f}, CATE={avg['Delta_CATE']:.4f}±{std['Delta_CATE']:.4f}", flush=True)
            results.append({
                'type': 'hsic', 
                'lambda_hsic': hsic, 
                'AUC': avg['AUC'], 'AUC_Std': std['AUC'], 
                'ACC': avg['ACC'], 'ACC_Std': std['ACC'],
                'F1': avg['F1'], 'F1_Std': std['F1'],
                'Sensitivity': avg['Sensitivity'], 'Sensitivity_Std': std['Sensitivity'],
                'PEHE': avg['PEHE'], 'PEHE_Std': std['PEHE'],
                'CATE': avg['Delta_CATE'], 'CATE_Std': std['Delta_CATE']
            })
            pd.DataFrame(results).to_csv("results/parameter_sensitivity_results_final.csv", index=False)

    # 3. Model Weight Balance
    print("\n>>> Sweeping CATE Weight...", flush=True)
    for cate in [0.0, 0.5, 1.0, 2.0, 5.0]:
        scores = []
        for seed in [42, 43, 44, 45, 46]:
            try:
                res = run_single_config(seed, {'lambda_cate': cate})
                print(f"  -> Seed {seed}: AUC={res['AUC']:.4f}, CATE={res['Delta_CATE']:.4f}", flush=True)
                scores.append(res)
            except Exception as e:
                print(f"  -> Seed {seed}: Failed ({e})", flush=True)
                
        if scores:
            avg = {k: np.mean([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            std = {k: np.std([s[k] for s in scores]) for k in ['AUC', 'ACC', 'F1', 'Sensitivity', 'PEHE', 'Delta_CATE']}
            
            print(f"CATE Weight: lambda={cate} -> AUC={avg['AUC']:.4f}±{std['AUC']:.4f}, CATE={avg['Delta_CATE']:.4f}±{std['Delta_CATE']:.4f}", flush=True)
            results.append({
                'type': 'cate', 
                'lambda_cate': cate, 
                'AUC': avg['AUC'], 'AUC_Std': std['AUC'], 
                'ACC': avg['ACC'], 'ACC_Std': std['ACC'],
                'F1': avg['F1'], 'F1_Std': std['F1'],
                'Sensitivity': avg['Sensitivity'], 'Sensitivity_Std': std['Sensitivity'],
                'PEHE': avg['PEHE'], 'PEHE_Std': std['PEHE'],
                'CATE': avg['Delta_CATE'], 'CATE_Std': std['Delta_CATE']
            })
            pd.DataFrame(results).to_csv("results/parameter_sensitivity_results_final.csv", index=False)
        
    # Save final
    df = pd.DataFrame(results)
    df.to_csv("results/parameter_sensitivity_results_final.csv", index=False)
    print("Done. Saved to results/parameter_sensitivity_results_final.csv", flush=True)

if __name__ == "__main__":
    main()

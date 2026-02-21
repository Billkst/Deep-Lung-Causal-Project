
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Project Imports
# Ensure project root is in path
sys.path.append("/home/UserData/ljx/Project_1")

from src.dlc.dlc_net import DLCNet
from src.dlc.ground_truth import GroundTruthGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
BLACKLIST_PATH = DATA_DIR / "PANCAN" / "pancan_leakage_blacklist.csv"

RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data_two_stage(random_state=42):
    """
    Load PANCAN and LUAD data separately for Transfer Learning.
    Returns:
        data_dict: {
            'pancan': (X, y, ite, egfr),
            'luad_train': (X, y, ite, egfr),
            'luad_test': (X, y, ite, egfr),
            'scaler': scaler_fitted_on_pancan
        }
    """
    print(f"[Data] Loading PANCAN...", flush=True)
    df_p = pd.read_csv(DATA_DIR / "pancan_synthetic_interaction.csv")
    
    # Filter Blacklist
    if BLACKLIST_PATH.exists():
        try:
            blacklist_df = pd.read_csv(BLACKLIST_PATH)
            if 'sampleID' in blacklist_df.columns:
                leak_ids = set(blacklist_df['sampleID'].astype(str).values)
            else:
                 leak_ids = set(blacklist_df.iloc[:,0].astype(str).values)
            
            if 'sampleID' in df_p.columns:
                initial_len = len(df_p)
                df_p = df_p[~df_p['sampleID'].astype(str).isin(leak_ids)].reset_index(drop=True)
                print(f"[Data] Filtered {initial_len - len(df_p)} leakage samples from PANCAN.", flush=True)
        except Exception as e:
            print(f"[Data] Warning: Blacklist filtering failed: {e}", flush=True)

    if 'sampleID' in df_p.columns: df_p = df_p.drop(columns=['sampleID'])

    print(f"[Data] Loading LUAD...", flush=True)
    df_l = pd.read_csv(DATA_DIR / "luad_synthetic_interaction.csv")
    if 'sampleID' in df_l.columns: df_l = df_l.drop(columns=['sampleID'])
    
    gt_gen = GroundTruthGenerator()

    # Process PANCAN
    ite_p = gt_gen.compute_true_ite(df_p).astype(np.float32)
    egfr_p = df_p['EGFR'].values.astype(np.float32) if 'EGFR' in df_p.columns else np.zeros(len(df_p), dtype=np.float32)
    y_p = df_p['Outcome_Label'].values.astype(np.float32)
    
    exclude = ['sampleID', 'Outcome_Label', 'True_Prob', 'True_ITE', 'Treatment']
    # Align columns
    common_cols = [c for c in df_l.columns if c not in exclude]
    X_p = df_p[common_cols].values.astype(np.float32)
    
    # Process LUAD
    ite_l = gt_gen.compute_true_ite(df_l).astype(np.float32)
    egfr_l = df_l['EGFR'].values.astype(np.float32)
    y_l = df_l['Outcome_Label'].values.astype(np.float32)
    X_l = df_l[common_cols].values.astype(np.float32)
    
    # Split LUAD
    X_l_train, X_l_test, y_l_train, y_l_test, ite_l_train, ite_l_test, egfr_l_train, egfr_l_test = train_test_split(
        X_l, y_l, ite_l, egfr_l, test_size=0.2, random_state=random_state, stratify=y_l
    )
    
    # Scale on PANCAN (The Domain Source)
    scaler = StandardScaler()
    scaler.fit(X_p)
    
    X_p_scaled = scaler.transform(X_p)
    X_l_train_scaled = scaler.transform(X_l_train)
    X_l_test_scaled = scaler.transform(X_l_test)
    
    return {
        'pancan': (X_p_scaled, y_p, ite_p, egfr_p),
        'luad_train': (X_l_train_scaled, y_l_train, ite_l_train, egfr_l_train),
        'luad_test': (X_l_test_scaled, y_l_test, ite_l_test, egfr_l_test)
    }

class DiamondTrainer:
    def __init__(self, seed, d_hidden=256, lr=1e-3):
        self.device = DEVICE
        self.seed = seed
        set_global_seed(seed)
        
        # Model Init
        self.model = DLCNet(
            input_dim=23,
            d_hidden=d_hidden,
            num_layers=3,
            dropout=0.006 
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Hyperparams (from Previous SOTA)
        self.lambda_pred = 2.5
        self.lambda_ite = 1.0
        self.lambda_hsic = 0.06
        self.lambda_cate = 5.0
        
    def train_epoch(self, loader, stage_name, epoch):
        self.model.train()
        loss_total_sum = 0
        loss_pred_sum = 0
        loss_ite_sum = 0
        
        for bx, by, bite, begfr in loader:
            self.optimizer.zero_grad()
            out = self.model(bx)
            
            # T inference (Standardized PM2.5 > 0 -> T=1)
            t_true = (bx[:, 22] > 0).float()
            
            # Assign lambdas
            self.model.lambda_pred = self.lambda_pred
            self.model.lambda_ite = self.lambda_ite
            self.model.lambda_hsic = self.lambda_hsic
            
            # Base compute_loss (Recon + HSIC + Pred)
            loss_dict = self.model.compute_loss(bx, by.flatten(), out, t_true)
            loss_base = loss_dict['loss_total']
            
            # Manual ITE Supervision
            loss_ite = F.mse_loss(out['ITE'].squeeze(), bite)
            
            # Manual CATE Supervision (EGFR based)
            loss_cate = torch.tensor(0.0, device=self.device)
            mask_mut = begfr > 0.5
            mask_wt = ~mask_mut
            if mask_mut.any() and mask_wt.any():
                d_pred = out['ITE'].squeeze()[mask_mut].mean() - out['ITE'].squeeze()[mask_wt].mean()
                d_true = bite[mask_mut].mean() - bite[mask_wt].mean()
                loss_cate = F.mse_loss(d_pred, d_true)
            
            # Combine
            loss_final = loss_base + self.lambda_ite * loss_ite + self.lambda_cate * loss_cate
            
            loss_final.backward()
            self.optimizer.step()
            
            loss_total_sum += loss_final.item()
            loss_pred_sum += loss_dict['loss_pred'].item()
            loss_ite_sum += loss_ite.item()
            
        return loss_total_sum / len(loader), loss_pred_sum / len(loader), loss_ite_sum / len(loader)

    def fit_pipeline(self, data):
        # Unpack
        X_p, y_p, ite_p, egfr_p = data['pancan']
        X_l, y_l, ite_l, egfr_l = data['luad_train']
        
        # Prepare Loaders
        # PANCAN Loader
        ds_p = TensorDataset(
            torch.tensor(X_p).float().to(self.device),
            torch.tensor(y_p).float().unsqueeze(1).to(self.device),
            torch.tensor(ite_p).float().to(self.device),
            torch.tensor(egfr_p).float().to(self.device)
        )
        loader_p = DataLoader(ds_p, batch_size=64, shuffle=True)
        
        # LUAD Loader
        ds_l = TensorDataset(
            torch.tensor(X_l).float().to(self.device),
            torch.tensor(y_l).float().unsqueeze(1).to(self.device),
            torch.tensor(ite_l).float().to(self.device),
            torch.tensor(egfr_l).float().to(self.device)
        )
        loader_l = DataLoader(ds_l, batch_size=32, shuffle=True)
        
        # === STAGE 1: Pre-training (PANCAN) ===
        epochs_s1 = 100
        print(f"\n[Seed {self.seed}] Starting Stage 1: PANCAN Pre-training ({epochs_s1} epochs)...", flush=True)
        
        for ep in range(1, epochs_s1 + 1):
            # Warmup
            if ep <= 10:
                self.lambda_pred = 3.5 # Boost Pred initially (from SOTA report)
                self.lambda_ite = 0.0
                self.lambda_cate = 0.0
            else:
                self.lambda_pred = 2.5
                self.lambda_ite = 1.0 # Standard ITE
                self.lambda_cate = 5.0
            
            l_tot, l_pred, l_ite = self.train_epoch(loader_p, "Stage1", ep)
            
            if ep % 20 == 0:
                print(f"  [Stage 1] Ep {ep} | Loss: {l_tot:.4f} (Pred: {l_pred:.4f}, ITE: {l_ite:.4f})", flush=True)
        
        # === STAGE 2: Fine-tuning (LUAD) ===
        epochs_s2 = 50
        print(f"\n[Seed {self.seed}] Starting Stage 2: LUAD Fine-tuning ({epochs_s2} epochs)...", flush=True)
        
        # Reduce LR for fine-tuning
        for pg in self.optimizer.param_groups: pg['lr'] = 0.0005
        
        for ep in range(1, epochs_s2 + 1):
            # Keep Lambdas constant
            self.lambda_pred = 2.5
            self.lambda_ite = 1.0
            self.lambda_cate = 5.0
            
            l_tot, l_pred, l_ite = self.train_epoch(loader_l, "Stage2", ep)
            
            if ep % 10 == 0:
                print(f"  [Stage 2] Ep {ep} | Loss: {l_tot:.4f} (Pred: {l_pred:.4f}, ITE: {l_ite:.4f})", flush=True)

    def evaluate(self, X_test, y_test, ite_true, egfr):
        self.model.eval()
        X_ten = torch.tensor(X_test).float().to(self.device)
        
        with torch.no_grad():
            out = self.model(X_ten)
            y0 = out['Y_0'].squeeze()
            y1 = out['Y_1'].squeeze()
            t_inferred = (X_ten[:, 22] > 0).float()
            
            # Predict Proba
            y_prob = t_inferred * y1 + (1 - t_inferred) * y0
            y_prob_np = y_prob.cpu().numpy()
            
            # ITE
            cate_pred = (y1 - y0).cpu().numpy()
            
        y_pred = (y_prob_np > 0.5).astype(int)
        
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        try:
            auc = roc_auc_score(y_test, y_prob_np)
        except:
            auc = 0.5
            
        pehe = np.sqrt(np.mean((ite_true - cate_pred)**2))
        
        # Delta CATE
        mask_high = egfr > 0.5
        cate_high = np.mean(cate_pred[mask_high])
        cate_low = np.mean(cate_pred[~mask_high])
        delta_cate = cate_high - cate_low
        
        return {
            'AUC': float(auc),
            'PEHE': float(pehe),
            'Delta CATE': float(delta_cate),
            'Acc': float(accuracy_score(y_test, y_pred)),
            'F1': float(f1_score(y_test, y_pred))
        }

def main():
    print("=== Diamond Protocol: Two-Stage Transfer Learning ===", flush=True)
    seeds = [42, 43, 44, 45, 46]
    results = []
    
    for seed in seeds:
        print(f"\n>>> Processing Seed {seed}", flush=True)
        set_global_seed(seed)
        
        # Reload data each time to clear splits logic
        data = load_data_two_stage(seed)
        
        trainer = DiamondTrainer(seed, d_hidden=256) # Hidden 256 from SOTA report
        trainer.fit_pipeline(data)
        
        # Evaluate
        X_test, y_test, ite_test, egfr_test = data['luad_test']
        metrics = trainer.evaluate(X_test, y_test, ite_test, egfr_test)
        metrics['Seed'] = seed
        results.append(metrics)
        
        print(f"[Seed {seed} Result] AUC: {metrics['AUC']:.4f} | PEHE: {metrics['PEHE']:.4f}", flush=True)
        
    # Summary
    aucs = [r['AUC'] for r in results]
    pehes = [r['PEHE'] for r in results]
    
    print("\n=== FINAL DIAMOND SUMMARY ===", flush=True)
    print(f"AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"PEHE: {np.mean(pehes):.4f} ± {np.std(pehes):.4f}", flush=True)
    
    # Save
    out_path = RESULTS_DIR / "diamond_protocol_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()

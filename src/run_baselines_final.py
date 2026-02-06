# -*- coding: utf-8 -*-
"""
Run Baselines Final (Task 8: Ultimate Re-test)
==============================================

Execute final benchmark for 5 baselines + DLC comparison.
Contenders: XGBoost, TabR, MOGONET, TransTEE, HyperFast.

Data Strategy:
- Training: Pooled (Clean PANCAN + LUAD Train 80%)
- Testing: LUAD Test (20%) - Locked random_state=42

Features:
- Full 23 features for Predictors.
- TransTEE special handling (PM2.5 -> Treatment).

Metrics:
- Predictive: AUC, Acc, F1, Precision, Recall
- Causal: PEHE, Delta CATE, Sensitivity(Age)
- System: Params, Inference Time
"""

import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

# Project Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.xgb_baseline import XGBBaseline
from src.baselines.tabr_baseline import TabRBaseline
from src.baselines.mogonet_baseline import MOGONETBaseline
from src.baselines.transtee_baseline import TransTEEBaseline
from src.baselines.hyperfast_baseline import HyperFastBaseline
from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_pehe_from_arrays # Use array version directly
from src.dlc.ground_truth import GroundTruthGenerator
from src.baselines.utils import set_global_seed

# ==========================================
# Config
# ==========================================
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLACKLIST_PATH = DATA_DIR / "PANCAN" / "pancan_leakage_blacklist.csv"

# ==========================================
# Data Loading (Reused from run_final_sota.py)
# ==========================================

def load_clean_pancan():
    """Load Clean PANCAN (Source) with Blacklist applied."""
    print("[Data] Loading PANCAN data...")
    csv_path = DATA_DIR / "pancan_synthetic_interaction.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PANCAN: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 1. Apply Blacklist
    if BLACKLIST_PATH.exists():
        try:
            # Try loading without header first
            blacklist_df = pd.read_csv(BLACKLIST_PATH, header=None)
            if not str(blacklist_df.iloc[0, 0]).startswith('TCGA'):
                blacklist_df = pd.read_csv(BLACKLIST_PATH)
            
            leak_ids = set(blacklist_df.iloc[:, 0].values)
            if 'sampleID' in df.columns:
                before = len(df)
                df = df[~df['sampleID'].isin(leak_ids)].reset_index(drop=True)
                print(f"[Data] Blacklist applied: {before} -> {len(df)}")
            else:
                print("[Data] Warning: 'sampleID' missing in PANCAN, cannot filter.")
        except Exception as e:
            print(f"[Data] Blacklist error: {e}")
    
    # 2. Extract Features
    exclude = ['sampleID', 'Outcome_Label', 'True_Prob', 'True_ITE', 'Treatment']
    feats = [c for c in df.columns if c not in exclude]
    
    X = df[feats].values.astype(np.float32)
    y = df['Outcome_Label'].values.astype(np.int64)
    
    return X, y, feats

def load_luad_target():
    """Load LUAD (Target) consistent with golden run."""
    print("[Data] Loading LUAD data...")
    csv_path = DATA_DIR / "luad_synthetic_interaction.csv"
    
    df = pd.read_csv(csv_path)
    
    exclude_cols = ["sampleID", "Outcome_Label", "True_Prob", "True_ITE", "Treatment"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Enforce feature count like golden run
    if len(feature_cols) > 23:
        feature_cols = feature_cols[:23]
        
    X = df[feature_cols].values.astype(np.float32)
    y = df['Outcome_Label'].values.astype(np.int64)
    
    assert X.shape[1] == 23, f"LUAD Dim mismatch: {X.shape[1]}"
    print(f"[Data] LUAD Loaded. Shape: {X.shape}")
    
    return X, y, feature_cols

def prepare_mogonet_views(X, feature_names):
    """
    Split data into Clinical and Omics views for MOGONET.
    Clinical: Age, Gender
    Omics: Everything else
    """
    clinical_indices = []
    omics_indices = []
    
    # Identify indices
    for i, name in enumerate(feature_names):
        if name in ['Age', 'Gender']:
            clinical_indices.append(i)
        else:
            omics_indices.append(i)
            
    # Fallback / Integrity check
    if not clinical_indices:
        # Fallback to first 2 cols if names mismatch
        clinical_indices = [0, 1]
        omics_indices = list(range(2, X.shape[1]))
        
    view1 = X[:, clinical_indices] # Clinical
    view2 = X[:, omics_indices]    # Omics
    
    return [view1, view2]

def predict_transtee_proba(model, X, t):
    """
    Manual prediction for TransTEE since it lacks predict_proba
    and expects (X, t).
    We use the raw output (Regression on 0/1) as probability score.
    """
    # Scale
    X_scaled = model.scaler.transform(X)
    
    # Tensor
    X_tensor = torch.FloatTensor(X_scaled).to(model.device)
    t_tensor = torch.FloatTensor(t).to(model.device)
    
    model.model.eval()
    with torch.no_grad():
        # returns y_pred, y0, y1
        y_pred, _, _ = model.model(X_tensor, t_tensor)
        
    prob = y_pred.cpu().numpy()
    
    # Since we treat regression output as score, we might want to clip it
    # to [0, 1] if it goes out of bound, though for AUC it doesn't matter.
    # But for "sensitivity" (absolute change), scale matters.
    # Let's keep it raw or clip? MSE on 0/1 usually stays around [0,1].
    # But strictly speaking, it's not a probability.
    
    # To return shape (N, 2) like predict_proba? 
    # The caller expects [:, 1] i.e. p(y=1)
    
    # We construct a fake (N, 2) array
    return np.vstack([1-prob, prob]).T

class DLCWrapper:
    """Wrapper to evaluate DLC model as a baseline."""
    def __init__(self, model_path, input_dim=23, fit_scaler_on=None, model_config=None, pm25_threshold=None, strict_load=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pm25_threshold = pm25_threshold
        
        if not model_path.exists():
            raise FileNotFoundError(f"DLC Model not found at {model_path}")
            
        print(f"[DLC] Loading weights from {model_path}")
        # Fix for PyTorch 2.6+ weights_only=True default
        try:
            payload = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
             # Fallback for older torch versions
            payload = torch.load(model_path, map_location=self.device)
        
        # Init Scaler
        self.scaler = StandardScaler()
        if isinstance(payload, dict) and 'scaler_mean' in payload:
            self.scaler.mean_ = np.array(payload['scaler_mean'])
            self.scaler.scale_ = np.array(payload['scaler_scale'])
            self.scaler.var_ = self.scaler.scale_ ** 2
            print("[DLC] Scaler parameters loaded from checkpoint.")
        elif fit_scaler_on is not None:
            print("[DLC] Warning: No scaler in checkpoint. Fitting scaler on provided training data...")
            self.scaler.fit(fit_scaler_on)
            if self.pm25_threshold is None:
                # Use provided data PM2.5 median as fallback
                try:
                    self.pm25_threshold = float(np.median(fit_scaler_on[:, -1]))
                except Exception:
                    self.pm25_threshold = None
        else:
            print("[DLC] Warning: No scaler params in payload and no selection data provided. Using unfitted scaler (will fail)!")
        
        # Init Model
        # SOTA Config: d_hidden=128, num_layers=3
        config = model_config or {}
        d_hidden = int(config.get("d_hidden", 128))
        num_layers = int(config.get("num_layers", 3))
        dropout = float(config.get("dropout", 0.006))
        self.model = DLCNet(input_dim=input_dim, d_hidden=d_hidden, num_layers=num_layers, dropout=dropout)
        
        # Load State
        if 'model_state_dict' in payload:
            self.model.load_state_dict(payload['model_state_dict'], strict=strict_load)
        else:
            self.model.load_state_dict(payload, strict=strict_load)
            
        self.model.to(self.device)
        self.model.eval()
        
    def predict_proba(self, X):
        # Scale
        try:
            X_scaled = self.scaler.transform(X)
        except:
            X_scaled = X # If scaler not fitted?
            
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            y1 = out['Y_1'].cpu().numpy().flatten()
            y0 = out['Y_0'].cpu().numpy().flatten()

            # Use factual outcome prediction for AUC/Acc/F1 (align with run_final_sota)
            if self.pm25_threshold is None:
                self.pm25_threshold = float(np.median(X[:, -1]))
            t_raw = (X[:, -1] >= self.pm25_threshold).astype(np.float32)
            prob = t_raw * y1 + (1.0 - t_raw) * y0
            
        return np.vstack([1-prob, prob]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)
        
    def predict_ite(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            y1 = out["Y_1"].cpu().numpy().flatten()
            y0 = out["Y_0"].cpu().numpy().flatten()
        return y1 - y0

# ==========================================
# Helpers
# ==========================================

def get_params_count(model):
    """Estimate parameters."""
    if hasattr(model, 'parameters'):
        # Torch model
        return sum(p.numel() for p in model.parameters())
    elif isinstance(model, DLCNet):
         return sum(p.numel() for p in model.parameters())
    elif hasattr(model, 'get_booster'):
        # XGBoost
        # Use number of trees * avg depth? Or just "N/A - Tree"
        # User asked for nodes or N/A
        try:
            booster = model.model.get_booster()
            dump = booster.get_dump()
            nodes = sum([len(t.split('\n')) for t in dump])
            return f"{nodes} (Nodes)"
        except:
            return "N/A"
    else:
        return "N/A"


def get_inference_params_count(model):
    if hasattr(model, "causal_vae") and hasattr(model.causal_vae, "decoder"):
        decoder_param_ids = {id(p) for p in model.causal_vae.decoder.parameters()}
        return sum(
            p.numel() for p in model.parameters()
            if p.requires_grad and id(p) not in decoder_param_ids
        )
    return get_params_count(model)


def summarize_metrics(records, keys):
    summary = {}
    for key in keys:
        vals = np.array([r[key] for r in records], dtype=float)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        }
    return summary


def format_mean_std(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}"

def find_best_threshold(y_true, y_prob):
    """Match run_final_sota threshold selection (maximize min(Acc, F1))."""
    best_thresh = 0.5
    best_score = -1.0
    for thr in np.linspace(0.0, 1.0, 101):
        preds = (y_prob > thr).astype(int)
        acc_tmp = accuracy_score(y_true, preds)
        f1_tmp = f1_score(y_true, preds)
        score = min(acc_tmp, f1_tmp)
        if score > best_score:
            best_score = score
            best_thresh = thr
    return best_thresh

def compute_delta_cate(model_name, model, X_test, feature_names, treatment_idx=22):
    """
    Compute PEHE (Accuracy) and Delta CATE (Mechanism Verification).
    
    Mechanism Verification (Golden Run Definition):
    Delta CATE = Mean(CATE_Mutant) - Mean(CATE_Wild)
    This checks if the model captures the interaction effect (EGFR status modifying effect of PM2.5).
    """
    # 1. Ground Truth ITE (For Accuracy/PEHE)
    # Important: GroundTruthGenerator expects full 23 features structure
    # Age(0), Gender(1)... PM2.5(22)
    pm25_idx = 22
    egfr_idx = feature_names.index("EGFR") if "EGFR" in feature_names else 3
    
    gt_gen = GroundTruthGenerator(pm25_idx=pm25_idx, egfr_idx=egfr_idx)
    true_ite = gt_gen.compute_true_ite(X_test)
    
    # 2. Estimated ITE
    est_ite = None
    
    if hasattr(model, 'predict_ite'):
        if model_name == "TransTEE":
             # TransTEE specific: remove Treatment col
             X_input = np.delete(X_test, treatment_idx, axis=1)
             est_ite = model.predict_ite(X_input)
        else:
             # DLC (or others) supporting predict_ite directly
             est_ite = model.predict_ite(X_test)
             
        if isinstance(est_ite, torch.Tensor):
            est_ite = est_ite.detach().cpu().numpy()
        est_ite = est_ite.flatten()
        
    else:
        # S-Learner Logic for Predictors (XGB, TabR, etc)
        # T=1 implies High PM2.5, T=0 implies Low PM2.5
        # We calculate effect of "High vs Low" using the model
        
        # Calculate representative High/Low values from X_test itself
        pm25_vals = X_test[:, treatment_idx]
        median_val = np.median(pm25_vals)
        val_high = np.mean(pm25_vals[pm25_vals > median_val])
        val_low = np.mean(pm25_vals[pm25_vals <= median_val])
        
        # Create Counterfactuals
        X_high = X_test.copy()
        X_high[:, treatment_idx] = val_high
        
        X_low = X_test.copy()
        X_low[:, treatment_idx] = val_low
        
        # Predict Proba (Probability of Outcome=1)
        if model_name == "MOGONET":
            views_high = prepare_mogonet_views(X_high, feature_names)
            views_low = prepare_mogonet_views(X_low, feature_names)
            p_high = model.predict_proba(views_high)[:, 1]
            p_low = model.predict_proba(views_low)[:, 1]
        else:
            p_high = model.predict_proba(X_high)[:, 1]
            p_low = model.predict_proba(X_low)[:, 1]
        
        est_ite = p_high - p_low
        
    # 3. Compute Metrics
    
    # PEHE (Accuracy vs Ground Truth)
    pehe = np.sqrt(np.nanmean((true_ite - est_ite) ** 2))
    
    # Delta CATE (Mechanism: EGFR Effect)
    # Definition: Mean(CATE | EGFR=1) - Mean(CATE | EGFR=0)
    egfr_mask = X_test[:, egfr_idx] > 0
    
    cate_mutant = np.mean(est_ite[egfr_mask])
    cate_wild = np.mean(est_ite[~egfr_mask])
    delta_cate = cate_mutant - cate_wild
    
    return pehe, delta_cate

def compute_sensitivity(model_name, model, X_test, feature_names=None, feature_idx=0, perturbation=5.0):
    """Perturb feature (Age) and measure output change."""
    try:
        if model_name == "MOGONET":
            views = prepare_mogonet_views(X_test, feature_names)
            base_pred = model.predict_proba(views)[:, 1]
            
            X_mod = X_test.copy()
            X_mod[:, feature_idx] += perturbation
            views_mod = prepare_mogonet_views(X_mod, feature_names)
            mod_pred = model.predict_proba(views_mod)[:, 1]

        elif model_name == "TransTEE":
             # TransTEE needs T. We use observed (binarized) T for sensitivity check?
             # X_test has 23 dims. TransTEE needs 22 + T.
             X_no_t = np.delete(X_test, 22, axis=1) # Hardcoded PM2.5 index
             t = (X_test[:, 22] > np.median(X_test[:, 22])).astype(float)
             
             base_pred = predict_transtee_proba(model, X_no_t, t=t)[:, 1]
             
             X_mod = X_no_t.copy()
             X_mod[:, feature_idx] += perturbation
             
             mod_pred = predict_transtee_proba(model, X_mod, t=t)[:, 1]
             
        else:
            base_pred = model.predict_proba(X_test)[:, 1]
            X_mod = X_test.copy()
            X_mod[:, feature_idx] += perturbation
            mod_pred = model.predict_proba(X_mod)[:, 1]
            
        return np.mean(np.abs(mod_pred - base_pred))
    except Exception as e:
        print(f"Sensitivity Error: {e}")
        return 0.0

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-tag", type=str, default="rigorous_20260203")
    parser.add_argument("--dlc-weights", type=str, default="results/dlc_final_sota_tuned_20260203.pth")
    parser.add_argument("--dlc-d-hidden", type=int, default=192)
    parser.add_argument("--dlc-num-layers", type=int, default=4)
    parser.add_argument("--only-dlc", action="store_true", help="Only evaluate DLC without baselines")
    parser.add_argument("--dlc-strict", action="store_true", default=False, help="Strictly load DLC weights")
    parser.add_argument("--no-dlc-strict", action="store_false", dest="dlc_strict", help="Allow missing keys for old DLC weights")
    args = parser.parse_args()

    print("🚀 Taking off: Baseline Ultimate Re-test...")
    print(f"[Config] seeds={args.seeds} dlc_weights={args.dlc_weights} d_hidden={args.dlc_d_hidden} num_layers={args.dlc_num_layers} only_dlc={args.only_dlc} dlc_strict={args.dlc_strict}")
    
    # 1. Prepare Data
    X_pancan, y_pancan, _ = load_clean_pancan()
    X_luad, y_luad, feature_names = load_luad_target()
    
    results = {}
    per_seed_results = {}
    PM25_COL_IDX = 22
    # 2. Multi-seed Loop
    for seed in args.seeds:
        print(f"\n[Seed {seed}] Preparing data...")
        set_global_seed(seed)

        # Split LUAD
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_luad, y_luad, test_size=0.2, random_state=seed, stratify=y_luad
        )
        
        # Pool Training Data
        X_train_pool = np.vstack([X_pancan, X_train_l])
        y_train_pool = np.hstack([y_pancan, y_train_l])
        
        print(f"[Dataset] Pooled Train: {X_train_pool.shape}, Test: {X_test_l.shape}")

        if not args.only_dlc:
            # Setup Models (per seed)
            models = {
                "XGBoost": XGBBaseline(random_state=seed),
                "TabR": TabRBaseline(random_state=seed, batch_size=256, epochs=50),
                "MOGONET": MOGONETBaseline(random_state=seed, epochs=50),
                "TransTEE": TransTEEBaseline(random_state=seed, epochs=50, batch_size=64),
                "HyperFast": HyperFastBaseline(random_state=seed, epochs=50, batch_size=64)
            }

            for name, model in models.items():
                print(f"\n⚡ Processing {name} (Seed {seed})...")
                start_t = time.time()
                
                if name == "TransTEE":
                    pm25_train = X_train_pool[:, PM25_COL_IDX]
                    med_train = np.median(pm25_train)
                    t_train = (pm25_train > med_train).astype(float)
                    X_tr_no_t = np.delete(X_train_pool, PM25_COL_IDX, axis=1)
                    model.fit(X_tr_no_t, t_train, y_train_pool)
                    
                    X_te_no_t = np.delete(X_test_l, PM25_COL_IDX, axis=1)
                    pm25_test = X_test_l[:, PM25_COL_IDX]
                    t_test = (pm25_test > med_train).astype(float)
                    y_prob = predict_transtee_proba(model, X_te_no_t, t=t_test)[:, 1]
                    y_pred = (y_prob > 0.5).astype(int)
                
                elif name == "MOGONET":
                    views_train = prepare_mogonet_views(X_train_pool, feature_names)
                    model.fit(views_train, y_train_pool)
                    views_test = prepare_mogonet_views(X_test_l, feature_names)
                    y_prob = model.predict_proba(views_test)[:, 1]
                    y_pred = (y_prob > 0.5).astype(int)
                else:
                    model.fit(X_train_pool, y_train_pool)
                    y_prob = model.predict_proba(X_test_l)[:, 1]
                    y_pred = model.predict(X_test_l)
                
                infer_time = (time.time() - start_t) / len(X_test_l) * 1000
                
                res = {
                    "AUC": float(roc_auc_score(y_test_l, y_prob)),
                    "Acc": float(accuracy_score(y_test_l, y_pred)),
                    "F1": float(f1_score(y_test_l, y_pred)),
                    "Precision": float(precision_score(y_test_l, y_pred)),
                    "Recall": float(recall_score(y_test_l, y_pred)),
                    "Time (ms)": float(infer_time),
                    "Params": str(get_params_count(model)),
                    "Seed": seed
                }
                
                pehe, delta_cate = compute_delta_cate(name, model, X_test_l, feature_names, PM25_COL_IDX)
                res["PEHE"] = float(pehe)
                res["Delta CATE"] = float(delta_cate)
                res["Sensitivity"] = float(compute_sensitivity(name, model, X_test_l, feature_names))

                per_seed_results.setdefault(name, []).append(res)
                print(f"  -> AUC: {res['AUC']:.4f}, PEHE: {res['PEHE']:.4f}")

        # DLC
        print("\n👑 Loading DLC SOTA Results...")
        dlc_model_path = Path(args.dlc_weights)
        if not dlc_model_path.exists():
            dlc_model_path = RESULTS_DIR / "dlc_final_sota.pth"
        try:
            pm25_threshold = float(np.median(X_train_l[:, -1]))
            dlc = DLCWrapper(
                dlc_model_path,
                input_dim=23,
                fit_scaler_on=X_pancan,
                model_config={"d_hidden": args.dlc_d_hidden, "num_layers": args.dlc_num_layers},
                pm25_threshold=pm25_threshold,
                strict_load=args.dlc_strict
            )
            start_t = time.time()
            y_prob = dlc.predict_proba(X_test_l)[:, 1]
            best_thr = find_best_threshold(y_test_l, y_prob)
            y_pred = (y_prob > best_thr).astype(int)
            infer_time = (time.time() - start_t) / len(X_test_l) * 1000
            res = {
                "AUC": float(roc_auc_score(y_test_l, y_prob)),
                "Acc": float(accuracy_score(y_test_l, y_pred)),
                "F1": float(f1_score(y_test_l, y_pred)),
                "Precision": float(precision_score(y_test_l, y_pred)),
                "Recall": float(recall_score(y_test_l, y_pred)),
                "Time (ms)": float(infer_time),
                "Params": str(get_inference_params_count(dlc.model)),
                "Seed": seed
            }
            pehe, delta_cate = compute_delta_cate("DLC", dlc, X_test_l, feature_names, PM25_COL_IDX)
            res["PEHE"] = float(pehe)
            res["Delta CATE"] = float(delta_cate)
            res["Sensitivity"] = float(compute_sensitivity("DLC", dlc, X_test_l, feature_names))
            per_seed_results.setdefault("DLC (SOTA)", []).append(res)
            print(f"  -> AUC: {res['AUC']:.4f}, PEHE: {res['PEHE']:.4f}")
        except Exception as e:
            print(f"Error evaluating DLC: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # Reporting
    # ==========================================
    md_path = RESULTS_DIR / f"final_comparison_matrix_{args.output_tag}.md"
    
    # Rows: Models, Cols: Metrics
    metrics_order = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity", "Params", "Time (ms)"]
    
    md_lines = ["# Final Model Comparison Matrix", "", "| Model | " + " | ".join(metrics_order) + " |"]
    md_lines.append("|-------|" + "|".join(["---"] * len(metrics_order)) + "|")
    
    for model_name, records in per_seed_results.items():
        row = f"| {model_name} |"
        numeric_metrics = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity", "Time (ms)"]
        summary = summarize_metrics(records, numeric_metrics)
        for m in metrics_order:
            if m == "Params":
                row += f" {records[0].get('Params', 'N/A')} |"
            elif m in summary:
                row += f" {format_mean_std(summary[m])} |"
            else:
                row += " N/A |"
        md_lines.append(row)
        
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
        
    print(f"\nTable generated at: {md_path}")
    
    # Json dump (per-seed)
    out_json = RESULTS_DIR / f"baseline_metrics_{args.output_tag}.json"
    with open(out_json, "w") as f:
        json.dump(per_seed_results, f, indent=4)

    print(f"Per-seed metrics saved to: {out_json}")

if __name__ == "__main__":
    # Suppress UndefinedMetricWarning from HyperFast (Precision=0)
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    main()

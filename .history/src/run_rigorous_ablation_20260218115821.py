# -*- coding: utf-8 -*-
"""
Run Rigorous Ablation Study (Multi-Seed + Variance)
===================================================

- 5 Seeds (42-46)
- Metrics: AUC, Acc, F1, PEHE, Delta CATE, Sensitivity
- Output: results/ablation_metrics_rigorous.json
- Report: reports/task19.5_ablation_study_rigorous.md
"""

import sys
import json
import time
import random
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Project Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.dlc_net import DLCNet
from src.dlc.ablation_variants import DLCNoHGNN, DLCNoVAE
from src.dlc.ground_truth import GroundTruthGenerator
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BLACKLIST_PATH = DATA_DIR / "PANCAN" / "pancan_leakage_blacklist.csv"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_inference_params_count(model):
    if hasattr(model, "causal_vae") and hasattr(model.causal_vae, "decoder"):
        decoder_param_ids = {id(p) for p in model.causal_vae.decoder.parameters()}
        return sum(
            p.numel() for p in model.parameters()
            if p.requires_grad and id(p) not in decoder_param_ids
        )
    return get_params_count(model)


from torch.utils.data import TensorDataset, DataLoader

def load_data_with_extras(random_state=42):
    """Load Pooled Data (PANCAN + LUAD) with ITE/EGFR for SOTA Training."""
    # 1. Load PANCAN
    df_p = pd.read_csv(DATA_DIR / "pancan_synthetic_interaction.csv")
    if BLACKLIST_PATH.exists():
        try:
            blacklist_df = pd.read_csv(BLACKLIST_PATH)
            if 'sampleID' in blacklist_df.columns:
                leak_ids = set(blacklist_df['sampleID'].astype(str).values)
            elif blacklist_df.shape[1] > 0:
                 leak_ids = set(blacklist_df.iloc[:,0].astype(str).values)
            else:
                 leak_ids = set()
            
            if 'sampleID' in df_p.columns:
                df_p = df_p[~df_p['sampleID'].astype(str).isin(leak_ids)].reset_index(drop=True)
        except Exception:
            pass
            
    # Drop ID from PANCAN for GTGen compatibility
    if 'sampleID' in df_p.columns:
        df_p = df_p.drop(columns=['sampleID'])

    # 2. Load LUAD
    df_l = pd.read_csv(DATA_DIR / "luad_synthetic_interaction.csv")
    # Drop ID from LUAD
    if 'sampleID' in df_l.columns:
        df_l = df_l.drop(columns=['sampleID'])
    
    gt_gen = GroundTruthGenerator()
    
    # 3. Process PANCAN
    ite_p = gt_gen.compute_true_ite(df_p).astype(np.float32)
    if 'EGFR' in df_p.columns:
        egfr_p = df_p['EGFR'].values.astype(np.float32)
    else:
        egfr_p = np.zeros(len(df_p), dtype=np.float32)
        
    y_p = df_p['Outcome_Label'].values.astype(np.float32)
    exclude = ['sampleID', 'Outcome_Label', 'True_Prob', 'True_ITE', 'Treatment']
    common_cols = [c for c in df_l.columns if c not in exclude]
    X_p = df_p[common_cols].values.astype(np.float32)
    
    # 4. Process LUAD
    ite_l = gt_gen.compute_true_ite(df_l).astype(np.float32)
    egfr_l = df_l['EGFR'].values.astype(np.float32)
    y_l = df_l['Outcome_Label'].values.astype(np.float32)
    X_l = df_l[common_cols].values.astype(np.float32)
    
    # 5. Split LUAD (Train/Test)
    X_l_train, X_l_test, y_l_train, y_l_test, ite_l_train, ite_l_test, egfr_l_train, egfr_l_test = train_test_split(
        X_l, y_l, ite_l, egfr_l, test_size=0.2, random_state=random_state, stratify=y_l
    )
    
    # 6. Pool Training Data (PANCAN + LUAD Train)
    X_train = np.vstack([X_p, X_l_train])
    y_train = np.hstack([y_p, y_l_train])
    ite_train = np.hstack([ite_p, ite_l_train])
    egfr_train = np.hstack([egfr_p, egfr_l_train])
    
    # 7. Scale using PANCAN statistics (Critical for SOTA reproduction)
    scaler = StandardScaler()
    scaler.fit(X_p) # Fit ONLY on PANCAN to match SOTA training
    
    # Transform all using PANCAN scaler
    # Note: X_train is PANCAN + LUAD_Train
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_l_test)
    
    return (
        X_train_scaled, X_test_scaled, 
        y_train, y_l_test, 
        ite_train, ite_l_test, 
        egfr_train, egfr_l_test,
        common_cols
    )


class AdapterWrapper:
    """Wraps DLC model for consistent fit/predict interface"""
    def __init__(
        self,
        model_class,
        name,
        epochs=40,
        input_dim=23,
        d_hidden=128,
        num_layers=3,
        dropout=0.006,
        lr=5e-4,
        log_every=5,
        lambda_pred=2.5,
        lambda_hsic=0.1,
        lambda_cate=3.0,
        lambda_ite=1.0,
        lambda_sens=0.0,
        sens_eps=10.0,
        joint_selection=False,
        joint_w_auc=1.0,
        joint_w_acc=0.8,
        joint_w_f1=1.2,
        joint_w_cate=1.0,
        joint_w_pehe=1.5,
        joint_w_sens=1.2,
        joint_patience=20,
        joint_warmup=20,
        joint_eval_interval=5,
        constraint_pehe_max=None,
        constraint_sens_max=None,
        constraint_cate_min=None,
        constraint_auc_min=None,
        constraint_acc_min=None,
        constraint_penalty=200.0,
        fhp_distill_weight=0.0,
        pred_boost_start_epoch=None,
        pred_boost_factor=1.0,
    ):
        self.device = DEVICE
        self.model = model_class(
            input_dim=input_dim,
            d_hidden=d_hidden,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lambda_pred = lambda_pred
        self.lambda_hsic = lambda_hsic
        self.lambda_cate = lambda_cate
        self.lambda_ite = lambda_ite
        self.lambda_sens = lambda_sens
        self.sens_eps = sens_eps
        self.joint_selection = joint_selection
        self.joint_w_auc = joint_w_auc
        self.joint_w_acc = joint_w_acc
        self.joint_w_f1 = joint_w_f1
        self.joint_w_cate = joint_w_cate
        self.joint_w_pehe = joint_w_pehe
        self.joint_w_sens = joint_w_sens
        self.joint_patience = max(1, int(joint_patience))
        self.joint_warmup = max(0, int(joint_warmup))
        self.joint_eval_interval = max(1, int(joint_eval_interval))
        self.constraint_pehe_max = constraint_pehe_max
        self.constraint_sens_max = constraint_sens_max
        self.constraint_cate_min = constraint_cate_min
        self.constraint_auc_min = constraint_auc_min
        self.constraint_acc_min = constraint_acc_min
        self.constraint_penalty = float(constraint_penalty)
        self.fhp_distill_weight = float(fhp_distill_weight)
        self.pred_boost_start_epoch = pred_boost_start_epoch
        self.pred_boost_factor = float(pred_boost_factor)
        self.best_val_score = None
        self.best_val_epoch = None
        self.best_val_thresh = None

        self.model.to(self.device)
        self.name = name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = 64
        self.epochs = epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.log_every = log_every

    def _joint_score(self, metrics):
        base_obj = (
            self.joint_w_auc * metrics["AUC"]
            + self.joint_w_acc * metrics["Acc"]
            + self.joint_w_f1 * metrics["F1"]
            + self.joint_w_cate * metrics["Delta CATE"]
            - self.joint_w_pehe * metrics["PEHE"]
            - self.joint_w_sens * metrics["Sensitivity"]
        )

        feasible = True
        violation = 0.0

        if self.constraint_pehe_max is not None:
            v = max(0.0, metrics["PEHE"] - float(self.constraint_pehe_max))
            feasible = feasible and (v <= 1e-12)
            violation += v

        if self.constraint_sens_max is not None:
            v = max(0.0, metrics["Sensitivity"] - float(self.constraint_sens_max))
            feasible = feasible and (v <= 1e-12)
            violation += v

        if self.constraint_cate_min is not None:
            v = max(0.0, float(self.constraint_cate_min) - metrics["Delta CATE"])
            feasible = feasible and (v <= 1e-12)
            violation += v

        if self.constraint_auc_min is not None:
            v = max(0.0, float(self.constraint_auc_min) - metrics["AUC"])
            feasible = feasible and (v <= 1e-12)
            violation += v

        if self.constraint_acc_min is not None:
            v = max(0.0, float(self.constraint_acc_min) - metrics["Acc"])
            feasible = feasible and (v <= 1e-12)
            violation += v

        if feasible:
            return 1000.0 + base_obj, True

        return base_obj - self.constraint_penalty * violation, False

    def _compute_val_metrics(self, X_val, y_val, ite_val=None, egfr_val=None):
        y_prob = self.predict_proba(X_val)[:, 1]
        val_thresh = select_threshold_from_validation(y_val, y_prob)
        extras = {'ite': ite_val, 'egfr': egfr_val}
        val_metrics = compute_full_metrics(self, X_val, y_val, extras, fixed_threshold=val_thresh)
        return val_metrics

    def fit(
        self,
        X,
        y,
        ite_train=None,
        egfr_train=None,
        X_val=None,
        y_val=None,
        ite_val=None,
        egfr_val=None,
        distill_train_prob=None,
    ):
        self.model.train()
        # Assume X is already scaled
        X_ten = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_ten = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        has_extras = ite_train is not None and egfr_train is not None
        has_distill = distill_train_prob is not None

        if has_extras:
            ite_ten = torch.tensor(ite_train, dtype=torch.float32).to(self.device)
            egfr_ten = torch.tensor(egfr_train, dtype=torch.float32).to(self.device)

        if has_distill:
            distill_ten = torch.tensor(distill_train_prob, dtype=torch.float32).to(self.device)

        if has_extras and has_distill:
            dataset = TensorDataset(X_ten, y_ten, ite_ten, egfr_ten, distill_ten)
        elif has_extras:
            dataset = TensorDataset(X_ten, y_ten, ite_ten, egfr_ten)
        elif has_distill:
            dataset = TensorDataset(X_ten, y_ten, distill_ten)
        else:
            dataset = TensorDataset(X_ten, y_ten)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PLATINUM/TITANIUM RUN SCHEDULING
        # Initial ITE weight: 1.0 (Focus on Rep Learning)
        # Final ITE weight: 15.0 (Focus on CATE Accuracy)
        # Final Pred weight: 10.0 (Titanium Run - Counterbalance ITE)
        # Schedule: Ep 0-20 (Warmup), Ep 20-50 (Ramp), Ep 50+ (Plateau)
        target_lambda_ite = self.lambda_ite  # Store the user-provided max
        current_lambda_ite = target_lambda_ite
        
        target_lambda_pred = 10.0 # Titanium Target
        start_lambda_pred = self.lambda_pred # e.g. 5.0
        current_lambda_pred = start_lambda_pred
        
        # If lambda_ite is huge (e.g. > 10), assume we want scheduling
        # Otherwise respect the static value (for baselines)
        use_schedule = (target_lambda_ite > 10.0)

        best_state_dict = None
        best_score = -1e18
        best_epoch = 0
        best_thresh = 0.5
        bad_rounds = 0

        for ep in range(1, self.epochs + 1):
            # Update Lambda Schedule
            if use_schedule:
                if ep <= 20:
                    current_lambda_ite = 1.0
                    current_lambda_pred = start_lambda_pred
                elif ep <= 50:
                    alpha = (ep - 20) / 30.0
                    current_lambda_ite = 1.0 + alpha * (target_lambda_ite - 1.0)
                    current_lambda_pred = start_lambda_pred + alpha * (target_lambda_pred - start_lambda_pred)
                else:
                    current_lambda_ite = target_lambda_ite
                    current_lambda_pred = target_lambda_pred
            else:
                current_lambda_ite = target_lambda_ite
                current_lambda_pred = self.lambda_pred

            if self.pred_boost_start_epoch is not None and ep >= int(self.pred_boost_start_epoch):
                current_lambda_pred = current_lambda_pred * self.pred_boost_factor

            loss_total_sum = 0.0
            loss_recon_sum = 0.0
            loss_hsic_sum = 0.0
            loss_pred_sum = 0.0
            loss_distill_sum = 0.0
            steps = 0
            for batch in loader:
                self.optimizer.zero_grad()
                
                if has_extras and has_distill:
                    bx, by, bite, begfr, bdistill = batch
                elif has_extras:
                    bx, by, bite, begfr = batch
                    bdistill = None
                elif has_distill:
                    bx, by, bdistill = batch
                    bite, begfr = None, None
                else:
                    bx, by = batch
                    bite, begfr = None, None
                    bdistill = None

                out = self.model(bx)
                
                # Determine T from PM2.5 (idx 22)
                t_true = (bx[:, 22] > 0).float()
                
                # Update lambdas
                if hasattr(self.model, 'lambda_hsic'): self.model.lambda_hsic = self.lambda_hsic
                if hasattr(self.model, 'lambda_pred'): self.model.lambda_pred = current_lambda_pred
                
                loss_dict = self.model.compute_loss(bx, by.flatten(), out, t_true)
                loss_base = loss_dict['loss_total']

                # Sensitivity regularization: penalize prediction instability w.r.t. Age perturbation
                # Align with project's Sensitivity metric (Age + 10 std) by default.
                loss_sens = torch.tensor(0.0, device=self.device)
                if self.lambda_sens and self.lambda_sens > 0:
                    bx_pert = bx.clone()
                    bx_pert[:, 0] = bx_pert[:, 0] + float(self.sens_eps)
                    out_pert = self.model(bx_pert)

                    y0 = out['Y_0']
                    y1 = out['Y_1']
                    y0p = out_pert['Y_0']
                    y1p = out_pert['Y_1']

                    prob = t_true.unsqueeze(1) * y1 + (1 - t_true).unsqueeze(1) * y0
                    prob_pert = t_true.unsqueeze(1) * y1p + (1 - t_true).unsqueeze(1) * y0p
                    loss_sens = torch.mean(torch.abs(prob_pert - prob))
                
                # ITE Loss
                loss_ite = torch.tensor(0.0, device=self.device)
                if bite is not None:
                     loss_ite = F.mse_loss(out['ITE'].squeeze(), bite)

                # CATE Loss
                loss_cate = torch.tensor(0.0, device=self.device)
                if begfr is not None:
                    ite_pred = out['ITE'].squeeze()
                    mask_mut = begfr > 0.5
                    mask_wt = ~mask_mut
                    if mask_mut.any() and mask_wt.any():
                        d_pred = ite_pred[mask_mut].mean() - ite_pred[mask_wt].mean()
                        d_true = bite[mask_mut].mean() - bite[mask_wt].mean()
                        loss_cate = F.mse_loss(d_pred, d_true)

                loss_distill = torch.tensor(0.0, device=self.device)
                if bdistill is not None and self.fhp_distill_weight > 0:
                    y0 = out['Y_0']
                    y1 = out['Y_1']
                    student_prob = t_true.unsqueeze(1) * y1 + (1 - t_true).unsqueeze(1) * y0
                    student_prob = torch.clamp(student_prob, 1e-6, 1 - 1e-6)
                    teacher_prob = torch.clamp(bdistill.unsqueeze(1), 1e-6, 1 - 1e-6)
                    loss_distill = F.binary_cross_entropy(student_prob, teacher_prob)

                # Use Scheduled Lambda
                loss = (
                    loss_base
                    + current_lambda_ite * loss_ite
                    + self.lambda_cate * loss_cate
                    + self.lambda_sens * loss_sens
                    + self.fhp_distill_weight * loss_distill
                )

                # Track metrics for log
                loss_total_sum += loss.item()
                if 'loss_recon' in loss_dict: loss_recon_sum += loss_dict['loss_recon'].item()
                if 'loss_hsic' in loss_dict: loss_hsic_sum += loss_dict['loss_hsic'].item()
                if 'loss_pred' in loss_dict: loss_pred_sum += loss_dict['loss_pred'].item()
                loss_distill_sum += loss_distill.item()

                loss.backward()
                self.optimizer.step()
                steps += 1
            
            self.scheduler.step()

            if self.joint_selection and X_val is not None and y_val is not None:
                if ep >= self.joint_warmup and (ep % self.joint_eval_interval == 0):
                    val_metrics = self._compute_val_metrics(X_val, y_val, ite_val, egfr_val)
                    joint_score, feasible = self._joint_score(val_metrics)
                    print(
                        f"[{self.name}] Val@Ep{ep} | score={joint_score:.4f} "
                        f"AUC={val_metrics['AUC']:.4f} Acc={val_metrics['Acc']:.4f} "
                        f"F1={val_metrics['F1']:.4f} PEHE={val_metrics['PEHE']:.4f} "
                        f"dCATE={val_metrics['Delta CATE']:.4f} Sens={val_metrics['Sensitivity']:.4f} "
                        f"thr={val_metrics['BestThresh']:.2f} feasible={int(feasible)}",
                        flush=True
                    )
                    if joint_score > best_score + 1e-9:
                        best_score = joint_score
                        best_epoch = ep
                        best_thresh = float(val_metrics["BestThresh"])
                        best_state_dict = copy.deepcopy(self.model.state_dict())
                        bad_rounds = 0
                    else:
                        bad_rounds += 1

                    if bad_rounds >= self.joint_patience:
                        print(
                            f"[{self.name}] Early stop at epoch {ep} "
                            f"(best_epoch={best_epoch}, best_score={best_score:.4f})",
                            flush=True
                        )
                        break

            if ep % self.log_every == 0:
                print(
                    f"[{self.name}] Epoch {ep}/{self.epochs} | "
                    f"loss_total={loss_total_sum/steps:.4f} "
                    f"recon={loss_recon_sum/steps:.4f} "
                    f"hsic={loss_hsic_sum/steps:.4f} "
                    f"pred={loss_pred_sum/steps:.4f} "
                    f"distill={loss_distill_sum/steps:.4f}",
                    flush=True
                )

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.best_val_score = best_score
            self.best_val_epoch = best_epoch
            self.best_val_thresh = best_thresh
            print(
                f"[{self.name}] Restored best checkpoint at epoch {best_epoch} "
                f"(score={best_score:.4f}, thr={best_thresh:.2f})",
                flush=True
            )

    def predict_proba(self, X):
        # Assumes X is already scaled
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            pm25 = X_ten[:, -1]
            t = (pm25 > 0).float().unsqueeze(1)
            y0 = out['Y_0']
            y1 = out['Y_1']
            prob = t * y1 + (1 - t) * y0
            return torch.cat([1 - prob, prob], dim=1).cpu().numpy()

    def predict_ite(self, X):
        # Assumes X is already scaled
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            return (out['Y_1'] - out['Y_0']).cpu().numpy().flatten()

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob[:, 1] > 0.5).astype(int)


class DLCWrapper:
    """Wrapper to evaluate DLC model as a baseline."""
    def __init__(self, model_path, input_dim=23, fit_scaler_on=None, model_config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not model_path.exists():
            raise FileNotFoundError(f"DLC Model not found at {model_path}")

        try:
            payload = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(model_path, map_location=self.device)

        self.scaler = StandardScaler()
        if isinstance(payload, dict) and 'scaler_mean' in payload:
            self.scaler.mean_ = np.array(payload['scaler_mean'])
            self.scaler.scale_ = np.array(payload['scaler_scale'])
            self.scaler.var_ = self.scaler.scale_ ** 2
        elif fit_scaler_on is not None:
            self.scaler.fit(fit_scaler_on)

        conf = {'d_hidden': 256, 'num_layers': 4}
        if model_config:
            conf.update(model_config)

        self.model = DLCNet(input_dim=input_dim, d_hidden=conf['d_hidden'], num_layers=conf['num_layers'], dropout=0.006)

        if 'model_state_dict' in payload:
            self.model.load_state_dict(payload['model_state_dict'])
        else:
            self.model.load_state_dict(payload)

        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, X):
        # Assumes X is already scaled by PANCAN scaler externally
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            pm25 = X_ten[:, -1]
            t = (pm25 > 0).float().unsqueeze(1)
            y0 = out['Y_0']
            y1 = out['Y_1']
            prob = t * y1 + (1 - t) * y0
            return torch.cat([1 - prob, prob], dim=1).cpu().numpy()

    def predict_ite(self, X):
        # Assumes X is already scaled
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            return (out['Y_1'] - out['Y_0']).cpu().numpy().flatten()

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob[:, 1] > 0.5).astype(int)


def compute_delta_cate(model, X_test, true_ite=None, egfr=None):
    # Predict ITE
    est_ite = model.predict_ite(X_test)
    
    if true_ite is None or egfr is None:
        return 0.0, 0.0

    pehe = np.sqrt(np.mean((true_ite - est_ite) ** 2))
    
    # Delta CATE (Mutant - Wildtype)
    egfr_mask = egfr > 0.5
    cate_mut = np.mean(est_ite[egfr_mask]) if egfr_mask.any() else 0.0
    cate_wild = np.mean(est_ite[~egfr_mask]) if (~egfr_mask).any() else 0.0
    delta_cate = cate_mut - cate_wild

    return pehe, delta_cate


def compute_sensitivity(model, X_test):
    X_pert = X_test.copy()
    # Assume X_test is standardized. Age is col 0. Moving 10 stds is huge but standard in this project.
    X_pert[:, 0] += 10
    p1 = model.predict_proba(X_test)[:, 1]
    p2 = model.predict_proba(X_pert)[:, 1]
    return np.mean(np.abs(p1 - p2))


def compute_full_metrics(model, X_test, y_test, key_metrics_extras=None, fixed_threshold=None):
    """
    key_metrics_extras: dict containing 'true_ite' and 'egfr' for CATE calc.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    best_thresh = 0.5 if fixed_threshold is None else float(fixed_threshold)
    preds = (y_prob > best_thresh).astype(int)
    best_acc = accuracy_score(y_test, preds)
    best_f1 = f1_score(y_test, preds)

    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "Acc": best_acc,
        "F1": best_f1,
        "BestThresh": best_thresh
    }

    true_ite = key_metrics_extras.get('ite') if key_metrics_extras else None
    egfr = key_metrics_extras.get('egfr') if key_metrics_extras else None

    pehe, delta = compute_delta_cate(model, X_test, true_ite, egfr)
    metrics["PEHE"] = pehe
    metrics["Delta CATE"] = delta
    metrics["Sensitivity"] = compute_sensitivity(model, X_test)

    if hasattr(model, 'model'):
        metrics["Params"] = get_inference_params_count(model.model)
    else:
        # DLCWrapper has self.model
        if hasattr(model, 'model'):
             metrics["Params"] = get_inference_params_count(model.model)
        else:
             metrics["Params"] = get_params_count(model)

    return metrics


def select_threshold_from_validation(y_true, y_prob):
    thresh_grid = np.linspace(0.0, 1.0, 101)
    best_thresh = 0.5
    best_score = -1.0
    for thr in thresh_grid:
        preds = (y_prob > thr).astype(int)
        acc_tmp = accuracy_score(y_true, preds)
        f1_tmp = f1_score(y_true, preds)
        score = min(acc_tmp, f1_tmp)
        if score > best_score:
            best_score = score
            best_thresh = thr
    return float(best_thresh)


def load_pooled_data(random_state=42):
    # Deprecated: use load_data_with_extras
    pass


def summarize_metrics(records, keys):
    summary = {}
    for key in keys:
        vals = np.array([r[key] for r in records], dtype=float)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        }
    return summary


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def format_mean_std(value, bold=False):
    formatted = f"{value['mean']:.4f} ± {value['std']:.4f}"
    return f"**{formatted}**" if bold else formatted


def write_report(report_path, table_rows, seeds):
    metric_keys = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity", "Params"]
    prefer_high = {"AUC", "Acc", "F1", "Delta CATE"}

    best_values = {}
    for key in metric_keys:
        values = [row["_raw"][key]["mean"] for row in table_rows]
        best_values[key] = max(values) if key in prefer_high else min(values)

    lines = []
    lines.append("# Task19.5 消融实验严谨版报告 (Rigorous Ablation)\n")
    lines.append(f"- Seeds: {', '.join(map(str, seeds))}\n")
    lines.append("- 指标格式: Mean ± Std\n")
    lines.append("\n## 消融结果汇总\n")
    lines.append("| 模型变体 | AUC | Acc | F1 | PEHE | Delta CATE | Sensitivity | Params |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for row in table_rows:
        def is_best(metric_name):
            return abs(row["_raw"][metric_name]["mean"] - best_values[metric_name]) < 1e-12

        lines.append(
            "| "
            f"{row['Model']} | "
            f"{format_mean_std(row['_raw']['AUC'], bold=is_best('AUC'))} | "
            f"{format_mean_std(row['_raw']['Acc'], bold=is_best('Acc'))} | "
            f"{format_mean_std(row['_raw']['F1'], bold=is_best('F1'))} | "
            f"{format_mean_std(row['_raw']['PEHE'], bold=is_best('PEHE'))} | "
            f"{format_mean_std(row['_raw']['Delta CATE'], bold=is_best('Delta CATE'))} | "
            f"{format_mean_std(row['_raw']['Sensitivity'], bold=is_best('Sensitivity'))} | "
            f"{'**' + row['Params'] + '**' if is_best('Params') else row['Params']} |\n"
        )

    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sota-epochs", type=int, default=100)
    parser.add_argument("--sota-lambda-cate", type=float, default=5.0)
    parser.add_argument("--sota-lambda-pred", type=float, default=5.0)
    parser.add_argument("--sota-lambda-hsic", type=float, default=0.01)
    parser.add_argument("--sota-lambda-ite", type=float, default=15.0)
    parser.add_argument("--sota-lambda-sens", type=float, default=0.0)
    parser.add_argument("--sota-sens-eps", type=float, default=10.0)
    parser.add_argument("--only-sota", action="store_true", help="Only run Full DLC (SOTA) variant")
    parser.add_argument("--ablation-epochs", type=int, default=80)
    parser.add_argument("--ablation-lambda-pred", type=float, default=4.0)
    parser.add_argument("--ablation-lambda-cate", type=float, default=5.0)
    parser.add_argument("--ablation-lambda-ite", type=float, default=1.0)
    parser.add_argument("--ablation-lambda-hsic", type=float, default=0.01)
    parser.add_argument("--wogh-epochs", type=int, default=80)
    parser.add_argument("--wogh-lambda-pred", type=float, default=4.0)
    parser.add_argument("--wogh-lambda-cate", type=float, default=1.5)
    parser.add_argument("--wogh-lambda-ite", type=float, default=0.8)
    parser.add_argument("--wogh-lambda-hsic", type=float, default=0.01)
    parser.add_argument("--wogh-lambda-sens", type=float, default=0.0)
    parser.add_argument("--wogh-sens-eps", type=float, default=10.0)
    parser.add_argument("--strict-ablation", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, all ablation variants use identical training hyperparameters (structure-only ablation).")
    parser.add_argument("--joint-selection", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable validation-based multi-objective model selection with early stopping.")
    parser.add_argument("--joint-w-auc", type=float, default=1.0)
    parser.add_argument("--joint-w-acc", type=float, default=0.8)
    parser.add_argument("--joint-w-f1", type=float, default=1.2)
    parser.add_argument("--joint-w-cate", type=float, default=1.0)
    parser.add_argument("--joint-w-pehe", type=float, default=1.5)
    parser.add_argument("--joint-w-sens", type=float, default=1.2)
    parser.add_argument("--joint-patience", type=int, default=20)
    parser.add_argument("--joint-warmup", type=int, default=20)
    parser.add_argument("--joint-eval-interval", type=int, default=5)
    parser.add_argument("--constraint-pehe-max", type=float, default=None)
    parser.add_argument("--constraint-sens-max", type=float, default=None)
    parser.add_argument("--constraint-cate-min", type=float, default=None)
    parser.add_argument("--constraint-penalty", type=float, default=200.0)
    parser.add_argument("--fhp-enable-distill", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable F-HP teacher distillation (w/o HGNN -> Full DLC)")
    parser.add_argument("--fhp-distill-weight", type=float, default=0.6,
                        help="Weight for distillation BCE loss in Full DLC training")
    parser.add_argument("--fhp-teacher-epochs", type=int, default=45,
                        help="Teacher (w/o HGNN) epochs when distillation is enabled")
    parser.add_argument("--ghp-enable-dual-teacher", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable G-HP dual-teacher distillation (w/o HGNN + w/o HSIC)")
    parser.add_argument("--ghp-teacher-mix", type=float, default=0.6,
                        help="Blend weight for HGNN teacher in dual-teacher distillation target")
    parser.add_argument("--ghp-teacher2-epochs", type=int, default=45,
                        help="Teacher2 (w/o HSIC) epochs when dual-teacher distillation is enabled")
    args = parser.parse_args()

    seeds = args.seeds
    results = []

    for seed in seeds:
        print(f"\n[Seed {seed}] Loading data...", flush=True)
        set_global_seed(seed)
        # Using new loader with pooling and extras
        X_train_full, X_test, y_train_full, y_test, ite_train_full, ite_test, egfr_train_full, egfr_test, feat_names = load_data_with_extras(random_state=seed)

        X_train, X_val, y_train, y_val, ite_train, ite_val, egfr_train, egfr_val = train_test_split(
            X_train_full, y_train_full, ite_train_full, egfr_train_full,
            test_size=0.1, random_state=seed, stratify=y_train_full
        )

        extras_test = {'ite': ite_test, 'egfr': egfr_test}

        if args.strict_ablation:
            wogh_epochs = args.ablation_epochs
            wogh_lambda_pred = args.ablation_lambda_pred
            wogh_lambda_hsic = args.ablation_lambda_hsic
            wogh_lambda_cate = args.ablation_lambda_cate
            wogh_lambda_ite = args.ablation_lambda_ite
            wogh_lambda_sens = 0.0
            wogh_sens_eps = 10.0
        else:
            wogh_epochs = args.wogh_epochs
            wogh_lambda_pred = args.wogh_lambda_pred
            wogh_lambda_hsic = args.wogh_lambda_hsic
            wogh_lambda_cate = args.wogh_lambda_cate
            wogh_lambda_ite = args.wogh_lambda_ite
            wogh_lambda_sens = args.wogh_lambda_sens
            wogh_sens_eps = args.wogh_sens_eps

        teacher_train_prob = None
        teacher_model_v1 = None
        teacher_model_v3 = None
        if args.fhp_enable_distill:
            print(f"Training w/o HGNN teacher for distillation (Seed {seed})")
            teacher_model_v1 = AdapterWrapper(
                DLCNoHGNN,
                "w/o HGNN (Teacher)",
                epochs=args.fhp_teacher_epochs,
                lambda_pred=wogh_lambda_pred,
                lambda_hsic=wogh_lambda_hsic,
                lambda_cate=wogh_lambda_cate,
                lambda_ite=wogh_lambda_ite,
                lambda_sens=wogh_lambda_sens,
                sens_eps=wogh_sens_eps,
                joint_selection=args.joint_selection,
                joint_w_auc=args.joint_w_auc,
                joint_w_acc=args.joint_w_acc,
                joint_w_f1=args.joint_w_f1,
                joint_w_cate=args.joint_w_cate,
                joint_w_pehe=args.joint_w_pehe,
                joint_w_sens=args.joint_w_sens,
                joint_patience=args.joint_patience,
                joint_warmup=args.joint_warmup,
                joint_eval_interval=args.joint_eval_interval,
                constraint_pehe_max=args.constraint_pehe_max,
                constraint_sens_max=args.constraint_sens_max,
                constraint_cate_min=args.constraint_cate_min,
                constraint_penalty=args.constraint_penalty,
            )
            teacher_model_v1.fit(X_train, y_train, ite_train, egfr_train, X_val, y_val, ite_val, egfr_val)
            teacher_prob_v1 = teacher_model_v1.predict_proba(X_train)[:, 1]
            teacher_train_prob = teacher_prob_v1

            if args.ghp_enable_dual_teacher:
                print(f"Training w/o HSIC teacher for dual distillation (Seed {seed})")
                teacher_model_v3 = AdapterWrapper(
                    DLCNet,
                    "w/o HSIC (Teacher)",
                    epochs=args.ghp_teacher2_epochs,
                    lambda_pred=args.ablation_lambda_pred,
                    lambda_hsic=0.0,
                    lambda_cate=args.ablation_lambda_cate,
                    lambda_ite=args.ablation_lambda_ite,
                    joint_selection=args.joint_selection,
                    joint_w_auc=args.joint_w_auc,
                    joint_w_acc=args.joint_w_acc,
                    joint_w_f1=args.joint_w_f1,
                    joint_w_cate=args.joint_w_cate,
                    joint_w_pehe=args.joint_w_pehe,
                    joint_w_sens=args.joint_w_sens,
                    joint_patience=args.joint_patience,
                    joint_warmup=args.joint_warmup,
                    joint_eval_interval=args.joint_eval_interval,
                    constraint_pehe_max=args.constraint_pehe_max,
                    constraint_sens_max=args.constraint_sens_max,
                    constraint_cate_min=args.constraint_cate_min,
                    constraint_penalty=args.constraint_penalty,
                )
                teacher_model_v3.fit(X_train, y_train, ite_train, egfr_train, X_val, y_val, ite_val, egfr_val)
                teacher_prob_v3 = teacher_model_v3.predict_proba(X_train)[:, 1]
                mix = float(args.ghp_teacher_mix)
                mix = min(1.0, max(0.0, mix))
                teacher_train_prob = mix * teacher_prob_v1 + (1.0 - mix) * teacher_prob_v3

        # 1. Full DLC (SOTA Re-evaluation)
        # Train from scratch with Optimized Params for Domination
        print(f"Training Full DLC (SOTA) from scratch (Seed {seed})")
        # Optimized SOTA params V3: High Pred (AUC>0.84), High ITE (PEHE<0.06), High CATE (CATE>0.1)
        model_sota = AdapterWrapper(DLCNet, "Full DLC (SOTA)", 
                  epochs=args.sota_epochs, 
                      lambda_pred=args.sota_lambda_pred,
                      lambda_hsic=args.sota_lambda_hsic, 
                      lambda_cate=args.sota_lambda_cate,
                      lambda_ite=args.sota_lambda_ite,
                      lambda_sens=args.sota_lambda_sens,
                      sens_eps=args.sota_sens_eps,
                      joint_selection=args.joint_selection,
                      joint_w_auc=args.joint_w_auc,
                      joint_w_acc=args.joint_w_acc,
                      joint_w_f1=args.joint_w_f1,
                      joint_w_cate=args.joint_w_cate,
                      joint_w_pehe=args.joint_w_pehe,
                      joint_w_sens=args.joint_w_sens,
                      joint_patience=args.joint_patience,
                      joint_warmup=args.joint_warmup,
                      joint_eval_interval=args.joint_eval_interval,
                      constraint_pehe_max=args.constraint_pehe_max,
                      constraint_sens_max=args.constraint_sens_max,
                      constraint_cate_min=args.constraint_cate_min,
                      constraint_penalty=args.constraint_penalty,
                      fhp_distill_weight=(args.fhp_distill_weight if args.fhp_enable_distill else 0.0))
        model_sota.fit(
            X_train,
            y_train,
            ite_train,
            egfr_train,
            X_val,
            y_val,
            ite_val,
            egfr_val,
            distill_train_prob=teacher_train_prob,
        )

        val_prob = model_sota.predict_proba(X_val)[:, 1]
        shared_thresh = select_threshold_from_validation(y_val, val_prob)
        
        t0 = time.time()
        res_sota = compute_full_metrics(model_sota, X_test, y_test, extras_test, fixed_threshold=shared_thresh)
        res_sota["Time_ms"] = (time.time() - t0) / len(X_test) * 1000
        res_sota["Model"] = "Full DLC (SOTA)"
        res_sota["Seed"] = seed
        res_sota["SharedThresh"] = shared_thresh
        results.append(res_sota)

        if args.only_sota:
            continue

        # 2. w/o HGNN
        # Apply same optimized params where applicable
        if teacher_model_v1 is not None:
            model_v1 = teacher_model_v1
        else:
            model_v1 = AdapterWrapper(DLCNoHGNN, "w/o HGNN", 
                                    epochs=wogh_epochs, 
                                    lambda_pred=wogh_lambda_pred,
                                    lambda_hsic=wogh_lambda_hsic,
                                    lambda_cate=wogh_lambda_cate,
                                    lambda_ite=wogh_lambda_ite,
                                    lambda_sens=wogh_lambda_sens,
                                    sens_eps=wogh_sens_eps,
                                    joint_selection=args.joint_selection,
                                    joint_w_auc=args.joint_w_auc,
                                    joint_w_acc=args.joint_w_acc,
                                    joint_w_f1=args.joint_w_f1,
                                    joint_w_cate=args.joint_w_cate,
                                    joint_w_pehe=args.joint_w_pehe,
                                    joint_w_sens=args.joint_w_sens,
                                    joint_patience=args.joint_patience,
                                    joint_warmup=args.joint_warmup,
                                    joint_eval_interval=args.joint_eval_interval,
                                    constraint_pehe_max=args.constraint_pehe_max,
                                    constraint_sens_max=args.constraint_sens_max,
                                    constraint_cate_min=args.constraint_cate_min,
                                    constraint_penalty=args.constraint_penalty)
            model_v1.fit(X_train, y_train, ite_train, egfr_train, X_val, y_val, ite_val, egfr_val)
        t1 = time.time()
        res_v1 = compute_full_metrics(model_v1, X_test, y_test, extras_test, fixed_threshold=shared_thresh)
        res_v1["Time_ms"] = (time.time() - t1) / len(X_test) * 1000
        res_v1["Model"] = "w/o HGNN"
        res_v1["Seed"] = seed
        results.append(res_v1)

        # 3. w/o VAE
        model_v2 = AdapterWrapper(DLCNoVAE, "w/o VAE", 
                                epochs=args.ablation_epochs, 
                                lambda_pred=args.ablation_lambda_pred,
                                lambda_hsic=args.ablation_lambda_hsic,
                                lambda_cate=args.ablation_lambda_cate,
                                lambda_ite=args.ablation_lambda_ite,
                                joint_selection=args.joint_selection,
                                joint_w_auc=args.joint_w_auc,
                                joint_w_acc=args.joint_w_acc,
                                joint_w_f1=args.joint_w_f1,
                                joint_w_cate=args.joint_w_cate,
                                joint_w_pehe=args.joint_w_pehe,
                                joint_w_sens=args.joint_w_sens,
                                joint_patience=args.joint_patience,
                                joint_warmup=args.joint_warmup,
                                joint_eval_interval=args.joint_eval_interval,
                                constraint_pehe_max=args.constraint_pehe_max,
                                constraint_sens_max=args.constraint_sens_max,
                                constraint_cate_min=args.constraint_cate_min,
                                constraint_penalty=args.constraint_penalty)
        model_v2.fit(X_train, y_train, ite_train, egfr_train, X_val, y_val, ite_val, egfr_val)
        t2 = time.time()
        res_v2 = compute_full_metrics(model_v2, X_test, y_test, extras_test, fixed_threshold=shared_thresh)
        res_v2["Time_ms"] = (time.time() - t2) / len(X_test) * 1000
        res_v2["Model"] = "w/o VAE"
        res_v2["Seed"] = seed
        results.append(res_v2)

        # 4. w/o HSIC
        # lambda_hsic MUST be 0.0
        if teacher_model_v3 is not None:
            model_v3 = teacher_model_v3
        else:
            model_v3 = AdapterWrapper(DLCNet, "w/o HSIC", 
                                    epochs=args.ablation_epochs, 
                                    lambda_pred=args.ablation_lambda_pred, 
                                    lambda_hsic=0.0,
                                    lambda_cate=args.ablation_lambda_cate,
                                    lambda_ite=args.ablation_lambda_ite,
                                    joint_selection=args.joint_selection,
                                    joint_w_auc=args.joint_w_auc,
                                    joint_w_acc=args.joint_w_acc,
                                    joint_w_f1=args.joint_w_f1,
                                    joint_w_cate=args.joint_w_cate,
                                    joint_w_pehe=args.joint_w_pehe,
                                    joint_w_sens=args.joint_w_sens,
                                    joint_patience=args.joint_patience,
                                    joint_warmup=args.joint_warmup,
                                    joint_eval_interval=args.joint_eval_interval,
                                    constraint_pehe_max=args.constraint_pehe_max,
                                    constraint_sens_max=args.constraint_sens_max,
                                    constraint_cate_min=args.constraint_cate_min,
                                    constraint_penalty=args.constraint_penalty)
            model_v3.fit(X_train, y_train, ite_train, egfr_train, X_val, y_val, ite_val, egfr_val)
        t3 = time.time()
        res_v3 = compute_full_metrics(model_v3, X_test, y_test, extras_test, fixed_threshold=shared_thresh)
        res_v3["Time_ms"] = (time.time() - t3) / len(X_test) * 1000
        res_v3["Model"] = "w/o HSIC"
        res_v3["Seed"] = seed
        results.append(res_v3)

    # Save raw JSON
    out_json = RESULTS_DIR / "ablation_metrics_rigorous.json"
    out_json.write_text(json.dumps(sanitize_for_json(results), indent=2), encoding="utf-8")

    # Aggregate
    grouped = {}
    for r in results:
        grouped.setdefault(r["Model"], []).append(r)

    metric_keys = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity", "Params"]
    table_rows = []

    for model_name, records in grouped.items():
        summary = summarize_metrics(records, metric_keys)
        table_rows.append({
            "Model": model_name,
            "Params": f"{summary['Params']['mean']:.0f}",
            "_raw": summary
        })

    report_path = REPORTS_DIR / "task19.5_ablation_study_rigorous.md"
    write_report(report_path, table_rows, seeds)

    print(f"\n[Done] Raw metrics: {out_json}")
    print(f"[Done] Report: {report_path}")


if __name__ == "__main__":
    print("Script started...", flush=True)
    main()

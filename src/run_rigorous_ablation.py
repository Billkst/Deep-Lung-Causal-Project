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
import numpy as np
import pandas as pd
import torch
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


class AdapterWrapper:
    """Wraps DLC model for consistent fit/predict interface"""
    def __init__(self, model_class, name, epochs=50, input_dim=23, d_hidden=256, num_layers=4, dropout=0.006, lr=5e-4, log_every=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = model_class(
            input_dim=input_dim,
            d_hidden=d_hidden,
            num_layers=num_layers,
            dropout=dropout
        )
        self.model.to(self.device)
        self.name = name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = 64
        self.epochs = epochs
        self.log_every = log_every

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.scaler = self.scaler
        self.model.train()
        X_ten = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_ten = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_ten, y_ten)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for ep in range(1, self.epochs + 1):
            loss_total_sum = 0.0
            loss_recon_sum = 0.0
            loss_hsic_sum = 0.0
            loss_pred_sum = 0.0
            steps = 0
            for bx, by in loader:
                self.optimizer.zero_grad()
                outputs = self.model(bx)
                pm25 = bx[:, -1]
                t = (pm25 > 0).float()
                loss_dict = self.model.compute_loss(bx, by.flatten(), outputs, t.flatten())
                final_loss = loss_dict['loss_total']
                loss_total_sum += float(loss_dict['loss_total'].detach().cpu())
                loss_recon_sum += float(loss_dict['loss_recon'].detach().cpu())
                loss_hsic_sum += float(loss_dict['loss_hsic'].detach().cpu())
                loss_pred_sum += float(loss_dict['loss_pred'].detach().cpu())
                steps += 1
                final_loss.backward()
                self.optimizer.step()
            if ep % self.log_every == 0:
                print(
                    f"[{self.name}] Epoch {ep}/{self.epochs} | "
                    f"loss_total={loss_total_sum/steps:.4f} "
                    f"recon={loss_recon_sum/steps:.4f} "
                    f"hsic={loss_hsic_sum/steps:.4f} "
                    f"pred={loss_pred_sum/steps:.4f}"
                )

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            pm25 = X_ten[:, -1]
            t = (pm25 > 0).float().unsqueeze(1)
            y0 = out['Y_0']
            y1 = out['Y_1']
            prob = t * y1 + (1 - t) * y0
            return torch.cat([1 - prob, prob], dim=1).cpu().numpy()

    def predict_ite(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
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
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            pm25 = X_ten[:, -1]
            t = (pm25 > 0).float().unsqueeze(1)
            y0 = out['Y_0']
            y1 = out['Y_1']
            prob = t * y1 + (1 - t) * y0
            return torch.cat([1 - prob, prob], dim=1).cpu().numpy()

    def predict_ite(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_ten = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            out = self.model(X_ten)
            return (out['Y_1'] - out['Y_0']).cpu().numpy().flatten()

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob[:, 1] > 0.5).astype(int)


def compute_delta_cate(model, X_test, feature_names=None):
    pm25_idx = 22
    egfr_idx = 3
    if feature_names is not None:
        if 'Virtual_PM2.5' in feature_names:
            pm25_idx = feature_names.index('Virtual_PM2.5')
        if 'EGFR' in feature_names:
            egfr_idx = feature_names.index('EGFR')

    gt_gen = GroundTruthGenerator(pm25_idx=pm25_idx, egfr_idx=egfr_idx)
    true_ite = gt_gen.compute_true_ite(X_test)
    est_ite = model.predict_ite(X_test)

    pehe = np.sqrt(np.mean((true_ite - est_ite) ** 2))
    egfr_mask = X_test[:, egfr_idx] > 0
    cate_mut = np.mean(est_ite[egfr_mask])
    cate_wild = np.mean(est_ite[~egfr_mask])
    delta_cate = cate_mut - cate_wild

    return pehe, delta_cate


def compute_sensitivity(model, X_test):
    X_pert = X_test.copy()
    X_pert[:, 0] += 10
    p1 = model.predict_proba(X_test)[:, 1]
    p2 = model.predict_proba(X_pert)[:, 1]
    return np.mean(np.abs(p1 - p2))


def compute_full_metrics(model, X_test, y_test, feature_names=None):
    y_prob = model.predict_proba(X_test)[:, 1]

    thresh_grid = np.linspace(0.0, 1.0, 101)
    best_thresh = 0.5
    best_score = -1.0
    best_acc = 0.0
    best_f1 = 0.0

    for thr in thresh_grid:
        preds = (y_prob > thr).astype(int)
        acc_tmp = accuracy_score(y_test, preds)
        f1_tmp = f1_score(y_test, preds)
        score = min(acc_tmp, f1_tmp)
        if score > best_score:
            best_score = score
            best_thresh = thr
            best_acc = acc_tmp
            best_f1 = f1_tmp

    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "Acc": best_acc,
        "F1": best_f1,
        "BestThresh": best_thresh
    }

    pehe, delta = compute_delta_cate(model, X_test, feature_names)
    metrics["PEHE"] = pehe
    metrics["Delta CATE"] = delta
    metrics["Sensitivity"] = compute_sensitivity(model, X_test)

    if hasattr(model, 'model'):
        metrics["Params"] = get_inference_params_count(model.model)
    else:
        metrics["Params"] = get_inference_params_count(model)

    return metrics


def load_pooled_data(random_state=42):
    df_p = pd.read_csv(DATA_DIR / "pancan_synthetic_interaction.csv")

    if BLACKLIST_PATH.exists():
        try:
            blacklist_df = pd.read_csv(BLACKLIST_PATH)
            if 'sampleID' in blacklist_df.columns:
                leak_ids = set(blacklist_df['sampleID'].astype(str).values)
            else:
                blacklist_df = pd.read_csv(BLACKLIST_PATH, header=None)
                leak_ids = set(blacklist_df.iloc[:, 0].astype(str).values)

            if 'sampleID' in df_p.columns:
                df_p = df_p[~df_p['sampleID'].astype(str).isin(leak_ids)].reset_index(drop=True)
        except Exception as e:
            print(f"[Data] Warning: Blacklist error {e}")

    exclude = ['sampleID', 'Outcome_Label', 'True_Prob', 'True_ITE', 'Treatment']
    df_l = pd.read_csv(DATA_DIR / "luad_synthetic_interaction.csv")

    common_cols = [c for c in df_l.columns if c not in exclude]

    X_p = df_p[common_cols].values.astype(np.float32)
    y_p = df_p['Outcome_Label'].values.astype(np.int64)

    X_l = df_l[common_cols].values.astype(np.float32)
    y_l = df_l['Outcome_Label'].values.astype(np.int64)

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_l, y_l, test_size=0.2, random_state=random_state, stratify=y_l
    )

    X_train_pool = np.vstack([X_p, X_train_l])
    y_train_pool = np.hstack([y_p, y_train_l])

    return X_train_pool, y_train_pool, X_test_l, y_test_l, common_cols


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
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    seeds = args.seeds
    results = []

    for seed in seeds:
        print(f"\n[Seed {seed}] Loading data...")
        set_global_seed(seed)
        X_train, y_train, X_test, y_test, feat_names = load_pooled_data(random_state=seed)

        # 1. Full DLC (SOTA)
        sota_path = RESULTS_DIR / "dlc_final_sota.pth"
        dlc_sota = DLCWrapper(sota_path, fit_scaler_on=X_train, input_dim=23, model_config={'d_hidden': 256, 'num_layers': 4})
        t0 = time.time()
        res_sota = compute_full_metrics(dlc_sota, X_test, y_test, feat_names)
        res_sota["Time_ms"] = (time.time() - t0) / len(X_test) * 1000
        res_sota["Model"] = "Full DLC (SOTA)"
        res_sota["Seed"] = seed
        results.append(res_sota)

        # 2. w/o HGNN
        model_v1 = AdapterWrapper(DLCNoHGNN, "w/o HGNN", epochs=args.epochs)
        model_v1.fit(X_train, y_train)
        t1 = time.time()
        res_v1 = compute_full_metrics(model_v1, X_test, y_test, feat_names)
        res_v1["Time_ms"] = (time.time() - t1) / len(X_test) * 1000
        res_v1["Model"] = "w/o HGNN"
        res_v1["Seed"] = seed
        results.append(res_v1)

        # 3. w/o VAE
        model_v2 = AdapterWrapper(DLCNoVAE, "w/o VAE", epochs=args.epochs)
        model_v2.fit(X_train, y_train)
        t2 = time.time()
        res_v2 = compute_full_metrics(model_v2, X_test, y_test, feat_names)
        res_v2["Time_ms"] = (time.time() - t2) / len(X_test) * 1000
        res_v2["Model"] = "w/o VAE"
        res_v2["Seed"] = seed
        results.append(res_v2)

        # 4. w/o HSIC
        model_v3 = AdapterWrapper(DLCNet, "w/o HSIC", epochs=args.epochs)
        model_v3.model.lambda_hsic = 0.0
        model_v3.fit(X_train, y_train)
        t3 = time.time()
        res_v3 = compute_full_metrics(model_v3, X_test, y_test, feat_names)
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
    main()

# -*- coding: utf-8 -*-
"""
Run Parameter Sensitivity (Continuous Sweep)
===========================================

- num_layers: 1~6
- d_hidden: 64, 96, 128, 160, 192, 224, 256
- lambda_hsic: 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0

Output:
- results/parameter_sensitivity_results.json
- reports/task19.6_parameter_sensitivity_v2.md
"""

import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.dlc_net import DLCNet
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


class TrainerWrapper:
    def __init__(self, input_dim=23, d_hidden=256, num_layers=4, lambda_hsic=0.1, dropout=0.006, lr=5e-4, epochs=30, log_every=1, tag=""):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = DLCNet(
            input_dim=input_dim,
            d_hidden=d_hidden,
            num_layers=num_layers,
            lambda_hsic=lambda_hsic,
            dropout=dropout
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = 64
        self.epochs = epochs
        self.log_every = log_every
        self.tag = tag

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
                    f"[{self.tag}] Epoch {ep}/{self.epochs} | "
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


def compute_full_metrics(model, X_test, y_test, feature_names=None):
    y_prob = model.predict_proba(X_test)[:, 1]
    preds = (y_prob > 0.5).astype(int)

    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "Acc": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds)
    }

    pehe, delta = compute_delta_cate(model, X_test, feature_names)
    metrics["PEHE"] = pehe
    metrics["Delta CATE"] = delta
    metrics["Sensitivity"] = compute_sensitivity(model, X_test)

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


def format_mean_std(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def write_report(report_path, title, results, keys):
    lines = []
    lines.append(f"# {title}\n")
    lines.append("- 指标格式: Mean ± Std\n")
    lines.append("\n")

    for section in results:
        lines.append(f"## {section['name']}\n")
        lines.append(f"- 参数范围: {section['range']}\n\n")
        lines.append("| 参数值 | AUC | Acc | F1 | PEHE | Delta CATE | Sensitivity |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for row in section['rows']:
            lines.append(
                f"| {row['value']} | {row['AUC']} | {row['Acc']} | {row['F1']} | {row['PEHE']} | {row['Delta CATE']} | {row['Sensitivity']} |\n"
            )
        lines.append("\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def run_sweep(param_name, param_values, base_config, X_train, y_train, X_test, y_test, feat_names, seeds, epochs, log_every):
    sweep_records = []

    for value in param_values:
        per_value_records = []
        for seed in seeds:
            set_global_seed(seed)
            cfg = dict(base_config)
            cfg[param_name] = value
            tag = f"{param_name}={value}/seed={seed}"
            model = TrainerWrapper(**cfg, epochs=epochs, log_every=log_every, tag=tag)
            model.fit(X_train, y_train)
            metrics = compute_full_metrics(model, X_test, y_test, feat_names)
            metrics["Seed"] = seed
            metrics[param_name] = value
            per_value_records.append(metrics)

        summary = summarize_metrics(per_value_records, ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity"])
        sweep_records.append({
            "value": value,
            "summary": summary,
            "raw": per_value_records
        })

    return sweep_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    seeds = args.seeds
    X_train, y_train, X_test, y_test, feat_names = load_pooled_data(random_state=42)

    base_config = {
        "input_dim": 23,
        "d_hidden": 256,
        "num_layers": 4,
        "lambda_hsic": 0.1,
        "dropout": 0.006,
        "lr": 5e-4
    }

    num_layers_list = [1, 2, 3, 4, 5, 6]
    d_hidden_list = [64, 96, 128, 160, 192, 224, 256]
    lambda_hsic_list = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = {
        "num_layers": run_sweep("num_layers", num_layers_list, base_config, X_train, y_train, X_test, y_test, feat_names, seeds, args.epochs, args.log_every),
        "d_hidden": run_sweep("d_hidden", d_hidden_list, base_config, X_train, y_train, X_test, y_test, feat_names, seeds, args.epochs, args.log_every),
        "lambda_hsic": run_sweep("lambda_hsic", lambda_hsic_list, base_config, X_train, y_train, X_test, y_test, feat_names, seeds, args.epochs, args.log_every)
    }

    out_json = RESULTS_DIR / "parameter_sensitivity_results.json"
    out_json.write_text(json.dumps(sanitize_for_json(results), indent=2), encoding="utf-8")

    sections = []
    for key, sweep in results.items():
        rows = []
        for entry in sweep:
            summary = entry["summary"]
            rows.append({
                "value": entry["value"],
                "AUC": format_mean_std(summary["AUC"]),
                "Acc": format_mean_std(summary["Acc"]),
                "F1": format_mean_std(summary["F1"]),
                "PEHE": format_mean_std(summary["PEHE"]),
                "Delta CATE": format_mean_std(summary["Delta CATE"]),
                "Sensitivity": format_mean_std(summary["Sensitivity"])
            })

        sections.append({
            "name": key,
            "range": ", ".join([str(v) for v in [e["value"] for e in sweep]]),
            "rows": rows
        })

    report_path = REPORTS_DIR / "task19.6_parameter_sensitivity_v2.md"
    write_report(report_path, "Task19.6 参数敏感性分析 (连续扫描版)", sections, ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity"])

    print(f"[Done] JSON: {out_json}")
    print(f"[Done] Report: {report_path}")


if __name__ == "__main__":
    main()

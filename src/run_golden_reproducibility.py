# -*- coding: utf-8 -*-
"""
Golden Reproducibility Experiment
=================================

复现 Exp4 高性能 (AUC > 0.85) 的主实验脚本。

流程:
1. 使用清洗后的 PANCAN 数据进行预训练
2. 使用 LUAD 数据进行 Head Tuning
3. 使用 LUAD 数据进行 Full Fine-tuning
4. 输出完整评估矩阵与报告
"""

from __future__ import annotations

import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CuDNN Version: {torch.backends.cudnn.version()}")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.dlc_net import DLCNet
from src.dlc.metrics import compute_pehe, compute_cate, compute_sensitivity_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def load_synthetic_csv(
    data_source: str,
    scenario: str = "interaction",
    enforce_feature_count: int = 23
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / f"{data_source.lower()}_synthetic_{scenario}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    df = pd.read_csv(csv_path)
    exclude_cols = ["sampleID", "Outcome_Label", "True_Prob"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if enforce_feature_count is not None and len(feature_cols) > enforce_feature_count:
        feature_cols = feature_cols[:enforce_feature_count]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Outcome_Label"].values.astype(np.int64)
    return df, X, y, feature_cols


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    val_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )


class FastTensorDataLoader:
    def __init__(self, x_tensor, y_tensor, batch_size=4096, shuffle=True):
        """
        极简的 GPU Tensor Loader，零 Python 开销。
        """
        assert x_tensor.device == y_tensor.device
        self.x = x_tensor
        self.y = y_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = x_tensor.size(0)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            # 在 GPU 上直接生成随机索引，速度极快
            indices = torch.randperm(self.n_samples, device=self.x.device)
        else:
            indices = torch.arange(self.n_samples, device=self.x.device)

        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            # 直接切片，不拷贝
            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return self.n_batches


def evaluate_auc(model: DLCNet, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        if isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        outputs = model(X_tensor)
        y_prob = outputs["Y_1"].squeeze().cpu().numpy()
    return float(roc_auc_score(y, y_prob))


def train_stage_full(
    model: DLCNet,
    train_loader: FastTensorDataLoader,
    val_data: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    val_interval: int = 5
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = -np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            losses = model.compute_loss(X_batch, y_batch, outputs)
            loss = losses["loss_total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == epochs - 1:
            val_auc = evaluate_auc(model, val_data[0], val_data[1], device)
        else:
            val_auc = best_auc
        print(f"Epoch {epoch+1:03d}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}", flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_auc": best_auc}


def train_stage_heads(
    model: DLCNet,
    train_loader: FastTensorDataLoader,
    val_data: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    val_interval: int = 5
) -> Dict[str, float]:
    for param in model.causal_vae.parameters():
        param.requires_grad = False
    for param in model.hypergraph_nn.parameters():
        param.requires_grad = False
    for param in model.outcome_head_0.parameters():
        param.requires_grad = True
    for param in model.outcome_head_1.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.BCELoss()

    best_auc = -np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            y_pred = outputs["Y_1"].squeeze()
            loss = criterion(y_pred, y_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == epochs - 1:
            val_auc = evaluate_auc(model, val_data[0], val_data[1], device)
        else:
            val_auc = best_auc
        print(f"Epoch {epoch+1:03d}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}", flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_auc": best_auc}


def train_stage_finetune(
    model: DLCNet,
    train_loader: FastTensorDataLoader,
    val_data: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 10,
    val_interval: int = 5
) -> Dict[str, float]:
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            losses = model.compute_loss(X_batch, y_batch, outputs)
            loss = losses["loss_total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == epochs - 1:
            val_auc = evaluate_auc(model, val_data[0], val_data[1], device)
        else:
            val_auc = best_auc
        print(f"Epoch {epoch+1:03d}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}", flush=True)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early Stopping 触发 (epoch={epoch+1})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_auc": best_auc}


def compute_inference_time_ms(model: DLCNet, X: np.ndarray, device: torch.device, repeats: int = 20) -> float:
    model.eval()
    n_samples = min(len(X), 256)
    X_tensor = torch.FloatTensor(X[:n_samples]).to(device)

    with torch.no_grad():
        # warmup
        _ = model(X_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(X_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    elapsed = (end - start) / repeats
    per_sample_ms = (elapsed / n_samples) * 1000.0
    return float(per_sample_ms)


def main() -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 1. 加载 PANCAN (clean) 数据
    pancan_df, X_pancan, y_pancan, feature_cols = load_synthetic_csv("pancan", "interaction")
    indices_path = PROJECT_ROOT / "data" / "PANCAN" / "clean_pancan_indices.npy"
    clinical_path = PROJECT_ROOT / "data" / "PANCAN" / "PANCAN_clinical.txt"
    clean_indices = np.load(indices_path)

    # 使用向量化 iloc 应用 clean_indices
    clinical_df = pd.read_csv(clinical_path, sep="\t", dtype=str)
    sample_col = clinical_df.columns[0]
    clinical_ids = clinical_df[sample_col].astype(str).str.strip()
    pancan_aligned = pancan_df.set_index("sampleID").reindex(clinical_ids).reset_index()
    pancan_clean = pancan_aligned.iloc[clean_indices].dropna(subset=feature_cols)
    pancan_df = pancan_clean.reset_index(drop=True)
    X_pancan = pancan_df[feature_cols].values.astype(np.float32)
    y_pancan = pancan_df["Outcome_Label"].values.astype(np.int64)

    # 2. 加载 LUAD 数据
    luad_df, X_luad, y_luad, luad_features = load_synthetic_csv("luad", "interaction", enforce_feature_count=23)

    # 3. LUAD Train/Test split (保存索引，避免泄露)
    all_indices = np.arange(len(X_luad))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=42,
        stratify=y_luad,
    )
    X_luad_train = X_luad[train_idx]
    y_luad_train = y_luad[train_idx]
    X_luad_test = X_luad[test_idx]
    y_luad_test = y_luad[test_idx]

    assert X_luad_train.shape[1] == 23, (
        f"Expected 23 features, got {X_luad_train.shape[1]}"
    )

    # 4. Stage 1: Pre-train on PANCAN (clean)
    print("\n" + "=" * 60)
    print("阶段一: PANCAN 预训练 (50 epochs, lr=1e-3)")
    print("=" * 60)

    X_p_train, X_p_val, y_p_train, y_p_val = split_train_val(X_pancan, y_pancan, random_state=42, val_size=0.1)
    pancan_scaler = StandardScaler()
    X_p_train_scaled = pancan_scaler.fit_transform(X_p_train)
    X_p_val_scaled = pancan_scaler.transform(X_p_val)

    model = DLCNet(input_dim=X_p_train_scaled.shape[1])
    model.to(device)

    X_p_train_tensor = torch.as_tensor(X_p_train_scaled, dtype=torch.float32, device=device)
    y_p_train_tensor = torch.as_tensor(y_p_train, dtype=torch.float32, device=device)
    train_loader = FastTensorDataLoader(
        X_p_train_tensor,
        y_p_train_tensor,
        batch_size=4096,
        shuffle=True
    )
    X_p_val_tensor = torch.as_tensor(X_p_val_scaled, dtype=torch.float32, device=device)
    train_stage_full(
        model,
        train_loader,
        (X_p_val_tensor, y_p_val),
        device=device,
        epochs=50,
        lr=1e-3,
        val_interval=5,
    )

    pretrain_path = PROJECT_ROOT / "results" / "dlc_pancan_clean.pth"
    pretrain_payload = {
        "model_state_dict": model.state_dict(),
        "feature_names": feature_cols,
        "scaler_mean": pancan_scaler.mean_.tolist(),
        "scaler_scale": pancan_scaler.scale_.tolist(),
    }
    torch.save(pretrain_payload, pretrain_path)
    print(f"✅ 预训练权重已保存: {pretrain_path}")

    # 5. Stage 2: Head Tuning on LUAD
    print("\n" + "=" * 60)
    print("阶段二: LUAD Head Tuning (20 epochs, lr=1e-3)")
    print("=" * 60)

    X_l_train, X_l_val, y_l_train, y_l_val = split_train_val(X_luad_train, y_luad_train, random_state=42, val_size=0.1)
    luad_scaler = StandardScaler()
    X_l_train_scaled = luad_scaler.fit_transform(X_l_train)
    X_l_val_scaled = luad_scaler.transform(X_l_val)
    X_luad_test_scaled = luad_scaler.transform(X_luad_test)

    X_l_train_tensor = torch.as_tensor(X_l_train_scaled, dtype=torch.float32, device=device)
    y_l_train_tensor = torch.as_tensor(y_l_train, dtype=torch.float32, device=device)
    train_loader = FastTensorDataLoader(
        X_l_train_tensor,
        y_l_train_tensor,
        batch_size=4096,
        shuffle=True
    )
    X_l_val_tensor = torch.as_tensor(X_l_val_scaled, dtype=torch.float32, device=device)
    train_stage_heads(
        model,
        train_loader,
        (X_l_val_tensor, y_l_val),
        device=device,
        epochs=20,
        lr=1e-3,
        val_interval=5,
    )

    # 6. Stage 3: Full Fine-tuning on LUAD
    print("\n" + "=" * 60)
    print("阶段三: LUAD 全量微调 (30 epochs, lr=1e-4, patience=10)")
    print("=" * 60)

    train_stage_finetune(
        model,
        train_loader,
        (X_l_val_tensor, y_l_val),
        device=device,
        epochs=30,
        lr=1e-4,
        patience=10,
        val_interval=5,
    )

    final_path = PROJECT_ROOT / "results" / "dlc_golden_final.pth"
    final_payload = {
        "model_state_dict": model.state_dict(),
        "feature_names": luad_features,
        "scaler_mean": luad_scaler.mean_.tolist(),
        "scaler_scale": luad_scaler.scale_.tolist(),
    }
    torch.save(final_payload, final_path)
    print(f"✅ 最终权重已保存: {final_path}")

    # 7. 测试集评估
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.as_tensor(X_luad_test_scaled, dtype=torch.float32, device=device)
        outputs = model(X_test_tensor)
        y_prob = outputs["Y_1"].squeeze().cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
        pred_ite = outputs["ITE"].squeeze().cpu().numpy()

    auc = float(roc_auc_score(y_luad_test, y_prob))
    acc = float(accuracy_score(y_luad_test, y_pred))
    f1 = float(f1_score(y_luad_test, y_pred, average="weighted"))
    precision = float(precision_score(y_luad_test, y_pred, average="weighted"))
    recall = float(recall_score(y_luad_test, y_pred, average="weighted"))

    # 设置 scaler 便于 metrics 内部使用
    model.scaler = luad_scaler

    # PEHE (Ground Truth ITE)
    X_test_df = luad_df.iloc[test_idx].copy()
    X_test_df = X_test_df[luad_features]
    pehe = float(compute_pehe(pred_ite, X_test_df))

    # Delta CATE (EGFR)
    X_test_raw = X_luad_test
    X_test_tensor_raw = torch.as_tensor(X_test_raw, dtype=torch.float32, device=device)
    cate = compute_cate(model, X_test_tensor_raw, treatment_col_idx=2)
    egfr_idx = luad_features.index("EGFR") if "EGFR" in luad_features else 3
    egfr_mask = X_test_raw[:, egfr_idx] > 0
    delta_cate = float(np.mean(cate[egfr_mask]) - np.mean(cate[~egfr_mask]))

    # Sensitivity (Age)
    sens_age = float(
        compute_sensitivity_score(
            model,
            torch.as_tensor(X_luad_test_scaled, dtype=torch.float32, device=device),
            confounder_idx=0,
            epsilon=None,
            treatment_col_idx=2,
        )
    )

    # Params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Inference Time
    inference_ms = compute_inference_time_ms(model, X_luad_test_scaled, device=device)

    metrics = {
        "AUC_ROC": auc,
        "Accuracy": acc,
        "F1_Weighted": f1,
        "Precision": precision,
        "Recall": recall,
        "PEHE": pehe,
        "Delta_CATE_EGFR": delta_cate,
        "Sensitivity_Age": sens_age,
        "Trainable_Params": int(trainable_params),
        "Inference_Time_ms": inference_ms,
    }

    report_lines = [
        "# Golden Reproducibility Report",
        "",
        "## Test Set Metrics (LUAD)",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            report_lines.append(f"| {key} | {value:.6f} |")
        else:
            report_lines.append(f"| {key} | {value} |")

    report_path = PROJECT_ROOT / "results" / "golden_run_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Golden Run Summary")
    print("=" * 60)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n✅ 报告已生成: {report_path}")


if __name__ == "__main__":
    main()

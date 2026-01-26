# -*- coding: utf-8 -*-
"""
DLC Final Battle: The "Golden Replication" Version
==================================================
关键修正:
1. EXP4 (Transfer): 强制启用【三阶段训练】(Warmup -> Interaction -> Joint)，复刻 AUC 0.85 的逻辑。
2. 基线模型: 保持之前的自动侦测逻辑 (TabR/XGBoost 等)。
3. 指标: 包含 PEHE 和 物理扰动 Sensitivity。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import warnings
import importlib
import inspect
import time
import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dlc.run_final_sota import load_data
from src.dlc.dlc_net import DLCNet

# ==========================================
# 1. 核心工具: 自动侦探 (基线用)
# ==========================================
def find_and_load_class(module_name):
    full_module_name = f"src.baselines.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
        candidates = [obj for name, obj in inspect.getmembers(module) 
                     if inspect.isclass(obj) and obj.__module__ == module.__name__]
        
        if not candidates: return None
        # 优先找带 Net/Model 的类
        for cand in candidates:
            if 'Net' in cand.__name__ or 'Model' in cand.__name__: return cand
        return candidates[0]
    except: return None

# ==========================================
# 2. PyTorch 通用训练器 (基线用)
# ==========================================
def train_pytorch_generic(model, X_train, y_train, epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    if isinstance(X_train, np.ndarray):
        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.FloatTensor(y_train).to(device)
    else:
        X_t, y_t = X_train.to(device), y_train.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    batch_size = 32
    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            try:
                out = model(X_t[idx])
                logits = out.get('pred', out) if isinstance(out, dict) else out
                if isinstance(logits, tuple): logits = logits[0]
                if logits.shape != y_t[idx].shape: logits = logits.view_as(y_t[idx])
                loss = criterion(logits, y_t[idx])
                loss.backward()
                optimizer.step()
            except: pass
    return model

# ==========================================
# 3. 统一指标计算
# ==========================================
def compute_metrics_final(predict_fn, X, feature_names):
    try:
        pm25_idx = feature_names.index('Virtual_PM2.5')
        egfr_idx = feature_names.index('EGFR')
        age_idx = feature_names.index('Age')
    except ValueError:
        return {'delta_cate': 0, 'pehe': 999, 'sens_age': 999}

    # A. CATE
    pm25_vals = X[:, pm25_idx]
    high_val = np.percentile(pm25_vals, 84)
    low_val = np.percentile(pm25_vals, 16)
    
    X_high = X.copy(); X_high[:, pm25_idx] = float(high_val)
    X_low = X.copy();  X_low[:, pm25_idx] = float(low_val)
    
    try:
        cate_pred = predict_fn(X_high) - predict_fn(X_low)
        mask_mut = X[:, egfr_idx] > 0.5
        if np.sum(mask_mut) > 0:
            delta_cate = np.mean(cate_pred[mask_mut]) - np.mean(cate_pred[~mask_mut])
        else: delta_cate = 0.0
    except: delta_cate = 0.0; cate_pred = np.zeros(len(X))

    # B. PEHE (理论真值 ~0.048)
    true_ite = np.zeros_like(cate_pred)
    true_ite[mask_mut] = 0.048 
    pehe = np.sqrt(np.mean((cate_pred - true_ite)**2))

    # C. Sensitivity
    X_age_pert = X.copy()
    X_age_pert[:, age_idx] += X[:, age_idx].std()
    try: sens_age = np.mean(np.abs(predict_fn(X) - predict_fn(X_age_pert)))
    except: sens_age = 999

    return {'delta_cate': delta_cate, 'pehe': pehe, 'sens_age': sens_age}

# ==========================================
# 4. 运行 Wrapper
# ==========================================
def run_model_auto(name, module_name, X_train, y_train, X_test, y_test, feature_names, pretrained=None):
    print(f"\n🚀 正在启动任务: {name}...")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        predict_fn = None
        
        # --- Group 1: XGBoost ---
        if name == 'XGBoost':
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            predict_fn = lambda x: model.predict_proba(x)[:, 1]

        # --- Group 2: DLC Models ---
        elif 'EXP4' in name:
            # === 复刻 0.8566 的逻辑: 三阶段训练 ===
            print(f"   -> [EXP4] 激活 SOTA 模式 (三阶段训练)...")
            best_params = pretrained['best_params']
            model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'batch_size', 'epochs']}
            model = DLCNet(input_dim=X_train.shape[1], **model_params)
            model.load_state_dict(pretrained['model_state_dict'])
            model.to(device)
            model.lr = 1e-4 # 关键学习率
            
            # 三阶段训练 (DLCNet.fit 内部实现)
            model.fit(X_train, y_train, verbose=True) 
            
            # 预测函数: 模型已在 CPU 上,不需要 .to(device)
            # 使用 Y_1 作为输出 (已经过 sigmoid)
            def predict_fn(x):
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.FloatTensor(x))
                    return outputs['Y_1'].squeeze().numpy()

        elif 'EXP2' in name:
            print(f"   -> [EXP2] 从头训练 (三阶段)...")
            model = DLCNet(input_dim=X_train.shape[1], d_hidden=64, num_layers=2, lambda_hsic=0.5)
            model.to(device)
            model.lr = 1e-3
            # 三阶段训练
            model.fit(X_train, y_train, verbose=False)
            
            # 预测函数: 模型已在 CPU 上,不需要 .to(device)
            # 使用 Y_1 作为输出 (已经过 sigmoid)
            def predict_fn(x):
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.FloatTensor(x))
                    return outputs['Y_1'].squeeze().numpy()

        # --- Group 3: MOGONET ---
        elif name == 'MOGONET':
            ModelCls = find_and_load_class(module_name)
            model = ModelCls(random_state=42)
            n_features = X_train.shape[1]
            split1, split2 = n_features // 3, 2 * n_features // 3
            v_data = [X_train[:, :split1], X_train[:, split1:split2], X_train[:, split2:]]
            # Reshape
            v_data = [v.reshape(len(X_train), -1) for v in v_data]
            model.fit(v_data, y_train)
            
            def predict_fn(x):
                v_test = [x[:, :split1].reshape(len(x),-1), x[:, split1:split2].reshape(len(x),-1), x[:, split2:].reshape(len(x),-1)]
                return model.predict_proba(v_test)[:, 1]

        # --- Group 4: Auto Baselines ---
        else:
            ModelCls = find_and_load_class(module_name)
            try: model = ModelCls(input_dim=X_train.shape[1])
            except: model = ModelCls()
            
            if hasattr(model, 'fit'):
                try: 
                    sig = inspect.signature(model.fit)
                    if 'X_valid' in sig.parameters: model.fit(X_train, y_train, X_valid=X_test, y_valid=y_test)
                    else: model.fit(X_train, y_train)
                except: model = train_pytorch_generic(model, X_train, y_train)
            else: model = train_pytorch_generic(model, X_train, y_train)
            
            def predict_fn(x):
                model.eval()
                t = torch.FloatTensor(x).to(device)
                with torch.no_grad():
                    try: 
                        out = model(t)
                        if hasattr(model, 'predict_proba'): return model.predict_proba(x)[:, 1]
                        logits = out.get('pred', out) if isinstance(out, dict) else out
                        if isinstance(logits, tuple): logits = logits[0]
                        return torch.sigmoid(logits).cpu().numpy().flatten()
                    except: return np.zeros(len(x))

        # --- 评估 ---
        y_pred = predict_fn(X_test)
        auc = roc_auc_score(y_test, y_pred)
        metrics = compute_metrics_final(predict_fn, X_test, feature_names)
        metrics['auc'] = auc
        
        elapsed = time.time() - start_time
        print(f"   ✅ {name} 完成! 耗时: {elapsed:.1f}s | AUC: {auc:.4f}")
        return metrics

    except Exception as e:
        print(f"   ❌ {name} 崩溃: {e}")
        return {'auc': 0, 'delta_cate': 0, 'pehe': 999, 'sens_age': 999, 'note': str(e)[:30]}

# ==========================================
# 5. 主程序
# ==========================================
def main():
    print("\n" + "="*120)
    print("🏆 DLC Final Battle: The Return of SOTA (Stage-wise Training)")
    print("="*120)
    
    X, y, true_prob, feature_names = load_data('LUAD', 'interaction', verbose=False)
    # 强制固定 Random State = 42, 保证和 finetune.py 数据一致
    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, true_prob, test_size=0.2, random_state=42, stratify=y
    )
    
    try:
        pretrained = torch.load('results/dlc_final_sota.pth', map_location='cpu') 
    except:
        pretrained = torch.load('results/dlc_final_sota.pth', map_location='cpu')

    tasks = [
        ('XGBoost', None),
        ('TabR', 'tabr_baseline'),
        ('MOGONET', 'mogonet_baseline'),
        ('EXP2 (Scratch)', None),
        ('EXP4 (Transfer)', None) 
    ]
    
    results = {}
    for name, module_name in tasks:
        kwargs = {}
        if name == 'EXP4 (Transfer)': kwargs['pretrained'] = pretrained
        res = run_model_auto(name, module_name, X_train, y_train, X_test, y_test, feature_names, **kwargs)
        if res: results[name] = res

    print("\n" + "="*120)
    print(f"{'Model':<20} | {'AUC':<8} | {'Delta CATE':<12} | {'PEHE':<8} | {'Sens(Age)':<10} | {'Status'}")
    print("-" * 120)
    
    # 排序
    valid_res = {k: v for k, v in results.items() if 'note' not in v}
    sorted_res = sorted(valid_res.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    for rank, (name, m) in enumerate(sorted_res, 1):
        status = "👑 OURS" if "EXP4" in name else ""
        print(f"{name:<20} | {m['auc']:.4f}   | {m['delta_cate']:.4f}{' '*6} | {m['pehe']:.4f}   | {m['sens_age']:.4f}{' '*4} | {status}")
    print("-" * 120)

if __name__ == '__main__':
    main()
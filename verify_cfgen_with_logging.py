#!/usr/bin/env python3
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.baselines.cf_gen import CFGen

os.makedirs('logs', exist_ok=True)
log_file = 'logs/cfgen_training.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

for handler in logger.handlers:
    handler.flush = lambda: None
    if isinstance(handler, logging.FileHandler):
        handler.stream.reconfigure(line_buffering=True)

def load_ihdp(csv_path):
    df = pd.read_csv(csv_path, header=None)
    t = df.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
    y_fact = df.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)
    X = df.iloc[:, 5:].values.astype(np.float32)
    mu0 = df.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)
    mu1 = df.iloc[:, 4].values.reshape(-1, 1).astype(np.float32)
    true_ite = mu1 - mu0
    return X, t, y_fact, true_ite

def train_cfgen_with_logging(seed=1):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training CFGen - IHDP Replication {seed}")
    logger.info(f"{'='*60}\n")
    
    IHDP_PATH = f"data/baselines_official/IHDP/ihdp_npci_{seed}.csv"
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Loading: {IHDP_PATH}\n")
    
    X, t, y, ite_true = load_ihdp(IHDP_PATH)
    logger.info(f"Data: X={X.shape}, t={t.shape}, y={y.shape}\n")
    
    X_train, X_test, t_train, t_test, y_train, y_test, ite_train, ite_test = train_test_split(
        X, t, y, ite_true, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(t_train), torch.tensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CFGen(input_dim=25, hidden_dim=200, latent_dim=20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    logger.info("Training started...\n")
    train_start = time.time()
    
    model.train()
    for ep in tqdm(range(EPOCHS), desc="Training", ncols=80, file=sys.stderr):
        total_loss = 0
        for bx, bt, by in train_loader:
            bx, bt, by = bx.to(DEVICE), bt.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(bx, bt, by)
            loss, _, _, _, _ = model.compute_loss(outputs, bx, bt, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (ep + 1) % 20 == 0:
            msg = f"Epoch {ep+1}: Loss={total_loss/len(train_loader):.4f}"
            logger.info(msg)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
    
    train_time = time.time() - train_start
    logger.info(f"\nTraining completed: {train_time:.2f}s\n")
    
    logger.info("Measuring inference time...")
    model.eval()
    
    bx_test = torch.tensor(X_test).to(DEVICE)
    with torch.no_grad():
        _ = model.predict(bx_test)
    
    inference_times = []
    n_repeats = 10
    
    for _ in tqdm(range(n_repeats), desc="Inference", ncols=80, file=sys.stderr):
        with torch.no_grad():
            start = time.time()
            _ = model.predict(bx_test)
            elapsed = time.time() - start
            inference_times.append(elapsed)
    
    avg_inf = np.mean(inference_times)
    std_inf = np.std(inference_times)
    per_sample = (avg_inf / len(X_test)) * 1000
    
    logger.info(f"\nInference: {avg_inf:.4f}±{std_inf:.4f}s total")
    logger.info(f"Per sample: {per_sample:.4f}ms\n")
    
    with torch.no_grad():
        y0_pred, y1_pred = model.predict(bx_test)
        ite_pred = (y1_pred - y0_pred).cpu().numpy()
    
    pehe = np.sqrt(np.mean((ite_test.squeeze() - ite_pred.squeeze())**2))
    logger.info(f"PEHE: {pehe:.4f}")
    
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    
    return {
        'seed': seed,
        'train_time': train_time,
        'inference_time': per_sample,
        'pehe': pehe
    }

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("CFGen Training with Dual Output (Terminal + Log File)")
    logger.info("="*60 + "\n")
    
    seeds = [1, 2, 3]
    results = []
    
    for seed in seeds:
        result = train_cfgen_with_logging(seed)
        results.append(result)
        logger.info(f"\n{'='*60}\n")
    
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60 + "\n")
    
    train_times = [r['train_time'] for r in results]
    inf_times = [r['inference_time'] for r in results]
    pehes = [r['pehe'] for r in results]
    
    logger.info(f"Training: {np.mean(train_times):.2f}±{np.std(train_times):.2f}s")
    logger.info(f"Inference: {np.mean(inf_times):.4f}±{np.std(inf_times):.4f}ms/sample")
    logger.info(f"PEHE: {np.mean(pehes):.4f}±{np.std(pehes):.4f}\n")
    
    for r in results:
        logger.info(f"Seed {r['seed']}: Train={r['train_time']:.2f}s, Inf={r['inference_time']:.4f}ms, PEHE={r['pehe']:.4f}")
    
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    
    logger.info(f"\nLog file saved to: {log_file}")

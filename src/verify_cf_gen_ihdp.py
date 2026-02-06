
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root
sys.path.append(os.getcwd())

from src.baselines.cf_gen import CFGen
from src.dlc.metrics import compute_pehe_from_arrays

def load_ihdp(csv_path):
    """
    Load IHDP data.
    Format usually:
    Col 0: Treatment (t)
    Col 1: Y_factual (y) (or col with 'y' name)
    ...
    Columns for mu1, mu0 indicate Ground Truth
    Columns for x (features)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IHDP file not found: {csv_path}")
    
    # Read without header to check structure? Most IHDP npci have no header
    # Standard NPCI format:
    # 1. Treatment
    # 2. Response (Y)
    # 3. Y_cf (Counterfactual)
    # 4. mu0 (True outcome 0)
    # 5. mu1 (True outcome 1)
    # 6..N: Features (Covariates)
    
    # Let's verify by loading first few lines
    df = pd.read_csv(csv_path, header=None)
    
    # IHDP typically:
    # Col 0: T (Int)
    # Col 1: Y (Float) -- Wait, Y_factual?
    # Col 2: Y_cf
    # Col 3: mu0
    # Col 4: mu1
    # Col 5-29: X (25 features)
    
    t = df.iloc[:, 0].values.reshape(-1, 1).astype(np.float32)
    y_fact = df.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)
    y_cf = df.iloc[:, 2].values.reshape(-1, 1).astype(np.float32)
    mu0 = df.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)
    mu1 = df.iloc[:, 4].values.reshape(-1, 1).astype(np.float32)
    X = df.iloc[:, 5:].values.astype(np.float32)
    
    # True ITE
    true_ite = mu1 - mu0
    
    return X, t, y_fact, true_ite

def train_and_verify():
    # 1. Config
    IHDP_PATH = "data/baselines_official/IHDP/ihdp_npci_1.csv"
    EPOCHS = 200 # Sufficient for IHDP (small data)
    BATCH_SIZE = 64
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== CF-Gen Verification on IHDP (Replication Check) ===")
    print(f"Device: {DEVICE}")
    print(f"Data: {IHDP_PATH}")
    
    # 2. Load Data
    X, t, y, ite_true = load_ihdp(IHDP_PATH)
    print(f"Data Shape: X={X.shape}, t={t.shape}, y={y.shape}")
    
    # Split Train/Val/Test (60/30/10 standard for IHDP 10-runs, but here just 1 run)
    X_train, X_test, t_train, t_test, y_train, y_test, ite_train, ite_test = train_test_split(
        X, t, y, ite_true, test_size=0.2, random_state=42
    )
    
    # Standardize X (Important for VAE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(t_train), torch.tensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model
    model = CFGen(input_dim=25, hidden_dim=200, latent_dim=20).to(DEVICE) # IHDP input dim is 25
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 4. Train Loop
    model.train()
    for ep in range(EPOCHS):
        total_loss_val = 0
        for bx, bt, by in train_loader:
            bx, bt, by = bx.to(DEVICE), bt.to(DEVICE), by.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(bx, bt, by)
            loss, lx, lt, ly, lkl = model.compute_loss(outputs, bx, bt, by)
            loss.backward()
            optimizer.step()
            
            total_loss_val += loss.item()
            
        if (ep+1) % 50 == 0:
            print(f"Ep {ep+1}: Loss {total_loss_val / len(train_loader):.4f}")

    # --- Verification Step: Check Train PEHE as well (Sanity) ---
    model.eval()
    with torch.no_grad():
         # Train Set
        y0_tr, y1_tr = model.predict(torch.tensor(X_train).to(DEVICE))
        ite_tr = (y1_tr - y0_tr).cpu().numpy()
        pehe_tr = np.sqrt(np.mean((ite_train.squeeze() - ite_tr.squeeze())**2))
        print(f"Train PEHE: {pehe_tr:.4f}")

    # 5. Eval (Test Set)
    # model.eval() # Already eval
    with torch.no_grad():
        bx_test = torch.tensor(X_test).to(DEVICE)
        # Predict ITE
        y0_pred, y1_pred = model.predict(bx_test)
        ite_pred = (y1_pred - y0_pred).cpu().numpy()
        
    # 6. Metrics
    # pehe = np.sqrt(np.mean((ite_true[len(X_train):].squeeze() - ite_pred.squeeze())**2)) # PEHE formula
    # Or strict definition: RMSE of (True ITE - Pred ITE)
    
    # Just to be precise with indices, let's use ite_test
    pehe = np.sqrt(np.mean((ite_test.squeeze() - ite_pred.squeeze())**2))
    
    print(f"\n[Result] IHDP PEHE Score: {pehe:.4f}")
    
    # Verification Limit
    # Standard CVAE on IHDP is around 0.6 - 0.8 usually.
    # Bad models > 1.5.
    if pehe < 1.2: # Relaxed slightly for single run
        print("✅ SUCCESS: CF-Gen successfully reproduces Causal VAE performance on IHDP.")
    else:
        print("❌ FAILURE: CF-Gen PEHE is too high. Model definition needs fix.")

if __name__ == "__main__":
    train_and_verify()

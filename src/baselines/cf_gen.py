import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import numpy as np

class CFGen(nn.Module):
    """
    CF-Gen: Counterfactual Generative Modeling.
    Simplified VAE structure optimized for Causal Inference (similar to CEVAE).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 200, latent_dim: int = 20, outcome_type: str = 'continuous'):
        super(CFGen, self).__init__()
        
        self.input_dim = input_dim
        self.outcome_type = outcome_type
        
        # --- 1. Inference Network (Encoder) q(z|x,t,y) ---
        # Input: X + T + Y
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- 2. Prior Network p(z|x) --- 
        # Crucial for test time inference when we don't have true Y
        self.prior_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- 3. Generative/Decoder Networks ---
        
        # p(x|z)
        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # p(t|z)
        self.decoder_t = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # p(y|z, t) - TARNet style heads
        output_activation = nn.Sigmoid() if outcome_type == 'binary' else nn.Identity()
        
        self.outcome_net_0 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            output_activation
        )
        self.outcome_net_1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            output_activation
        )
    
    # ... (rest is same-ish but loss needs update)

    def compute_loss(self, outputs, x, t, y):
        # Reconstruction
        loss_x = F.mse_loss(outputs['x_recon'], x, reduction='sum')
        loss_t = F.binary_cross_entropy(outputs['t_prob'], t, reduction='sum')
        
        if self.outcome_type == 'binary':
            loss_y = F.binary_cross_entropy(outputs['y_pred'], y, reduction='sum')
        else:
            loss_y = F.mse_loss(outputs['y_pred'], y, reduction='sum')
        
        # ... KL ...


    def encode(self, x, t, y):
        inp = torch.cat([x, t, y], dim=1)
        h = self.encoder_backbone(inp)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t, y):
        # 1. Inference q(z|x,t,y)
        mu_q, logvar_q = self.encode(x, t, y)
        z_q = self.reparameterize(mu_q, logvar_q)
        
        # 2. Prior p(z|x) (for KL calculation)
        h_p = self.prior_backbone(x)
        mu_p = self.prior_mu(h_p)
        logvar_p = self.prior_logvar(h_p)
        
        # 3. Reconstruction
        x_recon = self.decoder_x(z_q)
        t_prob = self.decoder_t(z_q)
        
        y0 = self.outcome_net_0(z_q)
        y1 = self.outcome_net_1(z_q)
        y_pred = t * y1 + (1 - t) * y0
        
        return {
            'x_recon': x_recon, 't_prob': t_prob, 'y_pred': y_pred,
            'y0': y0, 'y1': y1,
            'mu_q': mu_q, 'logvar_q': logvar_q,
            'mu_p': mu_p, 'logvar_p': logvar_p,
            'z_q': z_q
        }
    
    def predict(self, x):
        """
        Test time prediction using Prior Network p(z|x).
        """
        h_p = self.prior_backbone(x)
        mu_p = self.prior_mu(h_p)
        # Use mean of prior for deterministic prediction
        z = mu_p 
        
        y0 = self.outcome_net_0(z)
        y1 = self.outcome_net_1(z)
        return y0, y1

    def compute_loss(self, outputs, x, t, y):
        # Reconstruction
        loss_x = F.mse_loss(outputs['x_recon'], x, reduction='sum')
        loss_t = F.binary_cross_entropy(outputs['t_prob'], t, reduction='sum')
        
        if self.outcome_type == 'binary':
            loss_y = F.binary_cross_entropy(outputs['y_pred'], y, reduction='sum')
        else:
            loss_y = F.mse_loss(outputs['y_pred'], y, reduction='sum')
        
        # KL Divergence: KL(q(z|x,t,y) || p(z|x))
        mu_q, logvar_q = outputs['mu_q'], outputs['logvar_q']
        mu_p, logvar_p = outputs['mu_p'], outputs['logvar_p']
        
        # General KL between two Gaussians
        var_p = torch.exp(logvar_p)
        var_q = torch.exp(logvar_q)
        kl_div = 0.5 * torch.sum(
            (var_q / var_p) + ((mu_p - mu_q)**2 / var_p) - 1.0 + (logvar_p - logvar_q)
        )
        
        # Weights
        # loss_y boosted to ensure ITE accuracy
        # loss_x kept low but non-zero to ground Z in covariates
        total_loss = loss_x + loss_t + 5.0 * loss_y + 1.0 * kl_div
        
        return total_loss, loss_x, loss_t, loss_y, kl_div

class CFGenAdapter(BaseModel):
    """
    Adapter for CFGen to fit the BaseModel loop.
    """
    def __init__(self, input_dim, hidden_dim=200, latent_dim=20, lr=0.001, epochs=50, batch_size=64, treatment_idx=22, outcome_type='binary'):
        self.treatment_idx = treatment_idx
        self.outcome_type = outcome_type
        # CFGen input_dim should be features count (input_dim - 1)
        self.model = CFGen(input_dim - 1, hidden_dim, latent_dim, outcome_type=outcome_type)
        
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.t_median = 0.0
        
    def fit(self, X, y):
        # Handle X (Features + T)
        t = X[:, self.treatment_idx]
        self.t_median = np.median(t)
        t_bin = (t > self.t_median).astype(np.float32)
        
        X_cov = np.delete(X, self.treatment_idx, axis=1) # Remove T
        
        self.model.train()
        X_ten = torch.tensor(X_cov, dtype=torch.float32).to(self.device)
        t_ten = torch.tensor(t_bin, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_ten = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_ten, t_ten, y_ten)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for ep in range(self.epochs):
            total_loss = 0
            for bx, bt, by in loader:
                self.optimizer.zero_grad()
                outputs = self.model(bx, bt, by)
                loss, _, _, _, _ = self.model.compute_loss(outputs, bx, bt, by)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def predict(self, X):
         # Predict Factual Class
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        t = X[:, self.treatment_idx]
        t_bin = (t > self.t_median).astype(np.float32)
        X_cov = np.delete(X, self.treatment_idx, axis=1)
        
        self.model.eval()
        with torch.no_grad():
            X_cov_ten = torch.tensor(X_cov, dtype=torch.float32).to(self.device)
            y0, y1 = self.model.predict(X_cov_ten) # shape [B, 1]
            
            y0 = y0.cpu().numpy().flatten()
            y1 = y1.cpu().numpy().flatten()
            
            # Factual probability
            y_prob = t_bin * y1 + (1 - t_bin) * y0
            
            return np.vstack([1 - y_prob, y_prob]).T

    def predict_ite(self, X):
        X_cov = np.delete(X, self.treatment_idx, axis=1)
        self.model.eval()
        with torch.no_grad():
            X_cov_ten = torch.tensor(X_cov, dtype=torch.float32).to(self.device)
            y0, y1 = self.model.predict(X_cov_ten)
            
        return y1.cpu().numpy().flatten() - y0.cpu().numpy().flatten()

    def evaluate(self, X, t, y):
        pass
    
    def get_params(self, deep=True):
        return {"input_dim": self.model.input_dim}
    
    def set_params(self, **params):
        pass

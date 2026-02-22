import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dlc.dlc_net import DLCNet

class DLCNoHGNN(DLCNet):
    """
    Ablation Variant 1: w/o HGNN
    Replaces DynamicHypergraphNN with a simple MLP merging Gene features and Z_effect.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Replace hypergraph_nn with simple projection
        # Input: X_gene (20) + Z_effect (16) -> Hidden (64)
        self.gene_mlp = nn.Sequential(
            nn.Linear(20 + self.d_effect, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU()
        )
        # Remove original HGNN to save memory (optional, but good practice)
        del self.hypergraph_nn

    def forward(self, X):
        # 1. Split
        X_gene = X[:, 2:-1]  # [B, 20]
        
        # 2. VAE
        vae_outputs = self.causal_vae(X)
        Z_conf = vae_outputs['Z_conf']
        Z_effect = vae_outputs['Z_effect']
        X_recon = vae_outputs['X_recon']
        
        # 3. Feature Fusion (MLP instead of Hypergraph)
        # Concatenate Gene features and Effect features
        combined = torch.cat([X_gene, Z_effect], dim=1) # [B, 36]
        H_global = self.gene_mlp(combined)              # [B, 64]

        # Adversarial (Keep it)
        Z_effect_rev = self.grl(Z_effect)
        Age_pred = self.adv_age_head(Z_effect_rev)
        
        # 4. Predict
        Y_0 = self.outcome_head_0(H_global)
        Y_1 = self.outcome_head_1(H_global)
        ITE = Y_1 - Y_0
        
        return {
            'Y_0': Y_0, 'Y_1': Y_1, 'ITE': ITE,
            'Z_conf': Z_conf, 'Z_effect': Z_effect,
            'X_recon': X_recon, 'Age_pred': Age_pred
        }

class DLCNoVAE(DLCNet):
    """
    Ablation Variant 2: w/o VAE (Deterministic Encoder)
    Keeps structure but removes stochastic sampling (reparameterization).
    Logic for removing KL/Recon loss will be handled in the training loop 
    (by setting weights to 0), but here we force deterministic encoding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We don't need to change __init__, just forward behavior
        # But to be strict, let's override forward to skip reparameterization?
        # Actually, standard DLC VAE code computes mu and logvar.
        # We can just use mu as Z.
        pass

    def forward(self, X):
         # 1. Split
        X_gene = X[:, 2:-1]
        
        # 2. Deterministic Encoding (Bypass VAE sampling)
        # Manually call encode to get Mu only
        mu_conf, _, mu_effect, _ = self.causal_vae.encode(X)
        
        # Use Mu directly as Z
        Z_conf = mu_conf
        Z_effect = mu_effect
        
        # Pseudo-reconstruction (optional, or just skip)
        # If we skip recon, we might break loss computation if it expects X_recon
        # Let's do a pass through decoder with Mu
        X_recon = self.causal_vae.decoder(torch.cat([Z_conf, Z_effect], dim=1))
        
        # 3. HGNN
        H_out = self.hypergraph_nn(X_gene, Z_effect)
        H_global = torch.mean(H_out, dim=1)

        # Adversarial
        Z_effect_rev = self.grl(Z_effect)
        Age_pred = self.adv_age_head(Z_effect_rev)
        
        # 4. Predict
        Y_0 = self.outcome_head_0(H_global)
        Y_1 = self.outcome_head_1(H_global)
        ITE = Y_1 - Y_0
        
        # Return dict - note we don't return logvars so KL will fail if attempted
        return {
            'Y_0': Y_0, 'Y_1': Y_1, 'ITE': ITE,
            'Z_conf': Z_conf, 'Z_effect': Z_effect,
            'X_recon': X_recon, 'Age_pred': Age_pred,
            # Add dummy logvars to avoid crash if loss tries to access them?
            # Or better: The loss function usually looks inside vae_outputs or separate?
            # DLCNet.forward returns flattened dict. 
            # Original forward calls vae_outputs = self.causal_vae(X) which returns dict with mu/logvar
            # We are constructing return dict manually.
        }

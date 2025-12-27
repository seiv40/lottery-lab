#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) for Module 5: Deep Learning Model Zoo

"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils_seed import set_global_seed
from utils_data import DATA_PATH, OUTPUT_DIR, MODELS_DIR
from utils_metrics import write_summary

# lottery name to file suffix mapping
LOTTERY_SUFFIX_MAP = {
    "powerball": "_pb",
    "megamillions": "_mm",
}


# VAE Architecture

class VAE(nn.Module):
    """
    Variational Autoencoder with reparameterization trick.
    
    The VAE tries to compress high-dimensional lottery features into a
    small latent space (usually 2D), then reconstruct them back.
    
    The magic is in the latent space - if lottery is random, it should
    be a featureless blob. If there are patterns, we'd see structure.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]  # default architecture
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # encoder: compresses features to latent space
        # we predict both mean and log_variance (for the reparameterization trick)
        enc_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h_dim))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(0.1))  # prevent overfitting
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)  # mean of latent distribution
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)  # log variance (more stable than variance directly)
        
        # decoder: reconstructs features from latent space
        dec_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))  # back to original feature space
        
        self.decoder = nn.Sequential(*dec_layers)
    
    def encode(self, x):
        """encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # clamp log_var to prevent numerical issues
        # without this, exp(log_var) can explode or vanish
        log_var = torch.clamp(log_var, min=-20, max=20)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick: sample from N(mu, var) in a differentiable way.
        
        Instead of sampling z ~ N(mu, var) directly (which breaks backprop),
        we sample epsilon ~ N(0,1) and compute z = mu + epsilon * std.
        
        This way gradients can flow through mu and std.
        """
        std = torch.exp(0.5 * log_var)  # convert log_var to std
        eps = torch.randn_like(std)  # sample noise
        z = mu + eps * std  # reparameterize
        return z
    
    def decode(self, z):
        """decode latent vector back to feature space"""
        return self.decoder(z)
    
    def forward(self, x):
        """full forward pass: encode, sample, decode"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    """
    VAE loss = reconstruction loss + KL divergence.
    
    Reconstruction loss: how well can we rebuild the input?
    KL divergence: how far is the latent distribution from standard normal?
    
    Beta is a hyperparameter that controls the tradeoff. beta > 1 emphasizes
    learning disentangled representations (beta-VAE).
    """
    # reconstruction loss - mean squared error
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence between learned distribution and standard normal
    # analytical formula: -0.5 * sum(1 + log(var) - mu^2 - var)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # total loss (ELBO - evidence lower bound)
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


# Dataset

class LotteryFeatureDataset(Dataset):
    """simple dataset wrapper for lottery features"""
    
    def __init__(self, features: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


# Training and Export

def train_and_export(dm, args) -> dict:
    """
    Train VAE and export results.
    
    The key outputs are:
    1. Performance metrics (reconstruction quality)
    2. 2D embeddings of all drawings (for visualization)
    3. Model weights (so we can reload and analyze later)
    """
    # reproducibility
    seed = getattr(args, "seed", 42)
    set_global_seed(seed, deterministic=True)
    
    lottery = args.lottery
    latent_dim = int(getattr(args, "latent_dim", 2))
    batch_size = int(getattr(args, "batch_size", 64))
    lr = float(getattr(args, "learning_rate", 1e-3))
    max_epochs = int(getattr(args, "epochs", 100))
    beta = float(getattr(args, "beta", 1.0))  # beta-VAE parameter
    
    # load features
    path = DATA_PATH / f"features_{lottery}.parquet"
    if not path.exists():
        print(f"FATAL: Cannot find features file: {path}")
        return {"ok": False, "error": f"File not found: {path}"}
    
    try:
        df = dm.load_features()
        if df is None:
            raise ValueError("dm.load_features() returned None")
    except Exception as e:
        print(f"Warning: dm.load_features() failed ({e}). Loading manually.")
        df = pd.read_parquet(path)
    
    # define features - exclude metadata columns
    exclude_cols = ["drawing_index", "date", "game", "special_ball", "mean"]
    feature_cols = [
        col for col in df.columns
        if df[col].dtype in ["float64", "float32", "int64", "int32"]
        and col not in exclude_cols
    ]
    input_dim = len(feature_cols)
    print(f"Using {input_dim} feature columns for VAE")
    
    # fill NaNs (from rolling window features at the start)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # chronological split to prevent temporal leakage
    n = len(df)
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_val]
    test_df = df.iloc[n_val:]
    
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # standardize features (fit on train only to prevent leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    X_all = scaler.transform(df[feature_cols].values)  # for embeddings
    
    # create dataloaders
    train_dataset = LotteryFeatureDataset(X_train)
    val_dataset = LotteryFeatureDataset(X_val)
    test_dataset = LotteryFeatureDataset(X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # training loop
    start_time = time.time()
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(max_epochs):
        # training
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        for batch_X in train_loader:
            batch_X = batch_X.to(device)
            
            optimizer.zero_grad()
            x_recon, mu, log_var = model(batch_X)
            loss, recon, kl = vae_loss(x_recon, batch_X, mu, log_var, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        # validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for batch_X in val_loader:
                batch_X = batch_X.to(device)
                x_recon, mu, log_var = model(batch_X)
                loss, recon, kl = vae_loss(x_recon, batch_X, mu, log_var, beta=beta)
                
                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()
        
        # average losses
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        
        if getattr(args, "verbose", False) and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"[VAE][{lottery}] Epoch {epoch+1}/{max_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    runtime = time.time() - start_time
    
    # test evaluation
    model.eval()
    test_loss = 0.0
    test_recon = 0.0
    test_kl = 0.0
    
    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X.to(device)
            x_recon, mu, log_var = model(batch_X)
            loss, recon, kl = vae_loss(x_recon, batch_X, mu, log_var, beta=beta)
            
            test_loss += loss.item()
            test_recon += recon.item()
            test_kl += kl.item()
    
    # average test metrics
    test_loss /= len(test_dataset)
    test_recon /= len(test_dataset)
    test_kl /= len(test_dataset)
    
    # generate 2D embeddings for ALL drawings (for visualization)
    all_dataset = LotteryFeatureDataset(X_all)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch_X in all_loader:
            batch_X = batch_X.to(device)
            mu, _ = model.encode(batch_X)  # use mean of latent distribution
            embeddings.append(mu.cpu().numpy())
    
    embeddings = np.vstack(embeddings)  # shape: (n_drawings, latent_dim)
    
    # save results
    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = LOTTERY_SUFFIX_MAP[lottery]
    
    # performance metrics
    metrics = {
        "model_name": "vae",
        "lottery": lottery,
        "test_loss": float(test_loss),
        "test_reconstruction": float(test_recon),
        "test_kl_divergence": float(test_kl),
        "runtime_sec": float(runtime),
        "params": {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "beta": beta,
            "learning_rate": lr,
            "epochs": max_epochs,
        },
        "embeddings": embeddings.tolist(),  # 2D coordinates for each drawing
    }
    
    perf_path = out_dir / f"vae_perf{suffix}.json"
    perf_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    # save model weights
    vae_dir = MODELS_DIR / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    weights_path = vae_dir / f"{lottery}_vae.pt"
    torch.save(model.state_dict(), weights_path)
    
    # append to summary
    summary_path = write_summary("vae", lottery, metrics, args=args)
    
    print(f"[VAE] Training complete for {lottery}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Reconstruction: {test_recon:.4f}")
    print(f"  Test KL Divergence: {test_kl:.4f}")
    print(f"  Embeddings saved: {embeddings.shape}")
    
    return {
        "model": "vae",
        "perf_path": str(perf_path),
        "weights_path": str(weights_path),
        "summary_path": str(summary_path),
    }


# Standalone Execution

if __name__ == "__main__":
    import argparse
    from utils_data import LotteryFeatureDataModule
    
    parser = argparse.ArgumentParser(description="Train VAE on lottery data")
    
    # required arguments
    parser.add_argument("--lottery", type=str, required=True,
                       choices=["powerball", "megamillions"])
    
    # model hyperparameters
    parser.add_argument("--latent_dim", type=int, default=2,
                       help="Dimension of latent space (2 for visualization)")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Beta parameter for beta-VAE (1.0 = standard VAE)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    # compatibility arguments (not used but accepted for compatibility with zoo launcher)
    parser.add_argument("--window_len", type=int, default=64)
    parser.add_argument("--bnn_seq_len", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--conformal_mode", type=str, default="pooled")
    parser.add_argument("--eprocess_mode", type=str, default="standard")
    parser.add_argument("--safe_e_gamma", type=float, default=0.4)
    parser.add_argument("--baseline", type=str, default="copy_last")
    
    args = parser.parse_args()
    
    print(f"[VAE] Training on {args.lottery} dataset")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Beta: {args.beta}")
    
    set_global_seed(args.seed, deterministic=True)
    
    dm = LotteryFeatureDataModule(
        lottery=args.lottery,
        batch_size=args.batch_size,
        window_len=getattr(args, "window_len", 64)
    )
    
    args.suffix = LOTTERY_SUFFIX_MAP[args.lottery]
    
    try:
        result = train_and_export(dm, args)
        print(f"[VAE] Success! Results saved to: {result.get('perf_path')}")
    except Exception as e:
        print(f"[VAE] Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("[VAE] Done.")

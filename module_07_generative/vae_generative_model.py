#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) for Module 7 - GENERATIVE VERSION
Trained on 6D ball positions (not 103D features) for universe generation.

This is separate from Module 5's feature-based VAE because:
- Module 5: Pattern detection (features -> features) 
- Module 7: Generation (latent -> actual balls)

"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# PATHS - UPDATE THESE FOR YOUR SYSTEM

DATA_PATH = Path(r'C:\jackpotmath\lottery-lab\data\raw')
OUTPUT_DIR = Path(r'C:\jackpotmath\lottery-lab\output')
MODELS_DIR = Path(r'C:\jackpotmath\lottery-lab\models\vae')


# VAE MODEL (same architecture as Module 5)

class VAE(nn.Module):
    """
    Standard Variational Autoencoder (VAE) with MLP encoder/decoder.
    Same architecture as Module 5, but trained on ball positions.
    """
    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder: X -> z_mu, z_log_var
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # decoder: z -> X_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick with stabilization."""
        log_var = torch.clamp(log_var, -20, 20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode x to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        """Decode latent z to reconstructed x."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode, reparameterize, decode."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss_function(x_recon, x, mu, log_var, beta=1.0):
    """
    ELBO loss: Reconstruction loss + beta * KL divergence.
    """
    # reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return (recon_loss + beta * kld) / x.size(0)


# TRAINING FUNCTION

def train_generative_vae(lottery: str,
                         latent_dim: int = 2,
                         hidden_dim: int = 128,
                         beta: float = 1.0,
                         epochs: int = 100,
                         batch_size: int = 64,
                         lr: float = 1e-3,
                         weight_decay: float = 0.0,
                         seed: int = 42,
                         verbose: bool = True):
    """
    Train VAE on 6D ball positions for generative modeling.
    
    Args:
        lottery: 'powerball' or 'megamillions'
        latent_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
        beta: Beta-VAE coefficient
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: L2 regularization
        seed: Random seed
        verbose: Print training progress
    
    Returns:
        Dictionary with model, metrics, and paths
    """
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n--- TRAINING GENERATIVE VAE - {lottery.upper()} ---")
    
    # load raw ball data
    data_file = DATA_PATH / f"{lottery}_draws.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_parquet(data_file)
    print(f" Loaded {len(df)} draws from {data_file.name}")
    
    # extract ball columns
    if lottery == 'powerball':
        ball_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'powerball']
    elif lottery == 'megamillions':
        ball_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'megaball']
    else:
        raise ValueError(f"Unknown lottery: {lottery}")
    
    # verify columns exist
    missing = [col for col in ball_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    X = df[ball_cols].values.astype(np.float32)
    input_dim = X.shape[1]  # should be 6
    
    print(f" Extracted {input_dim} ball columns: {ball_cols}")
    print(f"  Shape: {X.shape}")
    print(f"  Range: [{X.min():.0f}, {X.max():.0f}]")
    
    # sequential split (70/15/15)
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)  # 70% + 15% = 85%
    
    X_train = X[:n_train]
    X_val = X[n_train:n_val]
    X_test = X[n_val:]
    
    print(f" Sequential split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # standardize (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f" Standardized features")
    print(f"  Mean: {scaler.mean_[:3].round(2)}...")
    print(f"  Scale: {scaler.scale_[:3].round(2)}...")
    
    # create datasets
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"\n Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # training loop
    print("\n--- TRAINING ---")
    
    start_time = time.time()
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            
            x_recon, mu, log_var = model(X_batch)
            loss = vae_loss_function(x_recon, X_batch, mu, log_var, beta=beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)
        
        train_loss /= train_count
        
        # validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(device)
                x_recon, mu, log_var = model(X_batch)
                loss = vae_loss_function(x_recon, X_batch, mu, log_var, beta=beta)
                val_loss += loss.item() * X_batch.size(0)
                val_count += X_batch.size(0)
        
        val_loss /= val_count
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # print progress
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    runtime = time.time() - start_time
    
    print(f"\n Training complete ({runtime:.1f}s)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    
    # test evaluation
    model.eval()
    test_loss = 0.0
    test_count = 0
    
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            x_recon, mu, log_var = model(X_batch)
            loss = vae_loss_function(x_recon, X_batch, mu, log_var, beta=beta)
            test_loss += loss.item() * X_batch.size(0)
            test_count += X_batch.size(0)
    
    test_loss /= test_count
    
    print(f"  Test loss: {test_loss:.4f}")
    
    # extract embeddings for full dataset
    X_full_scaled = scaler.transform(X)
    full_dataset = TensorDataset(torch.tensor(X_full_scaled, dtype=torch.float32))
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    all_mu = []
    all_log_var = []
    
    with torch.no_grad():
        for (X_batch,) in full_loader:
            X_batch = X_batch.to(device)
            mu, log_var = model.encode(X_batch)
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
    
    all_mu = np.concatenate(all_mu, axis=0)
    all_log_var = np.concatenate(all_log_var, axis=0)
    
    # save outputs
    print("\n--- SAVING OUTPUTS ---")
    
    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_pb' if lottery == 'powerball' else '_mm'
    
    # save model weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{lottery}_vae_gen.pt"
    torch.save(model.state_dict(), model_path)
    print(f" Model saved: {model_path}")
    
    # save scaler
    scaler_path = MODELS_DIR / f"{lottery}_vae_gen_scaler.pkl"
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Scaler saved: {scaler_path}")
    
    # save performance metrics
    metrics = {
        "model_name": "vae_generative",
        "lottery": lottery,
        "purpose": "Module 7 - Universe Generation",
        "input_type": "ball_positions",
        "input_dim": input_dim,
        "train_loss": float(train_loss),
        "val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "runtime_sec": float(runtime),
        "params": {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "beta": beta,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed
        },
        "data": {
            "n_total": n,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test)
        }
    }
    
    perf_path = out_dir / f"vae_gen_perf{suffix}.json"
    with open(perf_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Metrics saved: {perf_path}")
    
    # save embeddings
    embed_payload = {
        "model_name": "vae_generative",
        "lottery": lottery,
        "latent_dim": latent_dim,
        "mu": all_mu.tolist(),
        "log_var": all_log_var.tolist()
    }
    
    embed_path = out_dir / f"vae_gen_embeddings{suffix}.json"
    with open(embed_path, 'w') as f:
        json.dump(embed_payload, f, indent=2)
    print(f" Embeddings saved: {embed_path}")
    
    print("\n---  TRAINING COMPLETE ---")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "perf_path": str(perf_path),
        "embed_path": str(embed_path)
    }


# MAIN

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train generative VAE for Module 7")
    parser.add_argument("--lottery", type=str, required=True, 
                       choices=['powerball', 'megamillions'],
                       help="Lottery type")
    parser.add_argument("--latent_dim", type=int, default=2,
                       help="Latent dimension (default: 2)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension (default: 128)")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Beta-VAE coefficient (default: 1.0)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print training progress")
    
    args = parser.parse_args()
    
    try:
        result = train_generative_vae(
            lottery=args.lottery,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            verbose=args.verbose
        )
        
        print(f"\nModel saved to: {result['model_path']}")
        print(f"Ready for Module 7 universe generation!")
        
    except Exception as e:
        print(f"\n Error during training:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

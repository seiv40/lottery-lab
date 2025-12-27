#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalizing Flow (MAF) for Module 7 - GENERATIVE VERSION
Trained on 6D ball positions for universe generation.

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


# PATHS

DATA_PATH = Path(r'C:\jackpotmath\lottery-lab\data\raw')
OUTPUT_DIR = Path(r'C:\jackpotmath\lottery-lab\output')
MODELS_DIR = Path(r'C:\jackpotmath\lottery-lab\models\flow')


# FLOW ARCHITECTURE

class MaskedAffineFlow(nn.Module):
    """Masked affine flow layer with stabilization."""
    
    def __init__(self, dim, hidden_dim=128, mask_type='even'):
        super().__init__()
        self.dim = dim
        self.mask = self.build_mask(mask_type)
        
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim)
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim)
        )
        
        # initialize for stability
        nn.init.zeros_(self.s_net[-1].weight.data)
        nn.init.zeros_(self.s_net[-1].bias.data)

    def build_mask(self, mask_type):
        """Binary mask [0,1,0,1,...] or [1,0,1,0,...]."""
        mask = torch.zeros(self.dim)
        if mask_type == 'even':
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        return mask.float()

    def forward(self, x):
        """Forward: x -> z"""
        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)
            
        x_masked = x * self.mask
        s_raw = self.s_net(x_masked)
        s = torch.tanh(s_raw) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)
        
        z = (x * torch.exp(s) + t) * (1 - self.mask) + x_masked
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """Inverse: z -> x (for sampling)"""
        if self.mask.device != z.device:
            self.mask = self.mask.to(z.device)
            
        z_masked = z * self.mask
        s_raw = self.s_net(z_masked)
        s = torch.tanh(s_raw) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        
        x = ((z - t) * torch.exp(-s)) * (1 - self.mask) + z_masked
        log_det = -s.sum(dim=-1)
        return x, log_det


class MAF(nn.Module):
    """Masked Autoregressive Flow."""
    
    def __init__(self, dim, n_flows=5, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.base_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
        
        flows = []
        for i in range(n_flows):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            flows.append(MaskedAffineFlow(dim, hidden_dim, mask_type))
        self.flows = nn.Sequential(*flows)

    def to(self, *args, **kwargs):
        """Ensure base distribution is on correct device."""
        self = super().to(*args, **kwargs)
        self.base_dist = torch.distributions.Normal(
            torch.zeros(self.dim).to(*args, **kwargs),
            torch.ones(self.dim).to(*args, **kwargs)
        )
        return self

    def forward(self, x):
        """Compute log-likelihood."""
        log_det_sum = 0
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
            
        log_prob_z = self.base_dist.log_prob(z).sum(dim=-1)
        log_prob_x = log_prob_z + log_det_sum
        return log_prob_x

    def sample(self, n_samples=1):
        """Sample from flow."""
        z = self.base_dist.sample((n_samples,))
        x = z
        for flow in reversed(self.flows):
            x, _ = flow.inverse(x)
        return x


# TRAINING

def train_generative_flow(lottery: str,
                          n_flows: int = 5,
                          hidden_dim: int = 128,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3,
                          weight_decay: float = 0.0,
                          seed: int = 42,
                          verbose: bool = True):
    """Train Flow on 6D ball positions."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*70}")
    print(f"TRAINING GENERATIVE FLOW - {lottery.upper()}")
    print(f"{'='*70}")
    
    # load data
    data_file = DATA_PATH / f"{lottery}_draws.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_parquet(data_file)
    print(f" Loaded {len(df)} draws from {data_file.name}")
    
    # extract balls
    if lottery == 'powerball':
        ball_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'powerball']
    elif lottery == 'megamillions':
        ball_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'megaball']
    else:
        raise ValueError(f"Unknown lottery: {lottery}")
    
    X = df[ball_cols].values.astype(np.float32)
    input_dim = X.shape[1]
    
    print(f" Extracted {input_dim} ball columns")
    print(f"  Shape: {X.shape}")
    
    # sequential split
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    
    X_train = X[:n_train]
    X_val = X[n_train:n_val]
    X_test = X[n_val:]
    
    print(f" Sequential split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f" Standardized")
    
    # datasets
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAF(dim=input_dim, n_flows=n_flows, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"\n Model initialized")
    print(f"  Input dim: {input_dim}")
    print(f"  Flows: {n_flows}")
    print(f"  Hidden: {hidden_dim}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # training
    print("\n--- TRAINING ---")
    
    start_time = time.time()
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # train
        model.train()
        train_nll = 0.0
        train_count = 0
        
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            
            log_prob = model(X_batch)
            loss = -log_prob.mean()  # negative log-likelihood
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_nll += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)
        
        train_nll /= train_count
        
        # validation
        model.eval()
        val_nll = 0.0
        val_count = 0
        
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(device)
                log_prob = model(X_batch)
                loss = -log_prob.mean()
                val_nll += loss.item() * X_batch.size(0)
                val_count += X_batch.size(0)
        
        val_nll /= val_count
        
        if val_nll < best_val_loss:
            best_val_loss = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_nll={train_nll:.4f}, val_nll={val_nll:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    runtime = time.time() - start_time
    
    print(f"\n Training complete ({runtime:.1f}s)")
    print(f"  Best val NLL: {best_val_loss:.4f}")
    
    # test
    model.eval()
    test_nll = 0.0
    test_count = 0
    
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            log_prob = model(X_batch)
            loss = -log_prob.mean()
            test_nll += loss.item() * X_batch.size(0)
            test_count += X_batch.size(0)
    
    test_nll /= test_count
    print(f"  Test NLL: {test_nll:.4f}")
    
    # save
    print("\n--- SAVING ---")
    
    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_pb' if lottery == 'powerball' else '_mm'
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{lottery}_flow_gen.pt"
    torch.save(model.state_dict(), model_path)
    print(f" Model: {model_path}")
    
    scaler_path = MODELS_DIR / f"{lottery}_flow_gen_scaler.pkl"
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f" Scaler: {scaler_path}")
    
    metrics = {
        "model_name": "flow_generative",
        "lottery": lottery,
        "purpose": "Module 7 - Universe Generation",
        "input_type": "ball_positions",
        "input_dim": input_dim,
        "train_nll": float(train_nll),
        "val_nll": float(best_val_loss),
        "test_nll": float(test_nll),
        "runtime_sec": float(runtime),
        "params": {
            "n_flows": n_flows,
            "hidden_dim": hidden_dim,
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
    
    perf_path = out_dir / f"flow_gen_perf{suffix}.json"
    with open(perf_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Metrics: {perf_path}")
    
    print("\n---  TRAINING COMPLETE ---")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "perf_path": str(perf_path)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lottery", type=str, required=True, 
                       choices=['powerball', 'megamillions'])
    parser.add_argument("--n_flows", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    try:
        result = train_generative_flow(
            lottery=args.lottery,
            n_flows=args.n_flows,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            verbose=args.verbose
        )
        print(f"\nReady for universe generation!")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

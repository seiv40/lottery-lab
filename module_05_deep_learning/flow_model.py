#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalizing Flow (MAF) for Module 5: Deep Learning Model Zoo

"""

import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler  # <-- ADDED IMPORT

from utils_seed import set_global_seed
from utils_data import DATA_PATH, OUTPUT_DIR, MODELS_DIR
from utils_metrics import write_summary

# map lottery name to the file suffix
LOTTERY_SUFFIX_MAP = {
    'powerball': '_pb',
    'megamillions': '_mm'
}


class MaskedAffineFlow(nn.Module):
    """
    Implements a single masked affine flow layer:
    y = x * exp(s) + t
    where s and t are functions of the unmasked inputs.
    
    (MODIFIED with tanh activation on 's' to prevent explosion)
    """
    def __init__(self, dim, hidden_dim=128, mask_type='even'):
        super().__init__()
        self.dim = dim
        self.mask = self.build_mask(mask_type)
        
        # s and t networks
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
        
        # --- FIX ---
# initialize the final layer of s_net with zeros
# this makes 's' start at 0, so exp(s) starts at 1 (an identity)
# this is a common trick for stabilizing flow training.
        nn.init.zeros_(self.s_net[-1].weight.data)
        nn.init.zeros_(self.s_net[-1].bias.data)


    def build_mask(self, mask_type):
        """Creates a binary mask [0, 1, 0, 1, ...] or [1, 0, 1, 0, ...]."""
        mask = torch.zeros(self.dim)
        if mask_type == 'even':
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        return mask.float()

    def forward(self, x):
        """Forward pass (log-likelihood computation). x -> z"""
        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)
            
        x_masked = x * self.mask
        
        # --- STABILIZATION FIX ---
# use tanh to constrain the output of s_net
        # s_raw is unbounded, but s is now in [-1, 1]
# this prevents torch.exp(s) from becoming inf
        s_raw = self.s_net(x_masked)
        s = torch.tanh(s_raw) * (1 - self.mask) 
        
        t = self.t_net(x_masked) * (1 - self.mask)
        
        z = (x * torch.exp(s) + t) * (1 - self.mask) + x_masked
        
        # log|det(J)| = sum(s)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """Inverse pass (sampling). z -> x"""
        if self.mask.device != z.device:
            self.mask = self.mask.to(z.device)
            
        z_masked = z * self.mask
        
        # --- STABILIZATION FIX ---
        s_raw = self.s_net(z_masked)
        s = torch.tanh(s_raw) * (1 - self.mask)
        
        t = self.t_net(z_masked) * (1 - self.mask)
        
        # x = (z - t) * exp(-s)
        x = ((z - t) * torch.exp(-s)) * (1 - self.mask) + z_masked
        
        log_det = -s.sum(dim=-1)
        return x, log_det


class MAF(nn.Module):
    """Stack of Masked Affine Flows (RealNVP style)."""
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
        """Ensure base distribution is on the correct device."""
        self = super().to(*args, **kwargs)
        self.base_dist = torch.distributions.Normal(
            torch.zeros(self.dim).to(*args, **kwargs),
            torch.ones(self.dim).to(*args, **kwargs)
        )
        return self

    def forward(self, x):
        """Compute log-likelihood of x."""
        log_det_sum = 0
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
            
        log_prob_z = self.base_dist.log_prob(z).sum(dim=-1)
        log_prob_x = log_prob_z + log_det_sum
        return log_prob_x

    def sample(self, n_samples=1):
        """Sample from the flow (z -> x)."""
        z = self.base_dist.sample((n_samples,))
        x = z
        for flow in reversed(self.flows):
            x, _ = flow.inverse(x)
        return x


def train_and_export(dm, args) -> Dict:
    """
    MODIFIED to perform manual sequential data split AND feature standardization.
    Trains a Normalizing Flow (MAF) on the feature space.
    """
    seed = getattr(args, "seed", 42)
    set_global_seed(seed, deterministic=True)

    lottery = args.lottery
    batch_size = int(getattr(args, "batch_size", 64))
    lr = float(getattr(args, "learning_rate", 1e-3))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    max_epochs = int(getattr(args, "epochs", 100))
    
# flow specific
    flow_layers = int(getattr(args, "flow_layers", 5))
    flow_hidden = int(getattr(args, "flow_hidden", 128))


# load features parquet for this lottery
    path = DATA_PATH / f"features_{lottery}.parquet"
    if not path.exists():
        print(f"FATAL: Cannot find features file: {path}")
        return {"ok": False, "error": f"File not found: {path}"}
    
    try:
        df = dm.load_features()
        if df is None: raise ValueError("dm.load_features() returned None")
    except Exception as e:
        print(f"Error calling dm.load_features(): {e}. Falling back to manual load.")
        df = pd.read_parquet(path)

    # --- Manually Replicate Data Pipeline ---
    
    # 1. Create a dummy 'next_mean' column *only* to align splits
# this is non-standard for a VAE/Flow, but required for Module 6.
    raw_target_col = 'next_mean'
    df[raw_target_col] = df['mean'].shift(-1)
    
    # 2. Define feature columns (all numeric cols except metadata/targets)
    exclude_cols = ['drawing_index', 'date', 'game', 'special_ball', 
                    'mean', raw_target_col] # Exclude the raw target
    feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'float32', 'int64', 'int32'] and col not in exclude_cols]
    input_dim = len(feature_cols)
    print(f"Using {input_dim} feature columns.")
    
    # 3. Fill NaNs in features (e.g., from rolling windows)
    df[feature_cols] = df[feature_cols].fillna(0)
    print("Filled NaNs in feature columns with 0.")

    # 4. Drop the last row where the target is NaN (to align with BNN, etc.)
    df_clean = df.dropna(subset=[raw_target_col])
    n = len(df_clean)
    if n == 0:
        raise ValueError(f"No valid data remaining after dropna on '{raw_target_col}'.")

    # 5. Define split indices
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    
    # 6. Perform sequential split ON THE CLEANED DATA
    train_df = df_clean.iloc[:n_train]
    val_df = df_clean.iloc[n_train:n_val]
    test_df = df_clean.iloc[n_val:]
    
    # 7. --- ALIGN TEST SETS ---
# align with windowed models (Transformer)
    
    window_len = int(getattr(args, "window_len", 64)) # Get window_len from args
    n_chop = window_len - 1
    
# truncate the validation and test sets to match
    val_df = val_df.iloc[n_chop:]
    test_df = test_df.iloc[n_chop:]
    
    print(f"Aligned splits for window_len={window_len} -> Val: {len(val_df)}, Test: {len(test_df)}")

    # 8. Standardize FEATURES (was step 7)
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(train_df[feature_cols])
    X_val = feature_scaler.transform(val_df[feature_cols])
    X_test = feature_scaler.transform(test_df[feature_cols])
    print("Standardized features based on training set.")

    # 9. Create datasets
    ds_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    ds_val = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    ds_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    
# create a full dataset loader for final loglik export (using all *clean* data)
    X_full_scaled = feature_scaler.transform(df_clean[feature_cols])
    ds_full = TensorDataset(torch.tensor(X_full_scaled, dtype=torch.float32))
    full_loader = DataLoader(ds_full, batch_size=batch_size, shuffle=False)

    if len(ds_test) == 0:
         raise ValueError("Test dataset is empty after processing.")

    # --- END REPLACEMENT BLOCK ---
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MAF(
        dim=input_dim,
        n_flows=flow_layers,
        hidden_dim=flow_hidden
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start = time.time()
    best_val = float("inf")
    best_state = None

    for ep in range(max_epochs):
        model.train()
        running_nll = 0.0
        count = 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            
            log_prob_x = model(X_batch)
            loss = -log_prob_x.mean() # Minimize NLL
            
            opt.zero_grad()
            loss.backward()
            
            # --- ADDED GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            opt.step()
            
            running_nll += loss.item() * X_batch.size(0)
            count += X_batch.size(0)
        train_nll = running_nll / max(1, count)

# validation
        model.eval()
        val_nll = 0.0
        val_n = 0
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(device)
                log_prob_x = model(X_batch)
                loss = -log_prob_x.mean()
                val_nll += loss.item() * X_batch.size(0)
                val_n += X_batch.size(0)
        val_nll = val_nll / max(1, val_n)

        if getattr(args, "verbose", False) and ((ep + 1) % 20 == 0 or ep == 0):
            print(
                f"[Flow][{lottery}] epoch {ep+1}/{max_epochs} "
                f"train_nll={train_nll:.4f} val_nll={val_nll:.4f}"
            )

        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - start

# evaluation on test (NLL and log-likelihoods)
    model.eval()
    test_nll = 0.0
    test_n = 0
    test_loglik = []
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            log_prob_x = model(X_batch)
            loss = -log_prob_x.mean()
            test_nll += loss.item() * X_batch.size(0)
            test_n += X_batch.size(0)
            test_loglik.extend(log_prob_x.cpu().numpy().tolist())
            
    test_nll = test_nll / max(1, test_n)
    
# get loglik for *all* (clean) data for potential PIT analysis
    all_loglik = []
    with torch.no_grad():
        for (X_batch,) in full_loader:
            X_batch = X_batch.to(device)
            log_prob_x = model(X_batch)
            all_loglik.extend(log_prob_x.cpu().numpy().tolist())

# calculate statistics on test log-likelihoods
    ll_mean = float(np.mean(test_loglik))
    ll_median = float(np.median(test_loglik))
    ll_std = float(np.std(test_loglik))

    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = LOTTERY_SUFFIX_MAP[lottery]

    metrics = {
        "model_name": "flow",
        "lottery": lottery,
        "rmse": None,
        "coverage_95": None,
        "interval_width": { "mean": None, "p50": None, "p90": None },
        "runtime_sec": float(runtime),
        "params": {
            "input_dim": input_dim,
            "flow_layers": flow_layers,
            "flow_hidden": flow_hidden,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
            "nll_train": train_nll,
            "nll_val": val_nll,
            "nll_test": test_nll,
        },
        "y_true": [],
        "y_pred": [],
    }

    perf_path = out_dir / f"flow_perf{suffix}.json"
    perf_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# save likelihoods
    lik_payload = {
        "model_name": "flow",
        "lottery": lottery,
        "loglik_test_set": test_loglik, # Loglik for just the test set
        "loglik_all_clean_data": all_loglik, # Loglik for all data (train+val+test)
        "loglik_mean_test": ll_mean,
        "loglik_median_test": ll_median,
        "loglik_std_test": ll_std,
    }
    lik_path = out_dir / f"flow_likelihoods{suffix}.json"
    lik_path.write_text(json.dumps(lik_payload, indent=2), encoding="utf-8")

# save weights
    flow_dir = MODELS_DIR / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    weights_path = flow_dir / f"{lottery}_flow.pt"
    torch.save(model.state_dict(), weights_path)

    summary_path = write_summary("flow", lottery, metrics, args=args)
    
    print(f"[Flow] Training complete. Test NLL: {test_nll:.4f}, Test Samples: {test_n}")

    return {
        "model": "flow",
        "lottery": lottery,
        "perf_path": str(perf_path),
        "likelihoods_path": str(lik_path),
        "weights_path": str(weights_path),
        "summary_path": str(summary_path),
    }


if __name__ == "__main__":
    import argparse
    from utils_data import LotteryFeatureDataModule

    parser = argparse.ArgumentParser(description="Train Normalizing Flow (MAF) model.")
    
# add arguments matching the zoo launcher
    parser.add_argument("--lottery", type=str, required=True, choices=['powerball', 'megamillions'], help="Lottery name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

# model specific
    parser.add_argument("--flow_layers", type=int, default=5, help="Number of flow layers")
    parser.add_argument("--flow_hidden", type=int, default=128, help="Hidden dim in s/t nets")

# add dummy args for compatibility
    parser.add_argument("--window_len", type=int, default=64, help="Dummy arg")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dummy arg")
    parser.add_argument("--bnn_seq_len", type=int, default=1, help="Dummy arg")
    parser.add_argument("--alpha", type=float, default=0.05, help="Dummy arg")
    parser.add_argument("--conformal_mode", type=str, default='pooled', help="Dummy arg")
    parser.add_argument("--eprocess_mode", type=str, default='standard', help="Dummy arg")
    parser.add_argument("--safe_e_gamma", type=float, default=0.4, help="Dummy arg")
    parser.add_argument("--baseline", type=str, default='copy_last', help="Dummy arg")
    
    args = parser.parse_args()

    print(f"[Flow] Model: flow | Lottery: {args.lottery}")
    
    set_global_seed(args.seed, deterministic=True)

# initialize the DataModule
    dm = LotteryFeatureDataModule(
        lottery=args.lottery,
        batch_size=args.batch_size
    )
    
# we must add the file suffixes to the args for write_summary
    args.suffix = LOTTERY_SUFFIX_MAP[args.lottery]
    
    try:
        result = train_and_export(dm, args)
        print(f"[Flow] Training complete. Performance report: {result.get('perf_path')}")
    except Exception as e:
        print(f"[Flow] Worker crashed:\n{e}")
        import traceback
        traceback.print_exc()

    print("[Flow] Done.")
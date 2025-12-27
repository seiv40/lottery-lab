#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Sets for Module 5: Deep Learning Model Zoo

"""

import json
import re
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler  # <-- ADDED IMPORT

from utils_seed import set_global_seed
from utils_data import DATA_PATH, OUTPUT_DIR, MODELS_DIR
from utils_metrics import write_summary

# map lottery name to the file suffix
LOTTERY_SUFFIX_MAP = {
    'powerball': '_pb',
    'megamillions': '_mm'
}

# column detection

CAND_PATTERNS = [
    r"^position_[1-5]_value$",  # actual white ball columns in features_*.parquet
    r"^n[1-5]$",        # n1, n2, ..., n5
    r"^num[1-5]$",       # num1, ...
    r"^ball[1-5]$",      # ball1, ...
    r"^b[1-5]$",         # b1, ...
    r"^regular_[1-5]$",   # regular_1, ...
    r"^main_[1-5]$",      # main_1, ...
    r".*ball.*[1-5].*",    # anything containing "ball" + digit 1-5
    r".*num.*[1-5].*",     # anything containing "num" + digit 1-5
]


def detect_ball_columns(df: pd.DataFrame) -> List[str]:
    cols = df.columns.tolist()
    matched = set()
    for c in cols:
        for pat in CAND_PATTERNS:
            if re.match(pat, c, flags=re.IGNORECASE):
                matched.add(c)
                break

    if len(matched) >= 5:
        def sort_key(name: str):
            m = re.findall(r"[1-5]", name)
            return int(m[0]) if m else 999
        selected = sorted(matched, key=sort_key)[:5]
        print(f"[DeepSets] Detected ball columns by pattern: {selected}")
        return selected

    numeric_cols = []
    for c in cols:
        if c.lower() in ("date", "draw_date", "lottery", "jackpot", "multiplier"):
            continue
        if np.issubdtype(df[c].dtype, np.number):
            numeric_cols.append(c)

    if len(numeric_cols) >= 5:
        selected = numeric_cols[:5]
        print(f"[DeepSets] Fallback to first 5 numeric columns: {selected}")
        return selected

    raise ValueError(
        f"Could not detect 5 ball columns; numeric candidates = {numeric_cols}"
    )


# dataset

class BallPairsDataset(Dataset):
    """
    Dataset of (current_set, scalar_target) pairs.
    Takes TENSORS as input.
    """
    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor):
        super().__init__()
        if len(X_tensor) != len(y_tensor):
            raise ValueError("Mismatched tensor lengths")
        self.X_sets = X_tensor
        self.y_scalars = y_tensor

    def __len__(self) -> int:
        return len(self.X_sets)

    def __getitem__(self, idx: int):
        x_cur = self.X_sets[idx]        # This is a Tensor
        y_scalar = self.y_scalars[idx]  # This is a Tensor

        # random permutation of input set to encourage invariance
# we must use torch.randperm on the same device as the tensor
        perm = torch.randperm(len(x_cur), device=x_cur.device)
        x_perm = x_cur[perm] # This returns a permuted Tensor

        return (
            x_perm,   # <-- FIX: Return the tensor directly
            y_scalar  # y_scalar is already a tensor
        )

# model

class Phi(nn.Module):
    """Per-element embedding network φ: R -> R^embed"""
    def __init__(self, embed_dim: int = 64,
                 hidden1: int = 128,
                 hidden2: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.fc3 = nn.Linear(hidden2, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 5] -> [B, 5, 1]
        x = x.unsqueeze(-1)
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)
        return h  # [B, 5, embed_dim]


class DeepSetsHetero(nn.Module):
    """Permutation-invariant Deep Sets model with heteroscedastic Gaussian head."""
    def __init__(self,
                 embed_dim: int = 64,
                 phi_hidden1: int = 128,
                 phi_hidden2: int = 128,
                 rho_hidden1: int = 64,
                 rho_hidden2: int = 32,
                 dropout: float = 0.1):
        super().__init__()

        self.phi = Phi(embed_dim=embed_dim,
                       hidden1=phi_hidden1,
                       hidden2=phi_hidden2,
                       dropout=dropout)

        self.rho_fc1 = nn.Linear(embed_dim, rho_hidden1)
        self.rho_ln1 = nn.LayerNorm(rho_hidden1)
        self.rho_fc2 = nn.Linear(rho_hidden1, rho_hidden2)
        self.rho_fc_out = nn.Linear(rho_hidden2, 2)  # [mu, raw_scale]
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_set: torch.Tensor):
        z = self.phi(x_set)      # [B, 5, embed_dim]
        z_sum = z.sum(dim=1)     # [B, embed_dim]

        h = self.rho_fc1(z_sum)
        h = self.rho_ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.rho_fc2(h)
        h = F.relu(h)
        h = self.dropout(h)

        out = self.rho_fc_out(h)   # [B, 2]
        mu = out[..., 0]
        raw_scale = out[..., 1]

        sigma = F.softplus(raw_scale) + 1e-6
        sigma = torch.clamp(sigma, min=1e-4, max=10.0)

        return mu, sigma


# train & export

def train_and_export(dm, args) -> Dict:
    """
    MODIFIED to perform manual sequential data split AND target/feature creation.
    """
    seed = getattr(args, "seed", 42)
    set_global_seed(seed, deterministic=True)

    lottery = args.lottery
    batch_size = int(getattr(args, "batch_size", 64))
    lr = float(getattr(args, "learning_rate", 1e-3))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    max_epochs = int(getattr(args, "epochs", 200))

    embed_dim = int(getattr(args, "ds_embed", 64))
    phi_hidden1 = int(getattr(args, "ds_phi_hidden1", 128))
    phi_hidden2 = int(getattr(args, "ds_phi_hidden2", 128))
    rho_hidden1 = int(getattr(args, "ds_rho_hidden1", 64))
    rho_hidden2 = int(getattr(args, "ds_rho_hidden2", 32))
    dropout = float(getattr(args, "ds_dropout", 0.1))


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

# robust ball-column detection (patterns + fallback)
    ball_cols = detect_ball_columns(df)
    print(f"Found {len(ball_cols)} ball columns: {ball_cols}")

    # --- Manually Replicate Target Creation ---
    
    # 1. Create the raw target 'next_mean'
    raw_target_col = 'next_mean'
    df[raw_target_col] = df['mean'].shift(-1)
    
    # 2. Define the standardized target column name
    target_col = 'next_mean_std' # This is the canonical target
    
    # 3. Drop the last row where the target is NaN
    df_clean = df.dropna(subset=[raw_target_col])
    n = len(df_clean)
    if n == 0:
        raise ValueError(f"No valid data remaining after dropna on '{raw_target_col}'.")

    # 4. Define split indices
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    
    # 5. Perform sequential split ON THE CLEANED DATA
    train_df = df_clean.iloc[:n_train].copy()
    val_df = df_clean.iloc[n_train:n_val].copy()
    test_df = df_clean.iloc[n_val:].copy()
    
    # 6. --- ALIGN TEST SETS ---
# align with windowed models (Transformer)
    
    window_len = int(getattr(args, "window_len", 64)) # Get window_len from args
    n_chop = window_len - 1
    
# truncate the validation and test sets to match
    val_df = val_df.iloc[n_chop:]
    test_df = test_df.iloc[n_chop:]
    
    print(f"Aligned splits for window_len={window_len} -> Val: {len(val_df)}, Test: {len(test_df)}")

    # 7. Standardize TARGETS (was step 6)
    target_scaler = StandardScaler()
    train_df[target_col] = target_scaler.fit_transform(train_df[[raw_target_col]])
    val_df[target_col] = target_scaler.transform(val_df[[raw_target_col]])
    test_df[target_col] = target_scaler.transform(test_df[[raw_target_col]])
    print(f"Created canonical target: '{target_col}'")

    # 8. --- NEW: Standardize FEATURES (the ball columns) ---
    feature_scaler = StandardScaler()
    train_df[ball_cols] = feature_scaler.fit_transform(train_df[ball_cols])
    val_df[ball_cols] = feature_scaler.transform(val_df[ball_cols])
    test_df[ball_cols] = feature_scaler.transform(test_df[ball_cols])
    print(f"Created standardized features for ball columns.")

    # 9. Create datasets
    ds_train = BallPairsDataset(
        torch.tensor(train_df[ball_cols].values, dtype=torch.float32),
        torch.tensor(train_df[target_col].values, dtype=torch.float32)
    )
    ds_val = BallPairsDataset(
        torch.tensor(val_df[ball_cols].values, dtype=torch.float32),
        torch.tensor(val_df[target_col].values, dtype=torch.float32)
    )
    ds_test = BallPairsDataset(
        torch.tensor(test_df[ball_cols].values, dtype=torch.float32),
        torch.tensor(test_df[target_col].values, dtype=torch.float32)
    )

    if len(ds_test) == 0:
         raise ValueError("Test dataset is empty after processing.")


    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSetsHetero(
        embed_dim=embed_dim,
        phi_hidden1=phi_hidden1,
        phi_hidden2=phi_hidden2,
        rho_hidden1=rho_hidden1,
        rho_hidden2=rho_hidden2,
        dropout=dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training loop
    start = time.time()
    best_val = float("inf")
    best_state = None
    log2pi = np.log(2.0 * np.pi)

    for ep in range(max_epochs):
        model.train()
        running = 0.0
        count = 0
        for x_set, y_scalar in train_loader:
            x_set = x_set.to(device)
            y_scalar = y_scalar.to(device)  # [B]

            mu, sigma = model(x_set)      # [B], [B]
            sigma2 = sigma ** 2
            log_sigma2 = 2.0 * torch.log(sigma)
            nll = 0.5 * (((y_scalar - mu) ** 2) / sigma2 + log_sigma2 + log2pi)
            loss = nll.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item()) * x_set.size(0)
            count += x_set.size(0)
        train_loss = running / max(1, count)

# validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_n = 0
            for x_set, y_scalar in val_loader:
                x_set = x_set.to(device)
                y_scalar = y_scalar.to(device)
                mu, sigma = model(x_set)
                sigma2 = sigma ** 2
                log_sigma2 = 2.0 * torch.log(sigma)
                nll = 0.5 * (((y_scalar - mu) ** 2) / sigma2 + log_sigma2 + log2pi)
                loss = nll.mean()
                val_loss += float(loss.item()) * x_set.size(0)
                val_n += x_set.size(0)
        val_loss /= max(1, val_n)

        if getattr(args, "verbose", False) and ((ep + 1) % 50 == 0 or ep == 0):
            print(
                f"[DeepSets][{lottery}] epoch {ep+1}/{max_epochs} "
                f"train_nll={train_loss:.4f} val_nll={val_loss:.4f}"
            )

        # track best
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - start

# evaluation on test
    model.eval()
    y_true_list = []
    mu_list = []
    sigma_list = []

# get all test data
    X_test_all, y_true_all = ds_test[:]
    y_true = y_true_all.cpu().numpy()

    with torch.no_grad():
        for x_set, y_scalar in test_loader:
            x_set = x_set.to(device)
            y_scalar = y_scalar.to(device)
            mu, sigma = model(x_set)

            y_true_list.extend(y_scalar.cpu().numpy().tolist())
            mu_list.extend(mu.cpu().numpy().tolist())
            sigma_list.extend(sigma.cpu().numpy().tolist())

    y_true_np = np.array(y_true_list, dtype=np.float64)
    y_pred = np.array(mu_list, dtype=np.float64)
    sigma = np.array(sigma_list, dtype=np.float64)
    
# this check is needed because test_loader might be empty if test_df < 1 sample
    if len(y_true_np) > 0:
# re-align y_true just in case loader dropped last batch
        y_true = y_true_np
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        half_width_per = 1.96 * sigma
        y_lo = y_pred - half_width_per
        y_hi = y_pred + half_width_per
        coverage = float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))
        width = y_hi - y_lo
        iw_mean = float(np.mean(width))
        iw_p50 = float(np.quantile(width, 0.50))
        iw_p90 = float(np.quantile(width, 0.90))
    else:
        rmse = coverage = iw_mean = iw_p50 = iw_p90 = None
        y_lo = y_hi = np.array([])


# invariance check
    invariance_max_diff = 0.0
    with torch.no_grad():
# use the first batch from test_loader
        try:
            x_set, y_scalar = next(iter(test_loader))
            x_set = x_set.to(device)
            B, K = x_set.shape
            perm = torch.stack(
                [torch.randperm(K, device=device) for _ in range(B)], dim=0
            )
            x_perm = torch.gather(x_set, 1, perm)

            mu_orig, _ = model(x_set)
            mu_perm, _ = model(x_perm)
            diff = torch.abs(mu_orig - mu_perm).max().item()
            invariance_max_diff = max(invariance_max_diff, float(diff))
        except StopIteration:
            print("Test loader was empty, skipping invariance check.")


    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = LOTTERY_SUFFIX_MAP[lottery]

    metrics = {
        "model_name": "deepsets",
        "lottery": lottery,
        "rmse": rmse,
        "coverage_95": coverage,
        "interval_width": {
            "mean": iw_mean,
            "p50": iw_p50,
            "p90": iw_p90,
        },
        "runtime_sec": float(runtime),
        "params": {
            "ball_columns": ball_cols,
            "embed_dim": embed_dim,
            "phi_hidden1": phi_hidden1,
            "phi_hidden2": phi_hidden2,
            "rho_hidden1": rho_hidden1,
            "rho_hidden2": rho_hidden2,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
            "dropout": dropout,
            "invariance_max_diff": invariance_max_diff,
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

# perf + predictive exports
    perf_path = out_dir / f"deepset_perf{suffix}.json"
    perf_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pred_payload = {
        "model_name": "deepsets",
        "lottery": lottery,
        "y_mean": y_pred.tolist(),
        "y_lo": y_lo.tolist(),
        "y_hi": y_hi.tolist(),
    }
    pred_path = out_dir / f"deepset_predictive{suffix}.json"
    pred_path.write_text(
        json.dumps(pred_payload, indent=2), encoding="utf-8"
    )

# save weights
    ds_dir = MODELS_DIR / "deepsets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ds_dir / f"{lottery}_deepsets.pt"
    torch.save(model.state_dict(), weights_path)

    summary_path = write_summary("deepsets", lottery, metrics, args=args)
    
    print(f"[DeepSets] Training complete. Test RMSE: {rmse:.4f}, Coverage: {coverage:.4f}, Test Samples: {len(y_true)}")

    return {
        "model": "deepsets",
        "lottery": lottery,
        "perf_path": str(perf_path),
        "predictive_path": str(pred_path),
        "weights_path": str(weights_path),
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    import argparse
    from utils_data import LotteryFeatureDataModule

    parser = argparse.ArgumentParser(description="Train DeepSets v1 model.")
    
# add arguments matching the zoo launcher
    parser.add_argument("--lottery", type=str, required=True, choices=['powerball', 'megamillions'], help="Lottery name")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

# add model-specific args (optional, with defaults)
    parser.add_argument("--ds_embed", type=int, default=64)
    parser.add_argument("--ds_phi_hidden1", type=int, default=128)
    parser.add_argument("--ds_phi_hidden2", type=int, default=128)
    parser.add_argument("--ds_rho_hidden1", type=int, default=64)
    parser.add_argument("--ds_rho_hidden2", type=int, default=32)
    parser.add_argument("--ds_dropout", type=float, default=0.1)

# add dummy args for compatibility with other models
    parser.add_argument("--window_len", type=int, default=64, help="Dummy arg")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dummy arg")
    parser.add_argument("--bnn_seq_len", type=int, default=1, help="Dummy arg")
    parser.add_argument("--alpha", type=float, default=0.05, help="Dummy arg")
    parser.add_argument("--conformal_mode", type=str, default='pooled', help="Dummy arg")
    parser.add_argument("--eprocess_mode", type=str, default='standard', help="Dummy arg")
    parser.add_argument("--safe_e_gamma", type=float, default=0.4, help="Dummy arg")
    parser.add_argument("--baseline", type=str, default='copy_last', help="Dummy arg")
    
    args = parser.parse_args()

    print(f"[DeepSets] Model: deepsets | Lottery: {args.lottery}")
    print(f"[DeepSets] Optimizer: adam | Weight decay: {args.weight_decay}")
    
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
        print(f"[DeepSets] Training complete. Performance report: {result.get('perf_path')}")
    except Exception as e:
        print(f"[DeepSets] Worker crashed:\n{e}")
        import traceback
        traceback.print_exc()

    print("[DeepSets] Done.")
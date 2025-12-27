#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian Neural Network (BNN) for Module 5: Deep Learning Model Zoo

"""

import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler  # <-- REQUIRED

from utils_seed import set_global_seed
from utils_data import DATA_PATH, OUTPUT_DIR, MODELS_DIR 
from utils_metrics import write_summary

# map lottery name to the file suffix
LOTTERY_SUFFIX_MAP = {
    'powerball': '_pb',
    'megamillions': '_mm'
}

# dataset

class SupervisedLotteryDataset(Dataset):
    """
    Standard supervised dataset for (features, target) pairs.
    Takes TENSORS as input, not dataframes.
    """
    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor):
        
        if len(X_tensor) != len(y_tensor):
            raise ValueError("Mismatched tensor lengths")
            
        self.features = X_tensor
        self.targets = y_tensor
        
    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class MLP_BNN(nn.Module):
    """
    Simple MLP BNN with MC Dropout for epistemic uncertainty.
    Predicts a scalar value.
    """
    def __init__(self, input_dim: int, hidden: int = 256, dropout_p: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_p = dropout_p
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden // 2, 1) # Predict a single scalar value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) # [B, 1] -> [B]

    def predict_dist(self, x: torch.Tensor, mc_samples: int) -> torch.Tensor:
        """
        Perform MC Dropout to get a distribution of predictions.
        x: [B_test, Fin]
        Returns: [mc_samples, B_test]
        """
# enable dropout (train mode)
        self.train()
        
        preds = []
        for _ in range(mc_samples):
            preds.append(self.forward(x))
            
# stack predictions
# list of [B_test] -> [mc_samples, B_test]
        return torch.stack(preds, dim=0)


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
    max_epochs = int(getattr(args, "epochs", 100))
    
# bNN specific
    mc_samples = int(getattr(args, "bnn_mc_samples", 20))
    hidden_dim = int(getattr(args, "bnn_hidden", 256))
    dropout_p = float(getattr(args, "bnn_dropout", 0.1))



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


    # --- Manually Replicate Target Creation from utils_data.py ---
    
    # 1. Create the raw target 'next_mean'
    raw_target_col = 'next_mean'
    df[raw_target_col] = df['mean'].shift(-1)
    
    # 2. Define the standardized target column name
    target_col = 'next_mean_std' # This is the canonical target
    
    # 3. Define feature columns (all numeric cols except raw mean/target)
    exclude_cols = ['drawing_index', 'date', 'game', 'special_ball', 
                    'mean', raw_target_col, target_col]
    feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'float32', 'int64', 'int32'] and col not in exclude_cols]
    input_dim = len(feature_cols)
    print(f"Using {input_dim} feature columns.")
    
    # 4. Fill NaNs in features (e.g., from rolling windows)
    df[feature_cols] = df[feature_cols].fillna(0)
    print("Filled NaNs in feature columns with 0.")

    # 5. Drop the last row where the target is NaN
    df_clean = df.dropna(subset=[raw_target_col])
    n = len(df_clean)
    if n == 0:
        raise ValueError(f"No valid data remaining after dropna on '{raw_target_col}'.")

    # 6. Define split indices
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    
    # 7. Perform sequential split ON THE CLEANED DATA
    train_df = df_clean.iloc[:n_train].copy()
    val_df = df_clean.iloc[n_train:n_val].copy()
    test_df = df_clean.iloc[n_val:].copy()

    # 8. --- ALIGN TEST SETS ---
# we must align the test set with the windowed models (Transformer)
# the Transformer (window_len=64) can only start predicting at the 64th sample.
# its test set has (n_test - window_len + 1) samples.
# we must chop the first (window_len - 1) samples off the *other* models.
    
    window_len = int(getattr(args, "window_len", 64)) # Get window_len from args
    n_chop = window_len - 1
    
# truncate the validation and test sets to match
    val_df = val_df.iloc[n_chop:]
    test_df = test_df.iloc[n_chop:]
    
    print(f"Aligned splits for window_len={window_len} -> Val: {len(val_df)}, Test: {len(test_df)}")

    # 9. Standardize TARGETS (was step 8)
    target_scaler = StandardScaler()
    train_df[target_col] = target_scaler.fit_transform(train_df[[raw_target_col]])
    val_df[target_col] = target_scaler.transform(val_df[[raw_target_col]])
    test_df[target_col] = target_scaler.transform(test_df[[raw_target_col]])
    print(f"Created canonical target: '{target_col}'")

    # 10. --- NEW: Standardize FEATURES ---
    feature_scaler = StandardScaler()
    train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = feature_scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])
    print(f"Created standardized features.")

    # 11. Create datasets
    ds_train = SupervisedLotteryDataset(
        torch.tensor(train_df[feature_cols].values, dtype=torch.float32),
        torch.tensor(train_df[target_col].values, dtype=torch.float32)
    )
    ds_val = SupervisedLotteryDataset(
        torch.tensor(val_df[feature_cols].values, dtype=torch.float32),
        torch.tensor(val_df[target_col].values, dtype=torch.float32)
    )
    ds_test = SupervisedLotteryDataset(
        torch.tensor(test_df[feature_cols].values, dtype=torch.float32),
        torch.tensor(test_df[target_col].values, dtype=torch.float32)
    )

    if len(ds_test) == 0:
         raise ValueError("Test dataset is empty after processing.")

    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP_BNN(
        input_dim=input_dim,
        hidden=hidden_dim,
        dropout_p=dropout_p
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    start = time.time()
    best_val = float("inf")
    best_state = None

    for ep in range(max_epochs):
        model.train() # Keep train mode for dropout
        running = 0.0
        count = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running += loss.item() * X_batch.size(0)
            count += X_batch.size(0)
        train_loss = np.sqrt(running / max(1, count)) # Train RMSE

# validation
        model.eval() # Eval mode for consistent validation
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_n += X_batch.size(0)
        val_loss = np.sqrt(val_loss / max(1, val_n)) # Val RMSE

        if getattr(args, "verbose", False) and ((ep + 1) % 20 == 0 or ep == 0):
            print(
                f"[BNN][{lottery}] epoch {ep+1}/{max_epochs} "
                f"train_rmse={train_loss:.4f} val_rmse={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - start

# evaluation on test (with MC Dropout)
    
    # 1. Get all test data
    X_test_all, y_true_all = ds_test[:]
    X_test_all = X_test_all.to(device)
    
    # 2. Get MC predictions
    # [mc_samples, N_test]
    with torch.no_grad():
        pred_dist = model.predict_dist(X_test_all, mc_samples=mc_samples).cpu().numpy()

    # 3. Calculate metrics
    y_true = y_true_all.cpu().numpy()
    
# aleatoric uncertainty (model noise) is assumed 0 for this simple BNN
# epistemic uncertainty (MC variance)
    y_pred = np.mean(pred_dist, axis=0)
    
# quantiles for intervals
    y_lo = np.quantile(pred_dist, 0.025, axis=0)
    y_hi = np.quantile(pred_dist, 0.975, axis=0)
    
    if len(y_true) > 0:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        coverage = float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))
        width = y_hi - y_lo
        iw_mean = float(np.mean(width))
        iw_p50 = float(np.quantile(width, 0.50))
        iw_p90 = float(np.quantile(width, 0.90))
    else:
        rmse = coverage = iw_mean = iw_p50 = iw_p90 = None

    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = LOTTERY_SUFFIX_MAP[lottery]

    metrics = {
        "model_name": "bnn",
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
            "arch": "mlp_mcdropout",
            "seq_len": 1,
            "input_dim": input_dim,
            "mc_samples": mc_samples,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
            "hidden": hidden_dim,
            "dropout": dropout_p,
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

    perf_path = out_dir / f"bnn_perf{suffix}.json"
    perf_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_payload = {
        "model_name": "bnn",
        "lottery": lottery,
        "y_mean": y_pred.tolist(),
        "y_lo": y_lo.tolist(),
        "y_hi": y_hi.tolist(),
# optionally save full distribution
        # "y_dist": pred_dist.tolist(),
    }
    pred_path = out_dir / f"bnn_predictive{suffix}.json"
    pred_path.write_text(json.dumps(pred_payload, indent=2), encoding="utf-8")
    
    summary_path = write_summary("bnn", lottery, metrics, args=args)
    
    print(f"[BNN] Training complete. Test RMSE: {rmse:.4f}, Coverage: {coverage:.4f}, Test Samples: {len(y_true)}")

    return {
        "model": "bnn",
        "lottery": lottery,
        "perf_path": str(perf_path),
        "predictive_path": str(pred_path),
        "summary_path": str(summary_path),
    }


if __name__ == "__main__":
    import argparse
    from utils_data import LotteryFeatureDataModule

    parser = argparse.ArgumentParser(description="Train BNN (MC Dropout) model.")
    
# add arguments matching the zoo launcher
    parser.add_argument("--lottery", type=str, required=True, choices=['powerball', 'megamillions'], help="Lottery name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

# model specific
    parser.add_argument("--bnn_hidden", type=int, default=256, help="Hidden dim")
    parser.add_argument("--bnn_dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bnn_mc_samples", type=int, default=20, help="MC samples for inference")
    
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

    print(f"[BNN] Model: bnn | Lottery: {args.lottery}")
    
    set_global_seed(args.seed, deterministic=True)

# initialize the DataModule (it will be used for path info, but not splits)
    dm = LotteryFeatureDataModule(
        lottery=args.lottery,
        batch_size=args.batch_size
    )
    
# we must add the file suffixes to the args for write_summary
    args.suffix = LOTTERY_SUFFIX_MAP[args.lottery]
    
    try:
        result = train_and_export(dm, args)
        print(f"[BNN] Training complete. Performance report: {result.get('perf_path')}")
    except Exception as e:
        print(f"[BNN] Worker crashed:\n{e}")
        import traceback
        traceback.print_exc()

    print("[BNN] Done.")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer Model for Module 5: Deep Learning Model Zoo

"""

import json
import time
from typing import Dict, Tuple, List # <-- ADDED List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler  # <-- ADDED IMPORT

from utils_seed import set_global_seed
from utils_data import DATA_PATH, OUTPUT_DIR, MODELS_DIR # We will use DATA_PATH
from utils_metrics import write_summary

# map lottery name to the file suffix
LOTTERY_SUFFIX_MAP = {
    'powerball': '_pb',
    'megamillions': '_mm'
}


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Transformer Encoder model for sequence regression.
    Input: [B, L, Fin] -> Linear -> [B, L, D] -> PosEncode -> [L, B, D]
    Encoder: L, B, D -> L, B, D
    Output: L, B, D -> [B, L, D] -> mean pool -> [B, D] -> Linear -> [B, 1] (scalar)
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 d_hid: int = 256,
                 nlayers: int = 2,
                 dropout: float = 0.1,
                 window_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.window_len = window_len
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=False, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.output_head = nn.Linear(d_model, 1) # predict scalar target
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len]
        
        Returns:
            output: Tensor, shape [batch_size, 1]
        """
        # 1. Project input
        # [B, L, Fin] -> [B, L, D]
        src_proj = self.input_projection(src) * np.sqrt(self.d_model)
        
        # 2. Add positional encoding
        # [B, L, D] -> [L, B, D]
        src_pos = src_proj.permute(1, 0, 2)
        src_pos = self.pos_encoder(src_pos)
        
        # 3. Transformer Encoder
        # src_key_padding_mask: [B, L] (TransformerEncoder wants this)
        if src_key_padding_mask is not None:
# ensure mask is boolean
            src_key_padding_mask = src_key_padding_mask.bool()

        # [L, B, D] -> [L, B, D]
        memory = self.transformer_encoder(src_pos, src_key_padding_mask=src_key_padding_mask)
        
        # 4. Global Average Pooling over sequence length
        # [L, B, D] -> [B, L, D]
        memory_perm = memory.permute(1, 0, 2)
        
# handle padding in pooling:
        if src_key_padding_mask is not None:
# invert mask: True for valid tokens, False for padding
            mask = ~src_key_padding_mask.unsqueeze(-1).float() # [B, L, 1]
# sum valid tokens
            pooled = (memory_perm * mask).sum(dim=1) # [B, D]
# count valid tokens
            valid_counts = mask.sum(dim=1) # [B, 1]
            valid_counts = valid_counts.clamp(min=1.0) # Avoid division by zero
            pooled = pooled / valid_counts # Mean pool
        else:
# no mask, simple mean pool
            pooled = memory_perm.mean(dim=1) # [B, D]
            
        # 5. Output head
        # [B, D] -> [B, 1]
        output = self.output_head(pooled)
        return output.squeeze(-1) # [B]

# dataset

class SequentialLotteryDataset(Dataset):
    """
    Creates sequences (windows) from the feature DataFrame.
    X = window_len features
    y = scalar target (e.g., next_mean_std)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, window_len: int):
        self.window_len = window_len
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame.")
            
# this dataset is simpler: it just takes the pre-split, pre-processed df
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
        
        self.n_samples = len(self.features)
        
        if self.n_samples < window_len:
# this can happen for val/test sets, it's not a fatal error
             print(f"Warning: Not enough data ({self.n_samples}) for window_len={window_len}. Dataset will be empty.")
             self.num_sequences = 0
        else:
            self.num_sequences = self.n_samples - self.window_len + 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a window of features and the target *at the end* of that window.
        X = features[idx : idx + window_len]
        y = target[idx + window_len - 1]
        """
        end_idx = idx + self.window_len
        
        X_window = self.features[idx:end_idx, :] # [L, Fin]
        y_target = self.targets[end_idx - 1]     # scalar
        
        return X_window, y_target


# train & export

def train_and_export(dm, args) -> Dict:
    """
    MODIFIED to perform manual sequential data split AND target creation.
    """
    seed = getattr(args, "seed", 42)
    set_global_seed(seed, deterministic=True)

    lottery = args.lottery
    batch_size = int(getattr(args, "batch_size", 64))
    lr = float(getattr(args, "learning_rate", 1e-3))
    weight_decay = float(getattr(args, "weight_decay", 1e-4))
    max_epochs = int(getattr(args, "epochs", 60))
    window_len = int(getattr(args, "window_len", 64))


# load features parquet for this lottery
    path = DATA_PATH / f"features_{lottery}.parquet"
    if not path.exists():
        print(f"FATAL: Cannot find features file: {path}")
        return {"ok": False, "error": f"File not found: {path}"}
    
# we must load the raw features, as dm.load_features() is not available
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
# the sequential dataset logic handles this implicitly by starting
    # at the first valid target, but we fillna(0) for robustness.
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
    
    print(f"Performed sequential split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 8. Standardize the target *after* splitting
    scaler = StandardScaler()
    
# fit on training data
    train_df[target_col] = scaler.fit_transform(train_df[[raw_target_col]])
    
# transform validation and test data
    val_df[target_col] = scaler.transform(val_df[[raw_target_col]])
    test_df[target_col] = scaler.transform(test_df[[raw_target_col]])
    
    print(f"Created canonical target: '{target_col}'")

    # 9. Create datasets
    ds_train = SequentialLotteryDataset(train_df, feature_cols, target_col, window_len)
    ds_val = SequentialLotteryDataset(val_df, feature_cols, target_col, window_len)
    ds_test = SequentialLotteryDataset(test_df, feature_cols, target_col, window_len)

    print(f"Dataset sequences -> Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
    if len(ds_test) == 0:
         raise ValueError("Test dataset is empty after processing (not enough data for window_len).")

    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerModel(
        input_dim=input_dim,
        window_len=window_len
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    start = time.time()
    best_val = float("inf")
    best_state = None

    for ep in range(max_epochs):
        model.train()
        running = 0.0
        count = 0
        for X_seq, y_target in train_loader:
            X_seq, y_target = X_seq.to(device), y_target.to(device)
            
            y_pred = model(X_seq)
            loss = loss_fn(y_pred, y_target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running += loss.item() * X_seq.size(0)
            count += X_seq.size(0)
        train_loss = np.sqrt(running / max(1, count)) # Train RMSE

# validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for X_seq, y_target in val_loader:
                X_seq, y_target = X_seq.to(device), y_target.to(device)
                y_pred = model(X_seq)
                loss = loss_fn(y_pred, y_target)
                val_loss += loss.item() * X_seq.size(0)
                val_n += X_seq.size(0)
        val_loss = np.sqrt(val_loss / max(1, val_n)) # Val RMSE

        if getattr(args, "verbose", False) and ((ep + 1) % 20 == 0 or ep == 0):
            print(
                f"[Transformer][{lottery}] epoch {ep+1}/{max_epochs} "
                f"train_rmse={train_loss:.4f} val_rmse={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - start

# evaluation on test
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for X_seq, y_target in test_loader:
            X_seq, y_target = X_seq.to(device), y_target.to(device)
            y_pred = model(X_seq)
            y_true_list.extend(y_target.cpu().numpy().tolist())
            y_pred_list.extend(y_pred.cpu().numpy().tolist())

    y_true = np.array(y_true_list, dtype=np.float64)
    y_pred = np.array(y_pred_list, dtype=np.float64)

    if len(y_true) > 0:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    else:
        rmse = None

    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = LOTTERY_SUFFIX_MAP[lottery]

    metrics = {
        "model_name": "transformer",
        "lottery": lottery,
        "rmse": rmse,
        "coverage_95": None, # This model is deterministic
        "interval_width": {
            "mean": None,
            "p50": None,
            "p90": None,
        },
        "runtime_sec": float(runtime),
        "params": {
            "input_dim": input_dim,
            "d_model": model.d_model,
            "nhead": model.transformer_encoder.layers[0].self_attn.num_heads,
            "nlayers": model.transformer_encoder.num_layers,
            "window_len": window_len,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

    perf_path = out_dir / f"transformer_perf{suffix}.json"
    perf_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# save weights
    tx_dir = MODELS_DIR / "transformer"
    tx_dir.mkdir(parents=True, exist_ok=True)
    weights_path = tx_dir / f"{lottery}_transformer.pt"
    torch.save(model.state_dict(), weights_path)

    summary_path = write_summary("transformer", lottery, metrics, args=args)
    
    print(f"[Transformer] Training complete. Test RMSE: {rmse:.4f}, Test Samples: {len(y_true)}")

    return {
        "model": "transformer",
        "perf_path": str(perf_path),
        "weights_path": str(weights_path),
        "summary_path": str(summary_path),
    }


if __name__ == "__main__":
    import argparse
    from utils_data import LotteryFeatureDataModule

    parser = argparse.ArgumentParser(description="Train standard Transformer Encoder model.")
    
# add arguments matching the zoo launcher
    parser.add_argument("--lottery", type=str, required=True, choices=['powerball', 'megamillions'], help="Lottery name")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--window_len", type=int, default=64, help="Sequence window length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
# add dummy args for compatibility with other models
    parser.add_argument("--latent_dim", type=int, default=2, help="Dummy arg")
    parser.add_argument("--bnn_seq_len", type=int, default=1, help="Dummy arg")
    parser.add_argument("--alpha", type=float, default=0.05, help="Dummy arg")
    parser.add_argument("--conformal_mode", type=str, default='pooled', help="Dummy arg")
    parser.add_argument("--eprocess_mode", type=str, default='standard', help="Dummy arg")
    parser.add_argument("--safe_e_gamma", type=float, default=0.4, help="Dummy arg")
    parser.add_argument("--baseline", type=str, default='copy_last', help="Dummy arg")
    
    args = parser.parse_args()

    print(f"[Transformer] Model: transformer | Lottery: {args.lottery}")
    print(f"[Transformer] Optimizer: adamw | Weight decay: {args.weight_decay}")
    
    set_global_seed(args.seed, deterministic=True)

# initialize the DataModule
    dm = LotteryFeatureDataModule(
        lottery=args.lottery,
        batch_size=args.batch_size,
        window_len=args.window_len
    )
    
# we must add the file suffixes to the args for write_summary
    args.suffix = LOTTERY_SUFFIX_MAP[args.lottery]
    
    try:
        result = train_and_export(dm, args)
        print(f"[Transformer] Training complete. Performance report: {result.get('perf_path')}")
    except Exception as e:
        print(f"[Transformer] Worker crashed:\n{e}")
        import traceback
        traceback.print_exc()

    print("[Transformer] Done.")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer + Bayesian Head for Module 5: Deep Learning Model Zoo

This combines sequence modeling (Transformer) with uncertainty quantification
(Bayesian head). The key question: can the model produce CALIBRATED uncertainty?

We test 4 variants:
1. baseline: Gaussian, fixed variance (simplest)
2. hetero: Gaussian, per-sample variance (adapts to difficulty)
3. heavy: Student-t, fixed variance (robust to outliers)
4. hetero_heavy: Student-t, per-sample variance (most flexible)

"""

import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import StudentT, Normal
from sklearn.preprocessing import StandardScaler

from utils_seed import set_global_seed
from utils_data import DATA_PATH, OUTPUT_DIR, MODELS_DIR
from utils_metrics import write_summary

# map lottery name to the file suffix
LOTTERY_SUFFIX_MAP = {
    "powerball": "_pb",
    "megamillions": "_mm",
}

# positionalEncoding and TransformerEncoder


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (no weird indexing)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # position: [max_len, 1]
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # div_term: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # shape [max_len, 1, d_model] so it can broadcast over batch dimension
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Base Transformer Encoder stack.

    Input:  [B, L, Fin]
      -> Linear -> [B, L, D]
      -> PosEncode (on [L, B, D])
      -> Encoder -> [L, B, D]
      -> return [B, L, D]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        window_len: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_len = window_len
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=False,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        src: [B, L, Fin]
        src_key_padding_mask: [B, L] (True for padding)
        returns: [B, L, D]
        """
        # project to d_model
        src_proj = self.input_projection(src) * np.sqrt(self.d_model)  # [B, L, D]

        # [B, L, D] -> [L, B, D] for nn.Transformer
        src_pos = src_proj.permute(1, 0, 2)
        src_pos = self.pos_encoder(src_pos)

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.bool()

        memory = self.transformer_encoder(
            src_pos, src_key_padding_mask=src_key_padding_mask
        )  # [L, B, D]

        # back to [B, L, D]
        memory_perm = memory.permute(1, 0, 2)
        return memory_perm


# bayesHead and TransformerWithBayesHead


class BayesHead(nn.Module):
    """
    Flexible Bayesian output head for a Transformer encoder.
    Takes pooled features [B, D] and produces a predictive distribution.
    """

    def __init__(self, d_model: int, head_type: str = "baseline"):
        super().__init__()
        self.d_model = d_model
        self.head_type = head_type

        if head_type == "baseline":
# gaussian, homoscedastic
            self.fc_mu = nn.Linear(d_model, 1)
            self.log_var = nn.Parameter(torch.tensor(0.0))  # log(sigma^2)

        elif head_type == "hetero":
# gaussian, heteroscedastic
            self.fc_mu = nn.Linear(d_model, 1)
            self.fc_var = nn.Linear(d_model, 1)  # predicts log(sigma^2)

        elif head_type == "heavy":
# student-T, homoscedastic
            self.fc_mu = nn.Linear(d_model, 1)
            self.log_var = nn.Parameter(torch.tensor(0.0))
            # nu (degrees of freedom), learned
            self.log_nu = nn.Parameter(torch.tensor(np.log(28.0)))  # near Gaussian

        elif head_type == "hetero_heavy":
# student-T, heteroscedastic
            self.fc_mu = nn.Linear(d_model, 1)
            self.fc_var = nn.Linear(d_model, 1)
            self.log_nu = nn.Parameter(torch.tensor(np.log(28.0)))

        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def get_nu(self):
        """Returns positive degrees of freedom, clamped for stability."""
        if hasattr(self, "log_nu"):
            # softplus(log_nu) + 2 ensures nu > 2 (finite variance)
            return F.softplus(self.log_nu) + 2.0
        return None

    def get_dist(self, h: torch.Tensor) -> torch.distributions.Distribution:
        """
        Compute predictive distribution from features h: [B, D].
        """
        mu = self.fc_mu(h).squeeze(-1)  # [B]

        if self.head_type == "baseline":
            scale = (self.log_var.exp() + 1e-6).sqrt()
            return Normal(mu, scale)

        elif self.head_type == "hetero":
            log_var_pred = self.fc_var(h).squeeze(-1)
            scale = (log_var_pred.exp() + 1e-6).sqrt()
            return Normal(mu, scale)

        elif self.head_type == "heavy":
            scale = (self.log_var.exp() + 1e-6).sqrt()
            nu = self.get_nu()
            return StudentT(nu, mu, scale)

        elif self.head_type == "hetero_heavy":
            log_var_pred = self.fc_var(h).squeeze(-1)
            scale = (log_var_pred.exp() + 1e-6).sqrt()
            nu = self.get_nu()
            return StudentT(nu, mu, scale)

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss.

        h: [B, D]
        y: [B]
        """
        pred_dist = self.get_dist(h)
        nll = -pred_dist.log_prob(y)
        return nll


class TransformerWithBayesHead(nn.Module):
    """
    Transformer encoder + BayesHead output layer.
    """

    def __init__(self, encoder_cfg: Dict, head: BayesHead):
        super().__init__()
        self.encoder = TransformerEncoder(**encoder_cfg)
        self.head = head

    def pool(self, memory: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Global average pooling over sequence length, respecting padding mask.

        memory: [B, L, D]
        mask:   [B, L] (True for padding)
        """
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()  # [B, L, 1]
            pooled = (memory * valid_mask).sum(dim=1)  # [B, D]
            valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled / valid_counts
        else:
            pooled = memory.mean(dim=1)
        return pooled

    def forward(
        self, src: torch.Tensor, y: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Training forward pass (returns NLL loss).

        src: [B, L, Fin]
        y:   [B]
        """
        memory = self.encoder(src, src_key_padding_mask)  # [B, L, D]
        pooled = self.pool(memory, src_key_padding_mask)  # [B, D]
        nll = self.head(pooled, y)  # [B]
        return nll

    def predict_dist(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.distributions.Distribution:
        """
        Inference forward pass (returns predictive distribution).
        """
        memory = self.encoder(src, src_key_padding_mask)
        pooled = self.pool(memory, src_key_padding_mask)
        pred_dist = self.head.get_dist(pooled)
        return pred_dist


# sequentialLotteryDataset


class SequentialLotteryDataset(Dataset):
    """
    Creates sequences (windows) from the feature DataFrame.

    X = window_len features
    y = scalar target (e.g., next_mean_std)
    """

    def __init__(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str, window_len: int
    ):
        self.window_len = window_len

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame.")

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)

        self.n_samples = len(self.features)

        if self.n_samples < window_len:
            print(
                f"Warning: Not enough data ({self.n_samples}) for window_len={window_len}. "
                "Dataset will be empty."
            )
            self.num_sequences = 0
        else:
            self.num_sequences = self.n_samples - self.window_len + 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a window of features and the target at the end of that window.

        X = features[idx : idx + window_len]
        y = target[idx + window_len - 1]
        """
        end_idx = idx + self.window_len
        X_window = self.features[idx:end_idx, :]  # [L, Fin]
        y_target = self.targets[end_idx - 1]  # scalar
        return X_window, y_target


# train & export

def train_and_export(dm, args) -> Dict:
    """
    Manual sequential data split + standardized target `next_mean_std`.
    Reads --hetero_var / --heavy_tail flags to pick BayesHead variant.
    """
    seed = getattr(args, "seed", 42)
    set_global_seed(seed, deterministic=True)

    lottery = args.lottery
    batch_size = int(getattr(args, "batch_size", 64))
    lr = float(getattr(args, "learning_rate", 1e-3))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    max_epochs = int(getattr(args, "epochs", 60))
    window_len = int(getattr(args, "window_len", 64))

    # --- DATA PIPELINE ---

    path = DATA_PATH / f"features_{lottery}.parquet"
    if not path.exists():
        print(f"FATAL: Cannot find features file: {path}")
        return {"ok": False, "error": f"File not found: {path}"}

    try:
        df = dm.load_features()
        if df is None:
            raise ValueError("dm.load_features() returned None")
    except Exception as e:
        print(f"Error calling dm.load_features(): {e}. Falling back to manual load.")
        df = pd.read_parquet(path)

    # 1. Raw target
    raw_target_col = "next_mean"
    df[raw_target_col] = df["mean"].shift(-1)

    # 2. Standardized target name
    target_col = "next_mean_std"

    # 3. Feature columns
    exclude_cols = [
        "drawing_index",
        "date",
        "game",
        "special_ball",
        "mean",
        raw_target_col,
        target_col,
    ]
    feature_cols = [
        col
        for col in df.columns
        if df[col].dtype in ["float64", "float32", "int64", "int32"]
        and col not in exclude_cols
    ]
    input_dim = len(feature_cols)
    print(f"Using {input_dim} feature columns.")

    # 4. NaNs
    df[feature_cols] = df[feature_cols].fillna(0)
    print("Filled NaNs in feature columns with 0.")

    # 5. Drop last row with NaN target
    df_clean = df.dropna(subset=[raw_target_col])
    n = len(df_clean)
    if n == 0:
        raise ValueError(
            f"No valid data remaining after dropna on '{raw_target_col}'."
        )

    # 6. Sequential split
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    train_df = df_clean.iloc[:n_train].copy()
    val_df = df_clean.iloc[n_train:n_val].copy()
    test_df = df_clean.iloc[n_val:].copy()

    print(
        f"Performed sequential split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # 7. Align for windows
    n_chop = window_len - 1
    val_df = val_df.iloc[n_chop:]
    test_df = test_df.iloc[n_chop:]
    print(
        f"Aligned splits for window_len={window_len} -> Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # 8. Standardize targets
    target_scaler = StandardScaler()
    train_df[target_col] = target_scaler.fit_transform(train_df[[raw_target_col]])
    val_df[target_col] = target_scaler.transform(val_df[[raw_target_col]])
    test_df[target_col] = target_scaler.transform(test_df[[raw_target_col]])
    print(f"Created canonical target: '{target_col}'")

    # 9. Standardize features
    feature_scaler = StandardScaler()
    train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = feature_scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])
    print("Created standardized features.")

    # 10. Datasets
    ds_train = SequentialLotteryDataset(train_df, feature_cols, target_col, window_len)
    ds_val = SequentialLotteryDataset(val_df, feature_cols, target_col, window_len)
    ds_test = SequentialLotteryDataset(test_df, feature_cols, target_col, window_len)

    print(
        f"Dataset sequences -> Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}"
    )
    if len(ds_test) == 0:
        print(
            "WARNING: Test dataset is empty after processing (not enough data for window_len). "
            "This is expected for Mega Millions. Skipping test evaluation."
        )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- MODEL CONFIG ---

    model_cfg = {
        "input_dim": input_dim,
        "window_len": window_len,
        "d_model": 128,
        "nhead": 4,
        "d_hid": 256,
        "nlayers": 2,
        "dropout": 0.1,
    }

    use_hetero = getattr(args, "hetero_var", False)
    use_heavy = getattr(args, "heavy_tail", False)

    head_type = "baseline"
    model_key_out = "transformer_bayeshead_baseline"

    if use_hetero and use_heavy:
        head_type = "hetero_heavy"
        model_key_out = "transformer_bayeshead_hetero_heavy"
    elif use_hetero:
        head_type = "hetero"
        model_key_out = "transformer_bayeshead_hetero"
    elif use_heavy:
        head_type = "heavy"
        model_key_out = "transformer_bayeshead_heavy"

    print(
        f"Initializing BayesHead variant: {head_type} (hetero={use_hetero}, heavy={use_heavy})"
    )

    model_head = BayesHead(model_cfg["d_model"], head_type=head_type)
    model = TransformerWithBayesHead(model_cfg, model_head).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- TRAINING LOOP ---

    start = time.time()
    best_val = float("inf")
    best_state = None

    for ep in range(max_epochs):
        model.train()
        running_nll = 0.0
        count = 0

        for X_seq, y_target in train_loader:
            X_seq, y_target = X_seq.to(device), y_target.to(device)

            nll = model(X_seq, y_target)
            loss = nll.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            running_nll += loss.item() * X_seq.size(0)
            count += X_seq.size(0)

        train_nll = running_nll / max(1, count)

        model.eval()
        val_nll = 0.0
        val_n = 0
        with torch.no_grad():
            for X_seq, y_target in val_loader:
                X_seq, y_target = X_seq.to(device), y_target.to(device)
                nll = model(X_seq, y_target)
                loss = nll.mean()
                val_nll += loss.item() * X_seq.size(0)
                val_n += X_seq.size(0)
        val_nll = val_nll / max(1, val_n)

        if getattr(args, "verbose", False) and ((ep + 1) % 20 == 0 or ep == 0):
            print(
                f"[Tx+BayesHead][{lottery}] epoch {ep+1}/{max_epochs} "
                f"train_nll={train_nll:.4f} val_nll={val_nll:.4f}"
            )

        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - start

    # --- EVAL ON TEST ---

    model.eval()
    y_true_list = []
    y_pred_list = []
    y_lo_list = []
    y_hi_list = []

    with torch.no_grad():
        for X_seq, y_target in test_loader:
            X_seq, y_target = X_seq.to(device), y_target.to(device)

            pred_dist = model.predict_dist(X_seq)

# point predictions
            y_pred = pred_dist.mean.cpu().numpy()

            # 95% intervals via mean +/- 1.96 * stddev (no quantile/icdf calls)
            z = 1.96
            if hasattr(pred_dist, "stddev"):
                std = pred_dist.stddev
            elif hasattr(pred_dist, "scale"):
                std = pred_dist.scale
            elif hasattr(pred_dist, "variance"):
                std = pred_dist.variance.sqrt()
            else:
                std = torch.zeros_like(pred_dist.mean)

            y_lo = (pred_dist.mean - z * std).cpu().numpy()
            y_hi = (pred_dist.mean + z * std).cpu().numpy()

            y_true_list.extend(y_target.cpu().numpy().tolist())
            y_pred_list.extend(y_pred.tolist())
            y_lo_list.extend(y_lo.tolist())
            y_hi_list.extend(y_hi.tolist())

    y_true = np.array(y_true_list, dtype=np.float64)
    y_pred = np.array(y_pred_list, dtype=np.float64)
    y_lo = np.array(y_lo_list, dtype=np.float64)
    y_hi = np.array(y_hi_list, dtype=np.float64)

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

    perf_path = out_dir / f"{model_key_out}_perf{suffix}.json"
    pred_path = out_dir / f"{model_key_out}_predictive{suffix}.json"

    metrics = {
        "model_name": model_key_out,
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
            "head_type": head_type,
            "input_dim": input_dim,
            "d_model": model_cfg["d_model"],
            "nhead": model_cfg["nhead"],
            "nlayers": model_cfg["nlayers"],
            "window_len": window_len,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

    perf_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_payload = {
        "model_name": model_key_out,
        "lottery": lottery,
        "y_mean": y_pred.tolist(),
        "y_lo": y_lo.tolist(),
        "y_hi": y_hi.tolist(),
    }
    pred_path.write_text(json.dumps(pred_payload, indent=2), encoding="utf-8")

# save weights
    tx_dir = MODELS_DIR / "transformer_bayeshead"
    tx_dir.mkdir(parents=True, exist_ok=True)
    weights_path = tx_dir / f"{lottery}_transformer_encoder_{head_type}.pt"
    torch.save(model.state_dict(), weights_path)

    summary_path = write_summary(model_key_out, lottery, metrics, args=args)

    if rmse is not None and coverage is not None:
        print(
            f"[Tx+BayesHead] Training complete. "
            f"Test RMSE: {rmse:.4f}, Coverage: {coverage:.4f}, Test Samples: {len(y_true)}"
        )
    else:
        print("[Tx+BayesHead] Training complete, but no test samples were available.")

    return {
        "model": model_key_out,
        "variant": head_type,
        "perf_path": str(perf_path),
        "predictive_dump_path": str(pred_path),
        "encoder_weights_path": str(weights_path),
        "summary_path": str(summary_path),
    }


# main block / argument parsing

if __name__ == "__main__":
    import argparse
    from utils_data import LotteryFeatureDataModule

    parser = argparse.ArgumentParser(description="Train Transformer+BayesHead model.")
    parser.add_argument(
        "--lottery",
        type=str,
        required=True,
        choices=["powerball", "megamillions"],
        help="Lottery name",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--window_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--model", type=str, default="transformer_bayeshead")

# dummy args for zoo compatibility
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--bnn_seq_len", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--conformal_mode", type=str, default="pooled")
    parser.add_argument("--eprocess_mode", type=str, default="standard")
    parser.add_argument("--safe_e_gamma", type=float, default=0.4)
    parser.add_argument("--baseline", type=str, default="copy_last")

    args = parser.parse_args()

    print(f"[Tx+BayesHead] Model: {args.model} | Lottery: {args.lottery}")
    set_global_seed(args.seed, deterministic=True)

    dm = LotteryFeatureDataModule(
        lottery=args.lottery,
        batch_size=args.batch_size,
        window_len=args.window_len,
    )

    args.suffix = LOTTERY_SUFFIX_MAP[args.lottery]

    try:
        result = train_and_export(dm, args)
        print(
            f"[Tx+BayesHead] Training complete. Performance report: {result.get('perf_path')}"
        )
    except Exception as e:
        print(f"[Tx+BayesHead] Worker crashed:\n{e}")
        import traceback

        traceback.print_exc()

    print("[Tx+BayesHead] Done.")

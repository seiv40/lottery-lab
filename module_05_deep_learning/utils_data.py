#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Utilities for Module 5: Deep Learning Model Zoo

Why we need shared data utilities:

When running 7 different models, consistency is critical. If models use different:
- Train/val/test splits
- Feature normalization
- Window sizes
- Data loading logic

...then we can't fairly compare results. One model might perform better just because
it saw more training data or used different features.

This module ensures all models use identical data pipelines:
1. Same chronological 70/15/15 or 80/10/10 split
2. Same StandardScaler fit ONLY on training data (prevents leakage)
3. Same feature columns in same order
4. Consistent windowing for sequence models

We have two DataModule types:
- LotteryFeatureDataModule: Flat features (VAE, Flow, DeepSets, BNN)
- LotterySequenceDataModule: Windowed sequences (Transformer)

Both use PyTorch Lightning DataModules for clean train/val/test loaders.

"""

import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


class _SeqDataset(Dataset):
    """simple sequence dataset that returns (X, y) tensors for the model"""
    def __init__(self, X_np, y_np):
        # convert numpy arrays to tensors once up front
        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(y_np).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Path Configuration

def infer_base_path() -> Path:
    """
    figure out where the project root is for consistent paths
    
    we try three approaches in order:
    1. LOTTERY_LAB_BASE environment variable (optional override)
    2. hardcoded repo root (your Windows setup)
    3. fallback: relative traversal from this file location
    
    this flexibility helps when running on different machines or CI/CD
    """
    # option 1: allow override by environment variable
    env_base = os.environ.get("LOTTERY_LAB_BASE")
    if env_base:
        return Path(env_base).resolve()

    # option 2: hardcoded correct repo base (your known setup)
    hardcoded = Path("C:/jackpotmath/lottery-lab")
    if hardcoded.exists():
        return hardcoded

    # option 3: fallback heuristic (go up 2 directories from this file)
    # from modules/module_05_deep_learning/utils_data.py -> lottery-lab/
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        print(f"Warning: could not infer BASE_PATH. Assuming {hardcoded}")
        return hardcoded


# global paths used by all models
BASE_PATH = infer_base_path()
DATA_PATH = BASE_PATH / "data" / "processed" / "features"
OUTPUT_DIR = BASE_PATH / "outputs"
LOG_DIR = BASE_PATH / "logs"
MODELS_DIR = BASE_PATH / "models" / "deep_learning"

# ensure output directories exist
for d in [OUTPUT_DIR, LOG_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    make sure the 'date' column is actually datetime type
    
    sometimes dates get loaded as strings, which breaks chronological sorting
    """
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in features parquet.")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some 'date' values failed to parse to datetime.")
    return df


def split_and_scale(df: pd.DataFrame, feature_cols: List[str], scaler_path: Path):
    """
    chronological 80/10/10 split + standardization
    
    critical: fit scaler ONLY on training data to prevent temporal leakage
    
    we save the scaler so all models use the exact same normalization
    this ensures fair comparison - models see identically scaled features
    """
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    n = len(X)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    
    # chronological indices
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    # load or create scaler
    if scaler_path.exists():
        scaler: StandardScaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X[idx_train])  # fit ONLY on training data
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return (
        scaler.transform(X[idx_train]),
        scaler.transform(X[idx_val]),
        scaler.transform(X[idx_test]),
        idx_train, idx_val, idx_test, scaler
    )


# Data Modules

class LotteryFeatureDataModule(pl.LightningDataModule):
    """
    flat feature tensors (no sequences) for VAE/BNN/Flow/DeepSets
    
    these models don't need temporal windows - they work on single timestep features
    """
    
    def __init__(self, lottery: str, batch_size: int = 64, window_len: int = 64):
        super().__init__()
        self.lottery = lottery
        self.batch_size = batch_size
        self.window_len = window_len  # not used but kept for compatibility
        self.feature_file = DATA_PATH / f"features_{self.lottery}.parquet"
        self.meta_dir = OUTPUT_DIR / "meta"
        self.scaler_path = self.meta_dir / f"{self.lottery}_scaler.joblib"
        self.cols_path = self.meta_dir / f"{self.lottery}_feature_cols.json"

        self.input_dim = 0
        self.X_train = self.X_val = self.X_test = None
        self.X_all = None
        self.metadata_all = None
        self.feature_cols = None

    def setup(self, stage=None):
        """
        load data and prepare train/val/test splits
        
        we drop the first 20 rows because rolling window features are NaN there
        """
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Missing parquet: {self.feature_file}")
        
        df = pd.read_parquet(self.feature_file)
        df = ensure_datetime(df)
        df = df.iloc[20:].copy()  # drop early NaNs from rolling features

        # separate metadata from features
        meta_cols = {"date", "drawing_index", "game", "special_ball", 
                     "jackpot_annuitized", "jackpot_cash"}
        meta_cols = {c for c in meta_cols if c in df.columns}
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # save feature column order for consistency across models
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cols_path, "w") as f:
            json.dump(feature_cols, f, indent=2)

        # split and scale
        X_train, X_val, X_test, idx_tr, idx_va, idx_te, scaler = split_and_scale(
            df, feature_cols, self.scaler_path
        )

        self.input_dim = len(feature_cols)
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.X_all = np.concatenate([X_train, X_val, X_test], axis=0)

        # track split membership for each drawing
        dates = df["date"].reset_index(drop=True)
        self.metadata_all = pd.DataFrame({
            "date": dates,
            "split": (["train"] * len(range(*idx_tr.indices(len(df)))) +
                      ["val"] * len(range(*idx_va.indices(len(df)))) +
                      ["test"] * len(range(*idx_te.indices(len(df)))))
        })
        self.feature_cols = feature_cols

    def _make_loader(self, X, shuffle=False):
        """helper to create dataloader from numpy array"""
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self): 
        return self._make_loader(self.X_train, shuffle=True)
    
    def val_dataloader(self):   
        return self._make_loader(self.X_val, shuffle=False)
    
    def test_dataloader(self):  
        return self._make_loader(self.X_test, shuffle=False)
    
    def predict_dataloader(self): 
        return self._make_loader(self.X_all, shuffle=False)
    
    def load_features(self):
        """
        load raw features dataframe (for models that need custom processing)
        
        some models want to do their own windowing or target creation,
        so we provide access to the raw data
        """
        if not self.feature_file.exists():
            return None
        return pd.read_parquet(self.feature_file)


class SequenceDataset(Dataset):
    """
    sliding-window dataset for temporal models
    
    creates (window, next_row) pairs for sequence prediction
    window = rows [t-64, t-63, ..., t-1]
    target = row [t]
    
    this is how Transformers see the data
    """
    
    def __init__(self, X_split: np.ndarray, dates_split: pd.Series, window_len: int):
        self.X = X_split.astype(np.float32)
        self.dates = pd.to_datetime(dates_split).reset_index(drop=True)
        self.T = int(window_len)
        self.N = self.X.shape[0]
        self.K = max(0, self.N - self.T)  # number of valid windows
        self.target_dates = self.dates[self.T:].reset_index(drop=True)

    def __len__(self): 
        return self.K

    def __getitem__(self, idx: int):
        """
        returns:
        - x_seq: window of T timesteps [idx:idx+T]
        - y_next: the next timestep [idx+T]
        """
        x_seq = self.X[idx:idx + self.T]
        y_next = self.X[idx + self.T]
        return torch.from_numpy(x_seq), torch.from_numpy(y_next)


class LotterySequenceDataModule(pl.LightningDataModule):
    """
    sequence datamodule for transformer-like models
    
    creates sliding windows with NO TEMPORAL LEAKAGE:
    - train windows never see validation/test data
    - scaler fit only on training data
    - chronological ordering preserved
    """
    
    def __init__(self, lottery: str, window_len: int = 32, batch_size: int = 64):
        super().__init__()
        self.lottery = lottery
        self.window_len = int(window_len)
        self.batch_size = batch_size
        self.feature_file = DATA_PATH / f"features_{self.lottery}.parquet"
        self.meta_dir = OUTPUT_DIR / "meta"
        self.scaler_path = self.meta_dir / f"{self.lottery}_scaler.joblib"
        self.cols_path = self.meta_dir / f"{self.lottery}_feature_cols.json"

        self.input_dim = 0
        self.X_train = self.X_val = self.X_test = None
        self.d_train = self.d_val = self.d_test = None

    def setup(self, stage=None):
        """load and prepare windowed sequences"""
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Missing parquet: {self.feature_file}")
        
        df = pd.read_parquet(self.feature_file)
        df = ensure_datetime(df)
        df = df.iloc[20:].copy()

        # separate metadata from features
        meta_cols = {"date", "drawing_index", "game", "special_ball", 
                     "jackpot_annuitized", "jackpot_cash"}
        meta_cols = {c for c in meta_cols if c in df.columns}
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # save feature column order for consistency
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cols_path, "w") as f:
            json.dump(feature_cols, f, indent=2)

        # split and scale
        X_train, X_val, X_test, idx_tr, idx_va, idx_te, scaler = split_and_scale(
            df, feature_cols, self.scaler_path
        )

        # extract dates for each split
        dates = df["date"]
        self.d_train = dates.iloc[idx_tr].reset_index(drop=True)
        self.d_val = dates.iloc[idx_va].reset_index(drop=True)
        self.d_test = dates.iloc[idx_te].reset_index(drop=True)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.input_dim = len(feature_cols)

        # create windowed datasets
        self.train_ds = SequenceDataset(self.X_train, self.d_train, self.window_len)
        self.val_ds = SequenceDataset(self.X_val, self.d_val, self.window_len)
        self.test_ds = SequenceDataset(self.X_test, self.d_test, self.window_len)

    def train_dataloader(self): 
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):   
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):  
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)


# Baseline Helpers

def baseline_laststep_rmse(X_split: np.ndarray, window_len: int) -> float:
    """
    simple baseline: predict next row as copy of last row in window
    
    this gives us a "null model" to compare against. if our fancy deep learning
    models can't beat this simple baseline, they're not learning anything useful.
    """
    T = int(window_len)
    if len(X_split) <= T:
        return float("nan")
    
    Y_true = X_split[T:]  # actual next rows
    Y_pred = X_split[T-1:-1]  # last rows of each window (shifted)
    
    return float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))


def assert_feature_schema(X: np.ndarray, feature_cols=None, where: str = 'unknown'):
    """
    safety checks on scaled feature matrices
    
    catches common data issues before they break training:
    - inf or NaN values (breaks gradient descent)
    - extreme magnitudes (numerical instability)
    - dimension mismatches (wrong features loaded)
    
    better to fail fast with a clear error than get mysterious NaN losses later
    """
    assert isinstance(X, np.ndarray), f"{where}: X must be numpy array"
    assert np.isfinite(X).all(), f"{where}: non-finite values encountered after scaling"
    assert (np.abs(X) < 1e6).all(), (
        f"{where}: extreme magnitudes found (>1e6); scaler or units may be wrong"
    )
    if feature_cols is not None:
        assert X.shape[1] == len(feature_cols), (
            f"{where}: feature dimension mismatch: "
            f"X has {X.shape[1]}, feature_cols has {len(feature_cols)}"
        )
    return True

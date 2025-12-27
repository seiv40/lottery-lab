"""
Module 8: Data Loading and Preparation

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings

from config import (
    POWERBALL_FEATURES, MEGAMILLIONS_FEATURES,
    POWERBALL_RAW, MEGAMILLIONS_RAW,
    get_lottery_config, get_ball_columns,
    RANDOM_SEED
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


class LotteryDataLoader:
    """Load and prepare lottery data for causal inference analysis."""
    
    def __init__(self, lottery_name: str):
        """
        Initialize data loader for specific lottery.
        
        Parameters
        ----------
        lottery_name : str
            Either 'powerball' or 'megamillions'
        """
        self.lottery_name = lottery_name.lower()
        self.config = get_lottery_config(self.lottery_name)
        
        if self.config is None:
            raise ValueError(f"Unknown lottery: {lottery_name}")
        
        self.ball_columns = self.config['ball_columns']
        self.special_ball = self.config['special_ball']
        
    def load_features(self) -> pd.DataFrame:
        """Load engineered features from parquet file."""
        if self.lottery_name == 'powerball':
            path = POWERBALL_FEATURES
        else:
            path = MEGAMILLIONS_FEATURES
        
        print(f"Loading features from: {path}")
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} draws with {len(df.columns)} features")
        
        return df
    
    def load_raw_draws(self) -> pd.DataFrame:
        """Load raw lottery draws from parquet file."""
        if self.lottery_name == 'powerball':
            path = POWERBALL_RAW
        else:
            path = MEGAMILLIONS_RAW
        
        print(f"Loading raw draws from: {path}")
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} draws")
        
        return df
    
    def prepare_for_granger(self, 
                           features: List[str] = None,
                           max_lags: int = 20) -> pd.DataFrame:
        """
        Prepare data for Granger causality testing.
        
        Parameters
        ----------
        features : List[str], optional
            Features to include. If None, uses aggregate statistics.
        max_lags : int
            Maximum lag to include
            
        Returns
        -------
        pd.DataFrame
            Data with lagged features
        """
        df = self.load_features()
        
        if features is None:
            # use aggregate statistics - actual column names
            features = ['mean', 'sum', 'variance', 'range', 'std']
        
        # ensure features exist
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            print(f"Warning: Features not found: {missing}")
        
        # select features and sort by draw order
        df_granger = df[available_features].copy()
        
        # remove any NaN values
        df_granger = df_granger.dropna()
        
        print(f"Prepared {len(df_granger)} samples with {len(available_features)} features for Granger testing")
        
        return df_granger
    
    def prepare_for_conditional_independence(self,
                                            include_balls: bool = True,
                                            include_aggregate: bool = True) -> pd.DataFrame:
        """
        Prepare data for conditional independence testing.
        
        Parameters
        ----------
        include_balls : bool
            Whether to include individual ball positions (from raw files)
        include_aggregate : bool
            Whether to include aggregate statistics (from feature files)
            
        Returns
        -------
        pd.DataFrame
            Data ready for CI testing
        """
        columns_to_include = []
        dfs_to_merge = []
        
        if include_aggregate:
            # load feature file for aggregate statistics
            df_features = self.load_features()
            
            # get aggregate columns - actual column names from feature file
            agg_cols = ['mean', 'sum', 'variance', 'std', 'range', 'median', 'min', 'max']
            agg_cols = [c for c in agg_cols if c in df_features.columns]
            
            if agg_cols:
                dfs_to_merge.append(df_features[agg_cols])
        
        if include_balls:
            # load raw draws for individual ball positions
            df_raw = self.load_raw_draws()
            
            # get ball columns from raw file
            ball_cols = self.ball_columns
            ball_cols = [c for c in ball_cols if c in df_raw.columns]
            
            if ball_cols:
                dfs_to_merge.append(df_raw[ball_cols])
        
        # merge all selected data
        if len(dfs_to_merge) == 0:
            raise ValueError("No columns selected for CI testing")
        elif len(dfs_to_merge) == 1:
            df_ci = dfs_to_merge[0]
        else:
            # merge on index (assumes same draw order)
            df_ci = pd.concat(dfs_to_merge, axis=1)
        
        df_ci = df_ci.dropna()
        
        print(f"Prepared {len(df_ci)} samples with {len(df_ci.columns)} variables for CI testing")
        print(f"  Columns: {list(df_ci.columns)}")
        
        return df_ci
    
    def prepare_ball_time_series(self) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare individual ball time series for multivariate analysis.
        
        Returns
        -------
        data : np.ndarray
            Array of shape (n_draws, n_balls)
        ball_names : List[str]
            Names of each ball column
        """
        # load from raw draws (ball positions are here, not in feature file)
        df = self.load_raw_draws()
        
        # get ball column names from config
        ball_names = self.ball_columns
        ball_names = [c for c in ball_names if c in df.columns]
        
        if len(ball_names) == 0:
            raise ValueError(f"No ball columns found in raw data. Expected: {self.ball_columns}")
        
        data = df[ball_names].values
        
        print(f"Prepared ball time series: {data.shape[0]} draws x {data.shape[1]} balls")
        print(f"  Ball columns: {ball_names}")
        
        return data, ball_names
    
    def create_temporal_splits(self, 
                              n_splits: int = 3) -> List[Tuple[int, int]]:
        """
        Create temporal splits for causal invariance testing.
        
        Parameters
        ----------
        n_splits : int
            Number of temporal environments to create
            
        Returns
        -------
        List[Tuple[int, int]]
            List of (start_idx, end_idx) tuples defining each split
        """
        df = self.load_features()
        n_samples = len(df)
        
        split_size = n_samples // n_splits
        
        splits = []
        for i in range(n_splits):
            start = i * split_size
            end = start + split_size if i < n_splits - 1 else n_samples
            splits.append((start, end))
        
        print(f"Created {n_splits} temporal splits:")
        for i, (start, end) in enumerate(splits, 1):
            print(f"  Environment {i}: draws {start}-{end} ({end-start} samples)")
        
        return splits
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics about the dataset."""
        df = self.load_features()
        
        stats = {
            'n_draws': len(df),
            'n_features': len(df.columns),
            'lottery': self.lottery_name,
            'ball_config': self.config
        }
        
        # add some basic statistics
        if 'mean' in df.columns:
            stats['mean_of_means'] = df['mean'].mean()
            stats['std_of_means'] = df['mean'].std()
        
        return stats


def load_both_lotteries() -> Dict[str, pd.DataFrame]:
    """
    Load feature data for both lotteries.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with 'powerball' and 'megamillions' keys
    """
    loaders = {
        'powerball': LotteryDataLoader('powerball'),
        'megamillions': LotteryDataLoader('megamillions')
    }
    
    data = {}
    for name, loader in loaders.items():
        try:
            data[name] = loader.load_features()
            print(f"Loaded {name}: {len(data[name])} draws")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    return data


def validate_stationarity(data: pd.DataFrame,
                         columns: List[str] = None) -> Dict[str, float]:
    """
    Test stationarity using Augmented Dickey-Fuller test.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    columns : List[str], optional
        Columns to test. If None, tests all numeric columns.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of column name to ADF p-value
    """
    from statsmodels.tsa.stattools import adfuller
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    results = {}
    
    print("\nStationarity Testing (ADF):")
    print("-" * 60)
    
    for col in columns:
        try:
            series = data[col].dropna()
            adf_result = adfuller(series, maxlag=20, regression='ct')
            p_value = adf_result[1]
            results[col] = p_value
            
            status = "Stationary" if p_value < 0.05 else "Non-stationary"
            print(f"{col:30s} p={p_value:.4f}  [{status}]")
            
        except Exception as e:
            print(f"{col:30s} ERROR: {e}")
            results[col] = np.nan
    
    return results


if __name__ == "__main__":
    """Test data loading functionality."""
    
    print("\n--- MODULE 8: DATA LOADING TEST ---")
    
    # test loading both lotteries
    for lottery in ['powerball', 'megamillions']:
        print(f"\n--- Testing {lottery.upper()} ---")
        
        try:
            loader = LotteryDataLoader(lottery)
            
            # test feature loading
            print("\n1. Loading Features:")
            df_features = loader.load_features()
            print(f"   Shape: {df_features.shape}")
            print(f"   Columns: {list(df_features.columns[:10])}...")
            
            # test Granger preparation
            print("\n2. Preparing for Granger Causality:")
            df_granger = loader.prepare_for_granger()
            print(f"   Shape: {df_granger.shape}")
            
            # test CI preparation
            print("\n3. Preparing for Conditional Independence:")
            df_ci = loader.prepare_for_conditional_independence()
            print(f"   Shape: {df_ci.shape}")
            
            # test ball time series
            print("\n4. Preparing Ball Time Series:")
            ball_data, ball_names = loader.prepare_ball_time_series()
            print(f"   Shape: {ball_data.shape}")
            print(f"   Balls: {ball_names}")
            
            # test temporal splits
            print("\n5. Creating Temporal Splits:")
            splits = loader.create_temporal_splits(n_splits=3)
            
            # test stationarity
            print("\n6. Testing Stationarity:")
            stationarity = validate_stationarity(df_granger)
            
            print(f"\nAll tests passed for {lottery}")
            
        except Exception as e:
            print(f"\nError testing {lottery}: {e}")
            import traceback
            traceback.print_exc()

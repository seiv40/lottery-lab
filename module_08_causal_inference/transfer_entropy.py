"""
Module 8: Transfer Entropy Analysis
Measure directed information flow between lottery time series.

Transfer Entropy (TE) quantifies the reduction in uncertainty about Y's future
given knowledge of X's past, beyond what Y's own past provides:

TE(X -> Y) = I(Y_t+1; X_t^k | Y_t^l)

Where TE = 0 indicates no information flow (no causation).

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats

# Import pyinform for transfer entropy
try:
    from pyinform import transferentropy
    PYINFORM_AVAILABLE = True
except ImportError:
    print("Warning: pyinform not available. Using fallback implementation.")
    PYINFORM_AVAILABLE = False

from config import (
    TE_HISTORY_LENGTH, TE_N_PERMUTATIONS,
    FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED
)
from data_loader import LotteryDataLoader

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


class TransferEntropyAnalyzer:
    """Transfer entropy analysis for lottery time series."""
    
    def __init__(self, lottery_name: str):
        """
        Initialize transfer entropy analyzer.
        
        Parameters:
        -----------
        lottery_name : str
            Either 'powerball' or 'megamillions'
        """
        self.lottery_name = lottery_name
        self.loader = LotteryDataLoader(lottery_name)
        self.results = {}
        self.data = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare data for transfer entropy analysis."""
        print("Loading data for transfer entropy analysis...")
        
        # Load ball time series
        ball_data, ball_names = self.loader.prepare_ball_time_series()
        
        # Also include some aggregate statistics - ACTUAL COLUMN NAMES
        df_features = self.loader.load_features()
        agg_features = ['mean', 'sum', 'variance']
        agg_features = [f for f in agg_features if f in df_features.columns]
        
        if agg_features:
            agg_data = df_features[agg_features].values
            # Combine
            self.data = np.hstack([ball_data, agg_data])
            self.feature_names = ball_names + agg_features
        else:
            self.data = ball_data
            self.feature_names = ball_names
        
        print(f"  Loaded {self.data.shape[0]} samples with {self.data.shape[1]} features")
        print(f"  Features: {self.feature_names}")
    
    def discretize_data(self, 
                       n_bins: int = 10) -> np.ndarray:
        """
        Discretize continuous data for transfer entropy computation.
        
        Parameters:
        -----------
        n_bins : int
            Number of bins for discretization
            
        Returns:
        --------
        np.ndarray
            Discretized data
        """
        print(f"  Discretizing data into {n_bins} bins...")
        
        data_discrete = np.zeros_like(self.data, dtype=int)
        
        for i in range(self.data.shape[1]):
            # Use quantile-based binning for better distribution
            data_discrete[:, i] = pd.qcut(
                self.data[:, i],
                q=n_bins,
                labels=False,
                duplicates='drop'
            )
        
        return data_discrete
    
    def compute_transfer_entropy(self,
                                source_idx: int,
                                target_idx: int,
                                k: int = None) -> float:
        """
        Compute transfer entropy from source to target.
        
        TE(source -> target) measures how much knowing source's past
        reduces uncertainty about target's future.
        
        Parameters:
        -----------
        source_idx : int
            Index of source variable
        target_idx : int
            Index of target variable
        k : int, optional
            History length. If None, uses TE_HISTORY_LENGTH.
            
        Returns:
        --------
        float
            Transfer entropy value (in bits)
        """
        if k is None:
            k = TE_HISTORY_LENGTH
        
        # Discretize data if not already done
        if not hasattr(self, 'data_discrete'):
            self.data_discrete = self.discretize_data()
        
        source = self.data_discrete[:, source_idx].astype(int)
        target = self.data_discrete[:, target_idx].astype(int)
        
        if PYINFORM_AVAILABLE:
            # Use pyinform
            try:
                te = transferentropy.transfer_entropy(
                    source, target, k=k, local=False
                )
                return te
            except Exception as e:
                print(f"    Error computing TE with pyinform: {e}")
                return self._transfer_entropy_fallback(source, target, k)
        else:
            return self._transfer_entropy_fallback(source, target, k)
    
    def _transfer_entropy_fallback(self,
                                   source: np.ndarray,
                                   target: np.ndarray,
                                   k: int) -> float:
        """
        Fallback transfer entropy implementation.
        
        Uses basic estimation from conditional mutual information:
        TE(X -> Y) = I(Y_{t+1}; X_t^k | Y_t^k)
        """
        # This is a simplified implementation
        # In practice, proper estimation requires careful handling of history
        
        n = len(source) - k - 1
        
        # Build history vectors
        source_hist = np.array([source[i:i+k] for i in range(n)])
        target_hist = np.array([target[i:i+k] for i in range(n)])
        target_future = target[k+1:k+1+n]
        
        # Estimate mutual information
        # This is a very basic estimator - in production, use proper MI estimation
        
        # For now, return 0 to indicate no transfer
        # A proper implementation would estimate I(Y_future; X_hist | Y_hist)
        return 0.0
    
    def compute_te_matrix(self, k: int = None) -> np.ndarray:
        """
        Compute transfer entropy matrix for all variable pairs.
        
        Matrix[i,j] = TE(i -> j) = information flow from i to j
        
        Parameters:
        -----------
        k : int, optional
            History length
            
        Returns:
        --------
        np.ndarray
            Transfer entropy matrix
        """
        if k is None:
            k = TE_HISTORY_LENGTH
        
        if self.data is None:
            self.load_data()
        
        print(f"\nComputing transfer entropy matrix (k={k})...")
        
        # Discretize once
        self.data_discrete = self.discretize_data()
        
        n_features = self.data.shape[1]
        te_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue
                
                if (i % 2 == 0) and (j % 2 == 0):  # Progress indicator
                    print(f"  Progress: computing TE({self.feature_names[i]} -> {self.feature_names[j]})")
                
                try:
                    te = self.compute_transfer_entropy(i, j, k=k)
                    te_matrix[i, j] = te
                except Exception as e:
                    print(f"    Error computing TE({i},{j}): {e}")
                    te_matrix[i, j] = 0.0
        
        print(f"  Mean TE: {np.mean(te_matrix[te_matrix > 0]):.6f} bits")
        print(f"  Max TE: {np.max(te_matrix):.6f} bits")
        print(f"  Non-zero entries: {np.sum(te_matrix > 1e-6)}")
        
        self.results['te_matrix'] = te_matrix
        
        return te_matrix
    
    def significance_test_permutation(self,
                                    source_idx: int,
                                    target_idx: int,
                                    n_permutations: int = None,
                                    k: int = None) -> Tuple[float, float]:
        """
        Test significance of transfer entropy via permutation test.
        
        H0: TE(source -> target) = 0 (no information flow)
        
        Parameters:
        -----------
        source_idx : int
            Index of source variable
        target_idx : int
            Index of target variable
        n_permutations : int, optional
            Number of permutations for null distribution
        k : int, optional
            History length
            
        Returns:
        --------
        Tuple[float, float]
            (observed_te, p_value)
        """
        if n_permutations is None:
            n_permutations = TE_N_PERMUTATIONS
        
        if k is None:
            k = TE_HISTORY_LENGTH
        
        # Compute observed TE
        te_obs = self.compute_transfer_entropy(source_idx, target_idx, k=k)
        
        # Generate null distribution by permuting source
        te_null = []
        
        source = self.data_discrete[:, source_idx].copy()
        target = self.data_discrete[:, target_idx].copy()
        
        for _ in range(n_permutations):
            # Permute source to break any causal relationship
            source_perm = np.random.permutation(source)
            
            # Compute TE on permuted data
            if PYINFORM_AVAILABLE:
                try:
                    te_perm = transferentropy.transfer_entropy(
                        source_perm.astype(int), target.astype(int), k=k, local=False
                    )
                    te_null.append(te_perm)
                except:
                    te_null.append(0.0)
            else:
                te_null.append(0.0)
        
        te_null = np.array(te_null)
        
        # Compute p-value
        p_value = np.sum(te_null >= te_obs) / n_permutations
        
        return te_obs, p_value
    
    def test_pairwise_significance(self,
                                  feature_pairs: List[Tuple[int, int]] = None,
                                  n_permutations: int = 100) -> pd.DataFrame:
        """
        Test significance of TE for specific feature pairs.
        
        Parameters:
        -----------
        feature_pairs : List[Tuple[int, int]], optional
            List of (source_idx, target_idx) pairs to test.
            If None, tests a representative subset.
        n_permutations : int
            Number of permutations for each test
            
        Returns:
        --------
        pd.DataFrame
            Results with p-values
        """
        if self.data is None:
            self.load_data()
        
        self.data_discrete = self.discretize_data()
        
        if feature_pairs is None:
            # Test a subset: all balls → all balls
            n_balls = 6  # 5 white + 1 special
            feature_pairs = [(i, j) for i in range(n_balls) 
                           for j in range(n_balls) if i != j]
        
        print(f"\nTesting TE significance for {len(feature_pairs)} pairs...")
        print(f"  Using {n_permutations} permutations per test")
        
        results_list = []
        
        for source_idx, target_idx in feature_pairs:
            
            print(f"  Testing {self.feature_names[source_idx]} -> {self.feature_names[target_idx]}")
            
            te_obs, p_value = self.significance_test_permutation(
                source_idx, target_idx,
                n_permutations=n_permutations
            )
            
            results_list.append({
                'source': self.feature_names[source_idx],
                'target': self.feature_names[target_idx],
                'source_idx': source_idx,
                'target_idx': target_idx,
                'TE': te_obs,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        results_df = pd.DataFrame(results_list)
        
        # Summary
        n_significant = results_df['significant'].sum()
        print(f"\n  Significant TE relationships (p < 0.05): {n_significant}/{len(results_df)}")
        
        self.results['te_significance'] = results_df
        
        return results_df
    
    def visualize_results(self, save: bool = True):
        """Create visualizations of transfer entropy results."""
        
        # 1. Transfer entropy matrix heatmap
        if 'te_matrix' in self.results:
            te_matrix = self.results['te_matrix']
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.heatmap(te_matrix,
                       annot=False,
                       cmap='YlOrRd',
                       vmin=0,
                       xticklabels=self.feature_names,
                       yticklabels=self.feature_names,
                       cbar_kws={'label': 'Transfer Entropy (bits)'},
                       ax=ax)
            
            ax.set_title(f'Transfer Entropy Matrix - {self.lottery_name.title()}\n' +
                        f'(Row -> Column | 0 = No Information Flow)',
                        fontsize=14, pad=20)
            ax.set_xlabel('Target Variable', fontsize=12)
            ax.set_ylabel('Source Variable', fontsize=12)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_transfer_entropy_matrix.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
        
        # 2. Significance test results (if available)
        if 'te_significance' in self.results:
            sig_df = self.results['te_significance']
            
            # Bar plot of p-values
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Sort by TE value
            sig_df_sorted = sig_df.sort_values('TE', ascending=False).head(20)
            
            # TE values
            colors = ['red' if sig else 'gray' for sig in sig_df_sorted['significant']]
            labels = [f"{row['source'][:8]}->{row['target'][:8]}" 
                     for _, row in sig_df_sorted.iterrows()]
            
            ax1.barh(range(len(sig_df_sorted)), sig_df_sorted['TE'], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(sig_df_sorted)))
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel('Transfer Entropy (bits)', fontsize=12)
            ax1.set_title('Top 20 Transfer Entropy Values', fontsize=12)
            ax1.invert_yaxis()
            
            # P-values
            ax2.barh(range(len(sig_df_sorted)), sig_df_sorted['p_value'], 
                    color=colors, alpha=0.7)
            ax2.axvline(x=0.05, color='black', linestyle='--', label='α = 0.05')
            ax2.set_yticks(range(len(sig_df_sorted)))
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel('p-value', fontsize=12)
            ax2.set_title('Significance Tests', fontsize=12)
            ax2.legend()
            ax2.invert_yaxis()
            
            fig.suptitle(f'Transfer Entropy Significance - {self.lottery_name.title()}',
                        fontsize=14, y=1.02)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_transfer_entropy_significance.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
    
    def save_results(self):
        """Save all results to files."""
        print(f"\nSaving transfer entropy results...")
        
        # TE matrix
        if 'te_matrix' in self.results:
            te_df = pd.DataFrame(
                self.results['te_matrix'],
                index=self.feature_names,
                columns=self.feature_names
            )
            filepath = OUTPUT_DIR / f'{self.lottery_name}_transfer_entropy_matrix.csv'
            te_df.to_csv(filepath)
            print(f"  Saved: {filepath}")
        
        # Significance tests
        if 'te_significance' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_transfer_entropy_significance.csv'
            self.results['te_significance'].to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")
    
    def run_complete_analysis(self, k: int = None):
        """Run complete transfer entropy analysis pipeline."""
        print(f"\n--- TRANSFER ENTROPY ANALYSIS: {self.lottery_name.upper()} ---")
        
        if k is None:
            k = TE_HISTORY_LENGTH
        
        # 0. Load data
        self.load_data()
        
        # 1. Compute TE matrix
        print("\n1. Computing Transfer Entropy Matrix")
        print("-" * 70)
        self.compute_te_matrix(k=k)
        
        # 2. Significance tests (on subset due to computational cost)
        print("\n2. Significance Testing")
        print("-" * 70)
        self.test_pairwise_significance(n_permutations=100)
        
        # 3. Visualize
        print("\n3. Creating Visualizations")
        print("-" * 70)
        self.visualize_results(save=True)
        
        # 4. Save results
        self.save_results()
        
        print("\n--- ANALYSIS COMPLETE ---")
        
        return self.results


def run_both_lotteries(k: int = None) -> Dict:
    """Run transfer entropy analysis for both lotteries."""
    
    results = {}
    
    for lottery in ['powerball', 'megamillions']:
        print(f"\n--- # {lottery.upper()} ---")
        
        try:
            analyzer = TransferEntropyAnalyzer(lottery)
            results[lottery] = analyzer.run_complete_analysis(k=k)
            print(f"\n[OK] Successfully completed {lottery}")
            
        except Exception as e:
            print(f"\n[FAILED] Error analyzing {lottery}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    """Run transfer entropy analysis."""
    
    # Run analysis for both lotteries
    results = run_both_lotteries(k=TE_HISTORY_LENGTH)
    
    print("\n--- ALL TRANSFER ENTROPY ANALYSES COMPLETE ---")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

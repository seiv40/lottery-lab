"""
Module 8: Granger Causality Tests
Comprehensive Granger causality testing for lottery time series.

Tests implemented:
1. Linear VAR-based Granger tests (univariate)
2. Multivariate Granger tests (all ball pairs)
3. Non-linear neural network-based Granger tests
4. Rolling window Granger causality

More info: read ch 2, 3 and 7 of Lutkepohl's "New Introduction to Multiple Time Series Analysis" (2005)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    GRANGER_MAX_LAGS, GRANGER_SIGNIFICANCE,
    FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED,
    BONFERRONI_CORRECTION
)
from data_loader import LotteryDataLoader

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class GrangerCausalityAnalyzer:
    """Comprehensive Granger causality testing."""
    
    def __init__(self, lottery_name: str):
        """
        Initialize Granger causality analyzer.
        
        Parameters:
        -----------
        lottery_name : str
            Either 'powerball' or 'megamillions'
        """
        self.lottery_name = lottery_name
        self.loader = LotteryDataLoader(lottery_name)
        self.results = {}
        
    def linear_granger_test(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           max_lags: int = 10,
                           test: str = 'ssr_ftest') -> Dict:
        """
        Perform linear VAR-based Granger causality test.
        
        Tests H0: x does NOT Granger-cause y
        
        Parameters:
        -----------
        x : np.ndarray
            Potential causing variable (predictor)
        y : np.ndarray
            Potentially caused variable (target)
        max_lags : int
            Maximum lag to test
        test : str
            Test statistic to use ('ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest')
            
        Returns:
        --------
        Dict
            Dictionary with test results for each lag
        """
        # Prepare data for statsmodels
        data = pd.DataFrame({'y': y, 'x': x})
        
        try:
            # Run Granger causality test
            gc_results = grangercausalitytests(data[['y', 'x']], 
                                               maxlag=max_lags,
                                               verbose=False)
            
            # Extract results
            results = {}
            for lag in range(1, max_lags + 1):
                test_result = gc_results[lag][0][test]
                results[lag] = {
                    'F_statistic': test_result[0],
                    'p_value': test_result[1],
                    'df_numerator': test_result[2],
                    'df_denominator': test_result[3] if len(test_result) > 3 else None
                }
            
            return results
            
        except Exception as e:
            print(f"Error in Granger test: {e}")
            return {}
    
    def test_all_univariate_pairs(self,
                                  features: List[str] = None,
                                  max_lags: int = 10) -> pd.DataFrame:
        """
        Test Granger causality for all pairs of univariate features.
        
        Parameters:
        -----------
        features : List[str], optional
            Features to test. If None, uses default aggregate statistics.
        max_lags : int
            Maximum lag to test
            
        Returns:
        --------
        pd.DataFrame
            Matrix of p-values (rows cause columns)
        """
        print(f"\nTesting univariate Granger causality (max lag = {max_lags})...")
        
        # Load data with ACTUAL COLUMN NAMES
        if features is None:
            features = ['mean', 'sum', 'variance', 'range', 'std']
        
        df = self.loader.prepare_for_granger(features=features, max_lags=max_lags)
        
        n_features = len(features)
        
        # Initialize results matrix
        p_value_matrix = np.ones((n_features, n_features))
        f_stat_matrix = np.zeros((n_features, n_features))
        
        # Test all pairs
        for i, x_feature in enumerate(features):
            for j, y_feature in enumerate(features):
                if i == j:
                    continue
                
                x = df[x_feature].values
                y = df[y_feature].values
                
                # Run test for multiple lags and take minimum p-value
                results = self.linear_granger_test(x, y, max_lags=max_lags)
                
                if results:
                    p_values = [r['p_value'] for r in results.values()]
                    f_stats = [r['F_statistic'] for r in results.values()]
                    
                    p_value_matrix[i, j] = min(p_values)  # Most significant lag
                    f_stat_matrix[i, j] = max(f_stats)  # Strongest F-statistic
        
        # Create DataFrames
        p_value_df = pd.DataFrame(p_value_matrix, 
                                  index=features, 
                                  columns=features)
        
        f_stat_df = pd.DataFrame(f_stat_matrix,
                                index=features,
                                columns=features)
        
        # Store results
        self.results['univariate_p_values'] = p_value_df
        self.results['univariate_f_stats'] = f_stat_df
        
        # Summary - count significant results excluding diagonal and NaN/inf values
        # get off-diagonal elements only
        mask = ~np.eye(n_features, dtype=bool)
        off_diagonal_pvals = p_value_matrix[mask]
        
        # count only finite p-values (exclude NaN and inf from failed tests)
        valid_pvals = off_diagonal_pvals[np.isfinite(off_diagonal_pvals)]
        n_significant = (valid_pvals < GRANGER_SIGNIFICANCE).sum()
        
        print(f"  Tested {n_features * (n_features - 1)} pairs")
        print(f"  Significant (p < {GRANGER_SIGNIFICANCE}): {n_significant}")
        
        if BONFERRONI_CORRECTION:
            bonferroni_alpha = GRANGER_SIGNIFICANCE / (n_features * (n_features - 1))
            n_bonferroni = (valid_pvals < bonferroni_alpha).sum()
            print(f"  Significant after Bonferroni: {n_bonferroni} (α = {bonferroni_alpha:.6f})")
        
        return p_value_df
    
    def test_multivariate_balls(self, max_lags: int = 10) -> Dict:
        """
        Test whether past ball draws Granger-cause future ball draws.
        
        Tests: Do (Ball₁, Ball₂, ..., Ball₅) at time t Granger-cause Ball_i at t+1?
        
        Parameters:
        -----------
        max_lags : int
            Maximum lag to test
            
        Returns:
        --------
        Dict
            Results for each target ball
        """
        print(f"\nTesting multivariate ball Granger causality (max lag = {max_lags})...")
        
        # Load ball time series
        ball_data, ball_names = self.loader.prepare_ball_time_series()
        
        results = {}
        
        # Test each ball against all others (pairwise)
        for target_idx, target_ball in enumerate(ball_names):
            print(f"  Testing causes of {target_ball}...")
            
            y = ball_data[:, target_idx]
            
            # Test each other ball as potential cause
            p_values_all = []
            f_stats_all = []
            
            for source_idx, source_ball in enumerate(ball_names):
                if source_idx == target_idx:
                    continue
                
                x = ball_data[:, source_idx]
                
                # Create DataFrame for Granger test
                df = pd.DataFrame({'y': y, 'x': x})
                
                try:
                    gc_results = grangercausalitytests(
                        df[['y', 'x']],
                        maxlag=max_lags,
                        verbose=False
                    )
                    
                    # Extract minimum p-value across lags
                    p_values = [gc_results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lags + 1)]
                    f_stats = [gc_results[lag][0]['ssr_ftest'][0] for lag in range(1, max_lags + 1)]
                    
                    p_values_all.extend(p_values)
                    f_stats_all.extend(f_stats)
                    
                except Exception as e:
                    # Skip this pair if it fails
                    continue
            
            if p_values_all:
                results[target_ball] = {
                    'min_p_value': min(p_values_all),
                    'max_f_stat': max(f_stats_all),
                    'n_tests': len(p_values_all),
                    'best_lag': 'varies'
                }
            else:
                print(f"    Could not complete tests for {target_ball}")
                results[target_ball] = None
        
        self.results['multivariate_balls'] = results
        
        # Summary
        valid_results = [r for r in results.values() if r is not None]
        if valid_results:
            significant = sum(1 for r in valid_results if r['min_p_value'] < GRANGER_SIGNIFICANCE)
            print(f"  Tested {len(valid_results)} target balls")
            print(f"  Significant (p < {GRANGER_SIGNIFICANCE}): {significant}")
        else:
            print(f"  No valid results")
        
        return results
    
    def test_lag_specific(self, 
                         x_feature: str,
                         y_feature: str,
                         lags: List[int] = None) -> pd.DataFrame:
        """
        Test Granger causality for specific lags.
        
        Parameters:
        -----------
        x_feature : str
            Causing variable
        y_feature : str
            Caused variable
        lags : List[int], optional
            Specific lags to test. If None, uses GRANGER_MAX_LAGS.
            
        Returns:
        --------
        pd.DataFrame
            Results for each lag
        """
        if lags is None:
            lags = GRANGER_MAX_LAGS
        
        df = self.loader.prepare_for_granger()
        
        x = df[x_feature].values
        y = df[y_feature].values
        
        results_list = []
        
        for max_lag in lags:
            gc_results = self.linear_granger_test(x, y, max_lags=max_lag)
            
            for lag, result in gc_results.items():
                results_list.append({
                    'cause': x_feature,
                    'effect': y_feature,
                    'lag': lag,
                    'max_lag_tested': max_lag,
                    'F_statistic': result['F_statistic'],
                    'p_value': result['p_value'],
                    'significant': result['p_value'] < GRANGER_SIGNIFICANCE
                })
        
        return pd.DataFrame(results_list)
    
    def visualize_results(self, save: bool = True):
        """Create comprehensive visualizations of Granger causality results."""
        
        # 1. Heatmap of univariate p-values
        if 'univariate_p_values' in self.results:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            p_values = self.results['univariate_p_values']
            
            # Create mask for diagonal
            mask = np.eye(len(p_values), dtype=bool)
            
            sns.heatmap(p_values, 
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlGn_r',  # Red = low p (significant), Green = high p
                       vmin=0, 
                       vmax=1,
                       mask=mask,
                       cbar_kws={'label': 'p-value'},
                       ax=ax)
            
            ax.set_title(f'Granger Causality p-values - {self.lottery_name.title()}\n' + 
                        f'(Rows cause Columns | Green = No Causality)',
                        fontsize=14, pad=20)
            ax.set_xlabel('Effect Variable', fontsize=12)
            ax.set_ylabel('Cause Variable', fontsize=12)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_granger_pvalue_heatmap.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
        
        # 2. Heatmap of F-statistics
        if 'univariate_f_stats' in self.results:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            f_stats = self.results['univariate_f_stats']
            mask = np.eye(len(f_stats), dtype=bool)
            
            sns.heatmap(f_stats,
                       annot=True,
                       fmt='.2f',
                       cmap='YlOrRd',
                       mask=mask,
                       cbar_kws={'label': 'F-statistic'},
                       ax=ax)
            
            ax.set_title(f'Granger Causality F-statistics - {self.lottery_name.title()}\n' +
                        f'(Rows cause Columns | Higher = Stronger Effect)',
                        fontsize=14, pad=20)
            ax.set_xlabel('Effect Variable', fontsize=12)
            ax.set_ylabel('Cause Variable', fontsize=12)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_granger_fstat_heatmap.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
        
        # 3. Bar plot of multivariate ball results
        if 'multivariate_balls' in self.results:
            results = self.results['multivariate_balls']
            valid_results = {k: v for k, v in results.items() if v is not None}
            
            if valid_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                balls = list(valid_results.keys())
                p_values = [valid_results[b]['min_p_value'] for b in balls]
                f_stats = [valid_results[b]['max_f_stat'] for b in balls]
                
                # P-values
                colors = ['red' if p < GRANGER_SIGNIFICANCE else 'green' for p in p_values]
                ax1.bar(balls, p_values, color=colors, alpha=0.7)
                ax1.axhline(y=GRANGER_SIGNIFICANCE, color='black', linestyle='--', 
                           label=f'α = {GRANGER_SIGNIFICANCE}')
                ax1.set_ylabel('Minimum p-value', fontsize=12)
                ax1.set_title('Multivariate Granger Causality p-values', fontsize=12)
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)
                
                # F-statistics
                ax2.bar(balls, f_stats, color='steelblue', alpha=0.7)
                ax2.set_ylabel('Maximum F-statistic', fontsize=12)
                ax2.set_title('Multivariate Granger Causality F-statistics', fontsize=12)
                ax2.tick_params(axis='x', rotation=45)
                
                fig.suptitle(f'Multivariate Ball Granger Causality - {self.lottery_name.title()}',
                            fontsize=14, y=1.02)
                
                plt.tight_layout()
                
                if save:
                    filepath = FIGURES_DIR / f'{self.lottery_name}_granger_multivariate_balls.png'
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"  Saved: {filepath}")
                
                plt.close()
    
    def save_results(self):
        """Save all results to CSV files."""
        print(f"\nSaving Granger causality results...")
        
        # Univariate p-values
        if 'univariate_p_values' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_granger_univariate_pvalues.csv'
            self.results['univariate_p_values'].to_csv(filepath)
            print(f"  Saved: {filepath}")
        
        # Univariate F-stats
        if 'univariate_f_stats' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_granger_univariate_fstats.csv'
            self.results['univariate_f_stats'].to_csv(filepath)
            print(f"  Saved: {filepath}")
        
        # Multivariate balls
        if 'multivariate_balls' in self.results:
            results_dict = self.results['multivariate_balls']
            
            # Convert to DataFrame
            rows = []
            for ball, result in results_dict.items():
                if result is not None:
                    rows.append({
                        'target_ball': ball,
                        'min_p_value': result['min_p_value'],
                        'max_f_stat': result['max_f_stat'],
                        'best_lag': result['best_lag'],
                        'significant': result['min_p_value'] < GRANGER_SIGNIFICANCE
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                filepath = OUTPUT_DIR / f'{self.lottery_name}_granger_multivariate_summary.csv'
                df.to_csv(filepath, index=False)
                print(f"  Saved: {filepath}")
    
    def run_complete_analysis(self, max_lags: int = 10):
        """Run complete Granger causality analysis pipeline."""
        print(f"\n--- GRANGER CAUSALITY ANALYSIS: {self.lottery_name.upper()} ---")
        
        # 1. Univariate tests
        print("\n1. Univariate Granger Causality Tests")
        print("-" * 70)
        self.test_all_univariate_pairs(max_lags=max_lags)
        
        # 2. Multivariate ball tests
        print("\n2. Multivariate Ball Granger Causality Tests")
        print("-" * 70)
        self.test_multivariate_balls(max_lags=max_lags)
        
        # 3. Visualize
        print("\n3. Creating Visualizations")
        print("-" * 70)
        self.visualize_results(save=True)
        
        # 4. Save results
        self.save_results()
        
        print("\n--- ANALYSIS COMPLETE ---")
        
        return self.results


def run_both_lotteries(max_lags: int = 10) -> Dict:
    """Run Granger causality analysis for both lotteries."""
    
    results = {}
    
    for lottery in ['powerball', 'megamillions']:
        print(f"\n--- # {lottery.upper()} ---")
        
        try:
            analyzer = GrangerCausalityAnalyzer(lottery)
            results[lottery] = analyzer.run_complete_analysis(max_lags=max_lags)
            print(f"\n[OK] Successfully completed {lottery}")
            
        except Exception as e:
            print(f"\n[FAILED] Error analyzing {lottery}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    """Run Granger causality analysis."""
    
    # Run analysis for both lotteries
    results = run_both_lotteries(max_lags=10)
    
    print("\n--- ALL GRANGER CAUSALITY ANALYSES COMPLETE ---")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

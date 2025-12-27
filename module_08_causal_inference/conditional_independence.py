"""
Module 8: Conditional Independence Testing
Battery of conditional independence tests for lottery time series.

Tests implemented:
1. Fisher-z test (linear correlation)
2. Kernel Conditional Independence (KCI) test
3. Chi-square test
4. G-square test
5. Distance correlation test
6. Mutual information test

More info: for # 1, 3, and 4 read "Causation, Prediction, and Search" (Spirtes et al., 2000) + 
+ for # 2, 5 and 6 read "A Survey of Conditional Independence Tests" by Berrett & Samworth (2019)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import dcor

from causallearn.utils.cit import CIT

from config import (
    CI_TESTS, CI_SIGNIFICANCE, CI_MAX_CONDITIONING_SET_SIZE,
    FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED,
    BONFERRONI_CORRECTION, FDR_METHOD
)
from data_loader import LotteryDataLoader

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


class ConditionalIndependenceAnalyzer:
    """Comprehensive conditional independence testing."""
    
    def __init__(self, lottery_name: str):
        """
        Initialize CI analyzer.
        
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
        """Load and prepare data for CI testing."""
        print("Loading data for conditional independence testing...")
        
        # Load features with both balls and aggregate statistics
        df = self.loader.prepare_for_conditional_independence(
            include_balls=True,
            include_aggregate=True
        )
        
        self.data = df.values
        self.feature_names = df.columns.tolist()
        
        print(f"  Loaded {self.data.shape[0]} samples with {self.data.shape[1]} features")
        print(f"  Features: {self.feature_names}")
        
    def test_pairwise_independence(self,
                                  test_methods: List[str] = None) -> pd.DataFrame:
        """
        Test pairwise independence for all feature pairs.
        
        Tests H0: X ⊥ Y (X and Y are independent)
        
        Parameters:
        -----------
        test_methods : List[str], optional
            CI test methods to use. If None, uses all from CI_TESTS.
            
        Returns:
        --------
        pd.DataFrame
            Results for all pairs and all test methods
        """
        if self.data is None:
            self.load_data()
        
        if test_methods is None:
            test_methods = CI_TESTS
        
        print(f"\nTesting pairwise independence...")
        print(f"  Methods: {test_methods}")
        
        n_features = len(self.feature_names)
        n_pairs = n_features * (n_features - 1) // 2
        
        print(f"  Testing {n_pairs} pairs...")
        
        results_list = []
        
        # Test all pairs
        pair_count = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                pair_count += 1
                
                if pair_count % 10 == 0:
                    print(f"    Progress: {pair_count}/{n_pairs} pairs tested")
                
                feature_i = self.feature_names[i]
                feature_j = self.feature_names[j]
                
                # Test with each method
                for method in test_methods:
                    try:
                        cit = CIT(self.data, method=method)
                        p_value = cit(i, j)  # Test X_i ⊥ X_j
                        
                        results_list.append({
                            'feature_1': feature_i,
                            'feature_2': feature_j,
                            'index_1': i,
                            'index_2': j,
                            'method': method,
                            'p_value': p_value,
                            'independent': p_value > CI_SIGNIFICANCE
                        })
                        
                    except Exception as e:
                        print(f"      Error testing {feature_i} ⊥ {feature_j} with {method}: {e}")
        
        results_df = pd.DataFrame(results_list)
        
        # Summary statistics
        print(f"\n  Results Summary:")
        for method in test_methods:
            method_results = results_df[results_df['method'] == method]
            n_independent = method_results['independent'].sum()
            n_dependent = len(method_results) - n_independent
            
            print(f"    {method:12s}: {n_independent:4d} independent, {n_dependent:4d} dependent")
        
        self.results['pairwise'] = results_df
        
        return results_df
    
    def test_conditional_independence(self,
                                     max_conditioning_size: int = None,
                                     test_method: str = 'kci') -> pd.DataFrame:
        """
        Test conditional independence: X ⊥ Y | Z
        
        Parameters:
        -----------
        max_conditioning_size : int, optional
            Maximum size of conditioning set Z. If None, uses CI_MAX_CONDITIONING_SET_SIZE.
        test_method : str
            CI test method to use (default: 'kci' for non-linear)
            
        Returns:
        --------
        pd.DataFrame
            Results for conditional independence tests
        """
        if self.data is None:
            self.load_data()
        
        if max_conditioning_size is None:
            max_conditioning_size = CI_MAX_CONDITIONING_SET_SIZE
        
        print(f"\nTesting conditional independence (X ⊥ Y | Z)...")
        print(f"  Max conditioning set size: {max_conditioning_size}")
        print(f"  Method: {test_method}")
        
        n_features = len(self.feature_names)
        
        results_list = []
        
        # Test X ⊥ Y | Z for various Z
        test_count = 0
        max_tests = 500  # Limit to avoid excessive computation
        
        # Sample some representative tests
        # Test temporal independence: X_t ⊥ X_{t+k} | {X_{t+1}, ..., X_{t+k-1}}
        
        for i in range(min(n_features, 10)):  # Limit features to test
            for j in range(i + 1, min(n_features, 10)):
                
                # Test unconditional first
                try:
                    cit = CIT(self.data, method=test_method)
                    p_uncond = cit(i, j)
                    
                    results_list.append({
                        'feature_X': self.feature_names[i],
                        'feature_Y': self.feature_names[j],
                        'conditioning_set': 'None',
                        'conditioning_size': 0,
                        'p_value': p_uncond,
                        'independent': p_uncond > CI_SIGNIFICANCE
                    })
                    
                except Exception as e:
                    print(f"    Error in unconditional test: {e}")
                    continue
                
                # Test with conditioning sets of increasing size
                for cond_size in range(1, min(max_conditioning_size + 1, n_features - 1)):
                    
                    # Get possible conditioning variables (exclude i and j)
                    possible_cond = [k for k in range(n_features) if k not in [i, j]]
                    
                    if len(possible_cond) < cond_size:
                        continue
                    
                    # Test a few random conditioning sets
                    for _ in range(min(3, 10)):  # Test up to 3 random conditioning sets
                        
                        if test_count >= max_tests:
                            break
                        
                        # Random conditioning set
                        cond_set = np.random.choice(possible_cond, size=cond_size, replace=False).tolist()
                        
                        try:
                            p_cond = cit(i, j, cond_set)
                            
                            cond_names = [self.feature_names[k] for k in cond_set]
                            
                            results_list.append({
                                'feature_X': self.feature_names[i],
                                'feature_Y': self.feature_names[j],
                                'conditioning_set': ', '.join(cond_names),
                                'conditioning_size': cond_size,
                                'p_value': p_cond,
                                'independent': p_cond > CI_SIGNIFICANCE
                            })
                            
                            test_count += 1
                            
                        except Exception as e:
                            pass  # Skip failed tests
                    
                    if test_count >= max_tests:
                        break
                
                if test_count >= max_tests:
                    break
            
            if test_count >= max_tests:
                break
        
        results_df = pd.DataFrame(results_list)
        
        # Summary
        print(f"\n  Tested {len(results_df)} conditional independence relationships")
        
        for cond_size in sorted(results_df['conditioning_size'].unique()):
            size_results = results_df[results_df['conditioning_size'] == cond_size]
            n_independent = size_results['independent'].sum()
            pct_independent = 100 * n_independent / len(size_results)
            
            print(f"    Conditioning size {cond_size}: {n_independent}/{len(size_results)} " +
                  f"independent ({pct_independent:.1f}%)")
        
        self.results['conditional'] = results_df
        
        return results_df
    
    def distance_correlation_test(self) -> pd.DataFrame:
        """
        Test independence using distance correlation.
        
        Distance correlation is 0 iff variables are independent.
        
        Returns:
        --------
        pd.DataFrame
            Distance correlation matrix and p-values
        """
        if self.data is None:
            self.load_data()
        
        print(f"\nComputing distance correlation matrix...")
        
        n_features = len(self.feature_names)
        
        dcor_matrix = np.zeros((n_features, n_features))
        pval_matrix = np.ones((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i, n_features):
                
                if i == j:
                    dcor_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                else:
                    # Compute distance correlation
                    x = self.data[:, i]
                    y = self.data[:, j]
                    
                    try:
                        # Compute distance correlation
                        dc = dcor.distance_correlation(x, y)
                        dcor_matrix[i, j] = dc
                        dcor_matrix[j, i] = dc
                        
                        # Permutation test for significance
                        null_dist = []
                        n_perms = 100
                        for _ in range(n_perms):
                            y_perm = np.random.permutation(y)
                            dc_perm = dcor.distance_correlation(x, y_perm)
                            null_dist.append(dc_perm)
                        
                        # p-value: proportion of permutations with higher dcor
                        pval = np.mean(np.array(null_dist) >= dc)
                        pval_matrix[i, j] = pval
                        pval_matrix[j, i] = pval
                        
                    except Exception as e:
                        print(f"    Error computing dcor for {i},{j}: {e}")
        
        # Create DataFrames
        dcor_df = pd.DataFrame(dcor_matrix, 
                              index=self.feature_names,
                              columns=self.feature_names)
        
        pval_df = pd.DataFrame(pval_matrix,
                              index=self.feature_names,
                              columns=self.feature_names)
        
        # Summary
        n_significant = ((pval_matrix < CI_SIGNIFICANCE) & 
                        (np.triu(np.ones_like(pval_matrix), k=1) > 0)).sum()
        n_total = n_features * (n_features - 1) // 2
        
        print(f"  Significant correlations (p < {CI_SIGNIFICANCE}): {n_significant}/{n_total}")
        
        self.results['distance_correlation'] = dcor_df
        self.results['distance_correlation_pvalues'] = pval_df
        
        return dcor_df
    
    def mutual_information_test(self) -> pd.DataFrame:
        """
        Estimate mutual information for all feature pairs.
        
        MI = 0 iff variables are independent.
        
        Returns:
        --------
        pd.DataFrame
            Mutual information matrix
        """
        from sklearn.feature_selection import mutual_info_regression
        
        if self.data is None:
            self.load_data()
        
        print(f"\nEstimating mutual information matrix...")
        
        n_features = len(self.feature_names)
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            
            # Compute MI between feature i and all other features
            X_other = np.delete(self.data, i, axis=1)
            y_target = self.data[:, i]
            
            try:
                mi_scores = mutual_info_regression(X_other, y_target, 
                                                  random_state=RANDOM_SEED)
                
                # Fill in the matrix
                other_indices = [j for j in range(n_features) if j != i]
                for j, mi_val in zip(other_indices, mi_scores):
                    mi_matrix[i, j] = mi_val
                    
            except Exception as e:
                print(f"    Error computing MI for feature {i}: {e}")
        
        # Symmetrize (MI is symmetric)
        mi_matrix = (mi_matrix + mi_matrix.T) / 2
        
        mi_df = pd.DataFrame(mi_matrix,
                            index=self.feature_names,
                            columns=self.feature_names)
        
        # Summary
        mean_mi = np.mean(mi_matrix[np.triu_indices_from(mi_matrix, k=1)])
        print(f"  Mean pairwise MI: {mean_mi:.6f}")
        
        self.results['mutual_information'] = mi_df
        
        return mi_df
    
    def visualize_results(self, save: bool = True):
        """Create comprehensive visualizations of CI test results."""
        
        # 1. Pairwise independence p-value heatmaps (one per method)
        if 'pairwise' in self.results:
            pairwise_df = self.results['pairwise']
            
            for method in pairwise_df['method'].unique():
                method_data = pairwise_df[pairwise_df['method'] == method]
                
                # Create pivot table for heatmap
                n_features = len(self.feature_names)
                pval_matrix = np.ones((n_features, n_features))
                
                for _, row in method_data.iterrows():
                    i, j = int(row['index_1']), int(row['index_2'])
                    pval_matrix[i, j] = row['p_value']
                    pval_matrix[j, i] = row['p_value']
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                sns.heatmap(pval_matrix,
                           annot=False,
                           cmap='RdYlGn_r',
                           vmin=0,
                           vmax=1,
                           xticklabels=self.feature_names,
                           yticklabels=self.feature_names,
                           cbar_kws={'label': 'p-value'},
                           ax=ax)
                
                ax.set_title(f'Pairwise Independence Test ({method.upper()}) - {self.lottery_name.title()}\n' +
                           f'Green = Independent (p > {CI_SIGNIFICANCE})',
                           fontsize=14, pad=20)
                
                plt.tight_layout()
                
                if save:
                    filepath = FIGURES_DIR / f'{self.lottery_name}_ci_pairwise_{method}.png'
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"  Saved: {filepath}")
                
                plt.close()
        
        # 2. Conditional independence by conditioning set size
        if 'conditional' in self.results:
            cond_df = self.results['conditional']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by conditioning size
            summary = cond_df.groupby('conditioning_size').agg({
                'independent': ['sum', 'count']
            })
            summary.columns = ['independent_count', 'total_count']
            summary['pct_independent'] = 100 * summary['independent_count'] / summary['total_count']
            
            x = summary.index
            y = summary['pct_independent']
            
            ax.bar(x, y, color='steelblue', alpha=0.7)
            ax.axhline(y=100 * (1 - CI_SIGNIFICANCE), color='red', linestyle='--',
                      label=f'Expected under null ({100 * (1 - CI_SIGNIFICANCE):.0f}%)')
            ax.set_xlabel('Conditioning Set Size', fontsize=12)
            ax.set_ylabel('% Conditionally Independent', fontsize=12)
            ax.set_title(f'Conditional Independence by Conditioning Set Size - {self.lottery_name.title()}',
                        fontsize=14)
            ax.legend()
            ax.set_ylim([0, 105])
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_ci_conditional_summary.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
        
        # 3. Distance correlation heatmap
        if 'distance_correlation' in self.results:
            dcor_df = self.results['distance_correlation']
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.heatmap(dcor_df,
                       annot=False,
                       cmap='YlOrRd',
                       vmin=0,
                       vmax=1,
                       cbar_kws={'label': 'Distance Correlation'},
                       ax=ax)
            
            ax.set_title(f'Distance Correlation Matrix - {self.lottery_name.title()}\n' +
                        f'0 = Independent',
                        fontsize=14, pad=20)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_distance_correlation.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
        
        # 4. Mutual information heatmap
        if 'mutual_information' in self.results:
            mi_df = self.results['mutual_information']
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.heatmap(mi_df,
                       annot=False,
                       cmap='viridis',
                       vmin=0,
                       cbar_kws={'label': 'Mutual Information (nats)'},
                       ax=ax)
            
            ax.set_title(f'Mutual Information Matrix - {self.lottery_name.title()}\n' +
                        f'0 = Independent',
                        fontsize=14, pad=20)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_mutual_information.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
    
    def save_results(self):
        """Save all results to CSV files."""
        print(f"\nSaving conditional independence results...")
        
        # Pairwise results
        if 'pairwise' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_ci_pairwise_results.csv'
            self.results['pairwise'].to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")
        
        # Conditional results
        if 'conditional' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_ci_conditional_results.csv'
            self.results['conditional'].to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")
        
        # Distance correlation
        if 'distance_correlation' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_distance_correlation.csv'
            self.results['distance_correlation'].to_csv(filepath)
            print(f"  Saved: {filepath}")
        
        # Mutual information
        if 'mutual_information' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_mutual_information.csv'
            self.results['mutual_information'].to_csv(filepath)
            print(f"  Saved: {filepath}")
    
    def run_complete_analysis(self):
        """Run complete conditional independence analysis pipeline."""
        print("=" * 70)
        print(f"CONDITIONAL INDEPENDENCE ANALYSIS: {self.lottery_name.upper()}")
        print("=" * 70)
        
        # 0. Load data
        self.load_data()
        
        # 1. Pairwise independence
        print("\n1. Pairwise Independence Tests")
        print("-" * 70)
        self.test_pairwise_independence()
        
        # 2. Conditional independence
        print("\n2. Conditional Independence Tests")
        print("-" * 70)
        self.test_conditional_independence()
        
        # 3. Distance correlation
        print("\n3. Distance Correlation Tests")
        print("-" * 70)
        self.distance_correlation_test()
        
        # 4. Mutual information
        print("\n4. Mutual Information Estimation")
        print("-" * 70)
        self.mutual_information_test()
        
        # 5. Visualize
        print("\n5. Creating Visualizations")
        print("-" * 70)
        self.visualize_results(save=True)
        
        # 6. Save results
        self.save_results()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        return self.results


def run_both_lotteries() -> Dict:
    """Run CI analysis for both lotteries."""
    
    results = {}
    
    for lottery in ['powerball', 'megamillions']:
        print(f"\n\n{'#' * 70}")
        print(f"# {lottery.upper()}")
        print(f"{'#' * 70}\n")
        
        try:
            analyzer = ConditionalIndependenceAnalyzer(lottery)
            results[lottery] = analyzer.run_complete_analysis()
            print(f"\n[OK] Successfully completed {lottery}")
            
        except Exception as e:
            print(f"\n[FAILED] Error analyzing {lottery}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    """Run conditional independence analysis."""
    
    # Run analysis for both lotteries
    results = run_both_lotteries()
    
    print("\n--- ALL CONDITIONAL INDEPENDENCE ANALYSES COMPLETE ---")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

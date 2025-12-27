"""
Module 8: Causal Invariance Testing
Test whether causal relationships are invariant across different environments.

Invariant Causal Prediction (ICP) theory: True causal parents of Y remain
predictive across all environments, while spurious correlations break down.

For lottery data, we expect NO invariant predictors, confirming no causal structure.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from itertools import combinations

from config import (
    ICP_ALPHA, ICP_ENVIRONMENTS,
    FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED
)
from data_loader import LotteryDataLoader

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


class CausalInvarianceAnalyzer:
    """Test causal invariance across environments."""
    
    def __init__(self, lottery_name: str):
        """
        Initialize causal invariance analyzer.
        
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
        self.environments = None
        
    def load_data(self):
        """Load and prepare data for causal invariance testing."""
        print("Loading data for causal invariance testing...")
        
        # Load features
        df = self.loader.prepare_for_conditional_independence(
            include_balls=True,
            include_aggregate=True
        )
        
        self.data = df.values
        self.feature_names = df.columns.tolist()
        
        print(f"  Loaded {self.data.shape[0]} samples with {self.data.shape[1]} features")
    
    def create_environments(self, 
                          method: str = 'temporal',
                          n_envs: int = None) -> np.ndarray:
        """
        Create environment labels for ICP.
        
        Parameters:
        -----------
        method : str
            Method for creating environments:
            - 'temporal': Split data chronologically
            - 'random': Random assignment
            - 'weekday': Based on day of week (if available)
        n_envs : int, optional
            Number of environments
            
        Returns:
        --------
        np.ndarray
            Environment labels for each sample
        """
        if n_envs is None:
            n_envs = ICP_ENVIRONMENTS
        
        n_samples = self.data.shape[0]
        
        print(f"  Creating {n_envs} environments using '{method}' method...")
        
        if method == 'temporal':
            # Split chronologically
            env_labels = np.zeros(n_samples, dtype=int)
            
            split_size = n_samples // n_envs
            for i in range(n_envs):
                start = i * split_size
                end = start + split_size if i < n_envs - 1 else n_samples
                env_labels[start:end] = i
                
                print(f"    Environment {i}: samples {start}-{end} ({end-start} samples)")
        
        elif method == 'random':
            # Random assignment
            env_labels = np.random.randint(0, n_envs, size=n_samples)
            
            for i in range(n_envs):
                n_in_env = np.sum(env_labels == i)
                print(f"    Environment {i}: {n_in_env} samples")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.environments = env_labels
        return env_labels
    
    def test_invariance_single_predictor_set(self,
                                            target_idx: int,
                                            predictor_indices: List[int],
                                            alpha: float = None) -> Dict:
        """
        Test whether a predictor set is invariant across environments.
        
        H0: The residuals are identically distributed across all environments
        
        Parameters:
        -----------
        target_idx : int
            Index of target variable
        predictor_indices : List[int]
            Indices of predictor variables
        alpha : float, optional
            Significance level
            
        Returns:
        --------
        Dict
            Test results including p-value and invariance decision
        """
        if alpha is None:
            alpha = ICP_ALPHA
        
        if self.environments is None:
            self.create_environments()
        
        target = self.data[:, target_idx]
        predictors = self.data[:, predictor_indices]
        
        # Fit model in each environment and collect residuals
        residuals_by_env = []
        mse_by_env = []
        
        unique_envs = np.unique(self.environments)
        
        for env in unique_envs:
            mask = self.environments == env
            
            X_env = predictors[mask]
            y_env = target[mask]
            
            if len(y_env) < 10:  # Skip if too few samples
                continue
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X_env, y_env)
            
            # Compute residuals
            y_pred = model.predict(X_env)
            residuals = y_env - y_pred
            
            residuals_by_env.append(residuals)
            mse_by_env.append(mean_squared_error(y_env, y_pred))
        
        if len(residuals_by_env) < 2:
            return {'invariant': False, 'p_value': 1.0, 'test': 'insufficient_data'}
        
        # Test 1: Levene's test for equal variances
        levene_stat, levene_p = stats.levene(*residuals_by_env)
        
        # Test 2: Kruskal-Wallis test for equal distributions
        kw_stat, kw_p = stats.kruskal(*residuals_by_env)
        
        # Test 3: Compare MSEs across environments
        mse_var = np.var(mse_by_env)
        
        # Decision: Invariant if both tests fail to reject
        invariant = (levene_p > alpha) and (kw_p > alpha)
        
        return {
            'target': self.feature_names[target_idx],
            'predictors': [self.feature_names[i] for i in predictor_indices],
            'n_predictors': len(predictor_indices),
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'kruskal_statistic': kw_stat,
            'kruskal_p_value': kw_p,
            'mse_variance': mse_var,
            'mse_by_env': mse_by_env,
            'invariant': invariant,
            'alpha': alpha
        }
    
    def search_invariant_predictors(self,
                                   target_idx: int,
                                   max_predictor_set_size: int = 3) -> pd.DataFrame:
        """
        Search for invariant predictor sets for a target variable.
        
        Tests all possible predictor subsets up to max_predictor_set_size.
        
        Parameters:
        -----------
        target_idx : int
            Index of target variable
        max_predictor_set_size : int
            Maximum size of predictor sets to test
            
        Returns:
        --------
        pd.DataFrame
            Results for all tested predictor sets
        """
        print(f"\n  Searching for invariant predictors of {self.feature_names[target_idx]}...")
        
        # Get all possible predictors (exclude target)
        possible_predictors = [i for i in range(self.data.shape[1]) if i != target_idx]
        
        results_list = []
        
        # Test predictor sets of increasing size
        for size in range(1, min(max_predictor_set_size + 1, len(possible_predictors) + 1)):
            
            # Limit to avoid combinatorial explosion
            max_combinations = 100
            all_combos = list(combinations(possible_predictors, size))
            
            if len(all_combos) > max_combinations:
                print(f"    Size {size}: Testing {max_combinations} random combinations (of {len(all_combos)} total)")
                combos_to_test = [all_combos[i] for i in 
                                 np.random.choice(len(all_combos), max_combinations, replace=False)]
            else:
                print(f"    Size {size}: Testing {len(all_combos)} combinations")
                combos_to_test = all_combos
            
            for predictor_set in combos_to_test:
                result = self.test_invariance_single_predictor_set(
                    target_idx,
                    list(predictor_set)
                )
                results_list.append(result)
        
        results_df = pd.DataFrame(results_list)
        
        # Summary
        if len(results_df) > 0:
            n_invariant = results_df['invariant'].sum()
            print(f"    Total sets tested: {len(results_df)}")
            print(f"    Invariant sets: {n_invariant}")
        
        return results_df
    
    def test_all_targets(self,
                        target_indices: List[int] = None,
                        max_predictor_set_size: int = 2) -> Dict:
        """
        Test invariance for multiple target variables.
        
        Parameters:
        -----------
        target_indices : List[int], optional
            Indices of targets to test. If None, tests a subset.
        max_predictor_set_size : int
            Maximum predictor set size
            
        Returns:
        --------
        Dict
            Results for all targets
        """
        if self.data is None:
            self.load_data()
        
        if self.environments is None:
            self.create_environments()
        
        if target_indices is None:
            # Test first 5 features as targets
            target_indices = list(range(min(5, self.data.shape[1])))
        
        print(f"\nTesting causal invariance for {len(target_indices)} targets...")
        
        results_by_target = {}
        
        for target_idx in target_indices:
            results_df = self.search_invariant_predictors(
                target_idx,
                max_predictor_set_size=max_predictor_set_size
            )
            results_by_target[self.feature_names[target_idx]] = results_df
        
        self.results['by_target'] = results_by_target
        
        return results_by_target
    
    def compare_environments(self) -> pd.DataFrame:
        """
        Compare statistical properties across environments.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics for each environment
        """
        if self.data is None:
            self.load_data()
        
        if self.environments is None:
            self.create_environments()
        
        print("\nComparing environments...")
        
        unique_envs = np.unique(self.environments)
        
        comparison_data = []
        
        for env in unique_envs:
            mask = self.environments == env
            data_env = self.data[mask]
            
            comparison_data.append({
                'environment': env,
                'n_samples': np.sum(mask),
                'mean': np.mean(data_env),
                'std': np.std(data_env),
                'median': np.median(data_env),
                'min': np.min(data_env),
                'max': np.max(data_env)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n  Environment Statistics:")
        print(comparison_df.to_string(index=False))
        
        self.results['environment_comparison'] = comparison_df
        
        return comparison_df
    
    def visualize_results(self, save: bool = True):
        """Create visualizations of causal invariance results."""
        
        # 1. Environment comparison
        if 'environment_comparison' in self.results:
            comp_df = self.results['environment_comparison']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Sample sizes
            axes[0, 0].bar(comp_df['environment'], comp_df['n_samples'], 
                          color='steelblue', alpha=0.7)
            axes[0, 0].set_xlabel('Environment')
            axes[0, 0].set_ylabel('Number of Samples')
            axes[0, 0].set_title('Environment Sample Sizes')
            
            # Means
            axes[0, 1].bar(comp_df['environment'], comp_df['mean'],
                          color='coral', alpha=0.7)
            axes[0, 1].set_xlabel('Environment')
            axes[0, 1].set_ylabel('Mean Value')
            axes[0, 1].set_title('Environment Means')
            
            # Standard deviations
            axes[1, 0].bar(comp_df['environment'], comp_df['std'],
                          color='seagreen', alpha=0.7)
            axes[1, 0].set_xlabel('Environment')
            axes[1, 0].set_ylabel('Standard Deviation')
            axes[1, 0].set_title('Environment Standard Deviations')
            
            # Ranges
            axes[1, 1].bar(comp_df['environment'], 
                          comp_df['max'] - comp_df['min'],
                          color='purple', alpha=0.7)
            axes[1, 1].set_xlabel('Environment')
            axes[1, 1].set_ylabel('Range')
            axes[1, 1].set_title('Environment Ranges')
            
            fig.suptitle(f'Environment Comparison - {self.lottery_name.title()}',
                        fontsize=14, y=0.995)
            
            plt.tight_layout()
            
            if save:
                filepath = FIGURES_DIR / f'{self.lottery_name}_icp_environment_comparison.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            
            plt.close()
        
        # 2. Invariance test results summary
        if 'by_target' in self.results:
            
            # results
            all_results = []
            for target, results_df in self.results['by_target'].items():
                if len(results_df) > 0:
                    all_results.append({
                        'target': target,
                        'total_tests': len(results_df),
                        'invariant_sets': results_df['invariant'].sum(),
                        'pct_invariant': 100 * results_df['invariant'].sum() / len(results_df)
                    })
            
            if all_results:
                summary_df = pd.DataFrame(all_results)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = range(len(summary_df))
                ax.bar(x, summary_df['pct_invariant'], color='steelblue', alpha=0.7)
                ax.axhline(y=5, color='red', linestyle='--', 
                          label='Expected under null (5%)')
                ax.set_xticks(x)
                ax.set_xticklabels(summary_df['target'], rotation=45, ha='right')
                ax.set_ylabel('% Invariant Predictor Sets')
                ax.set_title(f'Causal Invariance Results - {self.lottery_name.title()}\n' +
                           f'(Low % = No causal structure)',
                           fontsize=14)
                ax.legend()
                ax.set_ylim([0, max(10, summary_df['pct_invariant'].max() * 1.1)])
                
                plt.tight_layout()
                
                if save:
                    filepath = FIGURES_DIR / f'{self.lottery_name}_icp_invariance_summary.png'
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"  Saved: {filepath}")
                
                plt.close()
    
    def save_results(self):
        """Save all results to files."""
        print(f"\nSaving causal invariance results...")
        
        # Environment comparison
        if 'environment_comparison' in self.results:
            filepath = OUTPUT_DIR / f'{self.lottery_name}_icp_environment_comparison.csv'
            self.results['environment_comparison'].to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")
        
        # Results by target
        if 'by_target' in self.results:
            for target, results_df in self.results['by_target'].items():
                safe_target = target.replace('/', '_')
                filepath = OUTPUT_DIR / f'{self.lottery_name}_icp_{safe_target}_results.csv'
                results_df.to_csv(filepath, index=False)
                print(f"  Saved: {filepath}")
    
    def run_complete_analysis(self):
        """Run complete causal invariance analysis pipeline."""
        print(f"\n--- CAUSAL INVARIANCE ANALYSIS: {self.lottery_name.upper()} ---")
        
        # 0. Load data
        self.load_data()
        
        # 1. Create environments
        print("\n1. Creating Environments")
        print("-" * 70)
        self.create_environments(method='temporal', n_envs=ICP_ENVIRONMENTS)
        
        # 2. Compare environments
        print("\n2. Comparing Environments")
        print("-" * 70)
        self.compare_environments()
        
        # 3. Test invariance for multiple targets
        print("\n3. Testing Causal Invariance")
        print("-" * 70)
        self.test_all_targets(max_predictor_set_size=2)
        
        # 4. Visualize
        print("\n4. Creating Visualizations")
        print("-" * 70)
        self.visualize_results(save=True)
        
        # 5. Save results
        self.save_results()
        
        print("\n--- ANALYSIS COMPLETE ---")
        
        return self.results


def run_both_lotteries() -> Dict:
    """Run causal invariance analysis for both lotteries."""
    
    results = {}
    
    for lottery in ['powerball', 'megamillions']:
        print(f"# {lottery.upper()}")
        
        try:
            analyzer = CausalInvarianceAnalyzer(lottery)
            results[lottery] = analyzer.run_complete_analysis()
            print(f"\n[OK] Successfully completed {lottery}")
            
        except Exception as e:
            print(f"\n[FAILED] Error analyzing {lottery}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    """Run causal invariance analysis."""
    
    # Run analysis for both lotteries
    results = run_both_lotteries()
    
    print("\n--- ALL CAUSAL INVARIANCE ANALYSES COMPLETE ---")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

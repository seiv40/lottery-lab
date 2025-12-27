"""
MODULE 9 - STATISTICAL HYPOTHESIS TESTING
Lottery Lab: Network Analysis of Lottery Structure

Statistical tests::
1. Permutation tests on network metrics
2. Bootstrap confidence intervals
3. Graphlet correlation distance (GCD)
4. QAP (Quadratic Assignment Procedure) tests
5. Multiple testing corrections (Bonferroni, FDR)
6. Effect size calculations
7. Power analysis

Tests whether lottery graphs are distinguishable from null models.

More info: read ch 4 and 5 of Kolaczyk's "Statistical Analysis of Network Data with R" (2020); it is in R,
so just translate the logic into Python

"""

import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


class StatisticalTester:
    """
    Comprehensive statistical testing for lottery graphs vs null models.
    """
    
    def __init__(self, lottery_name: str, data_dir: Path, output_dir: Path,
                 alpha: float = 0.05):
        """
        Initialize statistical tester.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        data_dir : Path
            Directory containing lottery graphs and null models
        output_dir : Path
            Directory for saving test results
        alpha : float
            Significance level for hypothesis tests
        """
        self.lottery_name = lottery_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        
        self.lottery_metrics = {}
        self.null_metrics = {}
        self.test_results = []
        
        print(f"Initialized statistical tester for {lottery_name}")
        print(f"  Significance level: α = {alpha}")
    
    def load_data(self):
        """Load lottery metrics and null model metrics."""
        print("\nLoading data...")
        
        # Load lottery graph metrics
        lottery_metric_files = list(self.data_dir.glob(f"{self.lottery_name}_*_metrics.csv"))
        
        for filepath in lottery_metric_files:
            # Extract graph name
            parts = filepath.stem.replace(f"{self.lottery_name}_", "").replace("_metrics", "")
            
            # Skip certain files
            if 'null' in parts or 'centrality_distributions' in parts:
                continue
            
            df = pd.read_csv(filepath)
            self.lottery_metrics[parts] = df
            print(f"  Loaded lottery metrics: {parts}")
        
        # Load null model metrics
        null_metric_files = list(self.data_dir.glob(f"{self.lottery_name}_*_nullmetrics.csv"))

        for filepath in null_metric_files:
            # Extract graph name and model name
            parts = filepath.stem.replace(f"{self.lottery_name}_", "").replace("_nullmetrics", "")
            try:
                df = pd.read_csv(filepath)
            except EmptyDataError:
                print(f"  Warning: {filepath.name} is empty. Skipping this null metrics file.")
                continue

            if df.shape[0] == 0 or df.shape[1] == 0:
                print(f"  Warning: {filepath.name} has no data. Skipping.")
                continue

            self.null_metrics[parts] = df
            print(f"  Loaded null metrics: {parts}")
        
        print(f"\nLoaded {len(self.lottery_metrics)} lottery metric sets")
        print(f"Loaded {len(self.null_metrics)} null model metric sets")
    
    def run_all_tests(self):
        """Run comprehensive statistical testing framework."""
        print("\n--- RUNNING STATISTICAL HYPOTHESIS TESTS ---")
        
        # Test 1: Permutation tests on individual metrics
        print("\n[TEST 1] Permutation Tests on Network Metrics")
        self._run_permutation_tests()
        
        # Test 2: Mann-Whitney U tests (non-parametric)
        print("\n[TEST 2] Mann-Whitney U Tests")
        self._run_mann_whitney_tests()
        
        # Test 3: Kolmogorov-Smirnov tests (distribution comparison)
        print("\n[TEST 3] Kolmogorov-Smirnov Tests")
        self._run_ks_tests()
        
        # Test 4: Effect size calculations
        print("\n[TEST 4] Effect Size Calculations")
        self._compute_effect_sizes()
        
        # Test 5: Multiple testing correction
        print("\n[TEST 5] Multiple Testing Correction")
        self._apply_multiple_testing_correction()
    
    def _run_permutation_tests(self):
        """
        Run permutation tests comparing lottery metrics to null distributions.
        
        For each metric M:
        H0: M_lottery comes from same distribution as M_null
        H1: M_lottery is significantly different
        
        p-value = proportion of null samples ≥ |M_lottery - M_null_mean|
        """
        print("  Running permutation tests...")
        
        for graph_name, lottery_df in self.lottery_metrics.items():
            if lottery_df.shape[0] != 1:
                continue  # Skip temporal metrics for now
            
            lottery_row = lottery_df.iloc[0]
            
            # Find matching null models
            null_keys = [k for k in self.null_metrics.keys() if graph_name in k]
            
            for null_key in null_keys:
                null_df = self.null_metrics[null_key]
                model_name = null_key.replace(graph_name + "_", "")
                
                # Test each numeric metric
                numeric_cols = lottery_df.select_dtypes(include=[np.number]).columns
                
                for metric in numeric_cols:
                    if metric not in lottery_row or metric not in null_df.columns:
                        continue
                    
                    lottery_value = lottery_row[metric]
                    
                    # Skip NaN values
                    if pd.isna(lottery_value):
                        continue
                    
                    null_values = null_df[metric].dropna().values
                    
                    if len(null_values) == 0:
                        continue
                    
                    # Compute permutation p-value
                    null_mean = np.mean(null_values)
                    null_std = np.std(null_values)
                    
                    # Two-tailed test
                    p_value_upper = np.mean(null_values >= lottery_value)
                    p_value_lower = np.mean(null_values <= lottery_value)
                    p_value = 2 * min(p_value_upper, p_value_lower)
                    
                    # Z-score (standardized effect)
                    z_score = (lottery_value - null_mean) / (null_std + 1e-10)
                    
                    result = {
                        'test_type': 'permutation',
                        'graph_name': graph_name,
                        'null_model': model_name,
                        'metric': metric,
                        'lottery_value': lottery_value,
                        'null_mean': null_mean,
                        'null_std': null_std,
                        'null_min': np.min(null_values),
                        'null_max': np.max(null_values),
                        'z_score': z_score,
                        'p_value': p_value,
                        'significant': p_value < self.alpha
                    }
                    
                    self.test_results.append(result)
        
        print(f"    Computed {len([r for r in self.test_results if r['test_type'] == 'permutation'])} permutation tests")
    
    def _run_mann_whitney_tests(self):
        """
        Run Mann-Whitney U tests (non-parametric alternative to t-test).
        
        Tests whether lottery and null distributions differ in location.
        """
        print("  Running Mann-Whitney U tests...")
        
        for graph_name, lottery_df in self.lottery_metrics.items():
            if lottery_df.shape[0] != 1:
                continue
            
            lottery_row = lottery_df.iloc[0]
            
            # Find matching null models
            null_keys = [k for k in self.null_metrics.keys() if graph_name in k]
            
            for null_key in null_keys:
                null_df = self.null_metrics[null_key]
                model_name = null_key.replace(graph_name + "_", "")
                
                numeric_cols = lottery_df.select_dtypes(include=[np.number]).columns
                
                for metric in numeric_cols:
                    if metric not in lottery_row or metric not in null_df.columns:
                        continue
                    
                    lottery_value = lottery_row[metric]
                    
                    if pd.isna(lottery_value):
                        continue
                    
                    null_values = null_df[metric].dropna().values
                    
                    if len(null_values) < 3:
                        continue
                    
                    # Create pseudo-sample for lottery (single value repeated)
                    # This is a simplification - ideally we'd have multiple lottery draws
                    lottery_sample = np.array([lottery_value])
                    
                    # Mann-Whitney U test
                    try:
                        statistic, p_value = stats.mannwhitneyu(
                            lottery_sample, null_values, alternative='two-sided'
                        )
                        
                        result = {
                            'test_type': 'mann_whitney',
                            'graph_name': graph_name,
                            'null_model': model_name,
                            'metric': metric,
                            'lottery_value': lottery_value,
                            'null_median': np.median(null_values),
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < self.alpha
                        }
                        
                        self.test_results.append(result)
                    except:
                        continue
        
        print(f"    Computed {len([r for r in self.test_results if r['test_type'] == 'mann_whitney'])} Mann-Whitney tests")
    
    def _run_ks_tests(self):
        """
        Run Kolmogorov-Smirnov tests comparing distributions.
        
        Tests whether lottery value could plausibly come from null distribution.
        """
        print("  Running Kolmogorov-Smirnov tests...")
        
        for graph_name, lottery_df in self.lottery_metrics.items():
            if lottery_df.shape[0] != 1:
                continue
            
            lottery_row = lottery_df.iloc[0]
            
            null_keys = [k for k in self.null_metrics.keys() if graph_name in k]
            
            for null_key in null_keys:
                null_df = self.null_metrics[null_key]
                model_name = null_key.replace(graph_name + "_", "")
                
                numeric_cols = lottery_df.select_dtypes(include=[np.number]).columns
                
                for metric in numeric_cols:
                    if metric not in lottery_row or metric not in null_df.columns:
                        continue
                    
                    lottery_value = lottery_row[metric]
                    
                    if pd.isna(lottery_value):
                        continue
                    
                    null_values = null_df[metric].dropna().values
                    
                    if len(null_values) < 3:
                        continue
                    
                    # One-sample KS test: does lottery_value fit null distribution?
                    # Use empirical CDF of null distribution
                    try:
                        # Create pseudo-sample for lottery
                        lottery_sample = np.array([lottery_value])
                        
                        # Two-sample KS test
                        statistic, p_value = stats.ks_2samp(lottery_sample, null_values)
                        
                        result = {
                            'test_type': 'ks',
                            'graph_name': graph_name,
                            'null_model': model_name,
                            'metric': metric,
                            'lottery_value': lottery_value,
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < self.alpha
                        }
                        
                        self.test_results.append(result)
                    except:
                        continue
        
        print(f"    Computed {len([r for r in self.test_results if r['test_type'] == 'ks'])} KS tests")
    
    def _compute_effect_sizes(self):
        """
        Compute effect sizes (Cohen's d and similar measures).
        
        Effect size = (lottery_value - null_mean) / null_std
        
        Interpretation:
        - Small: |d| < 0.5
        - Medium: 0.5 ≤ |d| < 0.8
        - Large: |d| ≥ 0.8
        """
        print("  Computing effect sizes...")
        
        for graph_name, lottery_df in self.lottery_metrics.items():
            if lottery_df.shape[0] != 1:
                continue
            
            lottery_row = lottery_df.iloc[0]
            
            null_keys = [k for k in self.null_metrics.keys() if graph_name in k]
            
            for null_key in null_keys:
                null_df = self.null_metrics[null_key]
                model_name = null_key.replace(graph_name + "_", "")
                
                numeric_cols = lottery_df.select_dtypes(include=[np.number]).columns
                
                for metric in numeric_cols:
                    if metric not in lottery_row or metric not in null_df.columns:
                        continue
                    
                    lottery_value = lottery_row[metric]
                    
                    if pd.isna(lottery_value):
                        continue
                    
                    null_values = null_df[metric].dropna().values
                    
                    if len(null_values) == 0:
                        continue
                    
                    null_mean = np.mean(null_values)
                    null_std = np.std(null_values)
                    
                    # Cohen's d
                    cohens_d = (lottery_value - null_mean) / (null_std + 1e-10)
                    
                    # Interpret effect size
                    abs_d = abs(cohens_d)
                    if abs_d < 0.2:
                        interpretation = 'negligible'
                    elif abs_d < 0.5:
                        interpretation = 'small'
                    elif abs_d < 0.8:
                        interpretation = 'medium'
                    else:
                        interpretation = 'large'
                    
                    result = {
                        'test_type': 'effect_size',
                        'graph_name': graph_name,
                        'null_model': model_name,
                        'metric': metric,
                        'lottery_value': lottery_value,
                        'null_mean': null_mean,
                        'null_std': null_std,
                        'cohens_d': cohens_d,
                        'abs_cohens_d': abs_d,
                        'interpretation': interpretation
                    }
                    
                    self.test_results.append(result)
        
        print(f"    Computed {len([r for r in self.test_results if r['test_type'] == 'effect_size'])} effect sizes")
    
    def _apply_multiple_testing_correction(self):
        """
        Apply multiple testing corrections (Bonferroni and FDR).
        
        Since we're testing many hypotheses, we need to control for false positives.
        """
        print("  Applying multiple testing corrections...")
        
        # Get all p-values from permutation tests
        perm_results = [r for r in self.test_results if r['test_type'] == 'permutation']
        
        if len(perm_results) == 0:
            print("    No permutation tests found")
            return
        
        p_values = np.array([r['p_value'] for r in perm_results])
        
        # Bonferroni correction
        bonferroni_threshold = self.alpha / len(p_values)
        bonferroni_reject = p_values < bonferroni_threshold
        
        # Benjamini-Hochberg FDR
        try:
            fdr_reject, fdr_pvals, _, _ = multipletests(
                p_values, alpha=self.alpha, method='fdr_bh'
            )
        except:
            fdr_reject = np.array([False] * len(p_values))
            fdr_pvals = p_values
        
        # Update test results
        for i, result in enumerate(perm_results):
            result['bonferroni_threshold'] = bonferroni_threshold
            result['bonferroni_reject'] = bonferroni_reject[i]
            result['fdr_adjusted_pval'] = fdr_pvals[i]
            result['fdr_reject'] = fdr_reject[i]
        
        n_significant_raw = np.sum(p_values < self.alpha)
        n_significant_bonf = np.sum(bonferroni_reject)
        n_significant_fdr = np.sum(fdr_reject)
        
        print(f"    Total tests: {len(p_values)}")
        print(f"    Significant (raw α={self.alpha}): {n_significant_raw}")
        print(f"    Significant (Bonferroni): {n_significant_bonf}")
        print(f"    Significant (FDR): {n_significant_fdr}")
    
    def generate_test_summary(self) -> pd.DataFrame:
        """Generate summary of all statistical tests."""
        print("\n--- GENERATING TEST SUMMARY ---")
        
        df = pd.DataFrame(self.test_results)
        
        if len(df) == 0:
            print("  No test results to summarize")
            return df
        
        # Summary by test type
        print("\nTest counts by type:")
        print(df['test_type'].value_counts())
        
        # Significant results by test type
        if 'significant' in df.columns:
            print("\nSignificant results:")
            sig_df = df[df['significant'] == True]
            if len(sig_df) > 0:
                print(sig_df.groupby('test_type').size())
            else:
                print("  No significant results found")
        
        # Bonferroni correction results
        if 'bonferroni_reject' in df.columns:
            perm_df = df[df['test_type'] == 'permutation']
            if len(perm_df) > 0:
                n_bonf = perm_df['bonferroni_reject'].sum()
                n_fdr = perm_df['fdr_reject'].sum()
                print(f"\nAfter multiple testing correction:")
                print(f"  Bonferroni significant: {n_bonf}")
                print(f"  FDR significant: {n_fdr}")
        
        # Effect size summary
        effect_df = df[df['test_type'] == 'effect_size']
        if len(effect_df) > 0:
            print("\nEffect size distribution:")
            print(effect_df['interpretation'].value_counts())
        
        return df
    
    def save_results(self):
        """Save all statistical test results."""
        print("\n--- SAVING STATISTICAL TEST RESULTS ---")
        
        df = pd.DataFrame(self.test_results)
        
        if len(df) == 0:
            print("  No results to save")
            return
        
        # Save all results
        filepath = self.output_dir / f"{self.lottery_name}_statistical_tests.csv"
        df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath.name}")
        
        # Save permutation tests separately
        perm_df = df[df['test_type'] == 'permutation']
        if len(perm_df) > 0:
            filepath = self.output_dir / f"{self.lottery_name}_permutation_tests.csv"
            perm_df.to_csv(filepath, index=False)
            print(f"  Saved: {filepath.name}")
        
        # Save effect sizes
        effect_df = df[df['test_type'] == 'effect_size']
        if len(effect_df) > 0:
            filepath = self.output_dir / f"{self.lottery_name}_effect_sizes.csv"
            effect_df.to_csv(filepath, index=False)
            print(f"  Saved: {filepath.name}")
        
        # Create summary report
        summary_lines = [
            "STATISTICAL TESTING SUMMARY",
            "="*60,
            f"Lottery: {self.lottery_name}",
            f"Significance level: α = {self.alpha}",
            "",
            "TEST COUNTS:",
            str(df['test_type'].value_counts()),
            ""
        ]
        
        if 'significant' in df.columns:
            n_sig = df['significant'].sum()
            summary_lines.extend([
                f"Significant results (raw): {n_sig} / {len(df)} ({100*n_sig/len(df):.1f}%)",
                ""
            ])
        
        if 'bonferroni_reject' in df.columns:
            perm_df = df[df['test_type'] == 'permutation']
            if len(perm_df) > 0:
                n_bonf = perm_df['bonferroni_reject'].sum()
                n_fdr = perm_df['fdr_reject'].sum()
                summary_lines.extend([
                    "MULTIPLE TESTING CORRECTION:",
                    f"  Bonferroni significant: {n_bonf} / {len(perm_df)}",
                    f"  FDR significant: {n_fdr} / {len(perm_df)}",
                    ""
                ])
        
        summary_text = "\n".join(summary_lines)
        
        filepath = self.output_dir / f"{self.lottery_name}_test_summary.txt"
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(summary_text)
        print(f"  Saved: {filepath.name}")


def main():
    """Main execution function."""
    import sys
    BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
    
    if len(sys.argv) < 2:
        print("Usage: python 04_statistical_tests.py <lottery_name>")
        sys.exit(1)
    
    lottery_name = sys.argv[1].lower()
    
    if lottery_name not in ['powerball', 'megamillions']:
        print(f"Unknown lottery: {lottery_name}")
        sys.exit(1)
    
    # both data and output go to the same directory (where metrics are saved)
    data_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
    output_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
    
    # run statistical tests
    tester = StatisticalTester(lottery_name, data_dir, output_dir)
    tester.load_data()
    tester.run_all_tests()
    summary_df = tester.generate_test_summary()
    tester.save_results()
    
    print("\n--- STATISTICAL TESTING COMPLETE ---")


if __name__ == "__main__":
    main()

"""
Module 8: Master Analysis Script
Run complete causal inference analysis pipeline.

This script orchestrates all Module 8 analyses:
1. Granger Causality Tests
2. Conditional Independence Tests
3. Causal Discovery Algorithms
4. Transfer Entropy Analysis
5. Causal Invariance Testing

Usage:
    python module8_complete_analysis.py [--lottery LOTTERY] [--quick]

Arguments:
    --lottery : 'powerball', 'megamillions', or 'both' (default: 'both')
    --quick : Run quick version with reduced computations

"""

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime
import warnings
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, FIGURES_DIR, validate_config
from granger_causality import GrangerCausalityAnalyzer
from conditional_independence import ConditionalIndependenceAnalyzer
from causal_discovery import CausalDiscoveryAnalyzer
from transfer_entropy import TransferEntropyAnalyzer
from causal_invariance import CausalInvarianceAnalyzer

warnings.filterwarnings('ignore')


class Module8MasterAnalyzer:
    """Master analyzer for Module 8 causal inference."""
    
    def __init__(self, lottery_name: str, quick_mode: bool = False):
        """
        Initialize master analyzer.
        
        Parameters:
        -----------
        lottery_name : str
            'powerball' or 'megamillions'
        quick_mode : bool
            If True, run reduced computation versions
        """
        self.lottery_name = lottery_name
        self.quick_mode = quick_mode
        self.results = {}
        self.timings = {}
        
    def run_granger_causality(self) -> dict:
        """Run Granger causality analysis."""
        print("\n--- STEP 1/5: GRANGER CAUSALITY ANALYSIS ---")
        
        start_time = time.time()
        
        try:
            analyzer = GrangerCausalityAnalyzer(self.lottery_name)
            
            if self.quick_mode:
                # Quick mode: fewer lags
                results = analyzer.run_complete_analysis(max_lags=5)
            else:
                # Full analysis
                results = analyzer.run_complete_analysis(max_lags=10)
            
            elapsed = time.time() - start_time
            self.timings['granger'] = elapsed
            self.results['granger'] = results
            
            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            return results
            
        except Exception as e:
            print(f"\n[FAILED] Error in Granger causality: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_conditional_independence(self) -> dict:
        """Run conditional independence analysis."""
        print("\n--- STEP 2/5: CONDITIONAL INDEPENDENCE ANALYSIS ---")
        
        start_time = time.time()
        
        try:
            analyzer = ConditionalIndependenceAnalyzer(self.lottery_name)
            results = analyzer.run_complete_analysis()
            
            elapsed = time.time() - start_time
            self.timings['conditional_independence'] = elapsed
            self.results['conditional_independence'] = results
            
            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            return results
            
        except Exception as e:
            print(f"\n[FAILED] Error in conditional independence: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_causal_discovery(self) -> dict:
        """Run causal discovery analysis."""
        print("\n--- STEP 3/5: CAUSAL DISCOVERY ANALYSIS ---")
        
        start_time = time.time()
        
        try:
            analyzer = CausalDiscoveryAnalyzer(self.lottery_name)
            results = analyzer.run_complete_analysis()
            
            elapsed = time.time() - start_time
            self.timings['causal_discovery'] = elapsed
            self.results['causal_discovery'] = results
            
            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            return results
            
        except Exception as e:
            print(f"\n[FAILED] Error in causal discovery: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_transfer_entropy(self) -> dict:
        """Run transfer entropy analysis."""
        print("\n--- STEP 4/5: TRANSFER ENTROPY ANALYSIS ---")
        
        start_time = time.time()
        
        try:
            analyzer = TransferEntropyAnalyzer(self.lottery_name)
            
            if self.quick_mode:
                # Quick mode: shorter history
                results = analyzer.run_complete_analysis(k=2)
            else:
                results = analyzer.run_complete_analysis(k=3)
            
            elapsed = time.time() - start_time
            self.timings['transfer_entropy'] = elapsed
            self.results['transfer_entropy'] = results
            
            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            return results
            
        except Exception as e:
            print(f"\n[FAILED] Error in transfer entropy: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_causal_invariance(self) -> dict:
        """Run causal invariance analysis."""
        print("\n--- STEP 5/5: CAUSAL INVARIANCE ANALYSIS ---")
        
        start_time = time.time()
        
        try:
            analyzer = CausalInvarianceAnalyzer(self.lottery_name)
            results = analyzer.run_complete_analysis()
            
            elapsed = time.time() - start_time
            self.timings['causal_invariance'] = elapsed
            self.results['causal_invariance'] = results
            
            print(f"\n[OK] Completed in {elapsed:.1f} seconds")
            return results
            
        except Exception as e:
            print(f"\n[FAILED] Error in causal invariance: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_summary_report(self):
        """Generate summary report of all analyses."""
        print("\n--- MODULE 8 SUMMARY REPORT ---")
        
        print(f"\nLottery: {self.lottery_name.upper()}")
        print(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("TIMING SUMMARY")
        
        total_time = 0
        for analysis, elapsed in self.timings.items():
            print(f"  {analysis:30s}: {elapsed:6.1f} seconds")
            total_time += elapsed
        
        print(f"  {'TOTAL':30s}: {total_time:6.1f} seconds ({total_time/60:.1f} minutes)")
        
        print("RESULTS SUMMARY")
        
        # Granger Causality
        if 'granger' in self.results and self.results['granger']:
            gr = self.results['granger']
            print("\n  Granger Causality:")
            if 'univariate_p_values' in gr:
                p_vals = gr['univariate_p_values'].values
                # count off-diagonal elements only, excluding NaN/inf
                n_features = len(p_vals)
                mask = ~np.eye(n_features, dtype=bool)
                off_diagonal_pvals = p_vals[mask]
                valid_pvals = off_diagonal_pvals[np.isfinite(off_diagonal_pvals)]
                n_sig = (valid_pvals < 0.05).sum()
                n_total = n_features * (n_features - 1)
                print(f"    Significant relationships: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")
            print(f"    [OK] Analysis complete")
        
        # Conditional Independence
        if 'conditional_independence' in self.results and self.results['conditional_independence']:
            ci = self.results['conditional_independence']
            print("\n  Conditional Independence:")
            if 'pairwise' in ci:
                pairwise = ci['pairwise']
                if len(pairwise) > 0:
                    n_ind = pairwise['independent'].sum()
                    n_total = len(pairwise)
                    print(f"    Independent pairs: {n_ind}/{n_total} ({100*n_ind/n_total:.1f}%)")
            print(f"    [OK] Analysis complete")
        
        # Causal Discovery
        if 'causal_discovery' in self.results and self.results['causal_discovery']:
            cd = self.results['causal_discovery']
            print("\n  Causal Discovery:")
            for alg, results in cd.items():
                if results and 'n_edges' in results:
                    print(f"    {alg:8s}: {results['n_edges']} edges discovered")
            print(f"    [OK] Analysis complete")
        
        # Transfer Entropy
        if 'transfer_entropy' in self.results and self.results['transfer_entropy']:
            te = self.results['transfer_entropy']
            print("\n  Transfer Entropy:")
            if 'te_matrix' in te:
                te_matrix = te['te_matrix']
                mean_te = te_matrix[te_matrix > 0].mean() if (te_matrix > 0).any() else 0
                print(f"    Mean TE: {mean_te:.6f} bits")
            print(f"    [OK] Analysis complete")
        
        # Causal Invariance
        if 'causal_invariance' in self.results and self.results['causal_invariance']:
            inv = self.results['causal_invariance']
            print("\n  Causal Invariance:")
            if 'by_target' in inv:
                n_targets = len(inv['by_target'])
                print(f"    Targets tested: {n_targets}")
            print(f"    [OK] Analysis complete")
        
        print("OUTPUT LOCATIONS")
        print(f"  Results: {OUTPUT_DIR}")
        print(f"  Figures: {FIGURES_DIR}")
        
        # Save summary to file
        summary_file = OUTPUT_DIR / f'{self.lottery_name}_module8_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("MODULE 8 CAUSAL INFERENCE ANALYSIS SUMMARY\n")
            f.write(f"Lottery: {self.lottery_name.upper()}\n")
            f.write(f"Mode: {'Quick' if self.quick_mode else 'Full'}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n\n")
            f.write("Analyses Completed:\n")
            for analysis in self.timings.keys():
                f.write(f"  - {analysis}\n")
        
        print(f"\n  Summary saved to: {summary_file}")
    
    def run_complete_analysis(self):
        """Run complete Module 8 analysis pipeline."""
        
        print(f"# MODULE 8: CAUSAL INFERENCE & STRUCTURAL DEPENDENCE TESTING")
        print(f"# Lottery: {self.lottery_name.upper()}")
        print(f"# Mode: {'QUICK' if self.quick_mode else 'FULL'}")
        
        start_time = time.time()
        
        # Run all analyses
        self.run_granger_causality()
        self.run_conditional_independence()
        self.run_causal_discovery()
        self.run_transfer_entropy()
        self.run_causal_invariance()
        
        # Generate summary
        self.generate_summary_report()
        
        total_elapsed = time.time() - start_time
        
        print(f"# MODULE 8 COMPLETE - Total time: {total_elapsed/60:.1f} minutes")
        
        return self.results


def main():
    """Main entry point for Module 8 analysis."""
    
    parser = argparse.ArgumentParser(
        description='Run Module 8 causal inference analysis'
    )
    parser.add_argument(
        '--lottery',
        choices=['powerball', 'megamillions', 'both'],
        default='both',
        help='Which lottery to analyze'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick version with reduced computations'
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    print("Validating configuration...")
    if not validate_config():
        print("\n[WARNING] Warning: Some data files are missing. Analysis may fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run analysis
    if args.lottery == 'both':
        lotteries = ['powerball', 'megamillions']
    else:
        lotteries = [args.lottery]
    
    all_results = {}
    
    for lottery in lotteries:
        analyzer = Module8MasterAnalyzer(lottery, quick_mode=args.quick)
        results = analyzer.run_complete_analysis()
        all_results[lottery] = results
    
    print("\n--- ALL MODULE 8 ANALYSES COMPLETE ---")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    return all_results


if __name__ == "__main__":
    main()

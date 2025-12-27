#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 6: Calibration Analysis - Testing if Models Know What They Do Not Know

We trained 7 deep learning models in Module 5. Some made probabilistic predictions
with uncertainty intervals (like "the next draw will be 35 +/- 4 with 95% confidence").
But do these intervals actually work? If a model says "95% confident", do 95% of
true values fall inside its intervals?

This is the calibration question: Does the model's stated confidence match reality?

What we test:
1. Expected Calibration Error (ECE) - Average gap between claimed vs actual confidence
2. Tail Calibration Error (TCE) - Does it work in extreme cases (very high/low values)?
3. Probability Integral Transform (PIT) - Are predictions uniformly distributed?
4. Interval Sharpness - How wide are confidence intervals?

Why this matters for lottery analysis:

For truly random data, well-calibrated models should:
- Have 95% intervals that contain approximately 95% of data points
- Show uniform PIT histograms (no systematic bias)
- Have wide intervals (admitting uncertainty)
- Have low ECE (confident intervals match reality)

If models are poorly calibrated (like BNN with 9% coverage instead of 95%),
it tells us something: even uncertainty quantification fails on
pure randomness. This is actually GOOD evidence for the randomness hypothesis.

Input files:
- Module 5 outputs: *_perf_*.json and *_predictive_*.json
- Located in output/powerball/ and output/megamillions/

Output files (saved to specified outputs directory):
- calibration_report_powerball.json - Per-model calibration metrics for Powerball
- calibration_report_megamillions.json - Per-model calibration metrics for Mega Millions
- calibration_summary.json - Combined summary of both lotteries
- calibration_summary.csv - Flat CSV for easy viewing
- reliability_*.png - Reliability diagrams (if matplotlib available)

Usage:
  python module6_calibration_analysis.py <powerball_dir> <megamillions_dir> <outputs_dir>

Example:
  python module6_calibration_analysis.py output/powerball output/megamillions module6_outputs

"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# matplotlib is used only for saving simple reliability diagrams
# if it is not installed, diagrams are skipped, but the rest still runs
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CalibrationAnalyzer:
    """
    calibration analysis for probabilistic predictions
    
    implements:
    - expected calibration error (ECE) with simple binning
    - tail calibration error (TCE) for extreme quantiles
    - probability integral transform (PIT) uniformity test
    - quantile calibration error (QCE) across multiple quantiles
    - reliability diagram data generation
    - interval sharpness metrics
    """

    def __init__(self, powerball_dir: str = None,
                 megamillions_dir: str = None,
                 outputs_dir: str = None):
        """
        initialize calibration analyzer
        
        args:
            powerball_dir: directory containing Powerball model outputs
            megamillions_dir: directory containing Mega Millions model outputs
            outputs_dir: directory for calibration reports
        """
        if powerball_dir is None:
            powerball_dir = "."
        if megamillions_dir is None:
            megamillions_dir = "."
        if outputs_dir is None:
            outputs_dir = "."

        self.powerball_dir = Path(powerball_dir)
        self.megamillions_dir = Path(megamillions_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # models with uncertainty quantification
        self.uncertainty_models = [
            "bnn",
            "deepset",
            "deepset_v2",
            "transformer_bayeshead_baseline",
            "transformer_bayeshead_heavy",
            "transformer_bayeshead_hetero",
            "transformer_bayeshead_hetero_heavy"
        ]

        print("="*80)
        print("MODULE 6: CALIBRATION ANALYSIS")
        print("="*80)
        print(f"Powerball dir: {self.powerball_dir}")
        print(f"Mega Millions dir: {self.megamillions_dir}")
        print(f"Output dir: {self.outputs_dir}")
        print()

    def load_model_data(self, model_name: str, lottery: str) -> Optional[Dict]:
        """
        load predictions and performance data for a model
        
        returns None if files do not exist or have issues
        """
        if lottery == 'powerball':
            data_dir = self.powerball_dir
            suffix = 'pb'
        else:
            data_dir = self.megamillions_dir
            suffix = 'mm'

        pred_file = data_dir / f"{model_name}_predictive_{suffix}.json"
        perf_file = data_dir / f"{model_name}_perf_{suffix}.json"

        if not pred_file.exists() or not perf_file.exists():
            return None

        try:
            with open(pred_file) as f:
                pred = json.load(f)
            with open(perf_file) as f:
                perf = json.load(f)

            # check if has uncertainty intervals
            if 'y_lo' not in pred or 'y_hi' not in pred:
                return None

            return {
                'predictions': pred,
                'performance': perf,
                'model_name': model_name
            }
        except Exception as e:
            print(f"  Warning: Error loading {model_name}: {e}")
            return None

    def compute_ece(self, y_true: np.ndarray, y_lo: np.ndarray, 
                   y_hi: np.ndarray, n_bins: int = 10) -> Dict:
        """
        compute expected calibration error (ECE)
        
        measures average gap between claimed confidence and actual coverage
        across binned confidence levels
        
        for 95% intervals, we expect ECE close to 0 (intervals well-calibrated)
        """
        # compute empirical coverage
        is_covered = (y_true >= y_lo) & (y_true <= y_hi)
        n = len(y_true)

        # compute interval widths as proxy for confidence
        widths = y_hi - y_lo
        
        # bin by width (narrower = more confident)
        bin_edges = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        bin_indices = np.digitize(widths, bin_edges) - 1

        ece = 0.0
        bin_stats = []

        for b in range(n_bins):
            mask = (bin_indices == b)
            if mask.sum() == 0:
                continue

            bin_coverage = is_covered[mask].mean()
            bin_fraction = mask.sum() / n
            
            # expected coverage for 95% intervals is 0.95
            gap = abs(bin_coverage - 0.95)
            ece += bin_fraction * gap

            bin_stats.append({
                'bin': int(b),
                'n_samples': int(mask.sum()),
                'coverage': float(bin_coverage),
                'expected': 0.95,
                'gap': float(gap)
            })

        return {
            'ece': float(ece),
            'n_bins': n_bins,
            'bin_stats': bin_stats
        }

    def compute_tce(self, y_true: np.ndarray, y_lo: np.ndarray,
                   y_hi: np.ndarray) -> Dict:
        """
        compute tail calibration error (TCE)
        
        measures calibration in distribution tails (extreme values)
        important for lottery analysis where extremes matter
        """
        is_covered = (y_true >= y_lo) & (y_true <= y_hi)
        
        # identify tail regions (top/bottom 10%)
        lower_tail = y_true <= np.percentile(y_true, 10)
        upper_tail = y_true >= np.percentile(y_true, 90)
        
        lower_coverage = is_covered[lower_tail].mean() if lower_tail.sum() > 0 else np.nan
        upper_coverage = is_covered[upper_tail].mean() if upper_tail.sum() > 0 else np.nan
        
        # TCE is average gap in tails
        gaps = []
        if not np.isnan(lower_coverage):
            gaps.append(abs(lower_coverage - 0.95))
        if not np.isnan(upper_coverage):
            gaps.append(abs(upper_coverage - 0.95))
        
        tce = np.mean(gaps) if gaps else np.nan

        return {
            'tce': float(tce) if not np.isnan(tce) else None,
            'lower_tail_coverage': float(lower_coverage) if not np.isnan(lower_coverage) else None,
            'upper_tail_coverage': float(upper_coverage) if not np.isnan(upper_coverage) else None
        }

    def compute_pit(self, y_true: np.ndarray, y_mean: np.ndarray,
                   y_std: np.ndarray) -> Dict:
        """
        compute probability integral transform (PIT)
        
        for well-calibrated probabilistic predictions, PIT should be uniform
        tests if model's uncertainty estimates match actual distribution
        """
        # compute PIT values
        pit_values = stats.norm.cdf(y_true, loc=y_mean, scale=y_std)
        
        # test uniformity with Kolmogorov-Smirnov
        ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
        
        # histogram for visualization
        hist, bins = np.histogram(pit_values, bins=10, range=(0, 1))
        
        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'pit_histogram': {
                'counts': hist.tolist(),
                'bins': bins.tolist()
            },
            'mean': float(np.mean(pit_values)),
            'std': float(np.std(pit_values))
        }

    def compute_sharpness(self, y_lo: np.ndarray, y_hi: np.ndarray) -> Dict:
        """
        compute interval sharpness metrics
        
        sharp intervals = narrow, confident predictions
        for random data, expect wide intervals (model admits uncertainty)
        """
        widths = y_hi - y_lo
        
        return {
            'mean': float(np.mean(widths)),
            'median': float(np.median(widths)),
            'std': float(np.std(widths)),
            'p90': float(np.percentile(widths, 90)),
            'p10': float(np.percentile(widths, 10))
        }

    def generate_reliability_data(self, y_true: np.ndarray, y_lo: np.ndarray,
                                 y_hi: np.ndarray, n_bins: int = 10) -> Dict:
        """
        generate data for reliability diagram
        
        plots observed coverage vs expected coverage
        perfect calibration = diagonal line
        """
        is_covered = (y_true >= y_lo) & (y_true <= y_hi)
        widths = y_hi - y_lo
        
        # bin by width
        bin_edges = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        bin_indices = np.digitize(widths, bin_edges) - 1
        
        observed_coverage = []
        expected_coverage = []
        bin_centers = []
        
        for b in range(n_bins):
            mask = (bin_indices == b)
            if mask.sum() == 0:
                continue
            
            obs_cov = is_covered[mask].mean()
            observed_coverage.append(obs_cov)
            expected_coverage.append(0.95)
            bin_centers.append(b / n_bins + 0.5 / n_bins)
        
        return {
            'observed': observed_coverage,
            'expected': expected_coverage,
            'bin_centers': bin_centers
        }

    def plot_reliability_diagram(self, reliability_data: Dict, model_name: str,
                                lottery: str, save_path: Path):
        """
        create and save reliability diagram
        
        skipped if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        obs = reliability_data['observed']
        exp = reliability_data['expected']
        
        ax.scatter(exp, obs, alpha=0.6, s=100)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        ax.set_xlabel('Expected Coverage (0.95)')
        ax.set_ylabel('Observed Coverage')
        ax.set_title(f'Reliability Diagram\n{model_name} - {lottery}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_model(self, model_name: str, lottery: str) -> Optional[Dict]:
        """
        comprehensive calibration analysis for single model
        
        returns None if model does not have predictive intervals
        """
        print(f"  Analyzing: {model_name}")
        
        data = self.load_model_data(model_name, lottery)
        if data is None:
            print(f"    No data available")
            return None
        
        pred = data['predictions']
        perf = data['performance']
        
        # extract arrays
        y_true = np.array(pred.get('y_true', perf.get('y_true', [])))
        y_mean = np.array(pred['y_mean'])
        y_lo = np.array(pred['y_lo'])
        y_hi = np.array(pred['y_hi'])
        
        if len(y_true) == 0:
            print(f"    No predictions available")
            return None
        
        # compute standard deviation from intervals (assume ~95% coverage)
        y_std = (y_hi - y_lo) / (2 * 1.96)
        
        # run all calibration analyses
        ece_result = self.compute_ece(y_true, y_lo, y_hi)
        tce_result = self.compute_tce(y_true, y_lo, y_hi)
        pit_result = self.compute_pit(y_true, y_mean, y_std)
        sharpness_result = self.compute_sharpness(y_lo, y_hi)
        reliability_data = self.generate_reliability_data(y_true, y_lo, y_hi)
        
        # save reliability diagram
        diagram_path = self.outputs_dir / f"reliability_{lottery}_{model_name}.png"
        self.plot_reliability_diagram(reliability_data, model_name, lottery, diagram_path)
        
        # compute existing metrics from performance file
        coverage_95 = perf.get('coverage_95')
        rmse = perf.get('rmse')
        
        print(f"    ECE: {ece_result['ece']:.4f}")
        print(f"    Coverage: {coverage_95:.3f}" if coverage_95 is not None else "    Coverage: N/A")
        
        return {
            'model_name': model_name,
            'lottery': lottery,
            'n_predictions': len(y_true),
            'ece': ece_result,
            'tce': tce_result,
            'pit': pit_result,
            'sharpness': sharpness_result,
            'reliability_diagram': reliability_data,
            'existing_metrics': {
                'coverage_95': coverage_95,
                'rmse': rmse
            }
        }

    def analyze_lottery(self, lottery: str) -> Dict:
        """
        analyze all models for a lottery
        
        returns dict mapping model_name to calibration results
        """
        print(f"\n--- ANALYZING: {lottery.upper()} ---")
        
        results = {}
        
        for model_name in self.uncertainty_models:
            result = self.analyze_model(model_name, lottery)
            if result:
                results[model_name] = result
        
        return results

    def run(self):
        """
        run calibration analysis for both lotteries
        
        generates reports and saves to outputs directory
        """
        print("\n--- STARTING CALIBRATION ANALYSIS ---")
        
        # analyze both lotteries
        pb_results = self.analyze_lottery('powerball')
        mm_results = self.analyze_lottery('megamillions')
        
        # save individual reports
        print("\n--- SAVING RESULTS ---")
        
        pb_file = self.outputs_dir / 'calibration_report_powerball.json'
        mm_file = self.outputs_dir / 'calibration_report_megamillions.json'
        
        with open(pb_file, 'w') as f:
            json.dump(pb_results, f, indent=2)
        print(f"Saved: {pb_file}")
        
        with open(mm_file, 'w') as f:
            json.dump(mm_results, f, indent=2)
        print(f"Saved: {mm_file}")
        
        # combined summary
        summary = {
            'powerball': pb_results,
            'megamillions': mm_results,
            'timestamp': '2025-11-16',
            'n_models': {
                'powerball': len(pb_results),
                'megamillions': len(mm_results)
            }
        }
        
        summary_file = self.outputs_dir / 'calibration_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {summary_file}")
        
        # flat CSV for easy viewing
        self.create_summary_csv(pb_results, mm_results)
        
        # print summary table
        self.print_summary(pb_results, mm_results)
        
        print("\n--- CALIBRATION ANALYSIS COMPLETE ---")

    def create_summary_csv(self, pb_results: Dict, mm_results: Dict):
        """
        create flat CSV summary of calibration metrics
        
        easier to view than nested JSON
        """
        rows = []
        
        for lottery, results in [('powerball', pb_results), ('megamillions', mm_results)]:
            for model_name, data in results.items():
                rows.append({
                    'lottery': lottery,
                    'model': model_name,
                    'n_predictions': data['n_predictions'],
                    'ece': data['ece']['ece'],
                    'tce': data['tce']['tce'],
                    'coverage_95': data['existing_metrics']['coverage_95'],
                    'rmse': data['existing_metrics']['rmse'],
                    'sharpness_mean': data['sharpness']['mean'],
                    'sharpness_p90': data['sharpness']['p90'],
                    'pit_ks_stat': data['pit']['ks_statistic'],
                    'pit_ks_pval': data['pit']['ks_pvalue']
                })
        
        df = pd.DataFrame(rows)
        csv_file = self.outputs_dir / 'calibration_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")

    def print_summary(self, pb_results: Dict, mm_results: Dict):
        """
        print summary table of calibration metrics
        
        shows key metrics for quick assessment
        """
        print("\n--- CALIBRATION SUMMARY ---")
        
        for lottery, results in [('Powerball', pb_results), ('Mega Millions', mm_results)]:
            if not results:
                continue
            
            print(f"{lottery}:")
            print(f"{'Model':<35} {'ECE':<8} {'Coverage':<10} {'Sharpness'}")
            print("-" * 70)
            
            for model_name, data in results.items():
                ece = data['ece']['ece']
                cov = data['existing_metrics']['coverage_95']
                sharp = data['sharpness']['mean']
                
                cov_str = f"{cov:.3f}" if cov is not None else "N/A"
                print(f"{model_name:<35} {ece:<8.4f} {cov_str:<10} {sharp:.3f}")
            print()


def main():
    """
    main execution
    
    accepts command line arguments for directories
    """
    import sys
    
    if len(sys.argv) == 4:
        analyzer = CalibrationAnalyzer(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        analyzer = CalibrationAnalyzer()
    else:
        print("Usage: python module6_calibration_analysis.py <powerball_dir> <megamillions_dir> <outputs_dir>")
        print("\nExample:")
        print('  python module6_calibration_analysis.py "output/powerball" "output/megamillions" "module6_outputs"')
        print("\nOr run with no arguments to use current directory")
        sys.exit(1)
    
    analyzer.run()


if __name__ == "__main__":
    main()

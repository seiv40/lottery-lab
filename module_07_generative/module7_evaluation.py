"""
Module 7: Universe Evaluation

Evaluates and compares all generated universes against baselines.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, pearsonr
import warnings


class UniverseEvaluator:
    """Comprehensive evaluation of lottery universes."""
    
    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type.lower()
        
        if self.lottery_type == 'powerball':
            self.white_range = range(1, 70)
            self.special_range = range(1, 27)
            self.n_white = 5
            self.special_name = 'powerball'
        elif self.lottery_type == 'megamillions':
            self.white_range = range(1, 71)
            self.special_range = range(1, 26)
            self.n_white = 5
            self.special_name = 'megaball'
        else:
            raise ValueError(f"Unknown lottery: {lottery_type}")
        
        self.white_cols = [f'white_{i+1}' for i in range(self.n_white)]
    
    def compute_full_statistics(self, universe: pd.DataFrame) -> Dict:
        """Compute comprehensive statistics for a universe."""
        
        stats_dict = {
            'n_draws': len(universe),
            'lottery_type': self.lottery_type
        }
        
        # white ball statistics
        all_white = universe[self.white_cols].values.flatten()
        
        stats_dict['white_balls'] = {
            'mean': float(all_white.mean()),
            'std': float(all_white.std()),
            'min': int(all_white.min()),
            'max': int(all_white.max()),
            'median': float(np.median(all_white)),
            'q25': float(np.percentile(all_white, 25)),
            'q75': float(np.percentile(all_white, 75)),
        }
        
        # special ball statistics
        special_balls = universe[self.special_name].values
        
        stats_dict['special_ball'] = {
            'mean': float(special_balls.mean()),
            'std': float(special_balls.std()),
            'min': int(special_balls.min()),
            'max': int(special_balls.max()),
            'unique_count': int(len(np.unique(special_balls))),
            'coverage': float(len(np.unique(special_balls)) / len(self.special_range))
        }
        
        # per-draw aggregates
        draw_means = universe[self.white_cols].mean(axis=1).values
        draw_sums = universe[self.white_cols].sum(axis=1).values
        draw_stds = universe[self.white_cols].std(axis=1).values
        
        stats_dict['per_draw'] = {
            'mean_of_means': float(draw_means.mean()),
            'std_of_means': float(draw_means.std()),
            'mean_of_sums': float(draw_sums.mean()),
            'std_of_sums': float(draw_sums.std()),
            'mean_of_stds': float(draw_stds.mean()),
        }
        
        # temporal structure
        if len(draw_means) > 10:
            autocorr, p_value = pearsonr(draw_means[:-1], draw_means[1:])
            stats_dict['temporal'] = {
                'lag1_autocorr': float(autocorr),
                'lag1_pvalue': float(p_value)
            }
        
        # ball frequency distribution
        white_counts = np.bincount(all_white.astype(int), minlength=max(self.white_range)+1)
        white_counts = white_counts[list(self.white_range)]
        
        stats_dict['white_frequency'] = {
            'counts': white_counts.tolist(),
            'chi2_uniformity': self._test_uniformity(white_counts)
        }
        
        special_counts = np.bincount(special_balls.astype(int), minlength=max(self.special_range)+1)
        special_counts = special_counts[list(self.special_range)]
        
        stats_dict['special_frequency'] = {
            'counts': special_counts.tolist(),
            'chi2_uniformity': self._test_uniformity(special_counts)
        }
        
        return stats_dict
    
    def _test_uniformity(self, counts: np.ndarray) -> Dict:
        """Chi-square test for uniformity."""
        expected = np.mean(counts)
        chi2_stat = np.sum((counts - expected)**2 / expected)
        df = len(counts) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'is_uniform': bool(p_value > 0.05)
        }
    
    def compare_universes(self, universe_a: pd.DataFrame, universe_b: pd.DataFrame,
                         name_a: str = 'A', name_b: str = 'B') -> Dict:
        """Compare two universes statistically."""
        
        comparison = {
            'name_a': name_a,
            'name_b': name_b
        }
        
        # white ball distribution comparison
        white_a = universe_a[self.white_cols].values.flatten()
        white_b = universe_b[self.white_cols].values.flatten()
        
        ks_stat, ks_pval = ks_2samp(white_a, white_b)
        
        comparison['white_balls_ks'] = {
            'statistic': float(ks_stat),
            'p_value': float(ks_pval),
            'similar': bool(ks_pval > 0.05)
        }
        
        # special ball comparison
        special_a = universe_a[self.special_name].values
        special_b = universe_b[self.special_name].values
        
        ks_stat_sp, ks_pval_sp = ks_2samp(special_a, special_b)
        
        comparison['special_ball_ks'] = {
            'statistic': float(ks_stat_sp),
            'p_value': float(ks_pval_sp),
            'similar': bool(ks_pval_sp > 0.05)
        }
        
        # mean comparison
        mean_diff = white_a.mean() - white_b.mean()
        std_diff = white_a.std() - white_b.std()
        
        comparison['distributional_shift'] = {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff)
        }
        
        return comparison
    
    def compute_uri_score(self, universe: pd.DataFrame, baseline: pd.DataFrame) -> Dict:
        """
        Universe Realism Index (URI): How realistic is generated universe?
        
        Combines multiple metrics into single score.
        URI = 1.0 means perfect realism (indistinguishable from baseline)
        URI < 0.5 means easily distinguishable
        """
        
        uri_components = {}
        
        # 1. Distribution similarity (KS test)
        white_gen = universe[self.white_cols].values.flatten()
        white_base = baseline[self.white_cols].values.flatten()
        
        ks_stat, ks_pval = ks_2samp(white_gen, white_base)
        uri_components['ks_score'] = float(ks_pval)  # higher p-value = more similar
        
        # 2. Frequency uniformity match
        white_counts_gen = np.bincount(white_gen.astype(int), minlength=max(self.white_range)+1)
        white_counts_base = np.bincount(white_base.astype(int), minlength=max(self.white_range)+1)
        
        # normalize to probabilities
        prob_gen = white_counts_gen[list(self.white_range)] / len(white_gen)
        prob_base = white_counts_base[list(self.white_range)] / len(white_base)
        
        # jensen-Shannon divergence (symmetric KL divergence)
        m = (prob_gen + prob_base) / 2
        kl_gen = np.sum(prob_gen * np.log(prob_gen / m + 1e-10))
        kl_base = np.sum(prob_base * np.log(prob_base / m + 1e-10))
        js_div = (kl_gen + kl_base) / 2
        
        uri_components['frequency_score'] = float(np.exp(-js_div))  # 1.0 = perfect match
        
        # 3. Moment matching
        mean_gen = white_gen.mean()
        mean_base = white_base.mean()
        std_gen = white_gen.std()
        std_base = white_base.std()
        
        mean_error = abs(mean_gen - mean_base) / mean_base
        std_error = abs(std_gen - std_base) / std_base
        
        uri_components['moment_score'] = float(np.exp(-(mean_error + std_error)))
        
        # 4. Temporal structure match
        draw_means_gen = universe[self.white_cols].mean(axis=1).values
        draw_means_base = baseline[self.white_cols].mean(axis=1).values
        
        if len(draw_means_gen) > 10 and len(draw_means_base) > 10:
            autocorr_gen, _ = pearsonr(draw_means_gen[:-1], draw_means_gen[1:])
            autocorr_base, _ = pearsonr(draw_means_base[:-1], draw_means_base[1:])
            
            temporal_error = abs(autocorr_gen - autocorr_base)
            uri_components['temporal_score'] = float(np.exp(-10 * temporal_error))
        else:
            uri_components['temporal_score'] = 1.0
        
        # aggregate URI (weighted average)
        weights = {
            'ks_score': 0.35,
            'frequency_score': 0.30,
            'moment_score': 0.20,
            'temporal_score': 0.15
        }
        
        uri = sum(uri_components[k] * weights[k] for k in weights)
        
        return {
            'uri': float(uri),
            'components': uri_components,
            'interpretation': self._interpret_uri(uri)
        }
    
    def _interpret_uri(self, uri: float) -> str:
        """Interpret URI score."""
        if uri >= 0.90:
            return "Excellent - Indistinguishable from real data"
        elif uri >= 0.75:
            return "Good - Minor detectable differences"
        elif uri >= 0.60:
            return "Fair - Noticeable differences"
        elif uri >= 0.40:
            return "Poor - Substantial differences"
        else:
            return "Very Poor - Easily distinguishable"


# MAIN ANALYSIS

def run_full_analysis(lottery: str):
    """Run comprehensive universe comparison."""
    
    lottery = lottery.lower()
    OUTPUT_DIR = Path(r'C:\jackpotmath\lottery-lab\output')
    
    suffix = '_pb' if lottery == 'powerball' else '_mm'
    
    print(f"\n--- MODULE 7 UNIVERSE EVALUATION - {lottery.upper()} ---")
    
    # load all universes
    universes = {}
    universe_files = {
        'Null': OUTPUT_DIR / f'universe_null{suffix}.parquet',
        'VAE': OUTPUT_DIR / f'universe_vae{suffix}.parquet',
        'Flow': OUTPUT_DIR / f'universe_flow{suffix}.parquet',
        'Empirical Bootstrap': OUTPUT_DIR / f'universe_bootstrap{suffix}.parquet',
        'Block Bootstrap': OUTPUT_DIR / f'universe_block_bootstrap{suffix}.parquet',
    }
    
    print(f"\nLoading universes...")
    for name, filepath in universe_files.items():
        if filepath.exists():
            universes[name] = pd.read_parquet(filepath)
            print(f"   {name}: {len(universes[name]):,} draws")
        else:
            print(f"   {name}: Not found")
    
    if not universes:
        print("\n No universes found!")
        return
    
    # initialize evaluator
    evaluator = UniverseEvaluator(lottery)
    
    # compute statistics for each universe
    print("\n--- INDIVIDUAL UNIVERSE STATISTICS ---")
    
    all_stats = {}
    for name, universe in universes.items():
        print(f"\n{name}:")
        print(f"-" * 40)
        stats_dict = evaluator.compute_full_statistics(universe)
        all_stats[name] = stats_dict
        
        wb = stats_dict['white_balls']
        print(f"  White balls: μ={wb['mean']:.2f}, sigma={wb['std']:.2f}")
        
        sb = stats_dict['special_ball']
        print(f"  Special ball: μ={sb['mean']:.2f}, unique={sb['unique_count']}")
        
        if 'temporal' in stats_dict:
            temp = stats_dict['temporal']
            print(f"  Temporal: rho={temp['lag1_autocorr']:.3f}")
    
    # pairwise comparisons
    print("\n--- PAIRWISE COMPARISONS (vs Empirical Bootstrap) ---")
    
    if 'Empirical Bootstrap' in universes:
        baseline = universes['Empirical Bootstrap']
        baseline_name = 'Empirical Bootstrap'
    elif 'Null' in universes:
        baseline = universes['Null']
        baseline_name = 'Null'
    else:
        baseline = list(universes.values())[0]
        baseline_name = list(universes.keys())[0]
    
    print(f"Baseline: {baseline_name}\n")
    
    comparisons = {}
    uri_scores = {}
    
    for name, universe in universes.items():
        if name == baseline_name:
            continue
        
        print(f"{name} vs {baseline_name}:")
        print(f"-" * 40)
        
        # statistical comparison
        comparison = evaluator.compare_universes(universe, baseline, name, baseline_name)
        comparisons[name] = comparison
        
        ks_white = comparison['white_balls_ks']
        print(f"  White balls KS: p={ks_white['p_value']:.4f} {'' if ks_white['similar'] else ''}")
        
        ks_special = comparison['special_ball_ks']
        print(f"  Special ball KS: p={ks_special['p_value']:.4f} {'' if ks_special['similar'] else ''}")
        
        # URI score
        uri_result = evaluator.compute_uri_score(universe, baseline)
        uri_scores[name] = uri_result
        
        print(f"\n  URI Score: {uri_result['uri']:.3f}")
        print(f"  -> {uri_result['interpretation']}")
        print()
    
    # summary ranking
    print("\n--- URI RANKING (Descending) ---")
    
    sorted_uri = sorted(uri_scores.items(), key=lambda x: x[1]['uri'], reverse=True)
    
    for rank, (name, uri_result) in enumerate(sorted_uri, 1):
        print(f"{rank}. {name}: {uri_result['uri']:.3f} - {uri_result['interpretation']}")
    
    # save results
    print("\n--- SAVING RESULTS ---")
    
    results = {
        'lottery': lottery,
        'statistics': all_stats,
        'comparisons': comparisons,
        'uri_scores': uri_scores,
        'baseline': baseline_name
    }
    
    output_file = OUTPUT_DIR / lottery / f'module7_evaluation{suffix}.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f" Results saved: {output_file}")
    
    print("\n--- EVALUATION COMPLETE ---")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python module7_evaluation.py <lottery>")
        print("  lottery: 'powerball' or 'megamillions'")
        sys.exit(1)
    
    lottery = sys.argv[1].lower()
    
    if lottery not in ['powerball', 'megamillions']:
        print(f"Error: Unknown lottery '{lottery}'")
        sys.exit(1)
    
    run_full_analysis(lottery)

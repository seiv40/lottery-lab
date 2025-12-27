"""
Module 7: Universe Generators

This implements universe generators - models that create large synthetic
datasets of lottery drawings ("universes").

Universe 0 (Null) is the gold standard: perfectly random i.i.d. sampling
following official lottery rules.

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class UniverseGenerator:
    """Base class for universe generators."""
    
    def __init__(self, lottery_type: str):
        """
        Initialize universe generator.
        
        Args:
            lottery_type: 'powerball' or 'megamillions'
        """
        self.lottery_type = lottery_type.lower()
        
        # set lottery-specific parameters
        if self.lottery_type == 'powerball':
            self.white_range = range(1, 70)  # 1-69
            self.special_range = range(1, 27)  # 1-26
            self.n_white = 5
            self.special_name = 'powerball'
        elif self.lottery_type == 'megamillions':
            self.white_range = range(1, 71)  # 1-70
            self.special_range = range(1, 26)  # 1-25
            self.n_white = 5
            self.special_name = 'megaball'
        else:
            raise ValueError(f"Unknown lottery type: {lottery_type}")
    
    def generate(self, n_draws: int, **kwargs) -> pd.DataFrame:
        """
        Generate a universe of lottery draws.
        
        To be implemented by subclasses.
        
        Args:
            n_draws: Number of drawings to generate
            **kwargs: Generator-specific parameters
        
        Returns:
            universe: DataFrame with columns for each ball position + metadata
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def save_universe(self, 
                  universe: pd.DataFrame, 
                  filepath: str,
                  metadata: Optional[Dict] = None):
        """
        Save universe to disk under the main output directory.
        """
        from pathlib import Path

        # base output directory
        OUTPUT_DIR = Path(r"C:\jackpotmath\lottery-lab\output")

        # ensure directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # convert to string if a Path object was passed
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # extract filename only - ignore caller's path
        filename = Path(filepath).name

        # ensure .parquet extension
        if not filename.endswith(".parquet"):
            filename += ".parquet"

        # final save paths
        save_path = OUTPUT_DIR / filename
        metadata_path = save_path.with_suffix("").as_posix() + "_metadata.json"

        # standard metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            "lottery_type": self.lottery_type,
            "n_draws": len(universe),
            "generated_at": datetime.now().isoformat(),
            "generator_class": self.__class__.__name__
        })

        # save parquet
        universe.to_parquet(save_path, index=False)

        # save metadata JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Universe saved to: {save_path}")
        print(f"Metadata saved to: {metadata_path}")



class NullUniverseGenerator(UniverseGenerator):
    """
    Universe 0: Pure i.i.d. random sampling.
    
    This is the gold standard - perfectly random lottery drawings
    following official rules. Any realistic model should produce
    universes statistically indistinguishable from this.
    """
    
    def generate(self, 
                n_draws: int = 10000,
                seed: Optional[int] = None,
                add_dates: bool = True) -> pd.DataFrame:
        """
        Generate null universe via pure i.i.d. sampling.
        
        Args:
            n_draws: Number of drawings to generate
            seed: Random seed for reproducibility
            add_dates: Whether to add synthetic draw dates
        
        Returns:
            universe: DataFrame with each row = one drawing
        """
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n--- Generating Null Universe ---")
        print(f"Lottery: {self.lottery_type.upper()}")
        print(f"Draws: {n_draws:,}")
        if seed is not None:
            print(f"Seed: {seed}")
        
        universe_data = []
        
        for i in range(n_draws):
            # generate white balls (sample without replacement)
            white_balls = sorted(
                np.random.choice(list(self.white_range), self.n_white, replace=False)
            )
            
            # generate special ball
            special_ball = np.random.choice(list(self.special_range))
            
            # create draw record
            draw = {
                'draw_id': i + 1
            }
            
            # add individual white ball columns
            for j, ball in enumerate(white_balls):
                draw[f'white_{j+1}'] = ball
            
            # add special ball
            draw[self.special_name] = special_ball
            
            # add synthetic date if requested
            if add_dates:
                # assume twice-weekly drawings (like real lotteries)
                # start from an arbitrary date
                base_date = datetime(2020, 1, 1)
                days_offset = (i // 2) * 7 + (3 if i % 2 == 0 else 6)  # wed & Sat
                draw['draw_date'] = base_date + timedelta(days=days_offset)
            
            universe_data.append(draw)
        
        universe = pd.DataFrame(universe_data)
        
        print(f" Generated {len(universe):,} draws")
        self._print_universe_summary(universe)
        
        return universe
    
    def _print_universe_summary(self, universe: pd.DataFrame):
        """Print summary statistics of generated universe."""
        print(f"\nUniverse Summary:")
        print(f"-" * 40)
        
        # ball frequency stats
        white_cols = [f'white_{i+1}' for i in range(self.n_white)]
        all_white_balls = universe[white_cols].values.flatten()
        
        print(f"White balls:")
        print(f"  Range: {all_white_balls.min()} - {all_white_balls.max()}")
        print(f"  Mean: {all_white_balls.mean():.2f}")
        print(f"  Std: {all_white_balls.std():.2f}")
        
        # check uniformity (Chi-square test would be ideal, but simple check here)
        unique, counts = np.unique(all_white_balls, return_counts=True)
        expected_count = len(all_white_balls) / len(self.white_range)
        max_deviation = np.abs(counts - expected_count).max()
        print(f"  Max deviation from uniform: {max_deviation:.1f} counts")
        
        # special ball stats
        special_balls = universe[self.special_name].values
        print(f"\n{self.special_name.capitalize()}:")
        print(f"  Range: {special_balls.min()} - {special_balls.max()}")
        print(f"  Mean: {special_balls.mean():.2f}")
        print(f"  Unique values: {len(np.unique(special_balls))}/{len(self.special_range)}")
        
        print(f"-" * 40)


class UniverseStatistics:
    """
    Compute and display basic statistics for any universe.
    
    This class provides quick sanity checks before running full audit suite.
    """
    
    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type.lower()
        
        if self.lottery_type == 'powerball':
            self.n_white = 5
            self.special_name = 'powerball'
        elif self.lottery_type == 'megamillions':
            self.n_white = 5
            self.special_name = 'megaball'
    
    def compute_statistics(self, universe: pd.DataFrame) -> Dict[str, any]:
        """
        Compute comprehensive statistics for a universe.
        
        Args:
            universe: DataFrame of lottery draws
        
        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            'n_draws': len(universe),
            'lottery_type': self.lottery_type
        }
        
        # white ball statistics
        white_cols = [f'white_{i+1}' for i in range(self.n_white)]
        all_white_balls = universe[white_cols].values.flatten()
        
        stats['white_balls'] = {
            'mean': float(all_white_balls.mean()),
            'std': float(all_white_balls.std()),
            'min': int(all_white_balls.min()),
            'max': int(all_white_balls.max()),
            'median': float(np.median(all_white_balls))
        }
        
        # per-draw statistics
        draw_means = universe[white_cols].mean(axis=1).values
        draw_sums = universe[white_cols].sum(axis=1).values
        draw_maxs = universe[white_cols].max(axis=1).values
        draw_mins = universe[white_cols].min(axis=1).values
        
        stats['per_draw'] = {
            'mean_of_means': float(draw_means.mean()),
            'mean_of_sums': float(draw_sums.mean()),
            'mean_of_maxs': float(draw_maxs.mean()),
            'mean_of_mins': float(draw_mins.mean())
        }
        
        # special ball statistics
        special_balls = universe[self.special_name].values
        stats['special_ball'] = {
            'mean': float(special_balls.mean()),
            'std': float(special_balls.std()),
            'min': int(special_balls.min()),
            'max': int(special_balls.max()),
            'unique_count': int(len(np.unique(special_balls)))
        }
        
        # temporal statistics (lag-1 autocorrelation of draw means)
        if len(draw_means) > 10:
            from scipy.stats import pearsonr
            autocorr_mean, _ = pearsonr(draw_means[:-1], draw_means[1:])
            stats['temporal'] = {
                'autocorr_lag1_mean': float(autocorr_mean)
            }
        
        return stats
    
    def print_statistics(self, stats: Dict[str, any]):
        """Print formatted statistics."""
        print(f"\n--- UNIVERSE STATISTICS ---")
        print(f"Lottery: {stats['lottery_type'].upper()}")
        print(f"Draws: {stats['n_draws']:,}")
        
        print("White Balls:")
        wb = stats['white_balls']
        print(f"  Mean: {wb['mean']:.2f}")
        print(f"  Std:  {wb['std']:.2f}")
        print(f"  Range: {wb['min']} - {wb['max']}")
        
        print(f"\nPer-Draw Aggregates:")
        pd_stats = stats['per_draw']
        print(f"  Avg draw mean: {pd_stats['mean_of_means']:.2f}")
        print(f"  Avg draw sum:  {pd_stats['mean_of_sums']:.2f}")
        print(f"  Avg draw max:  {pd_stats['mean_of_maxs']:.2f}")
        print(f"  Avg draw min:  {pd_stats['mean_of_mins']:.2f}")
        
        sb = stats['special_ball']
        print(f"\nSpecial Ball:")
        print(f"  Mean: {sb['mean']:.2f}")
        print(f"  Std:  {sb['std']:.2f}")
        print(f"  Unique values: {sb['unique_count']}")
        
        if 'temporal' in stats:
            temp = stats['temporal']
            print(f"\nTemporal Structure:")
            print(f"  Lag-1 autocorr (draw means): {temp['autocorr_lag1_mean']:.3f}")
            if abs(temp['autocorr_lag1_mean']) < 0.10:
                print(f"     No significant temporal structure")
            else:
                print(f"     Temporal structure detected")
        


# example usage
if __name__ == "__main__":
    print("\n--- Module 7: Universe Generators - Testing ---")
    
    # test 1: Generate Null Universe for Powerball
    print("\n[TEST 1] Generating Null Universe - Powerball")
    null_gen_pb = NullUniverseGenerator('powerball')
    universe_pb = null_gen_pb.generate(n_draws=10000, seed=42)
    
    # compute statistics
    stats_analyzer = UniverseStatistics('powerball')
    stats_pb = stats_analyzer.compute_statistics(universe_pb)
    stats_analyzer.print_statistics(stats_pb)
    
    # save universe
    null_gen_pb.save_universe(
        universe_pb,
        '/mnt/user-data/outputs/universe_null_pb.parquet',
        metadata={
            'description': 'Null universe (pure i.i.d.) for Powerball',
            'purpose': 'Gold standard benchmark for Module 7',
            'seed': 42
        }
    )
    
    
    # test 2: Generate Null Universe for Mega Millions
    print("\n[TEST 2] Generating Null Universe - Mega Millions")
    null_gen_mm = NullUniverseGenerator('megamillions')
    universe_mm = null_gen_mm.generate(n_draws=10000, seed=42)
    
    # create NEW stats analyzer for Mega Millions (not reuse Powerball one!)
    stats_analyzer_mm = UniverseStatistics('megamillions')
    stats_mm = stats_analyzer_mm.compute_statistics(universe_mm)
    stats_analyzer_mm.print_statistics(stats_mm)
    
    # save universe
    null_gen_mm.save_universe(
        universe_mm,
        '/mnt/user-data/outputs/universe_null_mm.parquet',
        metadata={
            'description': 'Null universe (pure i.i.d.) for Mega Millions',
            'purpose': 'Gold standard benchmark for Module 7',
            'seed': 42
        }
    )
    
    print("\n---  Testing complete! Null universes generated successfully. ---")
    print("\nNext steps:")
    print("1. Use these as benchmarks for all other universe evaluations")
    print("2. They should have URI approximately 1.0 (perfect realism)")
    print("3. Any model universe should be compared against these")

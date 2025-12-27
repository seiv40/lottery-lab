"""
Bootstrap Universe Generators - Module 7

Empirical resampling methods for establishing ground truth baselines.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class BootstrapUniverseGenerator:
    """Base class for bootstrap universe generators."""
    
    def __init__(self, lottery_type: str, training_data: pd.DataFrame):
        """
        Initialize bootstrap generator.
        
        Args:
            lottery_type: 'powerball' or 'megamillions'
            training_data: Historical draws to resample from
        """
        self.lottery_type = lottery_type.lower()
        
        if self.lottery_type == 'powerball':
            self.n_white = 5
            self.special_name = 'powerball'
        elif self.lottery_type == 'megamillions':
            self.n_white = 5
            self.special_name = 'megaball'
        else:
            raise ValueError(f"Unknown lottery type: {lottery_type}")
        
        # extract ball columns
        self.ball_cols = [f'white_{i+1}' for i in range(self.n_white)] + [self.special_name]
        self.training_data = training_data[self.ball_cols].copy()
        
        print(f"Bootstrap Generator initialized")
        print(f"  Lottery: {lottery_type}")
        print(f"  Training samples: {len(self.training_data)}")
    
    def save_universe(self, universe: pd.DataFrame, filepath: str, metadata: dict = None):
        """Save universe with metadata."""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'lottery_type': self.lottery_type,
            'n_draws': len(universe),
            'generated_at': datetime.now().isoformat(),
            'generator_class': self.__class__.__name__
        })
        
        if not filepath.endswith('.parquet'):
            filepath += '.parquet'
        
        universe.to_parquet(filepath, index=False)
        
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Universe saved to: {filepath}")
        print(f"Metadata saved to: {metadata_path}")


class EmpiricalBootstrapGenerator(BootstrapUniverseGenerator):
    """
    Empirical Bootstrap: Resample draws with replacement from real data.
    
    This is the empirical ground truth - what the actual lottery distribution
    looks like based on historical data. Models should match this, not beat it.
    """
    
    def generate(self, n_draws: int = 10000, seed: Optional[int] = None, verbose: bool = True):
        """
        Generate universe via empirical bootstrap (resampling with replacement).
        
        Args:
            n_draws: Number of draws to generate
            seed: Random seed
            verbose: Print progress
        
        Returns:
            universe: DataFrame of resampled draws
        """
        if seed is not None:
            np.random.seed(seed)
        
        if verbose:
            print(f"\nGenerating Empirical Bootstrap Universe")
            print(f"{'='*60}")
            print(f"Lottery: {self.lottery_type.upper()}")
            print(f"Draws: {n_draws:,}")
            print(f"Training pool: {len(self.training_data)}")
            if seed is not None:
                print(f"Seed: {seed}")
            print(f"{'='*60}\n")
        
        # resample indices with replacement
        indices = np.random.choice(len(self.training_data), size=n_draws, replace=True)
        
        # get resampled draws
        universe_draws = []
        for i, idx in enumerate(indices):
            row = self.training_data.iloc[idx]
            draw_row = {'draw_id': i + 1}
            draw_row.update(row.to_dict())
            universe_draws.append(draw_row)
        
        universe = pd.DataFrame(universe_draws)
        
        if verbose:
            print(f" Generated {len(universe):,} draws via resampling")
            self._print_summary(universe)
        
        return universe
    
    def _print_summary(self, universe: pd.DataFrame):
        """Print summary statistics."""
        print(f"\nEmpirical Bootstrap Summary:")
        print(f"-" * 40)
        
        white_cols = [f'white_{i+1}' for i in range(self.n_white)]
        all_white_balls = universe[white_cols].values.flatten()
        
        print(f"White balls:")
        print(f"  Mean: {all_white_balls.mean():.2f}")
        print(f"  Std: {all_white_balls.std():.2f}")
        print(f"  Range: {all_white_balls.min():.0f} - {all_white_balls.max():.0f}")
        
        special_balls = universe[self.special_name].values
        print(f"\n{self.special_name.capitalize()}:")
        print(f"  Mean: {special_balls.mean():.2f}")
        print(f"  Unique values: {len(np.unique(special_balls))}")
        print(f"-" * 40)


class BlockBootstrapGenerator(BootstrapUniverseGenerator):
    """
    Block Bootstrap: Resample contiguous blocks of draws.
    
    Preserves any short-term temporal dependencies that might exist.
    If block bootstrap approximately empirical bootstrap, confirms i.i.d. assumption.
    If different, suggests temporal structure.
    """
    
    def __init__(self, lottery_type: str, training_data: pd.DataFrame, block_size: int = 10):
        """
        Initialize block bootstrap generator.
        
        Args:
            lottery_type: 'powerball' or 'megamillions'
            training_data: Historical draws
            block_size: Number of consecutive draws per block
        """
        super().__init__(lottery_type, training_data)
        self.block_size = block_size
        print(f"  Block size: {block_size}")
    
    def generate(self, n_draws: int = 10000, seed: Optional[int] = None, verbose: bool = True):
        """
        Generate universe via block bootstrap.
        
        Args:
            n_draws: Number of draws to generate
            seed: Random seed
            verbose: Print progress
        
        Returns:
            universe: DataFrame of block-resampled draws
        """
        if seed is not None:
            np.random.seed(seed)
        
        if verbose:
            print(f"\nGenerating Block Bootstrap Universe")
            print(f"{'='*60}")
            print(f"Lottery: {self.lottery_type.upper()}")
            print(f"Draws: {n_draws:,}")
            print(f"Block size: {self.block_size}")
            print(f"Training pool: {len(self.training_data)}")
            if seed is not None:
                print(f"Seed: {seed}")
            print(f"{'='*60}\n")
        
        # calculate number of blocks needed
        n_blocks_needed = int(np.ceil(n_draws / self.block_size))
        
        # maximum starting index for blocks
        max_start_idx = len(self.training_data) - self.block_size
        
        if max_start_idx < 0:
            raise ValueError(f"Training data ({len(self.training_data)}) smaller than block size ({self.block_size})")
        
        universe_draws = []
        draw_id = 1
        
        for block_num in range(n_blocks_needed):
            # sample random starting index
            start_idx = np.random.randint(0, max_start_idx + 1)
            
            # extract block
            block = self.training_data.iloc[start_idx:start_idx + self.block_size]
            
            # add block to universe
            for _, row in block.iterrows():
                if draw_id > n_draws:
                    break
                
                draw_row = {'draw_id': draw_id}
                draw_row.update(row.to_dict())
                universe_draws.append(draw_row)
                draw_id += 1
            
            if draw_id > n_draws:
                break
        
        universe = pd.DataFrame(universe_draws[:n_draws])
        
        if verbose:
            print(f" Generated {len(universe):,} draws from {len(universe) // self.block_size} blocks")
            self._print_summary(universe)
        
        return universe
    
    def _print_summary(self, universe: pd.DataFrame):
        """Print summary statistics."""
        print(f"\nBlock Bootstrap Summary:")
        print(f"-" * 40)
        
        white_cols = [f'white_{i+1}' for i in range(self.n_white)]
        all_white_balls = universe[white_cols].values.flatten()
        
        print(f"White balls:")
        print(f"  Mean: {all_white_balls.mean():.2f}")
        print(f"  Std: {all_white_balls.std():.2f}")
        print(f"  Range: {all_white_balls.min():.0f} - {all_white_balls.max():.0f}")
        
        special_balls = universe[self.special_name].values
        print(f"\n{self.special_name.capitalize()}:")
        print(f"  Mean: {special_balls.mean():.2f}")
        print(f"  Unique values: {len(np.unique(special_balls))}")
        
        # check temporal correlation
        draw_means = universe[white_cols].mean(axis=1).values
        if len(draw_means) > 10:
            from scipy.stats import pearsonr
            autocorr, _ = pearsonr(draw_means[:-1], draw_means[1:])
            print(f"\nTemporal structure:")
            print(f"  Lag-1 autocorr: {autocorr:.3f}")
            if abs(autocorr) < 0.10:
                print(f"    -> Block structure not preserved")
            else:
                print(f"    -> Block structure detected")
        
        print(f"-" * 40)


# MAIN SCRIPT

def main(lottery: str, method: str = 'both'):
    """
    Generate bootstrap universes.
    
    Args:
        lottery: 'powerball' or 'megamillions'
        method: 'empirical', 'block', or 'both'
    """
    lottery = lottery.lower()
    if lottery not in ['powerball', 'megamillions']:
        raise ValueError(f"Unknown lottery: {lottery}")
    
    DATA_PATH = Path(r'C:\jackpotmath\lottery-lab\data\raw')
    OUTPUT_DIR = Path(r'C:\jackpotmath\lottery-lab\output')
    
    suffix = '_pb' if lottery == 'powerball' else '_mm'
    
    # load training data
    data_file = DATA_PATH / f"{lottery}_draws.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_parquet(data_file)
    
    print(f"\n--- GENERATING BOOTSTRAP UNIVERSES - {lottery.upper()} ---")
    
    # use 70% training data for resampling (same as models)
    n_train = int(len(df) * 0.70)
    training_data = df.iloc[:n_train]
    
    print(f"\nUsing {len(training_data)} training draws for bootstrap")
    
    # empirical Bootstrap
    if method in ['empirical', 'both']:
        print("\n--- EMPIRICAL BOOTSTRAP ---")
        
        emp_gen = EmpiricalBootstrapGenerator(lottery, training_data)
        universe_emp = emp_gen.generate(n_draws=10000, seed=42, verbose=True)
        
        output_path = OUTPUT_DIR / f"universe_bootstrap{suffix}.parquet"
        emp_gen.save_universe(
            universe_emp,
            str(output_path),
            metadata={
                'description': f'Empirical bootstrap universe for {lottery}',
                'method': 'empirical_bootstrap',
                'training_size': len(training_data),
                'resampling': 'with_replacement',
                'seed': 42
            }
        )
        print(f"\n Empirical bootstrap complete: {output_path}")
    
    # block Bootstrap
    if method in ['block', 'both']:
        print("\n--- BLOCK BOOTSTRAP ---")
        
        block_gen = BlockBootstrapGenerator(lottery, training_data, block_size=10)
        universe_block = block_gen.generate(n_draws=10000, seed=42, verbose=True)
        
        output_path = OUTPUT_DIR / f"universe_block_bootstrap{suffix}.parquet"
        block_gen.save_universe(
            universe_block,
            str(output_path),
            metadata={
                'description': f'Block bootstrap universe for {lottery}',
                'method': 'block_bootstrap',
                'block_size': 10,
                'training_size': len(training_data),
                'seed': 42
            }
        )
        print(f"\n Block bootstrap complete: {output_path}")
    
    print("\n--- BOOTSTRAP GENERATION COMPLETE ---")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bootstrap_universes.py <lottery> [method]")
        print("  lottery: 'powerball' or 'megamillions'")
        print("  method: 'empirical', 'block', or 'both' (default: 'both')")
        sys.exit(1)
    
    lottery = sys.argv[1].lower()
    method = sys.argv[2] if len(sys.argv) > 2 else 'both'
    
    main(lottery, method)

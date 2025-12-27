"""
Module 2: Feature Engineering

Generates features for lottery drawing analysis.

Creates 90+ features per drawing for statistical modeling and ML.

Input:  powerball_current_format.json, megamillions_current_format.json
Output: features_powerball.parquet, features_megamillions.parquet
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')  # suppress some annoying pandas warnings


class LotteryFeatureEngineer:
    """Feature engineering pipeline for lottery drawings."""
    
    def __init__(self, game_type: str, n_regular: int, regular_max: int, 
                 special_max: int):
        """
        Initialize for specific lottery game.
        
        game_type: 'powerball' or 'megamillions'
        n_regular: number of regular balls (5)
        regular_max: max regular ball (69 for PB, 70 for MM)
        special_max: max special ball (26 for PB, 25/24 for MM)
        """
        self.game_type = game_type
        self.n_regular = n_regular
        self.regular_max = regular_max
        self.special_max = special_max
        
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load and preprocess lottery data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['dateISO'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} drawings for {self.game_type}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def extract_basic_statistics(self, numbers: List[int]) -> Dict[str, float]:
        """
        Basic statistical features from a set of numbers.
        Returns 15 features.
        """
        numbers = np.array(numbers)
        
        features = {
            # central tendency
            'mean': np.mean(numbers),
            'median': np.median(numbers),
            'mode': stats.mode(numbers, keepdims=True)[0][0] if len(numbers) > 0 else 0,
            
            # spread measures
            'std': np.std(numbers),
            'variance': np.var(numbers),
            'range': np.max(numbers) - np.min(numbers),
            'iqr': np.percentile(numbers, 75) - np.percentile(numbers, 25),
            
            # extremes
            'min': np.min(numbers),
            'max': np.max(numbers),
            
            # sum-based features
            'sum': np.sum(numbers),
            'sum_squared': np.sum(numbers ** 2),
            
            # distribution shape
            'skewness': stats.skew(numbers),
            'kurtosis': stats.kurtosis(numbers),
            
            # coefficient of variation
            'cv': np.std(numbers) / np.mean(numbers) if np.mean(numbers) != 0 else 0,
            
            # geometric mean
            'geometric_mean': stats.gmean(numbers)
        }
        
        return features
    
    def extract_distribution_features(self, numbers: List[int]) -> Dict[str, float]:
        """
        Number distribution patterns.
        Returns 20 features.
        """
        numbers = np.array(sorted(numbers))
        
        features = {}
        
        # even/odd split
        even_count = np.sum(numbers % 2 == 0)
        features['even_count'] = even_count
        features['odd_count'] = len(numbers) - even_count
        features['even_ratio'] = even_count / len(numbers)
        
        # high/low split (above/below midpoint)
        midpoint = self.regular_max / 2
        high_count = np.sum(numbers > midpoint)
        features['high_count'] = high_count
        features['low_count'] = len(numbers) - high_count
        features['high_ratio'] = high_count / len(numbers)
        
        # decade distribution (0-9, 10-19, 20-29, etc)
        for decade in range(0, self.regular_max // 10 + 1):
            decade_count = np.sum((numbers >= decade * 10) & (numbers < (decade + 1) * 10))
            features[f'decade_{decade}_count'] = decade_count
        
        # consecutive numbers
        consecutive_count = 0
        for i in range(len(numbers) - 1):
            if numbers[i + 1] - numbers[i] == 1:
                consecutive_count += 1
        features['consecutive_count'] = consecutive_count
        features['has_consecutive'] = 1 if consecutive_count > 0 else 0
        
        # gaps between numbers
        gaps = np.diff(numbers)
        features['mean_gap'] = np.mean(gaps)
        features['std_gap'] = np.std(gaps)
        features['max_gap'] = np.max(gaps)
        features['min_gap'] = np.min(gaps)
        
        # prime numbers count
        # simple primality test
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        
        prime_count = sum(1 for n in numbers if is_prime(n))
        features['prime_count'] = prime_count
        features['prime_ratio'] = prime_count / len(numbers)
        
        return features
    
    def extract_temporal_features(self, df: pd.DataFrame, idx: int, 
                                  window_sizes: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Time-series features using historical data.
        Returns 25+ features.
        """
        features = {}
        
        # days since last drawing
        if idx > 0:
            days_since_last = (df.loc[idx, 'date'] - df.loc[idx - 1, 'date']).days
            features['days_since_last_drawing'] = days_since_last
        else:
            features['days_since_last_drawing'] = 0
        
        # lag features (look back at previous drawings)
        for lag in [1, 5, 10]:
            if idx >= lag:
                prev_numbers = df.loc[idx - lag, 'regularNumbers']
                features[f'lag_{lag}_mean'] = np.mean(prev_numbers)
                features[f'lag_{lag}_sum'] = np.sum(prev_numbers)
                features[f'lag_{lag}_range'] = np.max(prev_numbers) - np.min(prev_numbers)
            else:
                # not enough history yet
                features[f'lag_{lag}_mean'] = np.nan
                features[f'lag_{lag}_sum'] = np.nan
                features[f'lag_{lag}_range'] = np.nan
        
        # rolling window stats
        for window in window_sizes:
            if idx >= window:
                # calculate stats over the window
                window_means = []
                window_stds = []
                window_sums = []
                
                for i in range(idx - window, idx):
                    nums = df.loc[i, 'regularNumbers']
                    window_means.append(np.mean(nums))
                    window_stds.append(np.std(nums))
                    window_sums.append(np.sum(nums))
                
                features[f'rolling_{window}_mean'] = np.mean(window_means)
                features[f'rolling_{window}_std'] = np.std(window_means)
                features[f'rolling_{window}_sum_mean'] = np.mean(window_sums)
            else:
                # not enough history
                features[f'rolling_{window}_mean'] = np.nan
                features[f'rolling_{window}_std'] = np.nan
                features[f'rolling_{window}_sum_mean'] = np.nan
        
        return features
    
    def extract_gap_features(self, df: pd.DataFrame, idx: int, 
                            numbers: List[int]) -> Dict[str, float]:
        """
        Gap analysis - how long since each number was last drawn.
        Returns 15 features.
        """
        features = {}
        
        # for each number, find when it was last drawn
        gaps = []
        for num in numbers:
            # search backwards for this number
            gap_days = None
            for i in range(idx - 1, -1, -1):
                prev_numbers = df.loc[i, 'regularNumbers']
                if num in prev_numbers:
                    gap_days = (df.loc[idx, 'date'] - df.loc[i, 'date']).days
                    break
            
            if gap_days is not None:
                gaps.append(gap_days)
        
        # gap statistics
        if len(gaps) > 0:
            features['mean_gap_days'] = np.mean(gaps)
            features['median_gap_days'] = np.median(gaps)
            features['min_gap_days'] = np.min(gaps)
            features['max_gap_days'] = np.max(gaps)
            features['std_gap_days'] = np.std(gaps)
        else:
            # first few drawings won't have gap data
            features['mean_gap_days'] = np.nan
            features['median_gap_days'] = np.nan
            features['min_gap_days'] = np.nan
            features['max_gap_days'] = np.nan
            features['std_gap_days'] = np.nan
        
        # special ball gap
        special_num = df.loc[idx, 'specialNumber']
        special_gap = None
        for i in range(idx - 1, -1, -1):
            if df.loc[i, 'specialNumber'] == special_num:
                special_gap = (df.loc[idx, 'date'] - df.loc[i, 'date']).days
                break
        
        features['special_ball_gap_days'] = special_gap if special_gap is not None else np.nan
        
        # additional gap metrics
        if len(gaps) > 0:
            features['gap_range'] = np.max(gaps) - np.min(gaps)
            features['gap_cv'] = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
        else:
            features['gap_range'] = np.nan
            features['gap_cv'] = np.nan
        
        return features
    
    def extract_cooccurrence_features(self, df: pd.DataFrame, idx: int, 
                                     numbers: List[int]) -> Dict[str, float]:
        """
        Co-occurrence features - how often these numbers appeared together historically.
        Returns 10 features.
        """
        features = {}
        
        # count how many times each pair has appeared together
        if idx >= 10:  # need some history
            pair_counts = Counter()
            
            # look back at last 100 drawings (or all if less)
            lookback = min(100, idx)
            for i in range(idx - lookback, idx):
                prev_numbers = df.loc[i, 'regularNumbers']
                # count all pairs in this drawing
                for pair in combinations(prev_numbers, 2):
                    pair_counts[tuple(sorted(pair))] += 1
            
            # now check current drawing's pairs
            current_pairs = list(combinations(sorted(numbers), 2))
            pair_frequencies = [pair_counts.get(tuple(p), 0) for p in current_pairs]
            
            if len(pair_frequencies) > 0:
                features['mean_pair_frequency'] = np.mean(pair_frequencies)
                features['max_pair_frequency'] = np.max(pair_frequencies)
                features['min_pair_frequency'] = np.min(pair_frequencies)
                features['std_pair_frequency'] = np.std(pair_frequencies)
            else:
                features['mean_pair_frequency'] = 0
                features['max_pair_frequency'] = 0
                features['min_pair_frequency'] = 0
                features['std_pair_frequency'] = 0
        else:
            # not enough history
            features['mean_pair_frequency'] = np.nan
            features['max_pair_frequency'] = np.nan
            features['min_pair_frequency'] = np.nan
            features['std_pair_frequency'] = np.nan
        
        # count individual number frequencies
        if idx >= 10:
            number_counts = Counter()
            lookback = min(100, idx)
            for i in range(idx - lookback, idx):
                prev_numbers = df.loc[i, 'regularNumbers']
                for num in prev_numbers:
                    number_counts[num] += 1
            
            current_frequencies = [number_counts.get(n, 0) for n in numbers]
            features['mean_number_frequency'] = np.mean(current_frequencies)
            features['max_number_frequency'] = np.max(current_frequencies)
            features['min_number_frequency'] = np.min(current_frequencies)
            features['std_number_frequency'] = np.std(current_frequencies)
        else:
            features['mean_number_frequency'] = np.nan
            features['max_number_frequency'] = np.nan
            features['min_number_frequency'] = np.nan
            features['std_number_frequency'] = np.nan
        
        return features
    
    def extract_positional_features(self, numbers: List[int]) -> Dict[str, float]:
        """
        Positional features - where numbers fall in the range.
        Returns 10 features.
        """
        numbers = sorted(numbers)
        features = {}
        
        # position in range (normalized to 0-1)
        for i, num in enumerate(numbers):
            features[f'pos_{i+1}_normalized'] = num / self.regular_max
        
        # spacing uniformity
        # how evenly spaced are the numbers?
        ideal_spacing = self.regular_max / (len(numbers) + 1)
        actual_spacings = [numbers[0]] + list(np.diff(numbers)) + [self.regular_max - numbers[-1]]
        spacing_variance = np.var([s - ideal_spacing for s in actual_spacings])
        features['spacing_uniformity'] = -spacing_variance  # lower variance = more uniform
        
        return features
    
    def extract_seasonality_features(self, date: datetime) -> Dict[str, float]:
        """
        Date-based seasonal features.
        Returns 8 features.
        """
        features = {
            'day_of_week': date.dayofweek,  # 0=Monday, 6=Sunday
            'month': date.month,
            'year': date.year,
            'day_of_month': date.day,
            'day_of_year': date.timetuple().tm_yday,
            'quarter': (date.month - 1) // 3 + 1,
            'is_weekend': 1 if date.dayofweek >= 5 else 0,
            'week_of_year': date.isocalendar()[1]
        }
        
        return features
    
    def extract_jackpot_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Jackpot size features.
        Returns 3 features.
        """
        features = {
            'jackpot_annuitized': row.get('jackpotAnnuitized', 0),
            'jackpot_cash': row.get('jackpotCash', 0),
        }
        
        # log transform for better scale
        features['jackpot_log'] = np.log10(features['jackpot_cash'] + 1)
        
        return features
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for entire dataset.
        Returns DataFrame with 90+ features per drawing.
        """
        feature_list = []
        
        print(f"\nGenerating features for {len(df)} drawings...")
        print(f"This may take a few minutes...\n")
        
        for idx in range(len(df)):
            # progress updates
            if idx % 100 == 0:
                print(f"Processing drawing {idx + 1}/{len(df)}...")
            
            row = df.loc[idx]
            numbers = row['regularNumbers']
            
            # start with basic info
            features = {
                'drawing_index': idx,
                'date': row['date'],
                'game': self.game_type,
                'special_ball': row['specialNumber']
            }
            
            # add all feature categories
            features.update(self.extract_basic_statistics(numbers))
            features.update(self.extract_distribution_features(numbers))
            features.update(self.extract_temporal_features(df, idx))
            features.update(self.extract_gap_features(df, idx, numbers))
            features.update(self.extract_cooccurrence_features(df, idx, numbers))
            features.update(self.extract_positional_features(numbers))
            features.update(self.extract_seasonality_features(row['date']))
            features.update(self.extract_jackpot_features(row))
            
            feature_list.append(features)
        
        features_df = pd.DataFrame(feature_list)
        
        print(f"\nGenerated {len(features_df.columns)} features")
        print(f"Total drawings: {len(features_df)}")
        print(f"Missing values: {features_df.isnull().sum().sum()}")
        
        return features_df
    
    def save_features(self, features_df: pd.DataFrame, output_path: Path):
        """Save features to Parquet format"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(output_path, compression='snappy', index=False)
        
        print(f"\nFeatures saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def generate_feature_report(self, features_df: pd.DataFrame, 
                               output_path: Path):
        """Generate feature statistics report"""
        report = {
            'metadata': {
                'game': self.game_type,
                'total_drawings': len(features_df),
                'total_features': len(features_df.columns),
                'date_range': {
                    'start': features_df['date'].min().isoformat(),
                    'end': features_df['date'].max().isoformat()
                },
                'generation_timestamp': datetime.now().isoformat()
            },
            'feature_statistics': {},
            'feature_categories': {
                'basic_statistics': 15,
                'distribution_analysis': 20,
                'temporal_features': 25,
                'gap_analysis': 15,
                'cooccurrence_features': 10,
                'positional_features': 10,
                'seasonality_features': 8,
                'jackpot_features': 3,
                'metadata_fields': 4
            }
        }
        
        # stats for each feature
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['drawing_index', 'year']:
                report['feature_statistics'][col] = {
                    'mean': float(features_df[col].mean()),
                    'std': float(features_df[col].std()),
                    'min': float(features_df[col].min()),
                    'max': float(features_df[col].max()),
                    'missing_count': int(features_df[col].isnull().sum()),
                    'missing_pct': float(features_df[col].isnull().sum() / len(features_df) * 100)
                }
        
        # save report
        report_path = output_path.parent / f"feature_report_{self.game_type}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Feature report saved to: {report_path}")
        
        return report


def main():
    """Main execution"""
    print("\n--- MODULE 2: FEATURE ENGINEERING ---")
    print()
    
    # paths - update these if your structure is different
    base_path = Path(r"C:\jackpotmath\lottery-lab")
    data_dir = base_path / "data" / "processed"
    output_dir = data_dir / "features"
    
    # tried using relative paths first but absolute is clearer
    # base_path = Path(__file__).parent.parent.parent
    
    # game configs
    games_config = {
        'powerball': {
            'input_file': data_dir / 'powerball_current_format.json',
            'output_file': output_dir / 'features_powerball.parquet',
            'n_regular': 5,
            'regular_max': 69,
            'special_max': 26
        },
        'megamillions': {
            'input_file': data_dir / 'megamillions_current_format.json',
            'output_file': output_dir / 'features_megamillions.parquet',
            'n_regular': 5,
            'regular_max': 70,
            'special_max': 25  # note: changed to 24 on 2025-04-08
        }
    }
    
    # process each game
    for game_name, config in games_config.items():
        print(f"\n--- PROCESSING: {game_name.upper()} ---")
        
        # initialize engineer
        engineer = LotteryFeatureEngineer(
            game_type=game_name,
            n_regular=config['n_regular'],
            regular_max=config['regular_max'],
            special_max=config['special_max']
        )
        
        # load data
        df = engineer.load_data(config['input_file'])
        
        # generate features
        features_df = engineer.generate_features(df)
        
        # save
        engineer.save_features(features_df, config['output_file'])
        
        # generate report
        engineer.generate_feature_report(features_df, config['output_file'])
        
        print(f"\n {game_name.upper()} processing complete!")
    
    print("\n--- FEATURE ENGINEERING COMPLETE ---")
    print("\nNext Steps:")
    print("1. Review feature reports in data/processed/features/")
    print("2. Run quick_check.py to verify outputs")
    print("3. Proceed to Module 3 for visualizations")


if __name__ == "__main__":
    main()

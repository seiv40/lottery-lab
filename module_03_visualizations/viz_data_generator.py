"""
Module 3: Visualization Data Generator

Generates JSON files for the lottery visualizations on my website.
Takes the filtered data and features from Modules 1-2 and creates:
1. Heat maps - number frequency distributions with colors
2. Gap analysis - hot/cold number tracking
3. Quick stats - dashboard summary data

"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import sys


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy/pandas types.
    
    Without this, json.dump() throws errors on numpy int64, float64, etc.
    I ran into this issue and had to add this encoder.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


class VizDataGenerator:
    """Generate visualization JSON files for lottery analysis frontend."""
    
    def __init__(
        self,
        data_dir: Path,
        features_dir: Path,
        output_dir: Path,
        current_date: str = "2025-11-10"
    ):
        """
        Initialize the generator.
        
        data_dir: raw JSON files from Module 1
        features_dir: parquet files from Module 2
        output_dir: where to save the JSON files
        current_date: for gap calculations (ISO format)
        """
        self.data_dir = Path(data_dir)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.current_date = pd.to_datetime(current_date)
        
        # make sure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # game configs - these are the current formats
        self.game_config = {
            'powerball': {
                'regular_range': (1, 69),
                'special_range': (1, 26),
                'regular_count': 5,
                'special_name': 'powerball'
            },
            'megamillions': {
                'regular_range': (1, 70),
                'special_range': (1, 25),  # simplified to 1-25 for now
                'regular_count': 5,
                'special_name': 'megaball'
            }
        }
        
        print("\n--- MODULE 3: VISUALIZATION DATA GENERATOR ---")
        print(f"Input (raw data): {self.data_dir}")
        print(f"Input (features): {self.features_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Current date: {self.current_date.strftime('%Y-%m-%d')}")
        print()
    
    def load_data(self, game: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Load data for a specific game.
        
        Returns both the features DataFrame and raw drawings list.
        Need both because features has engineered stuff but raw has
        the actual drawing numbers in a cleaner format.
        """
        print(f"Loading data for {game.upper()}...")
        
        # load features from Module 2
        features_file = self.features_dir / f"features_{game}.parquet"
        features_df = pd.read_parquet(features_file)
        print(f"  Loaded {len(features_df)} drawings from features")
        
        # load raw drawings from Module 1
        raw_file = self.data_dir / f"{game}_current_format.json"
        with open(raw_file, 'r') as f:
            drawings = json.load(f)
        print(f"  Loaded {len(drawings)} drawings from raw data")
        
        return features_df, drawings
    
    def calculate_frequencies(
        self,
        drawings: List[Dict],
        game: str
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """
        Calculate how often each number has been drawn.
        
        This is the core of the heat map visualization.
        For each number, we track:
        - Total frequency (how many times drawn)
        - Percentile (compared to other numbers)
        - Color (based on percentile)
        - Position breakdown (which slot it appeared in)
        """
        config = self.game_config[game]
        regular_min, regular_max = config['regular_range']
        special_min, special_max = config['special_range']
        special_name = config['special_name']
        
        # counters for tracking occurrences
        regular_counter = Counter()
        special_counter = Counter()
        position_counters = [Counter() for _ in range(5)]  # 5 positions (0-4)
        
        # count how many times each number appears
        for drawing in drawings:
            # regular numbers
            numbers = drawing['regularNumbers']
            for pos, num in enumerate(numbers):
                regular_counter[num] += 1
                position_counters[pos][num] += 1  # track which position too
            
            # special ball
            special_counter[drawing['specialNumber']] += 1
        
        # calculate percentiles for coloring
        # this helps us scale colors from cold (blue) to hot (red)
        frequencies = list(regular_counter.values())
        if frequencies:
            # get percentile values for each frequency
            percentiles = np.percentile(frequencies, np.arange(0, 101))
        else:
            percentiles = np.zeros(101)  # shouldn't happen but just in case
        
        # build the frequency data structure for regular numbers
        regular_freq = {}
        for num in range(regular_min, regular_max + 1):
            freq = regular_counter.get(num, 0)
            # figure out what percentile this frequency is at
            percentile = self._get_percentile(freq, percentiles)
            
            regular_freq[num] = {
                'frequency': int(freq),
                'percentile': round(percentile, 2),
                'color': self._frequency_to_color(percentile),
                # track which position (1-5) this number appeared in
                'position_breakdown': {
                    str(pos + 1): int(position_counters[pos].get(num, 0))
                    for pos in range(5)
                }
            }
        
        # do the same for special numbers
        # tried to refactor this into a shared function but it's not much code
        special_frequencies = list(special_counter.values())
        if special_frequencies:
            special_percentiles = np.percentile(special_frequencies, np.arange(0, 101))
        else:
            special_percentiles = np.zeros(101)
        
        special_freq = {}
        for num in range(special_min, special_max + 1):
            freq = special_counter.get(num, 0)
            percentile = self._get_percentile(freq, special_percentiles)
            
            special_freq[num] = {
                'frequency': int(freq),
                'percentile': round(percentile, 2),
                'color': self._frequency_to_color(percentile)
            }
        
        return regular_freq, special_freq
    
    def calculate_gaps(
        self,
        drawings: List[Dict],
        game: str
    ) -> Dict[int, Dict]:
        """
        Calculate gap analysis - how long since each number was last drawn.
        
        This is for the "hot numbers" and "cold numbers" features.
        Numbers with small gaps are "hot" (recently drawn)
        Numbers with large gaps are "cold" (not drawn in a while)
        """
        config = self.game_config[game]
        regular_min, regular_max = config['regular_range']
        
        # we'll work backwards from most recent drawing
        # this is more efficient than searching forward
        gaps = {}
        
        # initialize gaps - if a number never appeared, gap is infinite
        for num in range(regular_min, regular_max + 1):
            gaps[num] = {
                'gap_days': None,  # will be set when we find it
                'last_date': None,
                'status': 'never_drawn'  # might be overwritten
            }
        
        # iterate through drawings from newest to oldest
        # stop once we've found all numbers at least once
        found_numbers = set()
        target_count = regular_max - regular_min + 1
        
        for i in range(len(drawings) - 1, -1, -1):
            drawing = drawings[i]
            drawing_date = pd.to_datetime(drawing['dateISO'])
            numbers = drawing['regularNumbers']
            
            # for each number in this drawing
            for num in numbers:
                # if we haven't found this number yet, record its gap
                if num not in found_numbers:
                    gap_days = (self.current_date - drawing_date).days
                    
                    gaps[num] = {
                        'gap_days': int(gap_days),
                        'last_date': drawing['dateISO'],
                        'status': self._categorize_gap(gap_days)
                    }
                    
                    found_numbers.add(num)
            
            # early exit if we found all numbers
            # this saves a lot of iterations for numbers that appear frequently
            if len(found_numbers) >= target_count:
                break
        
        # for numbers we never found, set gap to total days in dataset
        for num in range(regular_min, regular_max + 1):
            if gaps[num]['gap_days'] is None:
                # gap is from first drawing to current date
                first_date = pd.to_datetime(drawings[0]['dateISO'])
                gap_days = (self.current_date - first_date).days
                
                gaps[num] = {
                    'gap_days': int(gap_days),
                    'last_date': None,
                    'status': 'never_drawn'
                }
        
        return gaps
    
    def generate_heatmap(self, game: str):
        """
        Generate heatmap JSON file for a game.
        
        This file contains frequency and color data for all numbers.
        The frontend uses this to render the heat map visualization.
        """
        print(f"\n--- GENERATING HEATMAP: {game.upper()} ---")
        
        # load data
        features_df, drawings = self.load_data(game)
        
        # calculate frequencies
        print("\nCalculating frequencies...")
        regular_freq, special_freq = self.calculate_frequencies(drawings, game)
        print(f"  Processed {len(regular_freq)} regular numbers")
        print(f"  Processed {len(special_freq)} special numbers")
        
        # build JSON structure
        # converting int keys to strings because JSON requires string keys
        heatmap_data = {
            'game': game,
            'generated_at': datetime.now().isoformat(),
            'total_drawings': len(drawings),
            'date_range': {
                'start': drawings[0]['dateISO'],
                'end': drawings[-1]['dateISO']
            },
            'regular_numbers': {
                str(num): data for num, data in regular_freq.items()
            },
            'special_numbers': {
                str(num): data for num, data in special_freq.items()
            }
        }
        
        # save to file
        output_file = self.output_dir / f"{game}_heatmap.json"
        with open(output_file, 'w') as f:
            json.dump(heatmap_data, f, indent=2, cls=NumpyEncoder)
        
        # print file info
        file_size = output_file.stat().st_size / 1024  # convert to KB
        print(f"\n Heatmap generated: {output_file}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Regular numbers: {len(regular_freq)}")
        print(f"  Special numbers: {len(special_freq)}")
    
    def generate_gaps_file(self, game: str):
        """
        Generate gap analysis JSON file.
        
        This file shows which numbers are "hot" (recently drawn)
        and which are "cold" (overdue).
        """
        print(f"\n--- GENERATING GAP ANALYSIS: {game.upper()}")
        
        # load data
        features_df, drawings = self.load_data(game)
        
        # calculate gaps
        print("\nCalculating gaps...")
        gaps = self.calculate_gaps(drawings, game)
        print(f"  Calculated gaps for {len(gaps)} numbers")
        
        # categorize numbers by gap status
        hot_numbers = []  # gap < 7 days
        warm_numbers = []  # 7-30 days
        cool_numbers = []  # 30-90 days
        cold_numbers = []  # 90-180 days
        frozen_numbers = []  # > 180 days or never drawn
        
        for num, data in gaps.items():
            gap_days = data['gap_days']
            entry = {
                'number': num,
                'gap_days': gap_days,
                'last_date': data['last_date'],
                'status': data['status']
            }
            
            # categorize based on gap
            if data['status'] == 'hot':
                hot_numbers.append(entry)
            elif data['status'] == 'warm':
                warm_numbers.append(entry)
            elif data['status'] == 'cool':
                cool_numbers.append(entry)
            elif data['status'] == 'cold':
                cold_numbers.append(entry)
            else:  # frozen or never_drawn
                frozen_numbers.append(entry)
        
        # sort each category by gap (ascending for hot, descending for cold)
        hot_numbers.sort(key=lambda x: x['gap_days'])
        warm_numbers.sort(key=lambda x: x['gap_days'])
        cool_numbers.sort(key=lambda x: x['gap_days'])
        cold_numbers.sort(key=lambda x: x['gap_days'], reverse=True)
        frozen_numbers.sort(key=lambda x: x['gap_days'], reverse=True)
        
        # build JSON structure
        gaps_data = {
            'game': game,
            'generated_at': datetime.now().isoformat(),
            'analysis_date': self.current_date.isoformat(),
            'hot_numbers': hot_numbers,
            'warm_numbers': warm_numbers,
            'cool_numbers': cool_numbers,
            'cold_numbers': cold_numbers,
            'frozen_numbers': frozen_numbers,
            'summary': {
                'total_hot': len(hot_numbers),
                'total_warm': len(warm_numbers),
                'total_cool': len(cool_numbers),
                'total_cold': len(cold_numbers),
                'total_frozen': len(frozen_numbers)
            }
        }
        
        # save to file
        output_file = self.output_dir / f"{game}_gaps.json"
        with open(output_file, 'w') as f:
            json.dump(gaps_data, f, indent=2, cls=NumpyEncoder)
        
        # print summary
        file_size = output_file.stat().st_size / 1024
        print(f"\n Gap analysis generated: {output_file}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Hot: {len(hot_numbers)}, Warm: {len(warm_numbers)}, "
              f"Cool: {len(cool_numbers)}, Cold: {len(cold_numbers)}, "
              f"Frozen: {len(frozen_numbers)}")
    
    def generate_quickstats(self, game: str):
        """
        Generate quick stats JSON for dashboard.
        
        This file has summary statistics for the dashboard:
        - Most/least frequent numbers
        - Hot/cold numbers
        - Overall statistics
        """
        print(f"\n--- GENERATING QUICK STATS: {game.upper()} ---")
        
        # load data
        features_df, drawings = self.load_data(game)
        
        # get frequencies and gaps
        print("\nCalculating stats...")
        regular_freq, special_freq = self.calculate_frequencies(drawings, game)
        gaps = self.calculate_gaps(drawings, game)
        
        # combine frequency and gap data
        # this gives us a complete picture of each number
        combined_data = []
        for num, freq_data in regular_freq.items():
            gap_data = gaps[num]
            combined_data.append({
                'number': num,
                'frequency': freq_data['frequency'],
                'percentile': freq_data['percentile'],
                'gap_days': gap_data['gap_days'],
                'status': gap_data['status']
            })
        
        # sort by gap (ascending) to get hot numbers
        hot_numbers = sorted(
            [d for d in combined_data if d['gap_days'] is not None and d['gap_days'] < 30],
            key=lambda x: x['gap_days']
        )[:10]  # top 10 hottest
        
        # sort by gap (descending) to get cold numbers
        cold_numbers = sorted(
            [d for d in combined_data if d['gap_days'] is not None],
            key=lambda x: x['gap_days'],
            reverse=True
        )[:10]  # top 10 coldest
        
        # overdue numbers (high frequency but large gap)
        # these are numbers that usually appear often but haven't recently
        overdue_numbers = sorted(
            [d for d in combined_data 
             if d['percentile'] > 50 and d['gap_days'] is not None and d['gap_days'] > 30],
            key=lambda x: (x['percentile'], x['gap_days']),
            reverse=True
        )[:10]
        
        # get date range info
        start_date = drawings[0]['dateISO']
        end_date = drawings[-1]['dateISO']
        last_drawing = drawings[-1]
        
        # most frequent numbers (all time)
        most_frequent = sorted(
            [
                {
                    'number': num,
                    'frequency': data['frequency'],
                    'percentage': round(data['frequency'] / len(drawings) * 100, 2)
                }
                for num, data in regular_freq.items()
            ],
            key=lambda x: x['frequency'],
            reverse=True
        )[:10]
        
        # least frequent numbers
        least_frequent = sorted(
            [
                {
                    'number': num,
                    'frequency': data['frequency'],
                    'percentage': round(data['frequency'] / len(drawings) * 100, 2)
                }
                for num, data in regular_freq.items()
            ],
            key=lambda x: x['frequency']
        )[:10]
        
        # build JSON structure
        quickstats_data = {
            'game': game,
            'generated_at': datetime.now().isoformat(),
            'current_stats': {
                'total_drawings': len(drawings),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                },
                'last_drawing': {
                    'date': last_drawing['dateISO'],
                    'numbers': last_drawing['regularNumbers'],
                    self.game_config[game]['special_name']: last_drawing['specialNumber']
                }
            },
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'overdue_numbers': overdue_numbers,
            'most_frequent': most_frequent,
            'least_frequent': least_frequent,
            'statistics': {
                'avg_frequency': round(float(np.mean([d['frequency'] for d in regular_freq.values()])), 2),
                'std_frequency': round(float(np.std([d['frequency'] for d in regular_freq.values()])), 2),
                'total_hot': len([n for n in hot_numbers if n['gap_days'] < 7]),
                'total_cold': len([n for n in cold_numbers if n['gap_days'] > 90]),
                'total_overdue': len(overdue_numbers)
            }
        }
        
        # save to file
        output_file = self.output_dir / f"{game}_quickstats.json"
        with open(output_file, 'w') as f:
            json.dump(quickstats_data, f, indent=2, cls=NumpyEncoder)
        
        file_size = output_file.stat().st_size / 1024
        print(f"\n Quick stats generated: {output_file}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Hot numbers: {len(hot_numbers)}")
        print(f"  Cold numbers: {len(cold_numbers)}")
        print(f"  Overdue numbers: {len(overdue_numbers)}")
    
    def generate_all(self) -> None:
        """Generate all visualization files for both games."""
        print("\n--- GENERATING ALL VISUALIZATION FILES ---")
        
        games = ['powerball', 'megamillions']
        
        for game in games:
            try:
                # generate all three file types for this game
                self.generate_heatmap(game)
                self.generate_gaps_file(game)
                self.generate_quickstats(game)
            except Exception as e:
                # if one game fails, still try the other
                print(f"\n✗ Error processing {game}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n--- GENERATION COMPLETE ---")
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary of all generated files."""
        print("\nGenerated files:")
        
        # list all JSON files with sizes
        for file in sorted(self.output_dir.glob("*.json")):
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name:35s} {size_kb:>8.2f} KB")
        
        # total size
        total_size = sum(f.stat().st_size for f in self.output_dir.glob("*.json")) / 1024
        print(f"\nTotal size: {total_size:.2f} KB")
        print(f"Output directory: {self.output_dir}")
    
    @staticmethod
    def _get_percentile(value: float, percentiles: np.ndarray) -> float:
        """
        Figure out what percentile a value is at.
        
        This is used for coloring the heat map.
        Higher percentile = drawn more often = hotter color
        """
        if len(percentiles) == 0:
            return 50.0  # default to middle if no data
        
        # find where this value would fit in the sorted percentile array
        rank = np.searchsorted(percentiles, value)
        return min(100.0, max(0.0, rank))
    
    @staticmethod
    def _frequency_to_color(percentile: float) -> str:
        """
        Convert frequency percentile to a color.
        
        Color scale I'm using:
        - Hot (75-100): Red shades
        - Medium (25-75): Yellow to green
        - Cold (0-25): Blue shades
        
        I spent way too much time tweaking these colors to look good.
        """
        if percentile >= 75:
            # hot numbers - red shades
            t = (percentile - 75) / 25  # normalize to 0-1
            r = 255
            g = int(68 + t * (136 - 68))
            b = int(68 + t * (68 - 68))
            return f"#{r:02x}{g:02x}{b:02x}"
        elif percentile >= 50:
            # warm numbers - yellow to light green
            t = (percentile - 50) / 25
            r = int(255 - t * (255 - 136))
            g = int(204 - t * (204 - 204))
            b = int(68 + t * (68 - 68))
            return f"#{r:02x}{g:02x}{b:02x}"
        elif percentile >= 25:
            # cool numbers - light green to green
            t = (percentile - 25) / 25
            r = int(136 - t * (136 - 136))
            g = int(204 - t * (204 - 204))
            b = int(136 - t * (136 - 68))
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            # cold numbers - blue shades
            t = percentile / 25
            r = int(68 - t * (68 - 68))
            g = int(68 + t * (136 - 68))
            b = 255
            return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def _categorize_gap(gap_days: int) -> str:
        """
        Categorize a gap into hot/warm/cool/cold/frozen.
        
        These thresholds are somewhat arbitrary but seem reasonable:
        - hot: < 7 days (drawn this week)
        - warm: 7-30 days (drawn this month)
        - cool: 30-90 days (drawn in last 3 months)
        - cold: 90-180 days (drawn in last 6 months)
        - frozen: > 180 days (not drawn in 6+ months)
        """
        if gap_days < 7:
            return 'hot'
        elif gap_days < 30:
            return 'warm'
        elif gap_days < 90:
            return 'cool'
        elif gap_days < 180:
            return 'cold'
        else:
            return 'frozen'


def main():
    """Main execution"""
    # paths - update these for your setup
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    data_dir = base_dir / "data" / "processed"
    features_dir = data_dir / "features"
    output_dir = base_dir / "outputs" / "viz_data"
    
    # tried using relative paths but absolute is clearer
    # base_dir = Path(__file__).parent.parent.parent
    
    # initialize generator
    generator = VizDataGenerator(
        data_dir=data_dir,
        features_dir=features_dir,
        output_dir=output_dir,
        current_date="2025-11-10"  # update this to today's date when running
    )
    
    # generate all files
    generator.generate_all()
    
    print("\n--- MODULE 3 COMPLETE ---")
    print("\nNext steps:")
    print("1. Check the JSON files in the output directory")
    print("2. Run tests: python test_viz_data.py")
    print("3. Copy files to your frontend public folder if needed")


if __name__ == "__main__":
    main()

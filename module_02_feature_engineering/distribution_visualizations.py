"""
Module 2 Visualization: Feature Distribution Plots

Creates comparative distribution plots for key features.

Generates 3 stacked bar charts comparing Powerball vs Mega Millions:
- Even count distribution (0-5 even numbers per drawing)
- Range distribution (max - min of main numbers)
- Sum distribution (sum of 5 main numbers)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# set clean style
plt.style.use('seaborn-v0_8-whitegrid')


def load_features(features_dir: Path, game: str) -> pd.DataFrame:
    """Load feature parquet file for a game."""
    file_path = features_dir / f"features_{game}.parquet"
    
    print(f"Loading {game} features from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df)} drawings with {len(df.columns)} features")
    
    return df


def plot_even_count_distribution(pb_df: pd.DataFrame, mm_df: pd.DataFrame, 
                                 output_dir: Path):
    """
    Plot distribution of even number counts (0-5).
    
    This shows how often each drawing has 0, 1, 2, 3, 4, or 5 even numbers.
    If random, this should follow a binomial distribution.
    """
    print("\nCreating even count distribution plot...")
    
    # count occurrences for each game
    pb_counts = pb_df['even_count'].value_counts().sort_index()
    mm_counts = mm_df['even_count'].value_counts().sort_index()
    
    # make sure all values 0-5 are present
    all_values = range(6)
    pb_counts = pb_counts.reindex(all_values, fill_value=0)
    mm_counts = mm_counts.reindex(all_values, fill_value=0)
    
    # create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.8
    x = np.arange(len(all_values))
    
    # powerball on bottom (orange)
    ax.bar(x, pb_counts.values, width, label='Powerball', 
           color='#E69F00', alpha=0.8, edgecolor='white', linewidth=1)
    
    # mega millions on top (teal/blue)
    ax.bar(x, mm_counts.values, width, bottom=pb_counts.values, 
           label='Mega Millions', color='#7DBAAF', alpha=0.8, 
           edgecolor='white', linewidth=1)
    
    # labels and formatting
    ax.set_xlabel('Number of even main balls in a draw (0-5)', fontsize=14)
    ax.set_ylabel('Number of draws', fontsize=14)
    ax.set_title('Even count distribution', fontsize=20, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in all_values])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # save
    output_file = output_dir / "even_count_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file.name}")
    
    plt.close()


def plot_range_distribution(pb_df: pd.DataFrame, mm_df: pd.DataFrame, 
                           output_dir: Path):
    """
    Plot distribution of ranges (max - min).
    
    Range tells us the spread of numbers in each drawing.
    Small range = numbers clustered together
    Large range = numbers spread out
    """
    print("\nCreating range distribution plot...")
    
    # get range values
    pb_range = pb_df['range'].dropna()
    mm_range = mm_df['range'].dropna()
    
    # create bins for histogram
    # range can be from 4 (e.g., 65,66,67,68,69) to 68 (1,2,3,4,69)
    bins = np.arange(0, 75, 2)  # bins of width 2
    
    # calculate histograms
    pb_hist, _ = np.histogram(pb_range, bins=bins)
    mm_hist, _ = np.histogram(mm_range, bins=bins)
    
    # create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    width = 1.8  # slightly less than bin width for visual spacing
    x = bins[:-1] + 1  # center of each bin
    
    # powerball on bottom (orange)
    ax.bar(x, pb_hist, width, label='Powerball', 
           color='#E69F00', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # mega millions on top (teal)
    ax.bar(x, mm_hist, width, bottom=pb_hist, 
           label='Mega Millions', color='#7DBAAF', alpha=0.8, 
           edgecolor='white', linewidth=0.5)
    
    # labels and formatting
    ax.set_xlabel('Range of main numbers (max - min)', fontsize=14)
    ax.set_ylabel('Number of draws', fontsize=14)
    ax.set_title('Range distribution', fontsize=20, pad=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # set x-axis limits
    ax.set_xlim(0, 75)
    
    # clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # save
    output_file = output_dir / "range_main_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file.name}")
    
    plt.close()


def plot_sum_distribution(pb_df: pd.DataFrame, mm_df: pd.DataFrame, 
                         output_dir: Path):
    """
    Plot distribution of sums.
    
    Sum of the 5 main numbers should follow a roughly normal distribution
    centered around the expected value.
    
    For Powerball (1-69): Expected mean ≈ 175 (5 * 35)
    For Mega Millions (1-70): Expected mean ≈ 177.5 (5 * 35.5)
    """
    print("\nCreating sum distribution plot...")
    
    # get sum values
    pb_sum = pb_df['sum'].dropna()
    mm_sum = mm_df['sum'].dropna()
    
    # create bins for histogram
    # sum can range from 15 (1+2+3+4+5) to ~345 (65+66+67+68+69)
    bins = np.arange(40, 320, 5)  # bins of width 5
    
    # calculate histograms
    pb_hist, _ = np.histogram(pb_sum, bins=bins)
    mm_hist, _ = np.histogram(mm_sum, bins=bins)
    
    # create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    width = 4.5  # slightly less than bin width
    x = bins[:-1] + 2.5  # center of each bin
    
    # powerball on bottom (orange)
    ax.bar(x, pb_hist, width, label='Powerball', 
           color='#E69F00', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # mega millions on top (teal)
    ax.bar(x, mm_hist, width, bottom=pb_hist, 
           label='Mega Millions', color='#7DBAAF', alpha=0.8, 
           edgecolor='white', linewidth=0.5)
    
    # labels and formatting
    ax.set_xlabel('Sum of main numbers', fontsize=14)
    ax.set_ylabel('Number of draws', fontsize=14)
    ax.set_title('Distribution of draw sums', fontsize=20, pad=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # save
    output_file = output_dir / "sum_main_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file.name}")
    
    plt.close()


def create_all_distribution_plots(features_dir: Path, output_dir: Path):
    """Create all 3 distribution comparison plots."""
    print("\n--- MODULE 2 FEATURE DISTRIBUTION VISUALIZATIONS ---")
    
    # make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load both games
    print("\nLoading feature data...")
    pb_df = load_features(features_dir, 'powerball')
    mm_df = load_features(features_dir, 'megamillions')
    
    # create plots
    plot_even_count_distribution(pb_df, mm_df, output_dir)
    plot_range_distribution(pb_df, mm_df, output_dir)
    plot_sum_distribution(pb_df, mm_df, output_dir)
    
    print("\n--- VISUALIZATION COMPLETE ---")
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - even_count_distribution.png")
    print("  - range_main_distribution.png")
    print("  - sum_main_distribution.png")


def main():
    """Main execution"""
    # paths
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    features_dir = base_dir / "data" / "processed" / "features"
    output_dir = base_dir / "outputs" / "module_02_visualizations"
    
    # create plots
    create_all_distribution_plots(features_dir, output_dir)


if __name__ == "__main__":
    main()

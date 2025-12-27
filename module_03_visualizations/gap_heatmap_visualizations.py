"""
Module 3 Visualization: Gap Length and Frequency Heatmap Plots

Creates visualizations showing number appearance patterns.

Generates 3 plots:
- Gap length distribution (stacked histogram)
- Powerball frequency heatmap (main + special ball)
- Mega Millions frequency heatmap (main + special ball)

"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# set clean style
plt.style.use('seaborn-v0_8-whitegrid')


def load_lottery_data(data_dir: Path, game: str) -> list:
    """Load filtered lottery data."""
    file_path = data_dir / f"{game}_current_format.json"
    
    print(f"Loading {game} data from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} drawings")
    
    return data


def calculate_gap_lengths(data: list, max_number: int) -> list:
    """
    Calculate gap lengths (drawings between appearances) for all numbers.
    
    Returns a list of gap lengths across all numbers.
    """
    # track last appearance of each number
    last_seen = {}
    gap_lengths = []
    
    for draw_idx, drawing in enumerate(data):
        numbers = drawing['regularNumbers']
        
        for num in numbers:
            if num in last_seen:
                # calculate gap since last appearance
                gap = draw_idx - last_seen[num] - 1
                gap_lengths.append(gap)
            
            # update last seen
            last_seen[num] = draw_idx
    
    return gap_lengths


def plot_gap_length_distribution(pb_data: list, mm_data: list, output_dir: Path):
    """
    Plot distribution of gap lengths between consecutive appearances.
    
    Gap length = number of drawings between two consecutive appearances
    of the same number.
    
    If lottery is random, gap lengths should follow geometric distribution.
    """
    print("\nCreating gap length distribution plot...")
    
    # calculate gap lengths for both games
    pb_gaps = calculate_gap_lengths(pb_data, 69)
    mm_gaps = calculate_gap_lengths(mm_data, 70)
    
    print(f"  Powerball: {len(pb_gaps)} gaps calculated")
    print(f"  Mega Millions: {len(mm_gaps)} gaps calculated")
    
    # create bins for histogram
    max_gap = max(max(pb_gaps) if pb_gaps else 0, max(mm_gaps) if mm_gaps else 0)
    bins = np.arange(0, min(max_gap + 5, 105), 2)  # bins of width 2, cap at 100
    
    # calculate histograms
    pb_hist, _ = np.histogram(pb_gaps, bins=bins)
    mm_hist, _ = np.histogram(mm_gaps, bins=bins)
    
    # create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    width = 1.8
    x = bins[:-1] + 1  # center of each bin
    
    # powerball on bottom (orange)
    ax.bar(x, pb_hist, width, label='Powerball', 
           color='#E69F00', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # mega millions on top (teal)
    ax.bar(x, mm_hist, width, bottom=pb_hist, 
           label='Mega Millions', color='#7DBAAF', alpha=0.8, 
           edgecolor='white', linewidth=0.5)
    
    # labels and formatting
    ax.set_xlabel('Gap length in number of draws', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of gap lengths between appearances', fontsize=20, pad=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # set x-axis limits
    ax.set_xlim(0, 105)
    
    # clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # save
    output_file = output_dir / "gap_length_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file.name}")
    
    plt.close()


def calculate_frequencies(data: list, regular_max: int, special_max: int) -> tuple:
    """
    Calculate frequency of each number appearing.
    
    Returns:
        (regular_counts, special_counts) - arrays of frequencies
    """
    # initialize count arrays
    regular_counts = np.zeros(regular_max, dtype=int)
    special_counts = np.zeros(special_max, dtype=int)
    
    for drawing in data:
        # count regular numbers
        for num in drawing['regularNumbers']:
            if 1 <= num <= regular_max:
                regular_counts[num - 1] += 1
        
        # count special number
        special_num = drawing['specialNumber']
        if 1 <= special_num <= special_max:
            special_counts[special_num - 1] += 1
    
    return regular_counts, special_counts


def plot_frequency_heatmap(data: list, game: str, regular_max: int, 
                          special_max: int, output_dir: Path):
    """
    Create frequency heatmap showing main balls vs special ball.
    
    Two rows:
    - Top: All main numbers (69 or 70)
    - Bottom: Special ball (26 or 25)
    
    Uses viridis colormap (purple = low, yellow = high).
    """
    print(f"\nCreating {game} frequency heatmap...")
    
    # calculate frequencies
    regular_counts, special_counts = calculate_frequencies(data, regular_max, special_max)
    
    # create 2D array for heatmap
    # need to pad special_counts to match regular_counts length
    heatmap_data = np.zeros((2, regular_max))
    heatmap_data[0, :] = regular_counts
    heatmap_data[1, :special_max] = special_counts
    # rest of row 2 stays as 0 (will show as dark purple)
    
    # create heatmap
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # use viridis colormap
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    
    # labels
    if game == 'powerball':
        ball_name = "Powerball"
        title = "Powerball frequency heatmap (main vs Powerball)"
    else:
        ball_name = "Mega Ball"
        title = "Mega Millions frequency heatmap (main vs Mega Ball)"
    
    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel('Number index', fontsize=14)
    
    # y-axis labels
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Main balls', ball_name], fontsize=14)
    
    # x-axis ticks - show every 10
    x_ticks = list(range(0, regular_max, 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=12)
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax, label='Count')
    cbar.ax.tick_params(labelsize=12)
    
    # grid
    ax.set_xticks(np.arange(regular_max) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', size=0)
    
    plt.tight_layout()
    
    # save
    output_file = output_dir / f"{game}_frequency_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file.name}")
    
    plt.close()


def create_all_module3_visualizations(data_dir: Path, output_dir: Path):
    """Create all Module 3 visualizations."""
    print("\n--- MODULE 3 GAP LENGTH AND FREQUENCY VISUALIZATIONS ---")
    
    # make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load data for both games
    print("\nLoading lottery data...")
    pb_data = load_lottery_data(data_dir, 'powerball')
    mm_data = load_lottery_data(data_dir, 'megamillions')
    
    # create plots
    plot_gap_length_distribution(pb_data, mm_data, output_dir)
    plot_frequency_heatmap(pb_data, 'powerball', 69, 26, output_dir)
    plot_frequency_heatmap(mm_data, 'megamillions', 70, 25, output_dir)
    
    print("\n--- VISUALIZATION COMPLETE ---")
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - gap_length_distribution.png")
    print("  - powerball_frequency_heatmap.png")
    print("  - megamillions_frequency_heatmap.png")


def main():
    """Main execution"""
    # paths
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    data_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "module_03_visualizations"
    
    # create plots
    create_all_module3_visualizations(data_dir, output_dir)


if __name__ == "__main__":
    main()

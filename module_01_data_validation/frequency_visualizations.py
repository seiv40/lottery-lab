"""
Module 1 Visualization: Frequency Distribution Plots

Creates simple bar charts showing how often each number appears
in the current format lottery drawings.

Generates 4 plots:
- Powerball main numbers (1-69)
- Powerball special ball (1-26)
- Mega Millions main numbers (1-70)
- Mega Millions special ball (1-25)

"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# set clean style
plt.style.use('seaborn-v0_8-whitegrid')


def load_lottery_data(data_dir: Path, game: str) -> dict:
    """
    Load filtered lottery data from Module 1 output.
    """
    file_path = data_dir / f"{game}_current_format.json"
    
    print(f"Loading {game} data from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} drawings")
    
    return data


def calculate_frequencies(data: list, number_type: str, max_number: int, game: str) -> np.ndarray:
    """
    Calculate frequency of each number appearing in drawings.
    
    Args:
        data: List of drawing dictionaries
        number_type: 'regular' or 'special'
        max_number: Maximum number in range (69/70 for regular, 26/25 for special)
        game: 'powerball' or 'megamillions'
    
    Returns:
        Array of counts for each number (index 0 = number 1, etc.)
    """
    # initialize counts array
    counts = np.zeros(max_number, dtype=int)
    
    # count occurrences
    for drawing in data:
        if number_type == 'regular':
            numbers = drawing['regularNumbers']
        else:
            # both games use 'specialNumber' field
            numbers = [drawing['specialNumber']]
        
        for num in numbers:
            if 1 <= num <= max_number:
                counts[num - 1] += 1  # convert to 0-indexed
    
    return counts


def plot_frequency_distribution(
    counts: np.ndarray,
    game: str,
    number_type: str,
    output_file: Path
):
    """
    Create frequency distribution bar chart.
    
    """
    max_number = len(counts)
    numbers = np.arange(1, max_number + 1)  # 1-indexed numbers
    
    # create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # create bar chart with orange color
    ax.bar(numbers, counts, color='#E69F00', edgecolor='white', linewidth=0.5)
    
    # labels and title
    if number_type == 'main':
        title = f"{game.replace('_', ' ').title()} main number frequencies"
        xlabel = f"{game.replace('_', ' ').title()} main numbers (1-{max_number})"
    else:
        if game == 'powerball':
            ball_name = "Powerball"
        else:
            ball_name = "Mega Ball"
        title = f"{game.replace('_', ' ').title()} special ball frequencies"
        xlabel = f"{ball_name} number (1-{max_number})"
    
    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Count across all current-format draws', fontsize=14)
    
    # grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # grid behind bars
    
    # set x-axis to show all numbers
    # for main numbers, show every number
    # for special balls, show every number
    if max_number <= 30:
        # special balls - show every number
        ax.set_xticks(numbers)
    else:
        # main numbers - show every 10
        tick_positions = list(range(0, max_number + 1, 10))
        if tick_positions[0] == 0:
            tick_positions[0] = 1  # start at 1 not 0
        ax.set_xticks(tick_positions)
    
    # clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # save
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_file.name}")
    
    plt.close()


def create_all_frequency_plots(data_dir: Path, output_dir: Path):
    """
    Create all 4 frequency distribution plots.
    """
    print("\n--- MODULE 1 FREQUENCY VISUALIZATIONS ---")

    
    # make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # define games and their parameters
    games_config = {
        'powerball': {
            'main_max': 69,
            'special_max': 26
        },
        'megamillions': {
            'main_max': 70,
            'special_max': 25
        }
    }
    
    for game, config in games_config.items():
        print(f"\n{game.upper()}:")
        
        # load data
        data = load_lottery_data(data_dir, game)
        
        # main numbers frequency
        print("  Calculating main number frequencies...")
        main_counts = calculate_frequencies(data, 'regular', config['main_max'], game)
        output_file = output_dir / f"{game}_main_frequency.png"
        plot_frequency_distribution(main_counts, game, 'main', output_file)
        
        # special ball frequency
        print("  Calculating special ball frequencies...")
        special_counts = calculate_frequencies(data, 'special', config['special_max'], game)
        output_file = output_dir / f"{game}_special_frequency.png"
        plot_frequency_distribution(special_counts, game, 'special', output_file)
    
    print("\n--- VISUALIZATION COMPLETE ---")
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - powerball_main_frequency.png")
    print("  - powerball_special_frequency.png")
    print("  - megamillions_main_frequency.png")
    print("  - megamillions_special_frequency.png")


def main():
    """Main execution"""
    # paths
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    data_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "module_01_visualizations"
    
    # create plots
    create_all_frequency_plots(data_dir, output_dir)


if __name__ == "__main__":
    main()

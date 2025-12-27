"""
Module 4 Visualization: Bayesian Analysis Plots

Creates simple visualizations for the Bayesian analysis results.
Generates a bar chart showing Bayes factors for all models.

Note: This is a simplified version. For publication-quality plots,
you'd want to add more sophisticated visualizations.

"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results(results_path: Path) -> dict:
    """
    Load Bayesian results from JSON file.
    """
    print(f"Loading results from {results_path}...")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data


def plot_bayes_factors(data: dict, output_dir: Path):
    """
    Create bar chart of Bayes factors for models 1-4.
    
    Note: Model 5 uses R² instead of BF, so it's plotted separately.
    Model 4 is marked as flawed and shown in gray.
    Model 3's high BF is expected (balls are sorted).
    """
    print("\nCreating Bayes factor comparison plot...")
    
    # set up figure with 2 subplots (one for each game)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bayesian Analysis: Evidence for Randomness', 
                 fontsize=16, fontweight='bold')
    
    games = ['powerball', 'megamillions']
    
    for idx, game in enumerate(games):
        ax = axes[idx]
        
        if game not in data or 'error' in data[game]:
            ax.text(0.5, 0.5, f'No data for {game}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # extract model names and BFs (skip model 5 - it uses R²)
        model_names = []
        bayes_factors = []
        colors = []
        
        for model_key in ['model1', 'model2', 'model3', 'model4']:
            if model_key not in data[game]:
                continue
                
            model_data = data[game][model_key]
            
            if 'bayes_factor' not in model_data:
                continue
            
            # get display name
            name = model_data.get('model_name', model_key)
            name = name.replace('_', ' ').title()
            
            # add note for special cases
            if model_key == 'model3':
                name += '*'  # mark with asterisk
            elif model_key == 'model4':
                name += ' (flawed)'
            
            model_names.append(name)
            bf = model_data['bayes_factor']
            bayes_factors.append(bf)
            
            # color bars based on evidence and special cases
            if model_key == 'model4':
                # model 4 is flawed - gray it out
                colors.append('gray')
            elif model_key == 'model3':
                # model 3 has high BF but it's expected - use orange
                colors.append('orange')
            elif bf < 0.01:
                # green = evidence for randomness
                colors.append('green')
            elif bf < 100:
                # yellow = weak/moderate
                colors.append('gold')
            else:
                # red = evidence against randomness
                colors.append('red')
        
        if not model_names:
            ax.text(0.5, 0.5, f'No valid models for {game}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # create bar chart
        y_pos = np.arange(len(model_names))
        
        # plot on log scale because BFs can range from 0.0001 to 10^77
        bars = ax.barh(y_pos, bayes_factors, color=colors, alpha=0.7, edgecolor='black')
        
        # add vertical line at BF = 1 (no evidence either way)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1, label='BF = 1 (neutral)')
        
        # add vertical lines at interpretation thresholds
        ax.axvline(x=0.01, color='green', linestyle=':', linewidth=1, 
                  label='BF < 0.01 (random)')
        ax.axvline(x=100, color='red', linestyle=':', linewidth=1, 
                  label='BF > 100 (biased)')
        
        # labels and formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Bayes Factor (log scale)', fontsize=12)
        ax.set_title(f'{game.upper()}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')  # log scale for easier reading
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # add BF values as text on bars
        for i, (bar, bf) in enumerate(zip(bars, bayes_factors)):
            width = bar.get_width()
            # for very large BF, use scientific notation
            if bf > 1e10:
                text = f'{bf:.2e}'
            else:
                text = f'{bf:.4f}'
            ax.text(width * 1.1, i, text, va='center', fontsize=8)
    
    # add footnote about Model 3
    fig.text(0.5, 0.02, 
            '* Model 3 high BF is expected - lottery balls are drawn in sorted order',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # make room for footnote
    
    # save plot
    output_file = output_dir / "bayes_factors_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved plot: {output_file}")
    
    plt.close()


def plot_regression_results(data: dict, output_dir: Path):
    """
    Create separate plot for Model 5 regression results (R² metric).
    """
    print("\nCreating Model 5 regression results plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Model 5: Feature Predictive Power (R²)', 
                 fontsize=16, fontweight='bold')
    
    games = ['powerball', 'megamillions']
    r2_values = []
    r2_cis = []
    game_labels = []
    
    # extract R² values
    for game in games:
        if game not in data or 'error' in data[game]:
            continue
        
        if 'model5' not in data[game]:
            continue
        
        model_data = data[game]['model5']
        
        if 'r2_mean' in model_data:
            r2_values.append(model_data['r2_mean'])
            r2_cis.append(model_data.get('r2_ci', [0, 0]))
            game_labels.append(game.upper())
    
    if not r2_values:
        print("  ! No R² data to plot")
        return
    
    # create bar chart
    x_pos = np.arange(len(game_labels))
    bars = ax.bar(x_pos, r2_values, color='steelblue', alpha=0.7, edgecolor='black')
    
    # add error bars (credible intervals)
    for i, (r2, ci) in enumerate(zip(r2_values, r2_cis)):
        error = [[r2 - ci[0]], [ci[1] - r2]]
        ax.errorbar(i, r2, yerr=error, fmt='none', 
                   ecolor='black', capsize=5, capthick=2)
    
    # add horizontal line at R² = 0.05 (threshold for "negligible")
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
              label='R² = 0.05 (negligible threshold)')
    
    # add text showing R² values
    for i, (bar, r2) in enumerate(zip(bars, r2_values)):
        height = bar.get_height()
        ax.text(i, height + 0.002, f'{r2:.4f}', 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # labels and formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(game_labels)
    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12)
    ax.set_ylim(0, max(0.1, max(r2_values) * 1.3))  # scale y-axis appropriately
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=10)
    
    # add interpretation text
    ax.text(0.5, 0.95, 'R² < 0.05 = Features have negligible predictive power (random)',
           ha='center', va='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()
    
    # save plot
    output_file = output_dir / "model5_r2_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved plot: {output_file}")
    
    plt.close()


def plot_interpretation_summary(data: dict, output_dir: Path):
    """
    Create a simple text summary of interpretations.
    
    This is saved as a text file since it's easier to read than a plot.
    """
    print("\nCreating interpretation summary...")
    
    output_file = output_dir / "interpretation_summary.txt"
    
    # use UTF-8 encoding to handle Unicode characters (H₀, H₁ subscripts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n--- BAYESIAN ANALYSIS - INTERPRETATION SUMMARY ---\n")
        
        for game in ['powerball', 'megamillions']:
            if game not in data or 'error' in data[game]:
                f.write(f"\n{game.upper()}: No data\n")
                continue
            
            f.write(f"\n{game.upper()}:\n")
            
            for model_key, model_data in data[game].items():
                if 'model_name' not in model_data:
                    continue
                
                name = model_data['model_name']
                interp = model_data.get('interpretation', 'No interpretation')
                
                f.write(f"\n{name}:\n")
                
                # Model 5 uses R² instead of BF
                if model_key == 'model5':
                    r2 = model_data.get('r2_mean', 'N/A')
                    r2_ci = model_data.get('r2_ci', [0, 0])
                    n_sig = model_data.get('n_significant', 0)
                    n_total = model_data.get('n_features_selected', 0)
                    
                    f.write(f"  R²: {r2:.4f} (95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}])\n")
                    f.write(f"  Significant features: {n_sig}/{n_total}\n")
                    f.write(f"  Interpretation: {interp}\n")
                else:
                    # Models 1-4 use Bayes Factor
                    bf = model_data.get('bayes_factor', 'N/A')
                    
                    if isinstance(bf, (int, float)) and bf > 1e10:
                        bf_str = f"{bf:.2e}"
                    else:
                        bf_str = f"{bf}"
                    
                    f.write(f"  Bayes Factor: {bf_str}\n")
                    f.write(f"  Interpretation: {interp}\n")
                    
                    # Add special notes
                    if model_key == 'model3':
                        f.write(f"  NOTE: High BF expected - balls are sorted by position\n")
                    elif model_key == 'model4':
                        f.write(f"  NOTE: Model implementation flawed - ignore results\n")
        
        f.write("\n--- OVERALL CONCLUSION: ---\n")
        f.write("Models 1, 2, and 5 all provide evidence for lottery randomness:\n\n")
        f.write("Model 1 (Uniform): BF < 0.01 = Decisive evidence for randomness\n")
        f.write("Model 2 (Temporal): BF ≈ 0.2 = Moderate evidence for randomness\n")
        f.write("Model 3 (Position): High BF expected (balls sorted) - not a randomness test\n")
        f.write("Model 4 (Hierarchical): Implementation flawed - ignore\n")
        f.write("Model 5 (Regression): R² < 0.05 = Features cannot predict (random)\n\n")
        
        f.write("\n--- BAYES FACTOR INTERPRETATION (Jeffreys' Scale): ---\n")
        f.write("BF < 0.01   = Decisive evidence for randomness\n")
        f.write("BF < 0.10   = Strong evidence for randomness\n")
        f.write("BF < 0.33   = Moderate evidence for randomness\n")
        f.write("BF ≈ 1      = No evidence either way\n")
        f.write("BF > 3      = Moderate evidence against randomness\n")
        f.write("BF > 10     = Strong evidence against randomness\n")
        f.write("BF > 100    = Decisive evidence against randomness\n\n")
        
        f.write("\n--- R² INTERPRETATION: ---\n")
        f.write("R² < 0.05   = Negligible predictive power (random)\n")
        f.write("R² < 0.15   = Weak predictive power (mostly random)\n")
        f.write("R² > 0.15   = Some predictive power (potential patterns)\n")
    
    print(f" Saved summary: {output_file}")


def main():
    """Main execution"""
    # paths
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    results_dir = base_dir / "outputs" / "bayesian"
    output_dir = base_dir / "outputs" / "bayesian" / "plots"
    
    # make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- MODULE 4 VISUALIZATION ---")
    
    # load results
    results_file = results_dir / "bayesian_results_complete.json"
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run bayesian_analysis.py first.")
        return
    
    data = load_results(results_file)
    
    # create visualizations
    plot_bayes_factors(data, output_dir)
    plot_regression_results(data, output_dir)
    plot_interpretation_summary(data, output_dir)
    
    print("\n--- VISUALIZATION COMPLETE ---")
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - bayes_factors_comparison.png (Models 1-4)")
    print("  - model5_r2_results.png (Model 5 regression)")
    print("  - interpretation_summary.txt")


if __name__ == "__main__":
    main()

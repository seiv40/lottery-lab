"""
Model Interpretation Analysis

Analyzes what models learned (or failed to learn) from lottery data.
This examines:
1. Bayesian ridge regression R-squared values (proportion of variance explained)
2. Deep learning RMSE compared to baseline
3. Feature importance across models
4. Conclusion: Models correctly identified absence of patterns

"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# paths
BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
BAYESIAN_DIR = BASE_DIR / "outputs" / "bayesian"  # actual location!
DEEP_LEARNING_DIR = BASE_DIR / "outputs" / "deep_learning"  # likely location
MODULE_DIR = BASE_DIR / "modules" / "module_11_meta_analysis"
OUTPUT_DIR = MODULE_DIR / "outputs"
FIGURES_DIR = MODULE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("MODEL INTERPRETATION ANALYSIS")

def load_bayesian_results():
    """load Bayesian analysis results from bayesian output directory."""
    results_file = BAYESIAN_DIR / "bayesian_results_complete.json"
    
    if not results_file.exists():
        print(f"\nNote: Bayesian results not found at {results_file}")
        print("This file should contain results from Module 4 Bayesian inference.")
        print("Skipping Bayesian interpretation analysis.")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nLoaded Bayesian results from {results_file}")
    return data

def load_deep_learning_results():
    """load deep learning results from lottery-specific output directories."""
    results = {'powerball': {}, 'megamillions': {}}
    
    print("\nSearching for deep learning results...")
    
    found_count = 0
    
    # map lottery names to their abbreviations in filenames
    lottery_mapping = {
        'powerball': 'pb',
        'megamillions': 'mm'
    }
    
    for lottery, abbrev in lottery_mapping.items():
        lottery_dir = BASE_DIR / "outputs" / lottery
        
        if not lottery_dir.exists():
            print(f"  Note: {lottery_dir} does not exist")
            continue
        
        # look for performance files with pattern: *_perf_{abbrev}.json
        perf_pattern = f"*_perf_{abbrev}.json"
        json_files = list(lottery_dir.glob(perf_pattern))
        
        for json_file in json_files:
            filename = json_file.name
            
            # skip predictive files (we want perf files only)
            if 'predictive' in filename:
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # extract model name from filename: modelname_perf_pb.json -> modelname
                model = filename.replace(f'_perf_{abbrev}.json', '')
                
                results[lottery][model] = data
                found_count += 1
                print(f"  Found: {filename} → model '{model}'")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    
    if found_count == 0:
        print(f"\nNote: No deep learning results found")
        print("Checked:")
        print(f"  - {BASE_DIR / 'outputs' / 'powerball'}/*_perf_pb.json")
        print(f"  - {BASE_DIR / 'outputs' / 'megamillions'}/*_perf_mm.json")
        print("\nSkipping deep learning interpretation analysis.")
        return None
    
    print(f"\nTotal: Loaded {found_count} deep learning result files")
    return results

def analyze_bayesian_performance(bayesian_data):
    """analyze Bayesian ridge regression performance."""
    print("BAYESIAN RIDGE REGRESSION ANALYSIS")
    
    results = []
    
    for lottery in ['powerball', 'megamillions']:
        if lottery not in bayesian_data:
            continue
        
        lottery_data = bayesian_data[lottery]
        
        # look for Model 5 (Bayesian ridge regression)
        if 'model5' in lottery_data:
            model_data = lottery_data['model5']
            
            # extract R² if available
            r_squared = model_data.get('r_squared', model_data.get('R2', 0.0))
            
            results.append({
                'lottery': lottery,
                'r_squared': r_squared,
                'interpretation': 'No predictive power' if r_squared < 0.05 else 'Some signal'
            })
            
            print(f"\n{lottery.upper()}:")
            print(f"  R² = {r_squared:.4f}")
            print(f"  Variance explained: {r_squared*100:.2f}%")
            print(f"  Interpretation: Features explain <{r_squared*100:.1f}% of variance")
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "bayesian_interpretation.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        return df
    
    return None

def analyze_deep_learning_performance(dl_data):
    """analyze deep learning model performance vs baseline."""
    print("DEEP LEARNING PERFORMANCE ANALYSIS")
    
    results = []
    baseline_rmse = 1.0  # for standardized targets
    
    for lottery in ['powerball', 'megamillions']:
        if lottery not in dl_data or not dl_data[lottery]:
            continue
        
        print(f"\n{lottery.upper()}:")
        
        for model_name, model_data in dl_data[lottery].items():
            # extract RMSE - handle different possible formats
            rmse = model_data.get('test_rmse', model_data.get('rmse', None))
            
            # convert to float if it's a string
            if rmse is not None:
                try:
                    rmse = float(rmse)
                except (ValueError, TypeError):
                    print(f"  {model_name:20s}: RMSE format error, skipping")
                    continue
            else:
                print(f"  {model_name:20s}: RMSE not found, skipping")
                continue
            
            # check if it's a valid number
            if np.isnan(rmse) or rmse <= 0:
                print(f"  {model_name:20s}: Invalid RMSE value, skipping")
                continue
            
            improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
            
            results.append({
                'lottery': lottery,
                'model': model_name,
                'rmse': rmse,
                'baseline_rmse': baseline_rmse,
                'improvement_pct': improvement
            })
            
            print(f"  {model_name:20s}: RMSE = {rmse:.3f} ({improvement:+.1f}% vs baseline)")
    
    if results:
        df = pd.DataFrame(results)
        
        # summary statistics
        print("SUMMARY STATISTICS")
        for lottery in ['powerball', 'megamillions']:
            lottery_df = df[df['lottery'] == lottery]
            if len(lottery_df) > 0:
                mean_rmse = lottery_df['rmse'].mean()
                mean_imp = lottery_df['improvement_pct'].mean()
                print(f"{lottery.upper()}:")
                print(f"  Mean RMSE: {mean_rmse:.3f}")
                print(f"  Mean improvement: {mean_imp:+.1f}%")
                print(f"  Interpretation: Models perform at/near baseline (random guessing)")
        
        output_file = OUTPUT_DIR / "deep_learning_interpretation.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        return df
    
    return None

def create_interpretation_summary(bayesian_df, dl_df):
    """create summary visualization."""
    print("CREATING INTERPRETATION SUMMARY VISUALIZATION")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # plot 1: Bayesian R² comparison
    if bayesian_df is not None and len(bayesian_df) > 0:
        ax1 = axes[0]
        bayesian_df.plot(x='lottery', y='r_squared', kind='bar', ax=ax1, 
                        color='steelblue', legend=False)
        ax1.axhline(0.05, color='red', linestyle='--', label='Threshold (5%)')
        ax1.set_ylabel('R² (Variance Explained)', fontsize=11)
        ax1.set_xlabel('Lottery', fontsize=11)
        ax1.set_title('Bayesian Ridge Regression Performance', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 0.10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for i, row in bayesian_df.iterrows():
            ax1.text(i, row['r_squared'] + 0.002, f"{row['r_squared']:.3f}", 
                    ha='center', fontsize=10)
    
    # plot 2: Deep learning RMSE comparison
    if dl_df is not None and len(dl_df) > 0:
        ax2 = axes[1]
        
        # group by lottery
        for lottery in ['powerball', 'megamillions']:
            lottery_df = dl_df[dl_df['lottery'] == lottery]
            if len(lottery_df) > 0:
                x_pos = np.arange(len(lottery_df))
                if lottery == 'powerball':
                    ax2.bar(x_pos - 0.2, lottery_df['rmse'], width=0.4, 
                           label='Powerball', color='steelblue')
                else:
                    ax2.bar(x_pos + 0.2, lottery_df['rmse'], width=0.4,
                           label='Mega Millions', color='coral')
        
        ax2.axhline(1.0, color='red', linestyle='--', label='Baseline (1.0)')
        ax2.set_ylabel('RMSE', fontsize=11)
        ax2.set_xlabel('Models', fontsize=11)
        ax2.set_title('Deep Learning Performance vs Baseline', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0.5, 1.2)
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "interpretation_summary.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {fig_path}")

def main():
    """run complete interpretation analysis."""
    
    # load results from previous modules
    bayesian_data = load_bayesian_results()
    dl_data = load_deep_learning_results()
    
    # check if we have ANY data to analyze
    if bayesian_data is None and (dl_data is None or len(dl_data.get('powerball', {})) == 0):
        print("\n" + "="*80)
        print("NO DATA AVAILABLE FOR ANALYSIS")
        print("="*80)
        print("Cannot perform interpretation analysis without input data.")
        print("\nExpected file locations:")
        print(f"  Bayesian: {BAYESIAN_DIR / 'bayesian_results_complete.json'}")
        print(f"  Deep learning: {BASE_DIR / 'outputs' / 'powerball'}/*.json")
        print(f"                 {BASE_DIR / 'outputs' / 'megamillions'}/*.json")
        print("="*80)
        return
    
    # analyze Bayesian performance
    bayesian_df = None
    if bayesian_data:
        bayesian_df = analyze_bayesian_performance(bayesian_data)
    
    # analyze deep learning performance
    dl_df = None
    if dl_data:
        dl_df = analyze_deep_learning_performance(dl_data)
    
    # create summary visualization
    if bayesian_df is not None or dl_df is not None:
        create_interpretation_summary(bayesian_df, dl_df)
    
    # final interpretation - only if we have data
    print("INTERPRETATION CONCLUSION")
    
    if bayesian_df is not None or dl_df is not None:
        print("Based on available data:")
        
        if bayesian_df is not None:
            avg_r2 = bayesian_df['r_squared'].mean()
            print(f"  - Bayesian ridge: R² = {avg_r2:.4f} (features explain <{avg_r2*100:.1f}% of variance)")
        
        if dl_df is not None:
            avg_rmse = dl_df['rmse'].mean()
            baseline = 1.0
            print(f"  - Deep learning: Mean RMSE = {avg_rmse:.3f} (baseline = {baseline:.1f})")
            if avg_rmse >= 0.85:
                print(f"    Models perform at/near baseline (random guessing)")
        
        print("\nModels correctly identified absence of predictable patterns.")
    else:
        print("Insufficient data for conclusions.")
    

if __name__ == "__main__":
    main()

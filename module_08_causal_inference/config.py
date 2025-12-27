"""
Module 8: Configuration File

This file contains all paths and hyperparameters for Module 8 analysis.

"""

from pathlib import Path

# DATA PATHS

# base directory
BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")

# raw data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
POWERBALL_RAW = RAW_DATA_DIR / "powerball_draws.parquet"
MEGAMILLIONS_RAW = RAW_DATA_DIR / "megamillions_draws.parquet"

# processed data paths
PROCESSED_DIR = BASE_DIR / "data" / "processed"
POWERBALL_JSON = PROCESSED_DIR / "powerball_current_format.json"
MEGAMILLIONS_JSON = PROCESSED_DIR / "megamillions_current_format.json"

# feature data paths
FEATURES_DIR = PROCESSED_DIR / "features"
POWERBALL_FEATURES = FEATURES_DIR / "features_powerball.parquet"
MEGAMILLIONS_FEATURES = FEATURES_DIR / "features_megamillions.parquet"

# output paths
MODULE_DIR = BASE_DIR / "modules" / "module_08_causal_inference"
OUTPUT_DIR = MODULE_DIR / "outputs"
FIGURES_DIR = MODULE_DIR / "figures"

# create output directories if they do not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ANALYSIS PARAMETERS

# granger causality
GRANGER_MAX_LAGS = [1, 3, 5, 10, 15, 20]
GRANGER_SIGNIFICANCE = 0.05

# conditional independence
CI_TESTS = ['fisherz', 'kci', 'chisq', 'gsq']
CI_SIGNIFICANCE = 0.05
CI_MAX_CONDITIONING_SET_SIZE = 5

# causal discovery
PC_ALPHA = 0.05
GES_SCORE = 'local_score_BIC'
GFCI_ALPHA = 0.05

# transfer entropy
TE_HISTORY_LENGTH = 3  # k parameter
TE_N_PERMUTATIONS = 1000  # for significance testing

# causal invariance
ICP_ALPHA = 0.05
ICP_ENVIRONMENTS = 3  # split data into N environments

# multiple testing correction
BONFERRONI_CORRECTION = True
FDR_METHOD = 'fdr_bh'  # benjamini-hochberg

# visualization
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FIGURE_SIZE = (12, 8)

# random seeds
RANDOM_SEED = 42

# LOTTERY-SPECIFIC PARAMETERS

LOTTERY_CONFIG = {
    'powerball': {
        'white_balls': 5,
        'white_range': (1, 69),
        'special_ball': 'powerball',
        'special_range': (1, 26),
        # raw file columns (from powerball_draws.parquet)
        'ball_columns': ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'powerball']
    },
    'megamillions': {
        'white_balls': 5,
        'white_range': (1, 70),
        'special_ball': 'megaball',
        'special_range': (1, 25),
        # raw file columns (from megamillions_draws.parquet)
        'ball_columns': ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'megaball']
    }
}

# FEATURE COLUMNS FOR CAUSAL ANALYSIS

# these match the actual columns in your feature files
TEMPORAL_FEATURES = [
    'mean', 'sum', 'variance', 'std',
    'range', 'median', 'min', 'max'
]

# note: individual ball positions are NOT in feature files
# they must be loaded from raw files (powerball_draws.parquet, megamillions_draws.parquet)
# feature files only contain aggregate statistics

# all aggregate features available for multivariate analysis
ALL_AGGREGATE_FEATURES = TEMPORAL_FEATURES + [
    'skewness', 'kurtosis', 'cv', 'iqr',
    'even_count', 'odd_count', 'high_count', 'low_count',
    'consecutive_count', 'prime_count'
]

# UTILITY FUNCTIONS

def get_lottery_config(lottery_name):
    """Get configuration for specific lottery."""
    return LOTTERY_CONFIG.get(lottery_name.lower())

def get_ball_columns(lottery_name):
    """Get ball column names for specific lottery."""
    config = get_lottery_config(lottery_name)
    return config['ball_columns'] if config else None

def get_output_path(filename):
    """Get full output path for a file."""
    return OUTPUT_DIR / filename

def get_figure_path(filename):
    """Get full figure path for a file."""
    if not filename.endswith(f'.{FIGURE_FORMAT}'):
        filename = f"{filename}.{FIGURE_FORMAT}"
    return FIGURES_DIR / filename

# VALIDATION

def validate_config():
    """Validate that all required paths exist."""
    required_paths = [
        POWERBALL_FEATURES,
        MEGAMILLIONS_FEATURES,
        POWERBALL_RAW,
        MEGAMILLIONS_RAW
    ]
    
    missing_paths = [p for p in required_paths if not p.exists()]
    
    if missing_paths:
        print("WARNING: The following required data files are missing:")
        for p in missing_paths:
            print(f"  - {p}")
        return False
    return True

if __name__ == "__main__":
    print("\n--- Module 8 Configuration ---")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Figures Directory: {FIGURES_DIR}")
    print()
    print("Validating configuration...")
    if validate_config():
        print("All required data files found")
    else:
        print("Some data files are missing")

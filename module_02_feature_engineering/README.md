# Module 2: Feature Engineering

Generates 90+ features per drawing for statistical analysis and machine learning. Takes filtered data from Module 1 and creates features that capture statistics, distribution patterns, temporal patterns, gaps, co-occurrence, and seasonality.

## How to Run

```bash
# Run tests first (recommended)
python modules/module_02_feature_engineering/tests/test_features.py

# Generate features
python modules/module_02_feature_engineering/feature_engineering.py

# Quick validation
python modules/module_02_feature_engineering/quick_check.py
```

## Input/Output

**Input:**
- `data/processed/powerball_current_format.json`
- `data/processed/megamillions_current_format.json`

**Output:**
- `data/processed/features/features_powerball.parquet`
- `data/processed/features/features_megamillions.parquet`
- Feature reports (JSON)

## Feature Categories

- Basic statistics (mean, median, std, etc.)
- Distribution analysis (even/odd, high/low, consecutive numbers)
- Temporal features (lag features, rolling windows)
- Gap analysis (days since numbers were last drawn)
- Co-occurrence patterns
- Positional features
- Seasonality (day of week, month, etc.)
- Jackpot sizes

**Note:** Early drawings have NaN values for some features because they need historical data (lag features, rolling windows, gaps). This is expected and handled in later modules.

"""
Quick validation script - just a sanity check to make sure
the parquet files look reasonable after running feature engineering.

This is what I run to quickly verify things worked.
"""

import pandas as pd
from pathlib import Path

# adjust these paths if needed
base_path = Path(r"C:\jackpotmath\lottery-lab")
features_dir = base_path / "data" / "processed" / "features"

# load features
print("Loading features...")
df_pb = pd.read_parquet(features_dir / "features_powerball.parquet")
df_mm = pd.read_parquet(features_dir / "features_megamillions.parquet")

# quick checks
print("\n--- POWERBALL FEATURES ---")
print(f"Shape: {df_pb.shape}")  # should be ~(1269, 110)
print(f"Columns: {len(df_pb.columns)}")
print(f"Date range: {df_pb['date'].min()} to {df_pb['date'].max()}")
print(f"\nFirst few columns: {list(df_pb.columns[:5])}")
print(f"Sample features: {list(df_pb.columns[10:15])}")

print("\n--- MEGA MILLIONS FEATURES ---")
print(f"Shape: {df_mm.shape}")  # should be ~(838, 110)
print(f"Columns: {len(df_mm.columns)}")
print(f"Date range: {df_mm['date'].min()} to {df_mm['date'].max()}")

print("\n--- SAMPLE DATA (most recent Powerball drawing) ---")
print(df_pb.tail(1).T)  # transpose to see all features

print("\n--- FEATURE STATS (mean feature) ---")
print(df_pb['mean'].describe())

print("\n--- MISSING VALUES CHECK ---")
missing_pb = df_pb.isnull().sum().sum()
missing_mm = df_mm.isnull().sum().sum()
print(f"Powerball missing values: {missing_pb}")
print(f"Mega Millions missing values: {missing_mm}")

if missing_pb > 0 or missing_mm > 0:
    print("\nNote: Some missing values are expected for early drawings")
    print("(lag features, rolling windows need history)")

print("\nQuick check complete!")

"""
Quick diagnostic: What columns are actually in the feature files?

"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"

print("\n--- FEATURE FILE COLUMN DIAGNOSTIC ---")

for lottery in ['powerball', 'megamillions']:
    print(f"\n{lottery.upper()}:")
    print("-" * 70)
    
    # try both naming conventions
    for filename in [f'features_{lottery}.parquet', f'{lottery}_features.parquet']:
        filepath = FEATURES_DIR / filename
        
        if filepath.exists():
            print(f"\nFile: {filename}")
            df = pd.read_parquet(filepath)
            print(f"Shape: {df.shape}")
            print(f"Columns ({len(df.columns)}):")
            
            # show all columns
            for col in df.columns:
                print(f"  - {col}")
            
            # check for ball columns
            ball_cols = [c for c in df.columns if 'ball' in c.lower() or 'white' in c.lower() or 'powerball' in c.lower() or 'megaball' in c.lower()]
            print(f"\nBall columns found: {ball_cols}")
            
            # check for aggregate stats
            stat_cols = [c for c in df.columns if any(x in c.lower() for x in ['mean', 'sum', 'var', 'std', 'range', 'median'])]
            print(f"Aggregate stat columns found: {stat_cols}")
            
            break
    else:
        print(f"  File not found: tried both {lottery}_features.parquet and features_{lottery}.parquet")

print("\n--- DIAGNOSTIC COMPLETE ---")

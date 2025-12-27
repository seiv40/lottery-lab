"""
Test script for Module 2 feature engineering.

Run this BEFORE processing the full dataset to catch issues early.

Usage:
    python test_features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# add parent dir to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering import LotteryFeatureEngineer


def create_test_data():
    """Make some fake test data"""
    test_data = [
        {
            "game": "powerball",
            "dateISO": "2024-01-01",
            "regularNumbers": [5, 15, 25, 35, 45],
            "specialNumber": 10,
            "jackpotAnnuitized": 100000000,
            "jackpotCash": 50000000
        },
        {
            "game": "powerball",
            "dateISO": "2024-01-04",
            "regularNumbers": [2, 12, 22, 32, 42],
            "specialNumber": 15,
            "jackpotAnnuitized": 120000000,
            "jackpotCash": 60000000
        },
        {
            "game": "powerball",
            "dateISO": "2024-01-06",
            "regularNumbers": [8, 18, 28, 38, 48],
            "specialNumber": 20,
            "jackpotAnnuitized": 140000000,
            "jackpotCash": 70000000
        }
    ]
    
    return pd.DataFrame(test_data)


def test_basic_statistics():
    """Test basic stat extraction"""
    print("\n--- TEST 1: Basic Statistics ---")
    
    engineer = LotteryFeatureEngineer('powerball', 5, 69, 26)
    numbers = [10, 20, 30, 40, 50]
    
    features = engineer.extract_basic_statistics(numbers)
    
    # check results
    assert features['mean'] == 30.0, f"Mean wrong: {features['mean']}"
    assert features['sum'] == 150, f"Sum wrong: {features['sum']}"
    assert features['range'] == 40, f"Range wrong: {features['range']}"
    assert features['median'] == 30.0, f"Median wrong: {features['median']}"
    
    print("Mean correct")
    print("Sum correct")
    print("Range correct")
    print("Median correct")
    print("\nTEST 1 PASSED")


def test_distribution_features():
    """Test distribution features"""
    print("\n--- TEST 2: Distribution Features ---")
    
    engineer = LotteryFeatureEngineer('powerball', 5, 69, 26)
    numbers = [2, 11, 12, 21, 35]  # 2 even, 3 odd, 1 consecutive pair
    
    features = engineer.extract_distribution_features(numbers)
    
    # validate
    assert features['even_count'] == 2, f"Even count wrong: {features['even_count']}"
    assert features['odd_count'] == 3, f"Odd count wrong: {features['odd_count']}"
    assert features['consecutive_count'] == 1, f"Consecutive wrong: {features['consecutive_count']}"
    assert features['has_consecutive'] == 1
    
    print("Even/odd counting works")
    print(f"  Numbers: {numbers}")
    print(f"  Even: 2, 12 = {features['even_count']} ✓")
    print(f"  Odd: 11, 21, 35 = {features['odd_count']} ✓")
    print("Consecutive detection works")
    print(f"  Found {features['consecutive_count']} pair (11, 12)")
    print("\nTEST 2 PASSED")


def test_temporal_features():
    """Test temporal features"""
    print("\n--- TEST 3: Temporal Features ---")
    
    engineer = LotteryFeatureEngineer('powerball', 5, 69, 26)
    
    # test data
    df = create_test_data()
    df['date'] = pd.to_datetime(df['dateISO'])
    
    # test on 3rd drawing
    features = engineer.extract_temporal_features(df, idx=2)
    
    assert 'days_since_last_drawing' in features
    assert 'lag_1_mean' in features
    assert features['days_since_last_drawing'] == 2  # 2 days gap
    
    print("Days since last drawing calculated")
    print(f"  Gap: {features['days_since_last_drawing']} days")
    print("Lag features extracted")
    print("\nTEST 3 PASSED")


def test_gap_features():
    """Test gap analysis"""
    print("\n--- TEST 4: Gap Features ---")
    
    engineer = LotteryFeatureEngineer('powerball', 5, 69, 26)
    
    df = create_test_data()
    df['date'] = pd.to_datetime(df['dateISO'])
    
    # test gap features
    features = engineer.extract_gap_features(df, idx=2, numbers=[8, 18, 28, 38, 48])
    
    assert 'mean_gap_days' in features
    assert 'special_ball_gap_days' in features
    
    print("Gap analysis works")
    print(f"  Mean gap: {features['mean_gap_days']:.1f} days")
    print("\nTEST 4 PASSED")


def test_full_pipeline():
    """Test complete pipeline"""
    print("\n--- TEST 5: Full Pipeline ---")
    
    engineer = LotteryFeatureEngineer('powerball', 5, 69, 26)
    
    # test data
    df = create_test_data()
    df['date'] = pd.to_datetime(df['dateISO'])
    
    # generate features
    print("\nGenerating features for 3 test drawings...")
    features_df = engineer.generate_features(df)
    
    # validate
    assert len(features_df) == 3, f"Expected 3 rows, got {len(features_df)}"
    assert len(features_df.columns) >= 90, f"Expected 90+ features, got {len(features_df.columns)}"
    
    print(f"\nGenerated {len(features_df.columns)} features")
    print(f"Processed {len(features_df)} drawings")
    
    # check for required columns
    required_cols = ['mean', 'std', 'even_count', 'rolling_5_mean', 
                     'mean_gap_days', 'day_of_week']
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    
    if missing_cols:
        print(f"\nMissing columns: {missing_cols}")
        return False
    
    print("All required features present")
    
    # sample values
    print("\nSample values (Drawing 3):")
    print(f"  Mean: {features_df.loc[2, 'mean']:.1f}")
    print(f"  Std: {features_df.loc[2, 'std']:.1f}")
    print(f"  Even count: {features_df.loc[2, 'even_count']:.0f}")
    print(f"  Consecutive: {features_df.loc[2, 'consecutive_count']:.0f}")
    
    print("\nTEST 5 PASSED")
    
    return features_df


def test_save_load():
    """Test Parquet save/load"""
    print("\n--- TEST 6: Save/Load ---")
    
    engineer = LotteryFeatureEngineer('powerball', 5, 69, 26)
    df = create_test_data()
    df['date'] = pd.to_datetime(df['dateISO'])
    
    features_df = engineer.generate_features(df)
    
    # save
    test_path = Path("test_features.parquet")
    engineer.save_features(features_df, test_path)
    
    assert test_path.exists(), "Parquet file not created"
    print("Saved parquet file")
    
    # load it back
    loaded_df = pd.read_parquet(test_path)
    assert len(loaded_df) == len(features_df), "Row count mismatch"
    assert len(loaded_df.columns) == len(features_df.columns), "Column count mismatch"
    
    print("Loaded parquet file successfully")
    print(f"  File size: {test_path.stat().st_size / 1024:.2f} KB")
    
    # clean up
    test_path.unlink()
    print("Cleaned up test file")
    
    print("\nTEST 6 PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n--- MODULE 2: FEATURE ENGINEERING TESTS ---")
    print("\nRunning validation suite...")
    
    try:
        test_basic_statistics()
        test_distribution_features()
        test_temporal_features()
        test_gap_features()
        test_full_pipeline()
        test_save_load()
        
        print("\n--- ALL TESTS PASSED ---")
        print("\nModule 2 is ready!")
        print("You can now run: python feature_engineering.py")
        print("\n")
        
        return True
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        print("\nFix the error before proceeding.")
        return False
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

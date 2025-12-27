"""
Quick test to verify all fixes are working.
Run this before the full analysis.

"""

import sys
from pathlib import Path

# add parent directory to path (module scripts are one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n--- MODULE 8 QUICK TEST ---")

# Test 1: Config
print("\n1. Testing config...")
try:
    from config import POWERBALL_FEATURES, MEGAMILLIONS_FEATURES
    print(f"   [OK] Powerball features path: {POWERBALL_FEATURES}")
    print(f"   [OK] Mega Millions features path: {MEGAMILLIONS_FEATURES}")
    assert POWERBALL_FEATURES.exists(), "Powerball features not found!"
    assert MEGAMILLIONS_FEATURES.exists(), "Mega Millions features not found!"
    print("   [OK] Both feature files exist")
except Exception as e:
    print(f"   [FAILED] Config error: {e}")
    sys.exit(1)

# Test 2: Data Loading
print("\n2. Testing data loading...")
try:
    from data_loader import LotteryDataLoader
    
    loader = LotteryDataLoader('powerball')
    df = loader.load_features()
    print(f"   [OK] Loaded {len(df)} Powerball draws with {len(df.columns)} features")
    
    # Test Granger prep
    df_granger = loader.prepare_for_granger()
    print(f"   [OK] Prepared {len(df_granger)} samples for Granger testing")
    
    # Test CI prep
    df_ci = loader.prepare_for_conditional_independence()
    print(f"   [OK] Prepared {len(df_ci)} samples for CI testing")
    
    # Test ball time series
    ball_data, ball_names = loader.prepare_ball_time_series()
    print(f"   [OK] Prepared ball time series: {ball_data.shape}")
    print(f"   [OK] Ball columns: {ball_names}")
    
except Exception as e:
    print(f"   [FAILED] Data loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Quick Granger Test
print("\n3. Testing Granger causality...")
try:
    from granger_causality import GrangerCausalityAnalyzer
    
    analyzer = GrangerCausalityAnalyzer('powerball')
    # Just test that it can initialize and prepare data
    df = analyzer.loader.prepare_for_granger(features=['mean', 'sum'], max_lags=3)
    print(f"   [OK] Granger analyzer ready: {len(df)} samples")
    
except Exception as e:
    print(f"   [FAILED] Granger test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Conditional Independence
print("\n4. Testing conditional independence...")
try:
    from conditional_independence import ConditionalIndependenceAnalyzer
    
    analyzer = ConditionalIndependenceAnalyzer('powerball')
    analyzer.load_data()
    print(f"   [OK] CI analyzer ready: {analyzer.data.shape}")
    
except Exception as e:
    print(f"   [FAILED] CI test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Causal Discovery
print("\n5. Testing causal discovery...")
try:
    from causal_discovery import CausalDiscoveryAnalyzer
    
    analyzer = CausalDiscoveryAnalyzer('powerball')
    analyzer.load_data(max_features=5)
    print(f"   [OK] Causal discovery analyzer ready: {analyzer.data.shape}")
    
except Exception as e:
    print(f"   [FAILED] Causal discovery error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Transfer Entropy
print("\n6. Testing transfer entropy...")
try:
    from transfer_entropy import TransferEntropyAnalyzer
    
    analyzer = TransferEntropyAnalyzer('powerball')
    analyzer.load_data()
    print(f"   [OK] Transfer entropy analyzer ready: {analyzer.data.shape}")
    
except Exception as e:
    print(f"   [FAILED] Transfer entropy error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Causal Invariance
print("\n7. Testing causal invariance...")
try:
    from causal_invariance import CausalInvarianceAnalyzer
    
    analyzer = CausalInvarianceAnalyzer('powerball')
    analyzer.load_data()
    analyzer.create_environments()
    print(f"   [OK] Causal invariance analyzer ready: {analyzer.data.shape}")
    
except Exception as e:
    print(f"   [FAILED] Causal invariance error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- ALL TESTS PASSED ---")
print("\nYou're ready to run the full analysis:")
print("  python module8_complete_analysis.py")
print("\nOr start with quick mode:")
print("  python module8_complete_analysis.py --quick")

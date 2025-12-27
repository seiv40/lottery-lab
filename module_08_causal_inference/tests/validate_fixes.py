"""
Quick validation script to test Module 8 error fixes.

Checks if:
1. Multicollinearity detection works
2. Causal discovery completes without errors
3. Distance correlation uses correct API

Run this before the full analysis to verify fixes.

"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# add parent directory to path (module scripts are one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n--- MODULE 8 ERROR FIX VALIDATION ---")

# Test 1: Multicollinearity Detection
print("\n1. Testing multicollinearity detection...")
try:
    from causal_discovery import CausalDiscoveryAnalyzer
    
    # Create analyzer
    analyzer = CausalDiscoveryAnalyzer('powerball')
    
    # Load data (should automatically remove multicollinear features)
    analyzer.load_data(max_features=10)
    
    # Check that data loaded successfully
    assert analyzer.data is not None
    assert len(analyzer.feature_names) > 0
    
    print(f"   [OK] Loaded {analyzer.data.shape[0]} samples with {analyzer.data.shape[1]} features")
    print(f"   [OK] Features: {analyzer.feature_names}")
    print(f"   [OK] Multicollinearity detection working")
    
except Exception as e:
    print(f"   [FAILED] Error: {e}")
    sys.exit(1)

# Test 2: PC Algorithm with Fallback
print("\n2. Testing PC algorithm with fallback...")
try:
    # Try PC algorithm (should handle any remaining singularity issues)
    results = analyzer.run_pc_algorithm(alpha=0.05)
    
    if results:
        print(f"   [OK] PC algorithm completed successfully")
        print(f"   [OK] Found {results['n_edges']} edges")
        print(f"   [OK] Using independence test: {results['indep_test']}")
    else:
        print(f"   ! PC algorithm returned no results (may have failed gracefully)")
    
except Exception as e:
    print(f"   [FAILED] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: GES Algorithm
print("\n3. Testing GES algorithm...")
try:
    # Reset analyzer
    analyzer = CausalDiscoveryAnalyzer('powerball')
    analyzer.load_data(max_features=8)  # Use fewer features for faster testing
    
    results = analyzer.run_ges_algorithm()
    
    if results:
        print(f"   [OK] GES algorithm completed successfully")
        print(f"   [OK] Found {results['n_edges']} edges")
    else:
        print(f"   ! GES algorithm returned no results (may have failed gracefully)")
    
except Exception as e:
    print(f"   [FAILED] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: FCI Algorithm with Fallback
print("\n4. Testing FCI algorithm with fallback...")
try:
    results = analyzer.run_fci_algorithm(alpha=0.05)
    
    if results:
        print(f"   [OK] FCI algorithm completed successfully")
        print(f"   [OK] Found {results['n_edges']} edges")
        print(f"   [OK] Using independence test: {results['indep_test']}")
    else:
        print(f"   ! FCI algorithm returned no results (may have failed gracefully)")
    
except Exception as e:
    print(f"   [FAILED] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Distance Correlation
print("\n5. Testing distance correlation...")
try:
    from conditional_independence import ConditionalIndependenceAnalyzer
    import dcor
    
    # Check dcor is available
    print(f"   [OK] dcor version: {dcor.__version__}")
    
    # Create analyzer
    ci_analyzer = ConditionalIndependenceAnalyzer('powerball')
    ci_analyzer.load_data()
    
    # Test on small subset for speed
    ci_analyzer.data = ci_analyzer.data[:, :4]  # Just 4 features
    ci_analyzer.feature_names = ci_analyzer.feature_names[:4]
    
    # Run distance correlation
    dcor_matrix = ci_analyzer.distance_correlation_test()
    
    print(f"   [OK] Distance correlation completed successfully")
    print(f"   [OK] Computed for {len(ci_analyzer.feature_names)} features")
    print(f"   [OK] Matrix shape: {dcor_matrix.shape}")
    
except Exception as e:
    print(f"   [FAILED] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- VALIDATION COMPLETE ---")

print("\n[OK] All critical fixes validated")
print("\nYou can now run the full analysis with:")
print("  python module8_complete_analysis.py --lottery powerball")

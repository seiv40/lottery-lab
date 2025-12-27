"""
Test script for Module 4 Bayesian analysis.

Makes sure the Bayesian models run and produce valid results.
This is a simplified test - just checks basics.

Usage:
    python test_bayesian.py

"""

import json
import sys
from pathlib import Path
import numpy as np


def test_json_validity(output_dir: Path) -> bool:
    """
    Test 1: Check if results file exists and is valid JSON.
    """
    print("\n--- Test 1: JSON Validity ---")
    
    results_file = output_dir / "bayesian_results_complete.json"
    
    # check if file exists
    if not results_file.exists():
        print(f" FAILED: Results file not found: {results_file}")
        print("  Run bayesian_analysis.py first to generate results.")
        return False
    
    # try to load JSON
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f" FAILED: Invalid JSON: {e}")
        return False
    
    # check basic structure
    if 'metadata' not in data:
        print(" FAILED: Missing 'metadata' key")
        return False
    
    print(" PASSED: JSON file is valid")
    return True


def test_model_completeness(output_dir: Path) -> bool:
    """
    Test 2: Check if all models ran for both games.
    """
    print("\n--- Test 2: Model Completeness ---")
    
    results_file = output_dir / "bayesian_results_complete.json"
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # expected models
    expected_models = ['model1', 'model2', 'model3', 'model4', 'model5']
    games = ['powerball', 'megamillions']
    
    all_passed = True
    
    for game in games:
        if game not in data:
            print(f" FAILED: Missing results for {game}")
            all_passed = False
            continue
        
        # check for errors
        if 'error' in data[game]:
            print(f" FAILED: {game} has error: {data[game]['error']}")
            all_passed = False
            continue
        
        # check each model
        missing_models = []
        for model in expected_models:
            if model not in data[game]:
                missing_models.append(model)
        
        if missing_models:
            print(f" FAILED: {game} missing models: {missing_models}")
            all_passed = False
        else:
            print(f" {game}: All 5 models present")
    
    if all_passed:
        print(" PASSED: All models completed for both games")
    
    return all_passed


def test_bayes_factors(output_dir: Path) -> bool:
    """
    Test 3: Check Bayes factors (or R² for Model 5) are valid numbers.
    
    Note: Model 5 uses R² instead of BF since it's a regression model.
    """
    print("\n--- Test 3: Model Metrics Validity ---")
    
    results_file = output_dir / "bayesian_results_complete.json"
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    all_passed = True
    
    for game in ['powerball', 'megamillions']:
        if game not in data or 'error' in data[game]:
            continue
        
        for model_key, model_data in data[game].items():
            # Model 5 uses R² instead of Bayes Factor
            if model_key == 'model5':
                if 'r2_mean' not in model_data:
                    print(f" {game}/{model_key}: Missing r2_mean")
                    all_passed = False
                    continue
                
                r2 = model_data['r2_mean']
                
                # check if valid number
                if not isinstance(r2, (int, float)):
                    print(f" {game}/{model_key}: R² is not a number: {type(r2)}")
                    all_passed = False
                    continue
                
                # R² should be between 0 and 1
                if not (0 <= r2 <= 1):
                    print(f" {game}/{model_key}: R² out of range [0,1]: {r2}")
                    all_passed = False
                    continue
                
                # check if not nan
                if np.isnan(r2):
                    print(f" {game}/{model_key}: R² is nan")
                    all_passed = False
                    continue
                
                print(f" {game}/{model_key}: R² = {r2:.4f}")
                
            else:
                # Other models use Bayes Factor
                if 'bayes_factor' not in model_data:
                    print(f" {game}/{model_key}: Missing bayes_factor")
                    all_passed = False
                    continue
                
                bf = model_data['bayes_factor']
                
                # check if valid number
                if not isinstance(bf, (int, float)):
                    print(f" {game}/{model_key}: BF is not a number: {type(bf)}")
                    all_passed = False
                    continue
                
                # check if positive (BF should be > 0)
                if bf <= 0:
                    print(f" {game}/{model_key}: BF is not positive: {bf}")
                    all_passed = False
                    continue
                
                # check if reasonable (not nan, inf is ok for very large BF)
                if np.isnan(bf):
                    print(f" {game}/{model_key}: BF is nan")
                    all_passed = False
                    continue
                
                print(f" {game}/{model_key}: BF = {bf:.4f}")
    
    if all_passed:
        print(" PASSED: All model metrics are valid")
    
    return all_passed


def test_convergence(output_dir: Path) -> bool:
    """
    Test 4: Check MCMC convergence for models that use it.
    """
    print("\n--- Test 4: MCMC Convergence ---")
    
    results_file = output_dir / "bayesian_results_complete.json"
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # models that use MCMC
    mcmc_models = ['model2', 'model3', 'model5']
    
    all_passed = True
    
    for game in ['powerball', 'megamillions']:
        if game not in data or 'error' in data[game]:
            continue
        
        for model_key in mcmc_models:
            if model_key not in data[game]:
                continue
            
            model_data = data[game][model_key]
            
            if 'convergence' not in model_data:
                print(f" {game}/{model_key}: Missing convergence diagnostics")
                all_passed = False
                continue
            
            conv = model_data['convergence']
            
            # check if converged
            if 'converged' in conv and not conv['converged']:
                print(f" {game}/{model_key}: MCMC did not converge")
                print(f"   R-hat: {conv.get('r_hat', conv.get('r_hat_max', 'N/A'))}")
                all_passed = False
            else:
                print(f" {game}/{model_key}: MCMC converged")
    
    if all_passed:
        print(" PASSED: All MCMC models converged")
    
    return all_passed


def test_interpretations(output_dir: Path) -> bool:
    """
    Test 5: Check interpretations are present.
    """
    print("\n--- Test 5: Interpretation Strings ---")
    
    results_file = output_dir / "bayesian_results_complete.json"
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    all_passed = True
    
    for game in ['powerball', 'megamillions']:
        if game not in data or 'error' in data[game]:
            continue
        
        for model_key, model_data in data[game].items():
            if 'interpretation' not in model_data:
                print(f" {game}/{model_key}: Missing interpretation")
                all_passed = False
                continue
            
            interp = model_data['interpretation']
            
            # check if non-empty string
            if not isinstance(interp, str) or len(interp) == 0:
                print(f" {game}/{model_key}: Invalid interpretation: {interp}")
                all_passed = False
    
    if all_passed:
        print(" PASSED: All models have interpretations")
    
    return all_passed


def run_all_tests(output_dir: Path) -> bool:
    """Run all tests and return overall pass/fail."""
    print("\n--- MODULE 4 BAYESIAN ANALYSIS TESTS ---")
    print(f"Output directory: {output_dir}")
    
    # run tests
    tests = [
        ("JSON Validity", lambda: test_json_validity(output_dir)),
        ("Model Completeness", lambda: test_model_completeness(output_dir)),
        ("Model Metrics", lambda: test_bayes_factors(output_dir)),
        ("MCMC Convergence", lambda: test_convergence(output_dir)),
        ("Interpretations", lambda: test_interpretations(output_dir))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # print summary
    print("\n--- TEST SUMMARY ---")
    
    for test_name, passed in results:
        status = " PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print(" ALL TESTS PASSED")
    else:
        print(" SOME TESTS FAILED")
    
    
    return total_passed == total_tests


def main():
    """Run tests"""
    # paths
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    output_dir = base_dir / "outputs" / "bayesian"
    
    # check if output directory exists
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        print("Run bayesian_analysis.py first to generate results.")
        sys.exit(1)
    
    # run tests
    success = run_all_tests(output_dir)
    
    # exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

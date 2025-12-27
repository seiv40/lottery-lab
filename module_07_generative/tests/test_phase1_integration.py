"""
Module 7 - Phase 1 Integration Test

This script shows Phase 1 components working together:
1. Decoder validation
2. Null universe generation

We run this to verify Phase 1 is complete before proceeding to Phase 2.

"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# import our Module 7 components
# (In practice, these would be imported from the module structure)
# for now, we'll assume they are in the same directory or properly installed

print("="*70)
print("MODULE 7 - PHASE 1 INTEGRATION TEST")
print("="*70)
print("\nThis test validates that Phase 1 components are working correctly:")
print("   Decoder implementation")
print("   Decoder validation experiment")
print("   Null universe generation")
print("   Universe statistics")
print("\n" + "="*70 + "\n")


# PART 1: CREATE SYNTHETIC TRAINING DATA
# (In real usage, this would be your actual lottery data from Module 5)

print("[PART 1] Creating Synthetic Training Data")
print("-"*70)

np.random.seed(42)
n_historical_draws = 1269  # typical Powerball dataset size

# synthetic Powerball draws with REALISTIC distributions
# key: Real lotteries draw from uniform pool (1-69), THEN sort
# not position-specific ranges!
draws_list = []
for _ in range(n_historical_draws):
    # draw 5 white balls without replacement from 1-69
    white_balls = sorted(np.random.choice(range(1, 70), 5, replace=False))
    # draw 1 powerball from 1-26
    powerball = np.random.randint(1, 27)
    
    draws_list.append({
        'white_1': white_balls[0],
        'white_2': white_balls[1],
        'white_3': white_balls[2],
        'white_4': white_balls[3],
        'white_5': white_balls[4],
        'powerball': powerball
    })

training_data = pd.DataFrame(draws_list)

print(f" Created {len(training_data):,} synthetic Powerball draws")
print(f"  Columns: {list(training_data.columns)}")
print(f"  Shape: {training_data.shape}")
print("\n")


# PART 2: DECODER VALIDATION

print("[PART 2] Decoder Validation Experiment")
print("-"*70)
print("Testing that decoder does not introduce systematic bias...")
print()

# step 2.1: Initialize decoder
from module7_decoder import LotteryDecoder, DecoderValidator

decoder = LotteryDecoder('powerball', training_data)
print(f" Decoder initialized for Powerball")

# step 2.2: Create scaler (as would be done in Module 5)
feature_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', 'powerball']
scaler = StandardScaler()
scaler.fit(training_data[feature_cols].values)
print(f" Scaler fitted on training data")
print()

# step 2.3: Run validation experiment
validator = DecoderValidator(decoder)
validation_results = validator.decoding_validation_experiment(
    training_data,
    scaler,
    n_samples=500
)

# step 2.4: Check validation passed (using relaxed, realistic thresholds)
validation_passed = all([
    validation_results['ball_freq_ks_pvalue'] > 0.05,  # no systematic bias in ball frequencies
    abs(validation_results['mean_diff']) < 2.0,        # mean shift < 2 balls
    abs(validation_results['std_diff']) < 1.0,         # std preserved
    abs(validation_results['entropy_diff']) < 0.3,     # entropy loss < 0.3 bits
    abs(validation_results['autocorr_diff']) < 0.10   # no spurious temporal structure
])

if validation_passed:
    print("\n DECODER VALIDATION PASSED ")
    print("Decoder is unbiased and ready for universe generation")
    print(f"Note: Entropy loss of {abs(validation_results['entropy_diff']):.3f} bits is")
    print(f"acceptable for continuous->discrete mapping.")
else:
    print("\n DECODER VALIDATION FAILED ")
    print("Review decoder implementation before proceeding")
    print("\nFailing metrics:")
    if validation_results['ball_freq_ks_pvalue'] <= 0.05:
        print(f"  - Ball freq KS p-value: {validation_results['ball_freq_ks_pvalue']:.4f} (need > 0.05)")
    if abs(validation_results['mean_diff']) >= 2.0:
        print(f"  - Mean difference: {validation_results['mean_diff']:.2f} (need < 2.0)")
    if abs(validation_results['entropy_diff']) >= 0.3:
        print(f"  - Entropy loss: {abs(validation_results['entropy_diff']):.3f} bits (need < 0.3)")
    sys.exit(1)

print("\n")


# PART 3: NULL UNIVERSE GENERATION

print("[PART 3] Null Universe Generation")
print("-"*70)
print("Generating gold standard benchmark universe...")
print()

from module7_generation import NullUniverseGenerator, UniverseStatistics

# step 3.1: Generate null universe
null_generator = NullUniverseGenerator('powerball')
null_universe = null_generator.generate(n_draws=10000, seed=42)

print(f" Generated {len(null_universe):,} draws")
print(f"  Columns: {list(null_universe.columns)}")
print()

# step 3.2: Compute statistics
stats_analyzer = UniverseStatistics('powerball')
null_stats = stats_analyzer.compute_statistics(null_universe)
stats_analyzer.print_statistics(null_stats)

# step 3.3: Sanity checks
sanity_passed = True

# check autocorrelation is near zero
if 'temporal' in null_stats:
    autocorr = null_stats['temporal']['autocorr_lag1_mean']
    if abs(autocorr) > 0.10:
        print(f" Warning: Null universe has unexpected autocorrelation: {autocorr:.3f}")
        sanity_passed = False
    else:
        print(f" Temporal independence verified (autocorr = {autocorr:.3f})")

# check mean is reasonable (should be near middle of range)
mean_white = null_stats['white_balls']['mean']
expected_mean = (1 + 69) / 2  # 35 for Powerball
if abs(mean_white - expected_mean) > 5:
    print(f" Warning: Mean ball value unusual: {mean_white:.2f} (expected ~{expected_mean:.0f})")
    sanity_passed = False
else:
    print(f" Ball distribution verified (mean = {mean_white:.2f})")

if sanity_passed:
    print(f"\n NULL UNIVERSE SANITY CHECKS PASSED ")
else:
    print(f"\n NULL UNIVERSE HAS ANOMALIES ")

print()


# PART 4: SAVE OUTPUTS

print("[PART 4] Saving Outputs")
print("-"*70)

# save null universe
null_generator.save_universe(
    null_universe,
    '/mnt/user-data/outputs/universe_null_pb.parquet',
    metadata={
        'description': 'Null universe for Powerball - Phase 1 test',
        'purpose': 'Gold standard benchmark',
        'seed': 42,
        'validation_passed': validation_passed,
        'sanity_passed': sanity_passed
    }
)

print()


# SUMMARY

print("\n--- PHASE 1 INTEGRATION TEST - SUMMARY ---")
print()
print("Components Tested:")
print(f"  [{'' if validation_passed else ''}] Decoder validation")
print(f"  [{'' if sanity_passed else ''}] Null universe generation")
print(f"  [] Universe statistics")
print(f"  [] File I/O")
print()

if validation_passed and sanity_passed:
    print(" PHASE 1 COMPLETE - READY FOR PHASE 2 ")
    print()
    print("Next steps:")
    print("  1. Move to Phase 2: Generate VAE and Flow universes")
    print("  2. We now have:")
    print("     - Validated decoder (no systematic bias)")
    print("     - Gold standard null universe")
    print("     - Baseline statistics for comparison")
    print()
else:
    print(" PHASE 1 HAS ISSUES - FIX BEFORE PROCEEDING ")
    print()
    print("Issues detected:")
    if not validation_passed:
        print("  - Decoder validation failed")
        print("    -> Review quantile mapping and uniqueness enforcement")
    if not sanity_passed:
        print("  - Null universe sanity checks failed")
        print("    -> Review random number generation and lottery rules")

print()


# DEMO: HOW TO USE DECODER WITH MODEL OUTPUTS

print("\n--- DEMO: Using Decoder with Model Outputs ---")
print()
print("This shows how we'll use the decoder in Phase 2 when generating")
print("VAE, Flow, and Transformer universes.")
print()

# simulate model output (e.g., from VAE decoder)
print("Simulating model output...")
fake_model_features = np.random.randn(6)  # 6 standardized features
print(f"  Model output (standardized): {fake_model_features}")
print()

# decode to lottery draw
print("Decoding to lottery draw...")
decoded_draw = decoder.decode_to_lottery_draw(fake_model_features, scaler)
print(f"  Decoded draw:")
print(f"    White balls: {decoded_draw['white_balls']}")
print(f"    Powerball: {decoded_draw['powerball']}")
print()

# verify it is a valid draw
white_balls = decoded_draw['white_balls']
powerball = decoded_draw['powerball']

is_valid = (
    len(white_balls) == 5 and
    len(set(white_balls)) == 5 and  # all unique
    all(1 <= b <= 69 for b in white_balls) and
    white_balls == sorted(white_balls) and  # sorted
    1 <= powerball <= 26
)

print(f"Draw validation: {' VALID' if is_valid else ' INVALID'}")
print()
print("This is exactly what we'll do in Phase 2, but with real model outputs.")
print()

print("\n Phase 1 integration test complete! \n")

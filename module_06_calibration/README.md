# Module 6: Calibration Analysis and Ensembles

Tests whether models' uncertainty estimates are reliable and whether combining models helps. Module 5 trained 7 models, some with confidence intervals. This module checks if those intervals are trustworthy.

## What It Tests

1. **Calibration** - Do 95% confidence intervals actually contain 95% of true values?
2. **Ensembles** - Does combining multiple models improve predictions?

For random data, we expect poor calibration (models struggle with noise) and ensembles that perform similarly to individual models.

## How to Run

```bash
# Calibration analysis
python modules/module_06_calibration/calibration_analysis.py ../outputs/powerball ../outputs/megamillions module6_outputs

# Build ensembles
python modules/module_06_calibration/ensemble_builder.py ../outputs/powerball ../outputs/megamillions module6_outputs

# Validate
python modules/module_06_calibration/tests/test_calibration.py module6_outputs
```

## Calibration Metrics

**ECE (Expected Calibration Error)** - Average gap between claimed and actual confidence. Lower is better.

**Coverage** - Percentage of true values inside 95% intervals. Should be about 0.95 for well-calibrated models.

**Sharpness** - Average interval width. Narrow intervals mean confident predictions.

## Ensemble Methods

1. Mean - simple average
2. Variance-weighted - confident models get higher weight
3. Calibration-weighted - well-calibrated models get higher weight
4. Quantile - conservative ensemble using median and min/max bounds

## Expected Results

For random lottery data:
- BNN shows poor calibration (about 9% coverage instead of 95%)
- DeepSets shows good calibration (about 97% coverage)
- All ensembles perform within 0.01 RMSE of best individual model
- No ensemble significantly outperforms others

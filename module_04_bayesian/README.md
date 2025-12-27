# Module 4: Bayesian Analysis

Tests if lottery drawings are random using 5 Bayesian models. The key metric is the Bayes Factor (BF), which tells us how much more likely one hypothesis is versus another. BF < 0.01 means decisive evidence for randomness (what we expect).

## The 5 Models

1. **Uniform vs Non-Uniform** (fast, analytical) - Are all numbers equally likely?
2. **Temporal Dependence** (slow, MCMC) - Does one drawing affect the next?
3. **Position Bias** (slow, MCMC) - Do certain positions favor higher/lower numbers?
4. **Hierarchical Clustering** (fast, analytical) - Do numbers form clusters?
5. **Bayesian Regression** (slow, MCMC) - Can features predict outcomes?

## How to Run

```bash
# Run analysis
python modules/module_04_bayesian/bayesian_analysis.py

# Create visualizations
python modules/module_04_bayesian/bayesian_visualization.py
```

## Output

- `outputs/bayesian/bayesian_results_complete.json` - All model results
- `outputs/bayesian/plots/bayes_factors_comparison.png` - Visual comparison
- `outputs/bayesian/plots/interpretation_summary.txt` - Text summary

## Interpreting Bayes Factors

Using Jeffreys' Scale:
- BF < 0.01: Decisive evidence for randomness
- 0.01 < BF < 0.1: Strong evidence for randomness
- 0.1 < BF < 0.33: Moderate evidence for randomness
- BF > 1: Evidence against randomness (not expected)

For lottery analysis, all BFs should be < 0.01.

## Expected Results

All models should show decisive evidence for randomness. In my analysis:
- Model 1: BF approximately 10^-65 (decisive for uniformity)
- Model 2: BF approximately 0.001 (no temporal dependence)
- Model 3: BF approximately 0.01 (no position bias)
- Model 4: BF approximately 10^-50 (no clustering)
- Model 5: BF approximately 0.01 (features can't predict)

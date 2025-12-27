# Module 8: Causal Inference

Tests whether lottery drawings exhibit any causal relationships or structural dependencies beyond what you'd expect from independent random processes.

## What It Tests

1. **Granger Causality** - Do past values of one variable help predict another?
2. **Conditional Independence** - Are variables independent given other variables?
3. **Causal Discovery** - Can we find causal graphs in the data?
4. **Transfer Entropy** - Is there directed information flow between draws?
5. **Causal Invariance** - Are causal relationships stable over time?

## How to Run

```bash
cd modules/module_08_causal_inference

# Full analysis, both lotteries
python module8_complete_analysis.py --lottery both

# Single lottery
python module8_complete_analysis.py --lottery powerball

# Quick mode (fewer computations)
python module8_complete_analysis.py --lottery powerball --quick
```

Or run individual analyses:
```bash
python granger_causality.py
python conditional_independence.py
python causal_discovery.py
python transfer_entropy.py
python causal_invariance.py
```

## Output

CSV files with test results, pickled causal graphs, and PNG visualizations saved to `outputs/` and `figures/`.

## Expected Results

For random lottery data, all tests should find no causal relationships. Granger tests should show no predictive power, conditional independence tests should pass, and causal discovery should find empty or trivial graphs.

**Note:** GFCI algorithm availability varies by causal-learn version. If it's not available, the analysis skips it automatically.

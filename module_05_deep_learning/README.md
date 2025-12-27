# Module 5: Deep Learning

Tests lottery randomness with 7 different neural network architectures. If the lottery is truly random, all models should perform similarly and fail to beat baseline performance.

## Models Tested

- VAE (variational autoencoder)
- Transformer
- BNN (Bayesian neural network)
- Normalizing Flow
- DeepSets (v1 and v2)
- Transformer + Bayesian Head (4 variants)

## How to Run

Train a single model:
```bash
python modules/module_05_deep_learning/module5_zoo.py --model vae --lottery powerball
```

Train all models (PowerShell):
```bash
$models = @('vae', 'transformer', 'bnn', 'flow', 'deepsets', 'deepsets_v2', 'transformer_bayeshead')
foreach ($model in $models) {
    python modules/module_05_deep_learning/module5_zoo.py --model $model --lottery powerball --epochs 60
    python modules/module_05_deep_learning/module5_zoo.py --model $model --lottery megamillions --epochs 60
}
```

Generate visualizations:
```bash
python modules/module_05_deep_learning/visualize_results.py
```

## Output

Results saved to `outputs/<lottery>/`:
- `<model>_perf_<suffix>.json` - performance metrics
- `<model>_predictive_<suffix>.json` - predictions with uncertainty
- `dl_performance_summary.json` - aggregated results

## Expected Results

All models should cluster around RMSE values of 0.83-1.09, confirming that no architecture finds patterns.

**Note:** BNN achieves about 9% coverage instead of 95%. This is expected for random noise and documented in the research. Transformer+BayesHead can't run on Mega Millions because the test set (63 samples) is smaller than the required window length (64).

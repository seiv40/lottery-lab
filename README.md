# Lottery Lab

Testing whether machine learning and advanced statistics can predict lottery numbers. Spoiler: they can't.

I know a simple chi-square test would prove randomness in about 10 lines of code. This project uses 10 independent analytical methods as a way to demonstrate different statistical and ML techniques on a real dataset where we know the ground truth. All approaches converged on the same conclusion with 100% validity.

## Project Structure

The analysis is organized into 11 modules:

1. **Data Validation** - Filter to current lottery formats
2. **Feature Engineering** - Generate 90+ statistical features
3. **Visualization** - Create JSON data for JackpotMath.com
4. **Bayesian Analysis** - 5 Bayesian models testing randomness
5. **Deep Learning** - 7 neural network architectures
6. **Calibration** - Test uncertainty estimates and ensembles
7. **Universe Generation** - Synthetic data generation
8. **Causal Inference** - Test for causal relationships
9. **Network Analysis** - Graph theory on lottery data
10. **Manifold Geometry** - Topological and geometric analysis
11. **Meta-Analysis** - Synthesize results across all modules

## Main Findings

All 10 analytical approaches reached the same conclusion:

- Bayesian analysis: Bayes Factor = 10^-65 (decisive evidence for uniformity)
- Deep learning: RMSE approximately 1.0 (baseline, no patterns learned)
- Causal inference: No temporal dependencies detected
- Network analysis: No structure beyond random graphs
- Meta-analysis: Fisher's combined p = 0.51 (consistent with randomness)

Lottery drawings behave like well-calibrated random number generators. No feature-based, temporal, causal, network, or topological strategy can exploit any structure because no exploitable structure exists.

## Data

Analysis uses current-format drawings only:
- **Powerball:** 1,550 drawings (Oct 2015 - present)
- **Mega Millions:** 830 drawings (Oct 2017 - present)

Older formats excluded to avoid biasing frequency tests with different number ranges.

## Requirements

Core dependencies:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

Module-specific requirements listed in each module's README.

## Usage

Each module can run independently. Typical workflow:

```bash
# Step 1: Filter data
python modules/module_01_data_validation/filter_current_formats.py

# Step 2: Generate features
python modules/module_02_feature_engineering/feature_engineering.py

# Step 3+: Run any module
python modules/module_04_bayesian/bayesian_analysis.py
python modules/module_05_deep_learning/module5_zoo.py --model transformer --lottery powerball
# etc.

# Final: Synthesize results
python modules/module_11_meta_analysis/cross_module_synthesis.py
```

See individual module READMEs for details.

## Methods Used

- Bayesian inference (Dirichlet-Multinomial, hierarchical models, Bayes factors)
- Neural networks (VAE, Transformers, BNN, Normalizing Flows, DeepSets)
- Causal inference (Granger causality, transfer entropy, causal discovery)
- Graph theory (co-occurrence networks, null models, spectral analysis)
- Topology (persistent homology, manifold learning, curvature estimation)
- Meta-analysis (Fisher's method, convergent validity)

# Module 7: Universe Generation

Generates synthetic lottery datasets (called "universes") to test whether our models can produce realistic lottery drawings.

## Generation Methods

1. **Null Universe** - Pure random sampling (baseline)
2. **VAE Universe** - Generated from trained VAE model
3. **Flow Universe** - Generated from trained normalizing flow
4. **Empirical Bootstrap** - Resample real draws with replacement
5. **Block Bootstrap** - Resample blocks of consecutive draws

Each universe is evaluated to see if it's statistically indistinguishable from real lottery data.

## How to Run

```bash
cd modules/module_07_universes

# Generate null universe (baseline)
python module7_generation.py

# Train generative models
python vae_generative_model.py --lottery powerball --epochs 100
python flow_generative_model.py --lottery powerball --epochs 100

# Generate model universes
python run_vae_universe.py powerball
python run_flow_universe.py powerball

# Generate bootstrap universes
python bootstrap_universes.py powerball both

# Evaluate all
python module7_evaluation.py powerball
```

## Universe Realism Index (URI)

Measures how realistic a generated universe is:
- URI >= 0.90: Excellent (indistinguishable from real data)
- URI >= 0.75: Good (minor differences)
- URI >= 0.60: Fair (noticeable differences)
- URI < 0.60: Poor (easily distinguishable)

The null universe should achieve URI about 1.0. Model universes should match this.

## Output

Universes saved to `outputs/`:
- `universe_null_pb.parquet`
- `universe_vae_pb.parquet`
- `universe_flow_pb.parquet`
- `universe_bootstrap_pb.parquet`
- `universe_block_bootstrap_pb.parquet`

Evaluation results: `outputs/<lottery>/module7_evaluation_pb.json`

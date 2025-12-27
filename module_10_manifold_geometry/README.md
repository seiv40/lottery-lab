# Module 10: Manifold Geometry and Topology

Tests whether lottery data exhibits any intrinsic geometric structure beyond random noise using differential geometry, topological data analysis, and manifold learning.

## Analyses Performed

1. Distance concentration
2. UMAP stability
3. Diffusion maps
4. Intrinsic dimensionality
5. Persistent cohomology (TDA)
6. Random manifold tests
7. GNN embeddings
8. Contrastive embeddings
9. Curvature estimation
10. Geodesic entropy

## How to Run

```bash
cd modules/module_10_manifold_geometry

# Both lotteries, skip TDA (recommended)
python master_pipeline.py --lottery both --skip_tda

# Single lottery, include TDA (requires 16GB+ RAM)
python master_pipeline.py --lottery powerball
```

Or run individual analyses:
```bash
python distance_concentration.py --lottery powerball
python umap_stability.py --lottery powerball --n_seeds 5
python intrinsic_dimensionality.py --lottery powerball
python curvature_estimation.py --lottery powerball --max_samples 500
```

## Expected Results

For random data, we expect high intrinsic dimensionality (no manifold structure), unstable embeddings, trivial topological features, and curvature distributions consistent with noise.

**Note:** TDA (persistent cohomology) is memory-intensive. Use `--skip_tda` if you have less than 16GB RAM. PyTorch Geometric installation can be tricky, see their docs if it fails.

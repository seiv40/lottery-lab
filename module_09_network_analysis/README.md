# Module 9: Network Analysis

Applies graph theory to lottery data. Constructs 7 different network representations and tests whether they show any structure beyond random graphs.

## Graph Types

1. **Ball Co-Occurrence** - Which balls appear together
2. **Temporal Transition** - Sequence patterns between draws
3. **Position Correlation** - Correlations between ball positions
4. **K-Nearest Neighbor** - Draw similarity network
5. **Bipartite Draw-Ball** - Bipartite mapping
6. **Position-Specific** - Individual networks for each position
7. **Temporal Evolution** - Time-windowed snapshots

Compared against random graph null models (Erdos-Renyi, configuration model, rewired networks, etc.).

## How to Run

```bash
cd modules/module_09_network_analysis

# Full pipeline with 100 null samples
python master_pipeline.py powerball 100

# Quick test (10 samples)
python master_pipeline.py powerball 10
```

Or run individual steps:
```bash
python 01_graph_construction.py powerball
python 02_network_metrics.py powerball
python 03_null_models.py powerball 100
python 04_statistical_tests.py powerball
python 05_advanced_analysis.py powerball
python 06_visualizations.py powerball
```

## Metrics Computed

For each graph: centrality measures, clustering, path metrics, degree distributions, assortativity, small-world metrics, community structure, and network motifs.

## Expected Results

For random lottery data, observed networks should match null model distributions. No significant differences in metrics, no community structure, and degree distributions that look like random graphs.

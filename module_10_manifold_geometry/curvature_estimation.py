"""
Module 10: Manifold Curvature Estimation
Estimates Ricci curvature and sectional curvature of lottery manifold.

Implements:
- Forman-Ricci curvature (combinatorial, based on degrees)
- Ollivier-Ricci curvature (optimal transport approximation)
- Sectional curvature (PCA-based local estimation)

Read more: https://github.com/saibalmars/GraphRicciCurvature (doc + examples) +
+ https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 

"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
    get_module10_figures_dir,
)
from data import load_features, load_module9_graphs


def compute_forman_ricci_curvature(G: nx.Graph) -> Dict[tuple, float]:
    """
    Compute Forman-Ricci curvature for each edge in the graph.
    
    Forman curvature for edge (i,j):
    kappa(i,j) = w_ij * (deg(i) + deg(j) - 2)
    
    For unweighted graphs, w_ij = 1.
    """
    curvatures = {}
    
    for i, j in G.edges():
        deg_i = G.degree(i)
        deg_j = G.degree(j)
        
        # Forman curvature formula (simplified)
        curv = deg_i + deg_j - 2
        curvatures[(i, j)] = curv
    
    return curvatures


def compute_ollivier_ricci_curvature_approximate(
    G: nx.Graph, 
    alpha: float = 0.5,
    n_samples: int = 100,
    random_state: int = RANDOM_STATE_DEFAULT
) -> Dict[tuple, float]:
    """
    Approximate Ollivier-Ricci curvature using lazy random walks.
    
    This is a simplified version. Full Ollivier-Ricci requires optimal transport,
    which is computationally expensive. This uses a lazy random walk approximation.
    """
    rng = np.random.default_rng(random_state)
    curvatures = {}
    
    # Sample edges if graph is large
    edges = list(G.edges())
    if len(edges) > n_samples:
        edges = rng.choice(edges, size=n_samples, replace=False)
    
    for i, j in edges:
        # Get neighbors
        N_i = set(G.neighbors(i))
        N_j = set(G.neighbors(j))
        
        # Lazy random walk distributions
        deg_i = len(N_i)
        deg_j = len(N_j)
        
        if deg_i == 0 or deg_j == 0:
            curvatures[(i, j)] = 0.0
            continue
        
        # Compute Wasserstein-1 distance approximation
        # W1(μ_i, μ_j) approximately 1 - |N_i intersect N_j| / |N_i union N_j|
        intersection = len(N_i & N_j)
        union = len(N_i | N_j)
        
        if union > 0:
            w1_approx = 1.0 - (intersection / union)
        else:
            w1_approx = 1.0
        
        # Ollivier-Ricci curvature: kappa = 1 - W1(μ_i, μ_j)
        curv = 1.0 - w1_approx
        curvatures[(i, j)] = curv
    
    return curvatures


def compute_sectional_curvature_pca(
    X: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Estimate sectional curvature at each point using local PCA.
    
    Sectional curvature measures how the manifold curves in different
    2D planes. We approximate this by:
    1. Computing local tangent space via PCA on k-NN
    2. Computing explained variance ratios
    3. Curvature ~ (1 - lambda_1 - lambda_2) where lambda are top 2 eigenvalues
    
    Returns array of shape (n_samples,) with local curvature estimates.
    """
    from sklearn.decomposition import PCA
    
    n = X.shape[0]
    k = min(k, n - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    indices = nbrs.kneighbors(return_distance=False)
    
    curvatures = np.zeros(n)
    
    for i in range(n):
        # Get local neighborhood
        local_idx = indices[i, 1:]  # Exclude self
        local_points = X[local_idx]
        
        # Center the points
        local_centered = local_points - local_points.mean(axis=0)
        
        # PCA to get tangent space
        if local_centered.shape[0] > 2:
            pca = PCA(n_components=min(3, local_centered.shape[1]))
            pca.fit(local_centered)
            
            # Curvature estimate: 1 - sum of top 2 explained variance ratios
            # High curvature -> low explained variance in top components
            explained_var = pca.explained_variance_ratio_
            if len(explained_var) >= 2:
                curvatures[i] = 1.0 - (explained_var[0] + explained_var[1])
            else:
                curvatures[i] = 0.0
        else:
            curvatures[i] = 0.0
    
    return curvatures


def run_curvature_estimation(
    lottery: LotteryName,
    k_neighbors: int = 10,
    max_samples: int = 500,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    """Run manifold curvature estimation analysis."""
    
    print(f"\n[Module10][Curvature] Starting for {lottery}...")
    
    # Load feature data
    df = load_features(lottery, numeric_only=True)
    X_full = df.to_numpy(dtype=float)
    
    # Downsample if needed
    if X_full.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_full.shape[0], size=max_samples, replace=False)
        X = X_full[idx]
        print(f"  Downsampled from {X_full.shape[0]} to {X.shape[0]} samples.")
    else:
        X = X_full
    
    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)
    
    # ---- Graph-based curvature (Forman, Ollivier-Ricci) ----
    print("  Computing graph-based curvatures...")
    
    try:
        graphs = load_module9_graphs(lottery)
        knn_graph = graphs["knn_graph"]
        
        # Forman-Ricci curvature
        forman_curvatures = compute_forman_ricci_curvature(knn_graph)
        forman_values = list(forman_curvatures.values())
        forman_mean = float(np.mean(forman_values)) if forman_values else 0.0
        forman_std = float(np.std(forman_values)) if forman_values else 0.0
        
        # Ollivier-Ricci approximation
        ollivier_curvatures = compute_ollivier_ricci_curvature_approximate(
            knn_graph,
            n_samples=min(100, len(knn_graph.edges())),
            random_state=random_state,
        )
        ollivier_values = list(ollivier_curvatures.values())
        ollivier_mean = float(np.mean(ollivier_values)) if ollivier_values else 0.0
        ollivier_std = float(np.std(ollivier_values)) if ollivier_values else 0.0
        
        print(f"    Forman-Ricci: mean={forman_mean:.4f}, std={forman_std:.4f}")
        print(f"    Ollivier-Ricci: mean={ollivier_mean:.4f}, std={ollivier_std:.4f}")
        
    except Exception as e:
        print(f"  Warning: Could not load Module 9 graphs: {e}")
        forman_mean = forman_std = ollivier_mean = ollivier_std = np.nan
        forman_values = []
        ollivier_values = []
    
    # ---- Sectional curvature (PCA-based) ----
    print("  Computing sectional curvatures...")
    
    sectional_curvatures = compute_sectional_curvature_pca(X, k=k_neighbors)
    sectional_mean = float(np.mean(sectional_curvatures))
    sectional_std = float(np.std(sectional_curvatures))
    sectional_median = float(np.median(sectional_curvatures))
    
    print(f"    Sectional: mean={sectional_mean:.4f}, std={sectional_std:.4f}, median={sectional_median:.4f}")
    
    # Save summary
    summary_csv = os.path.join(output_dir, f"{lottery}_curvature_summary.csv")
    pd.DataFrame([{
        "lottery": lottery,
        "forman_ricci_mean": forman_mean,
        "forman_ricci_std": forman_std,
        "ollivier_ricci_mean": ollivier_mean,
        "ollivier_ricci_std": ollivier_std,
        "sectional_curvature_mean": sectional_mean,
        "sectional_curvature_std": sectional_std,
        "sectional_curvature_median": sectional_median,
        "k_neighbors": k_neighbors,
        "max_samples": max_samples,
    }]).to_csv(summary_csv, index=False)
    
    # Save detailed sectional curvatures
    sectional_csv = os.path.join(output_dir, f"{lottery}_sectional_curvatures.csv")
    pd.DataFrame({
        "sample_idx": np.arange(len(sectional_curvatures)),
        "sectional_curvature": sectional_curvatures,
    }).to_csv(sectional_csv, index=False)
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Forman-Ricci distribution
        if forman_values:
            axes[0].hist(forman_values, bins=30, alpha=0.7, edgecolor='black')
            axes[0].axvline(forman_mean, color='red', linestyle='--', label=f'Mean: {forman_mean:.2f}')
            axes[0].set_xlabel("Forman-Ricci Curvature")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Forman-Ricci Curvature (kNN Graph)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Ollivier-Ricci distribution
        if ollivier_values:
            axes[1].hist(ollivier_values, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1].axvline(ollivier_mean, color='red', linestyle='--', label=f'Mean: {ollivier_mean:.2f}')
            axes[1].set_xlabel("Ollivier-Ricci Curvature")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Ollivier-Ricci Curvature (Approx)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Sectional curvature distribution
        axes[2].hist(sectional_curvatures, bins=30, alpha=0.7, edgecolor='black', color='green')
        axes[2].axvline(sectional_mean, color='red', linestyle='--', label=f'Mean: {sectional_mean:.2f}')
        axes[2].set_xlabel("Sectional Curvature")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Sectional Curvature (PCA-based)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f"{lottery.capitalize()} - Manifold Curvature Analysis", y=1.02)
        
        fig_path = os.path.join(fig_dir, f"{lottery}_curvature_distributions.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved figure to {fig_path}")
        
    except ImportError:
        pass
    
    print(f"[Module10][Curvature] Completed for {lottery}.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Manifold Curvature Estimation"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_curvature_estimation(
        lottery=args.lottery,  # type: ignore
        k_neighbors=args.k_neighbors,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
"""
Module 10: Geodesic Neighborhood Entropy
Measures randomness in k-hop neighborhoods and random walks.

High entropy indicates uniform/random structure.
Low entropy indicates clustered/organized neighborhoods.

"""

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
    get_module10_figures_dir,
)
from data import load_module9_graphs


def compute_k_hop_neighborhoods(
    G: nx.Graph,
    k_max: int = 3,
) -> Dict[int, Dict[int, set]]:
    """
    Compute k-hop neighborhoods for all nodes.
    
    Returns:
        dict: {k: {node: set of neighbors at distance k}}
    """
    neighborhoods = {k: {} for k in range(1, k_max + 1)}
    
    for node in G.nodes():
        # BFS from this node
        distances = nx.single_source_shortest_path_length(G, node, cutoff=k_max)
        
        # Group by distance
        for k in range(1, k_max + 1):
            neighbors_at_k = {n for n, d in distances.items() if d == k}
            neighborhoods[k][node] = neighbors_at_k
    
    return neighborhoods


def compute_neighborhood_entropy(
    G: nx.Graph,
    neighborhoods: Dict[int, Dict[int, set]],
    k: int,
) -> np.ndarray:
    """
    Compute entropy of k-hop neighborhood degree distributions.
    
    For each node, compute the degree distribution of its k-hop neighbors,
    then compute the Shannon entropy of that distribution.
    
    High entropy -> uniform/random neighborhood structure
    Low entropy -> structured/clustered neighborhoods
    """
    entropies = []
    
    for node in G.nodes():
        neighbors_k = neighborhoods[k].get(node, set())
        
        if len(neighbors_k) == 0:
            entropies.append(0.0)
            continue
        
        # Get degrees of k-hop neighbors
        degrees = [G.degree(n) for n in neighbors_k]
        
        # Compute degree distribution
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        probs = counts / counts.sum()
        
        # Shannon entropy
        H = entropy(probs, base=2)
        entropies.append(H)
    
    return np.array(entropies)


def compute_random_walk_entropy(
    G: nx.Graph,
    walk_length: int = 5,
    n_walks_per_node: int = 100,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> Dict[int, float]:
    """
    Compute entropy of random walk visit distributions from each node.
    
    For each node:
    1. Perform multiple random walks
    2. Count visit frequencies to all nodes
    3. Compute Shannon entropy of visit distribution
    
    Returns:
        dict: {node: entropy}
    """
    rng = np.random.default_rng(random_state)
    entropies = {}
    
    for start_node in G.nodes():
        visit_counts = {n: 0 for n in G.nodes()}
        
        for _ in range(n_walks_per_node):
            current = start_node
            
            for step in range(walk_length):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = rng.choice(neighbors)
                visit_counts[current] += 1
        
        # Compute entropy
        counts = np.array(list(visit_counts.values()))
        counts = counts[counts > 0]  # Remove unvisited nodes
        
        if len(counts) > 0:
            probs = counts / counts.sum()
            H = entropy(probs, base=2)
            entropies[start_node] = float(H)
        else:
            entropies[start_node] = 0.0
    
    return entropies


def run_geodesic_entropy(
    lottery: LotteryName,
    k_max: int = 3,
    walk_length: int = 5,
    n_walks: int = 100,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    """Run geodesic neighborhood entropy analysis."""
    
    print(f"\n[Module10][GeodesicEntropy] Starting for {lottery}...")
    
    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)
    
    # Load Module 9 graphs
    graphs = load_module9_graphs(lottery)
    knn_graph = graphs["knn_graph"]
    
    print(f"  Graph: {knn_graph.number_of_nodes()} nodes, {knn_graph.number_of_edges()} edges")
    
    # ---- K-hop neighborhood entropy ----
    print(f"  Computing {k_max}-hop neighborhood entropies...")
    
    neighborhoods = compute_k_hop_neighborhoods(knn_graph, k_max=k_max)
    
    results = []
    for k in range(1, k_max + 1):
        entropies_k = compute_neighborhood_entropy(knn_graph, neighborhoods, k=k)
        
        mean_entropy = float(np.mean(entropies_k))
        std_entropy = float(np.std(entropies_k))
        median_entropy = float(np.median(entropies_k))
        
        results.append({
            "lottery": lottery,
            "k_hops": k,
            "mean_entropy": mean_entropy,
            "std_entropy": std_entropy,
            "median_entropy": median_entropy,
            "min_entropy": float(np.min(entropies_k)),
            "max_entropy": float(np.max(entropies_k)),
        })
        
        print(f"    k={k}: mean={mean_entropy:.4f}, std={std_entropy:.4f}, median={median_entropy:.4f}")
    
    # Save k-hop results
    khop_csv = os.path.join(output_dir, f"{lottery}_khop_entropy.csv")
    pd.DataFrame(results).to_csv(khop_csv, index=False)
    
    # ---- Random walk entropy ----
    print(f"  Computing random walk entropies (length={walk_length}, n_walks={n_walks})...")
    
    rw_entropies = compute_random_walk_entropy(
        knn_graph,
        walk_length=walk_length,
        n_walks_per_node=n_walks,
        random_state=random_state,
    )
    
    rw_values = np.array(list(rw_entropies.values()))
    rw_mean = float(np.mean(rw_values))
    rw_std = float(np.std(rw_values))
    rw_median = float(np.median(rw_values))
    
    print(f"    Random walk: mean={rw_mean:.4f}, std={rw_std:.4f}, median={rw_median:.4f}")
    
    # Save random walk results
    rw_csv = os.path.join(output_dir, f"{lottery}_random_walk_entropy.csv")
    pd.DataFrame([{
        "lottery": lottery,
        "walk_length": walk_length,
        "n_walks_per_node": n_walks,
        "mean_entropy": rw_mean,
        "std_entropy": rw_std,
        "median_entropy": rw_median,
        "min_entropy": float(np.min(rw_values)),
        "max_entropy": float(np.max(rw_values)),
    }]).to_csv(rw_csv, index=False)
    
    # Save per-node random walk entropies
    rw_detailed_csv = os.path.join(output_dir, f"{lottery}_random_walk_entropy_per_node.csv")
    pd.DataFrame([
        {"node": node, "entropy": ent}
        for node, ent in rw_entropies.items()
    ]).to_csv(rw_detailed_csv, index=False)
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # K-hop entropy trends
        df_khop = pd.DataFrame(results)
        axes[0].errorbar(
            df_khop["k_hops"],
            df_khop["mean_entropy"],
            yerr=df_khop["std_entropy"],
            marker='o',
            capsize=5,
            label="Mean +/- Std"
        )
        axes[0].plot(df_khop["k_hops"], df_khop["median_entropy"], marker='s', label="Median")
        axes[0].set_xlabel("k-hops")
        axes[0].set_ylabel("Entropy (bits)")
        axes[0].set_title(f"{lottery.capitalize()} - k-Hop Neighborhood Entropy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Random walk entropy distribution
        axes[1].hist(rw_values, bins=30, alpha=0.7, edgecolor='black', color='purple')
        axes[1].axvline(rw_mean, color='red', linestyle='--', label=f'Mean: {rw_mean:.2f}')
        axes[1].axvline(rw_median, color='orange', linestyle='--', label=f'Median: {rw_median:.2f}')
        axes[1].set_xlabel("Random Walk Entropy (bits)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"{lottery.capitalize()} - Random Walk Entropy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig_path = os.path.join(fig_dir, f"{lottery}_geodesic_entropy.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"  Saved figure to {fig_path}")
        
    except ImportError:
        pass
    
    print(f"[Module10][GeodesicEntropy] Completed for {lottery}.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Geodesic Neighborhood Entropy"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument("--k_max", type=int, default=3)
    parser.add_argument("--walk_length", type=int, default=5)
    parser.add_argument("--n_walks", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_geodesic_entropy(
        lottery=args.lottery,  # type: ignore
        k_max=args.k_max,
        walk_length=args.walk_length,
        n_walks=args.n_walks,
        random_state=args.random_state,
    )
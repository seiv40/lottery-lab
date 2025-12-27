"""
MODULE 9 - NETWORK METRICS COMPUTATION
Lottery Lab: Network Analysis of Lottery Structure

Computes comprehensive network metrics for all graph types:
- Centrality measures (degree, betweenness, closeness, eigenvector, PageRank, Katz)
- Clustering coefficients (local, global, transitivity)
- Community detection (Louvain, modularity, spectral clustering, label propagation)
- Path length distributions
- Assortativity and degree correlations
- Small-world properties (clustering vs path length)
- Graph entropy and information measures
- Network motifs (triads, cliques, cycles)

Read more: ch 2, 7, 9 of https://networksciencebook.com/ (it is free)

"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("Warning: python-louvain not available")


class NetworkMetricsCalculator:
    """
    Comprehensive network metrics computation for lottery graphs.
    """
    
    def __init__(self, lottery_name: str, graphs_dir: Path, output_dir: Path):
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        graphs_dir : Path
            Directory containing pickled graph objects
        output_dir : Path
            Directory for saving metrics CSV files
        """
        self.lottery_name = lottery_name
        self.graphs_dir = Path(graphs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.graphs = {}
        self.metrics = {}
        
        print(f"Initialized metrics calculator for {lottery_name}")
    
    def load_graphs(self):
        """Load all pickled graph objects."""
        print("\nLoading graphs...")
        
        graph_files = list(self.graphs_dir.glob(f"{self.lottery_name}_*.pkl"))
        
        for filepath in graph_files:
            graph_name = filepath.stem.replace(f"{self.lottery_name}_", "")
            
            with open(filepath, 'rb') as f:
                self.graphs[graph_name] = pickle.load(f)
            
            print(f"  Loaded: {graph_name}")
        
        print(f"\nTotal graphs loaded: {len(self.graphs)}")
    
    def compute_all_metrics(self):
        """Compute metrics for all graph types."""
        print("\n" + "="*60)
        print("COMPUTING NETWORK METRICS")
        print("="*60)
        
        # Skip certain graph types that need special handling
        skip_graphs = ['temporal_snapshots', 'position_specific', 'graph_summary']
        
        for graph_name, graph_obj in self.graphs.items():
            # Skip things like temporal snapshots, summaries, etc.
            if any(skip in graph_name for skip in skip_graphs):
                continue

            # Skip non-graph, non-snapshot objects (e.g. *_params dicts)
            if not isinstance(
                graph_obj,
                (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph, list),
            ):
                print(f"[{graph_name.upper()}] Skipping (not a NetworkX graph or snapshot list: {type(graph_obj)})")
                continue
            
            print(f"\n[{graph_name.upper()}]")
            
            if isinstance(graph_obj, list):
                # Temporal snapshots - compute for each window
                self.metrics[graph_name] = self._compute_temporal_metrics(graph_obj)
            else:
                # Single graph - compute comprehensive metrics
                self.metrics[graph_name] = self._compute_graph_metrics(graph_obj, graph_name)

    def _compute_graph_metrics(self, G: nx.Graph, graph_name: str) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a single graph.
        
        Returns
        -------
        metrics : dict
            All computed network metrics
        """
        metrics = {
            'graph_name': graph_name,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'is_directed': G.is_directed()
        }
        
        # Skip if graph is empty or too small
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            print("  Graph too small for meaningful metrics")
            return metrics
        
        # Basic graph properties
        print("  Computing basic properties...")
        metrics.update(self._compute_basic_properties(G))
        
        # Centrality measures
        print("  Computing centrality measures...")
        metrics.update(self._compute_centrality_measures(G))
        
        # Clustering and community structure
        print("  Computing clustering and community metrics...")
        metrics.update(self._compute_clustering_metrics(G))
        
        # Path and distance metrics
        print("  Computing path metrics...")
        metrics.update(self._compute_path_metrics(G))
        
        # Degree distribution
        print("  Computing degree distribution...")
        metrics.update(self._compute_degree_metrics(G))
        
        # Assortativity
        print("  Computing assortativity...")
        metrics.update(self._compute_assortativity(G))
        
        # Small-world properties
        print("  Computing small-world metrics...")
        metrics.update(self._compute_smallworld_metrics(G))
        
        # Graph entropy and information
        print("  Computing information-theoretic metrics...")
        metrics.update(self._compute_information_metrics(G))
        
        # Network motifs (if feasible)
        if G.number_of_nodes() <= 1000:  # Limit for computational tractability
            print("  Computing network motifs...")
            metrics.update(self._compute_motif_metrics(G))
        
        return metrics
    
    def _compute_basic_properties(self, G: nx.Graph) -> Dict[str, float]:
        """Compute basic graph properties."""
        metrics = {}
        
        metrics['density'] = nx.density(G)
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['std_degree'] = np.std(degrees)
        metrics['min_degree'] = np.min(degrees)
        metrics['max_degree'] = np.max(degrees)
        metrics['median_degree'] = np.median(degrees)
        
        # Weighted degree if available
        if nx.is_weighted(G):
            weighted_degrees = [d for n, d in G.degree(weight='weight')]
            metrics['avg_weighted_degree'] = np.mean(weighted_degrees)
            metrics['std_weighted_degree'] = np.std(weighted_degrees)
        
        # Connected components
        if G.is_directed():
            metrics['n_weakly_connected'] = nx.number_weakly_connected_components(G)
            metrics['n_strongly_connected'] = nx.number_strongly_connected_components(G)
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            metrics['largest_wcc_size'] = len(largest_wcc)
            metrics['largest_wcc_fraction'] = len(largest_wcc) / G.number_of_nodes()
        else:
            metrics['n_connected_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            metrics['largest_cc_size'] = len(largest_cc)
            metrics['largest_cc_fraction'] = len(largest_cc) / G.number_of_nodes()
        
        return metrics
    
    def _compute_centrality_measures(self, G: nx.Graph) -> Dict[str, float]:
        """Compute node centrality measures and their distributions."""
        metrics = {}
        
        # Get largest connected component for centrality calculations
        if G.is_directed():
            Gcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        else:
            Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if Gcc.number_of_nodes() < 2:
            return metrics
        
        # Degree centrality
        degree_cent = nx.degree_centrality(Gcc)
        metrics['degree_centrality_mean'] = np.mean(list(degree_cent.values()))
        metrics['degree_centrality_std'] = np.std(list(degree_cent.values()))
        metrics['degree_centrality_max'] = np.max(list(degree_cent.values()))
        
        # Betweenness centrality (sample if graph is large)
        if Gcc.number_of_nodes() <= 500:
            betweenness = nx.betweenness_centrality(Gcc, weight='weight' if nx.is_weighted(Gcc) else None)
        else:
            # Sample for computational efficiency
            k = min(100, Gcc.number_of_nodes())
            betweenness = nx.betweenness_centrality(Gcc, k=k, weight='weight' if nx.is_weighted(Gcc) else None)
        
        metrics['betweenness_centrality_mean'] = np.mean(list(betweenness.values()))
        metrics['betweenness_centrality_std'] = np.std(list(betweenness.values()))
        metrics['betweenness_centrality_max'] = np.max(list(betweenness.values()))
        
        # Closeness centrality
        if Gcc.number_of_nodes() <= 1000:
            closeness = nx.closeness_centrality(Gcc, distance='weight' if nx.is_weighted(Gcc) else None)
            metrics['closeness_centrality_mean'] = np.mean(list(closeness.values()))
            metrics['closeness_centrality_std'] = np.std(list(closeness.values()))
            metrics['closeness_centrality_max'] = np.max(list(closeness.values()))
        
        # Eigenvector centrality (if graph is undirected or weakly connected)
        if not G.is_directed() or nx.is_weakly_connected(G):
            try:
                eigenvector = nx.eigenvector_centrality(Gcc, max_iter=1000, weight='weight' if nx.is_weighted(Gcc) else None)
                metrics['eigenvector_centrality_mean'] = np.mean(list(eigenvector.values()))
                metrics['eigenvector_centrality_std'] = np.std(list(eigenvector.values()))
                metrics['eigenvector_centrality_max'] = np.max(list(eigenvector.values()))
            except:
                pass
        
        # PageRank (works for directed and undirected)
        try:
            pagerank = nx.pagerank(Gcc, alpha=0.85, weight='weight' if nx.is_weighted(Gcc) else None)
            metrics['pagerank_mean'] = np.mean(list(pagerank.values()))
            metrics['pagerank_std'] = np.std(list(pagerank.values()))
            metrics['pagerank_max'] = np.max(list(pagerank.values()))
        except:
            pass
        
        # Katz centrality (for directed graphs)
        if G.is_directed() and Gcc.number_of_nodes() <= 1000:
            try:
                katz = nx.katz_centrality(Gcc, alpha=0.1, weight='weight' if nx.is_weighted(Gcc) else None)
                metrics['katz_centrality_mean'] = np.mean(list(katz.values()))
                metrics['katz_centrality_std'] = np.std(list(katz.values()))
                metrics['katz_centrality_max'] = np.max(list(katz.values()))
            except:
                pass
        
        return metrics
    
    def _compute_clustering_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Compute clustering and community detection metrics."""
        metrics = {}
        
        # Skip for directed graphs (clustering not well-defined)
        if G.is_directed():
            return metrics
        
        # Global clustering coefficient
        metrics['avg_clustering'] = nx.average_clustering(G, weight='weight' if nx.is_weighted(G) else None)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Local clustering coefficient distribution
        local_clustering = list(nx.clustering(G, weight='weight' if nx.is_weighted(G) else None).values())
        metrics['clustering_std'] = np.std(local_clustering)
        metrics['clustering_max'] = np.max(local_clustering)
        
        # Community detection using Louvain
        if LOUVAIN_AVAILABLE and G.number_of_nodes() > 1:
            try:
                # Get largest connected component
                Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                
                if nx.is_weighted(Gcc):
                    partition = community_louvain.best_partition(Gcc, weight='weight')
                else:
                    partition = community_louvain.best_partition(Gcc)
                
                # Modularity
                metrics['louvain_modularity'] = community_louvain.modularity(partition, Gcc, weight='weight' if nx.is_weighted(Gcc) else None)
                
                # Number of communities
                metrics['louvain_n_communities'] = len(set(partition.values()))
                
                # Community size distribution
                community_sizes = Counter(partition.values())
                sizes = list(community_sizes.values())
                metrics['louvain_avg_community_size'] = np.mean(sizes)
                metrics['louvain_std_community_size'] = np.std(sizes)
                metrics['louvain_max_community_size'] = np.max(sizes)
                metrics['louvain_min_community_size'] = np.min(sizes)
            except Exception as e:
                print(f"    Louvain failed: {e}")
        
        # Spectral clustering (for smaller graphs)
        if G.number_of_nodes() <= 500:
            try:
                # Get adjacency matrix
                adj_matrix = nx.to_numpy_array(G)
                
                # Compute modularity using greedy algorithm
                communities = nx.community.greedy_modularity_communities(G, weight='weight' if nx.is_weighted(G) else None)
                metrics['greedy_n_communities'] = len(communities)
                metrics['greedy_modularity'] = nx.community.modularity(G, communities, weight='weight' if nx.is_weighted(G) else None)
            except Exception as e:
                print(f"    Spectral clustering failed: {e}")
        
        return metrics
    
    def _compute_path_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Compute shortest path and distance metrics."""
        metrics = {}
        
        # Get largest connected component
        if G.is_directed():
            Gcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        else:
            Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if Gcc.number_of_nodes() < 2:
            return metrics
        
        # Average shortest path length (sample if large)
        if Gcc.number_of_nodes() <= 500:
            try:
                if nx.is_weighted(Gcc):
                    avg_path = nx.average_shortest_path_length(Gcc, weight='weight')
                else:
                    avg_path = nx.average_shortest_path_length(Gcc)
                metrics['avg_shortest_path_length'] = avg_path
            except:
                pass
        
        # Diameter (for smaller graphs)
        if Gcc.number_of_nodes() <= 500:
            try:
                if G.is_directed():
                    diameter = nx.algorithms.distance_measures.diameter(Gcc)
                else:
                    diameter = nx.diameter(Gcc)
                metrics['diameter'] = diameter
            except:
                pass
        
        # Radius (for smaller graphs)
        if not G.is_directed() and Gcc.number_of_nodes() <= 500:
            try:
                metrics['radius'] = nx.radius(Gcc)
            except:
                pass
        
        # Eccentricity distribution (for smaller graphs)
        if not G.is_directed() and Gcc.number_of_nodes() <= 300:
            try:
                eccentricity = nx.eccentricity(Gcc)
                metrics['eccentricity_mean'] = np.mean(list(eccentricity.values()))
                metrics['eccentricity_std'] = np.std(list(eccentricity.values()))
            except:
                pass
        
        return metrics
    
    def _compute_degree_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Compute degree distribution properties."""
        metrics = {}
        
        degrees = [d for n, d in G.degree()]
        
        # Degree distribution statistics
        metrics['degree_skewness'] = pd.Series(degrees).skew()
        metrics['degree_kurtosis'] = pd.Series(degrees).kurtosis()
        
        # Gini coefficient (inequality measure)
        sorted_degrees = np.sort(degrees)
        n = len(sorted_degrees)
        cumsum = np.cumsum(sorted_degrees)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_degrees)) / (n * cumsum[-1]) - (n + 1) / n
        metrics['degree_gini'] = gini
        
        # Power-law fit attempt (Clauset et al. method approximation)
        degree_counts = Counter(degrees)
        unique_degrees = sorted(degree_counts.keys())
        if len(unique_degrees) > 10:
            # Simple log-log regression for power-law exponent estimation
            log_degrees = np.log(unique_degrees)
            log_counts = np.log([degree_counts[d] for d in unique_degrees])
            
            # Remove zeros
            valid = (log_degrees > 0) & (log_counts > 0) & np.isfinite(log_degrees) & np.isfinite(log_counts)
            if np.sum(valid) > 5:
                coef = np.polyfit(log_degrees[valid], log_counts[valid], 1)
                metrics['degree_powerlaw_exponent'] = -coef[0]  # Negative of slope
        
        return metrics
    
    def _compute_assortativity(self, G: nx.Graph) -> Dict[str, float]:
        """Compute assortativity coefficients."""
        metrics = {}
        
        try:
            # Degree assortativity
            metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight' if nx.is_weighted(G) else None)
        except:
            pass
        
        # Attribute assortativity (if nodes have attributes)
        if G.number_of_nodes() > 0:
            node_attrs = list(G.nodes(data=True))[0][1].keys()
            
            for attr in node_attrs:
                if attr != 'bipartite':  # Skip bipartite attribute
                    try:
                        # Check if attribute is categorical or numeric
                        values = [G.nodes[n][attr] for n in G.nodes() if attr in G.nodes[n]]
                        if len(values) > 0 and isinstance(values[0], (int, float)):
                            assortativity = nx.numeric_assortativity_coefficient(G, attr)
                            metrics[f'{attr}_assortativity'] = assortativity
                    except:
                        pass
        
        return metrics
    
    def _compute_smallworld_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Compute small-world properties.
        
        Small-world networks have:
        - High clustering coefficient (like regular lattice)
        - Short average path length (like random graph)
        """
        metrics = {}
        
        if G.is_directed() or G.number_of_nodes() < 10:
            return metrics
        
        # Get largest connected component
        Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if Gcc.number_of_nodes() < 10:
            return metrics
        
        # Observed metrics
        C_obs = nx.average_clustering(Gcc)
        
        if Gcc.number_of_nodes() <= 500:
            try:
                L_obs = nx.average_shortest_path_length(Gcc)
                
                # Generate random graph with same n, m
                n = Gcc.number_of_nodes()
                m = Gcc.number_of_edges()
                p = 2 * m / (n * (n - 1))
                
                # Sample random graphs
                C_rand_list = []
                L_rand_list = []
                
                for _ in range(10):  # 10 random samples
                    G_rand = nx.gnp_random_graph(n, p)
                    if nx.is_connected(G_rand):
                        C_rand_list.append(nx.average_clustering(G_rand))
                        L_rand_list.append(nx.average_shortest_path_length(G_rand))
                
                if len(C_rand_list) > 0 and len(L_rand_list) > 0:
                    C_rand = np.mean(C_rand_list)
                    L_rand = np.mean(L_rand_list)
                    
                    # Small-world coefficient: sigma = (C/C_rand) / (L/L_rand)
                    # Small-world: sigma >> 1
                    if C_rand > 0 and L_rand > 0:
                        sigma = (C_obs / C_rand) / (L_obs / L_rand)
                        metrics['smallworld_sigma'] = sigma
                        metrics['smallworld_C_ratio'] = C_obs / C_rand
                        metrics['smallworld_L_ratio'] = L_obs / L_rand
            except:
                pass
        
        return metrics
    
    def _compute_information_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Compute information-theoretic metrics."""
        metrics = {}
        
        # Degree distribution entropy
        degrees = [d for n, d in G.degree()]
        degree_counts = Counter(degrees)
        degree_probs = np.array(list(degree_counts.values())) / sum(degree_counts.values())
        metrics['degree_entropy'] = entropy(degree_probs, base=2)
        
        # Edge weight entropy (if weighted)
        if nx.is_weighted(G) and G.number_of_edges() > 0:
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            weight_counts = Counter(weights)
            weight_probs = np.array(list(weight_counts.values())) / sum(weight_counts.values())
            metrics['edge_weight_entropy'] = entropy(weight_probs, base=2)
        
        # Structural entropy (von Neumann entropy)
        if G.number_of_nodes() <= 500 and not G.is_directed():
            try:
                # Compute normalized Laplacian eigenvalues
                L_norm = nx.normalized_laplacian_matrix(G).toarray()
                eigenvalues = np.linalg.eigvalsh(L_norm)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero
                
                if len(eigenvalues) > 0:
                    eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
                    metrics['von_neumann_entropy'] = entropy(eigenvalues, base=2)
            except:
                pass
        
        return metrics
    
    def _compute_motif_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Compute network motif statistics."""
        metrics = {}
        
        if G.is_directed():
            return metrics  # Skip for directed graphs (complex)
        
        # Triangle count
        triangles = sum(nx.triangles(G).values()) / 3  # Each triangle counted 3 times
        metrics['n_triangles'] = triangles
        
        # Clique analysis (for smaller graphs)
        if G.number_of_nodes() <= 200:
            try:
                cliques = list(nx.find_cliques(G))
                metrics['n_cliques'] = len(cliques)
                
                clique_sizes = [len(c) for c in cliques]
                metrics['max_clique_size'] = np.max(clique_sizes)
                metrics['avg_clique_size'] = np.mean(clique_sizes)
            except:
                pass
        
        # K-core decomposition
        try:
            core_numbers = nx.core_number(G)
            metrics['max_k_core'] = max(core_numbers.values())
            metrics['avg_k_core'] = np.mean(list(core_numbers.values()))
        except:
            pass
        
        return metrics
    
    def _compute_temporal_metrics(self, snapshots: List[nx.Graph]) -> pd.DataFrame:
        """
        Compute metrics for temporal snapshots.
        
        Parameters
        ----------
        snapshots : list of nx.Graph
            List of temporal graph snapshots
        
        Returns
        -------
        metrics_df : pd.DataFrame
            Time series of network metrics
        """
        print(f"  Computing metrics for {len(snapshots)} temporal snapshots...")
        
        metrics_list = []
        
        for i, G in enumerate(snapshots):
            snapshot_metrics = {
                'snapshot_index': i,
                'window_start': G.graph.get('window_start', i),
                'window_end': G.graph.get('window_end', i),
                'date_start': G.graph.get('date_start', ''),
                'date_end': G.graph.get('date_end', '')
            }
            
            # Compute basic metrics for each snapshot
            snapshot_metrics.update(self._compute_basic_properties(G))
            
            # Add clustering if undirected
            if not G.is_directed():
                snapshot_metrics['avg_clustering'] = nx.average_clustering(G)
            
            metrics_list.append(snapshot_metrics)
        
        return pd.DataFrame(metrics_list)
    
    def save_metrics(self):
        """Save all computed metrics to CSV files."""
        print("\n--- SAVING NETWORK METRICS ---")
        
        for graph_name, metrics_data in self.metrics.items():
            if isinstance(metrics_data, pd.DataFrame):
                # Temporal metrics (already DataFrame)
                filepath = self.output_dir / f"{self.lottery_name}_{graph_name}_metrics.csv"
                metrics_data.to_csv(filepath, index=False)
            else:
                # Single graph metrics (dict)
                filepath = self.output_dir / f"{self.lottery_name}_{graph_name}_metrics.csv"
                pd.DataFrame([metrics_data]).to_csv(filepath, index=False)
            
            print(f"  Saved: {filepath.name}")
        
        # Save centrality distributions for co-occurrence graph
        self._save_centrality_distributions()
    
    def _save_centrality_distributions(self):
        """Save detailed centrality distributions for main graphs."""
        main_graphs = ['cooccurrence_graph', 'transition_graph']
        
        for graph_name in main_graphs:
            if graph_name in self.graphs:
                G = self.graphs[graph_name]
                
                if G.is_directed():
                    Gcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
                else:
                    Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                
                if Gcc.number_of_nodes() < 2:
                    continue
                
                # Compute all centralities
                centrality_data = []
                
                degree_cent = nx.degree_centrality(Gcc)
                
                # Sample for betweenness if large
                if Gcc.number_of_nodes() <= 500:
                    betweenness = nx.betweenness_centrality(Gcc)
                else:
                    betweenness = nx.betweenness_centrality(Gcc, k=100)
                
                pagerank = nx.pagerank(Gcc)
                
                for node in Gcc.nodes():
                    centrality_data.append({
                        'node': node,
                        'degree': Gcc.degree(node),
                        'degree_centrality': degree_cent[node],
                        'betweenness_centrality': betweenness[node],
                        'pagerank': pagerank[node]
                    })
                
                df = pd.DataFrame(centrality_data)
                filepath = self.output_dir / f"{self.lottery_name}_{graph_name}_centrality_distributions.csv"
                df.to_csv(filepath, index=False)
                print(f"  Saved centrality distribution: {filepath.name}")


def main():
    """Main execution function."""
    import sys
    BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
    
    if len(sys.argv) < 2:
        print("Usage: python 02_network_metrics.py <lottery_name>")
        sys.exit(1)
    
    lottery_name = sys.argv[1].lower()
    
    if lottery_name == 'powerball':
        graphs_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
        output_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
    elif lottery_name == 'megamillions':
        graphs_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
        output_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
    else:
        print(f"Unknown lottery: {lottery_name}")
        sys.exit(1)
    
    # Compute metrics
    calculator = NetworkMetricsCalculator(lottery_name, graphs_dir, output_dir)
    calculator.load_graphs()
    calculator.compute_all_metrics()
    calculator.save_metrics()
    
    print("\n--- NETWORK METRICS COMPUTATION COMPLETE ---")


if __name__ == "__main__":
    main()

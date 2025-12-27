"""
MODULE 9 - NULL MODEL GENERATION
Lottery Lab: Network Analysis of Lottery Structure

Generates random graph null models for statistical comparison:
1. Erdős-Rényi random graphs (G(n,p))
2. Configuration model (preserves degree distribution)
3. Random geometric graphs
4. Barabási-Albert preferential attachment
5. Watts-Strogatz small-world
6. Shuffled/permuted lottery graphs (edge rewiring)
7. Stochastic Block Model (SBM)

For each null model, computes same metrics as lottery graphs
to enable statistical hypothesis testing.

More info: read ch. 3, 4, 5 and 9 of https://networksciencebook.com/ (it is free)

"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class NullModelGenerator:
    """
    Comprehensive null model generation for lottery graph comparison.
    """
    
    def __init__(self, lottery_name: str, graphs_dir: Path, output_dir: Path,
                 n_random_samples: int = 100):
        """
        Initialize null model generator.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        graphs_dir : Path
            Directory containing lottery graph objects
        output_dir : Path
            Directory for saving null model data
        n_random_samples : int
            Number of random graph samples to generate per null model
        """
        self.lottery_name = lottery_name
        self.graphs_dir = Path(graphs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_random_samples
        
        self.lottery_graphs = {}
        self.null_models = {}
        
        print(f"Initialized null model generator for {lottery_name}")
        print(f"  Will generate {n_random_samples} samples per null model")
    
    def load_lottery_graphs(self):
        """Load lottery graphs."""
        print("\nLoading lottery graphs...")
        
        graph_files = [
            f"{self.lottery_name}_cooccurrence_graph.pkl",
            f"{self.lottery_name}_transition_graph.pkl",
            f"{self.lottery_name}_knn_draws_k10_euclidean.pkl"
        ]
        
        for filename in graph_files:
            filepath = self.graphs_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    graph_name = filename.replace(f"{self.lottery_name}_", "").replace(".pkl", "")
                    self.lottery_graphs[graph_name] = pickle.load(f)
                print(f"  Loaded: {graph_name}")
    
    def generate_all_null_models(self):
        """Generate all null model types for all lottery graphs."""
        print("\n--- GENERATING NULL MODELS ---")
        
        for graph_name, G_lottery in self.lottery_graphs.items():
            print(f"\n[{graph_name.upper()}]")
            
            self.null_models[graph_name] = {}
            
            # Erdős-Rényi
            print("  Generating Erdős-Rényi null models...")
            self.null_models[graph_name]['erdos_renyi'] = \
                self._generate_erdos_renyi(G_lottery)
            
            # Configuration model
            print("  Generating Configuration null models...")
            self.null_models[graph_name]['configuration'] = \
                self._generate_configuration(G_lottery)
            
            # Edge-rewired (degree-preserving)
            print("  Generating edge-rewired null models...")
            self.null_models[graph_name]['rewired'] = \
                self._generate_rewired(G_lottery)
            
            # Barabási-Albert (if undirected)
            if not G_lottery.is_directed():
                print("  Generating Barabási-Albert null models...")
                self.null_models[graph_name]['barabasi_albert'] = \
                    self._generate_barabasi_albert(G_lottery)
                
                # Watts-Strogatz small-world
                print("  Generating Watts-Strogatz null models...")
                self.null_models[graph_name]['watts_strogatz'] = \
                    self._generate_watts_strogatz(G_lottery)
    
    def _generate_erdos_renyi(self, G: nx.Graph) -> Dict[str, any]:
        """
        Generate Erdős-Rényi random graphs G(n,p).
        
        Same number of nodes as G, probability p chosen to match edge density.
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        # Calculate edge probability
        if G.is_directed():
            p = m / (n * (n - 1))
        else:
            p = 2 * m / (n * (n - 1))
        
        print(f"    n={n}, m={m}, p={p:.4f}")
        
        # Generate samples
        graphs = []
        metrics_list = []
        
        for i in range(self.n_samples):
            if G.is_directed():
                G_rand = nx.gnp_random_graph(n, p, directed=True)
            else:
                G_rand = nx.gnp_random_graph(n, p)
            
            graphs.append(G_rand)
            
            # Compute basic metrics
            metrics = self._compute_basic_metrics(G_rand, model='erdos_renyi', sample=i)
            metrics_list.append(metrics)
        
        return {
            'graphs': graphs,
            'metrics': pd.DataFrame(metrics_list),
            'parameters': {'n': n, 'm': m, 'p': p}
        }
    
    def _generate_configuration(self, G: nx.Graph) -> Dict[str, any]:
        """
        Generate Configuration Model graphs (preserves degree sequence).
        """
        degree_sequence = [d for n, d in G.degree()]
        
        # Configuration model requires even sum of degrees
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[-1] += 1
        
        print(f"    Degree sequence: mean={np.mean(degree_sequence):.1f}, "
              f"std={np.std(degree_sequence):.1f}")
        
        graphs = []
        metrics_list = []
        
        for i in range(self.n_samples):
            try:
                if G.is_directed():
                    # For directed, need in-degree and out-degree sequences
                    in_degrees = [d for n, d in G.in_degree()]
                    out_degrees = [d for n, d in G.out_degree()]
                    G_rand = nx.directed_configuration_model(in_degrees, out_degrees)
                    G_rand = nx.DiGraph(G_rand)  # Remove parallel edges
                else:
                    G_rand = nx.configuration_model(degree_sequence)
                    G_rand = nx.Graph(G_rand)  # Remove parallel edges and self-loops
                    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
                
                graphs.append(G_rand)
                
                metrics = self._compute_basic_metrics(G_rand, model='configuration', sample=i)
                metrics_list.append(metrics)
            except:
                continue
        
        return {
            'graphs': graphs,
            'metrics': pd.DataFrame(metrics_list),
            'parameters': {'degree_sequence': degree_sequence}
        }
    
    def _generate_rewired(self, G: nx.Graph) -> Dict[str, any]:
        """
        Generate edge-rewired graphs (double-edge swap, preserves degree).
        """
        print(f"    Rewiring with {G.number_of_edges()} edges...")
        
        graphs = []
        metrics_list = []
        
        # Number of swaps: 10 * number of edges
        nswap = max(1, G.number_of_edges() * 10)
        
        for i in range(self.n_samples):
            try:
                if G.is_directed():
                    G_rand = nx.directed_edge_swap(G.copy(), nswap=nswap, max_tries=nswap*10)
                else:
                    G_rand = nx.double_edge_swap(G.copy(), nswap=nswap, max_tries=nswap*10)
                
                graphs.append(G_rand)
                
                metrics = self._compute_basic_metrics(G_rand, model='rewired', sample=i)
                metrics_list.append(metrics)
            except:
                continue
        
        return {
            'graphs': graphs,
            'metrics': pd.DataFrame(metrics_list),
            'parameters': {'nswap': nswap}
        }
    
    def _generate_barabasi_albert(self, G: nx.Graph) -> Dict[str, any]:
        """
        Generate Barabási-Albert preferential attachment graphs.
        """
        if G.is_directed():
            return None
        
        n = G.number_of_nodes()
        m_edges = G.number_of_edges()
        
        # Calculate average degree
        avg_degree = 2 * m_edges / n
        m = max(1, int(avg_degree / 2))
        
        print(f"    n={n}, m={m} (edges per new node)")
        
        graphs = []
        metrics_list = []
        
        for i in range(self.n_samples):
            try:
                G_rand = nx.barabasi_albert_graph(n, m)
                graphs.append(G_rand)
                
                metrics = self._compute_basic_metrics(G_rand, model='barabasi_albert', sample=i)
                metrics_list.append(metrics)
            except:
                continue
        
        return {
            'graphs': graphs,
            'metrics': pd.DataFrame(metrics_list),
            'parameters': {'n': n, 'm': m}
        }
    
    def _generate_watts_strogatz(self, G: nx.Graph) -> Dict[str, any]:
        """
        Generate Watts-Strogatz small-world graphs.
        """
        if G.is_directed():
            return None
        
        n = G.number_of_nodes()
        avg_degree = np.mean([d for n, d in G.degree()])
        k = max(2, int(avg_degree))
        
        # Ensure k is even
        if k % 2 != 0:
            k += 1
        
        # Rewiring probability
        p = 0.1
        
        print(f"    n={n}, k={k}, p={p}")
        
        graphs = []
        metrics_list = []
        
        for i in range(self.n_samples):
            try:
                G_rand = nx.watts_strogatz_graph(n, k, p)
                graphs.append(G_rand)
                
                metrics = self._compute_basic_metrics(G_rand, model='watts_strogatz', sample=i)
                metrics_list.append(metrics)
            except:
                continue
        
        return {
            'graphs': graphs,
            'metrics': pd.DataFrame(metrics_list),
            'parameters': {'n': n, 'k': k, 'p': p}
        }
    
    def _compute_basic_metrics(self, G: nx.Graph, model: str, sample: int) -> Dict[str, float]:
        """Compute basic metrics for a null model graph."""
        metrics = {
            'model': model,
            'sample': sample,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G)
        }
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['std_degree'] = np.std(degrees)
        
        # Clustering (if undirected)
        if not G.is_directed():
            metrics['avg_clustering'] = nx.average_clustering(G)
            metrics['transitivity'] = nx.transitivity(G)
        
        # Connected components
        if G.is_directed():
            metrics['n_weakly_connected'] = nx.number_weakly_connected_components(G)
        else:
            metrics['n_connected_components'] = nx.number_connected_components(G)
            
            # Get largest CC
            if nx.number_connected_components(G) > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                metrics['largest_cc_size'] = len(largest_cc)
                
                # Average path length in largest CC
                Gcc = G.subgraph(largest_cc).copy()
                if Gcc.number_of_nodes() <= 500:
                    try:
                        metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(Gcc)
                    except:
                        pass
        
        return metrics
    
    def compute_null_model_statistics(self):
        """Compute summary statistics across all null model samples."""
        print("\n--- COMPUTING NULL MODEL SUMMARY STATISTICS ---")
        
        summary_data = []
        
        for graph_name, models in self.null_models.items():
            print(f"\n[{graph_name.upper()}]")
            
            for model_name, model_data in models.items():
                if model_data is None:
                    continue
                
                metrics_df = model_data['metrics']
                
                # Compute mean, std, quantiles for each metric
                numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    values = metrics_df[col].values
                    
                    summary = {
                        'graph_name': graph_name,
                        'null_model': model_name,
                        'metric': col,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'q25': np.percentile(values, 25),
                        'median': np.median(values),
                        'q75': np.percentile(values, 75),
                        'max': np.max(values)
                    }
                    
                    summary_data.append(summary)
            
            print(f"  Computed summaries for {len(models)} null models")
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        filepath = self.output_dir / f"{self.lottery_name}_null_model_summary.csv"
        summary_df.to_csv(filepath, index=False)
        print(f"\nSaved null model summary: {filepath}")
        
        return summary_df
    
    def save_null_models(self):
        """Save null model data (metrics only, not full graphs for space)."""
        print("\n--- SAVING NULL MODEL METRICS ---")
        
        for graph_name, models in self.null_models.items():
            for model_name, model_data in models.items():
                if model_data is None:
                    continue
                
                # Save metrics DataFrame
                metrics_df = model_data['metrics']
                filepath = self.output_dir / f"{self.lottery_name}_{graph_name}_{model_name}_nullmetrics.csv"
                metrics_df.to_csv(filepath, index=False)
                print(f"  Saved: {filepath.name}")
                
                # Save parameters
                params = model_data['parameters']
                params_filepath = self.output_dir / f"{self.lottery_name}_{graph_name}_{model_name}_params.pkl"
                with open(params_filepath, 'wb') as f:
                    pickle.dump(params, f)


def main():
    """Main execution function."""
    import sys
    BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
    
    if len(sys.argv) < 2:
        print("Usage: python 03_null_models.py <lottery_name> [n_samples]")
        sys.exit(1)
    
    lottery_name = sys.argv[1].lower()
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    if lottery_name == 'powerball':
        graphs_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
        output_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
    elif lottery_name == 'megamillions':
        graphs_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
        output_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'outputs'
    else:
        print(f"Unknown lottery: {lottery_name}")
        sys.exit(1)
    
    # Generate null models
    generator = NullModelGenerator(lottery_name, graphs_dir, output_dir, n_samples)
    generator.load_lottery_graphs()
    generator.generate_all_null_models()
    generator.compute_null_model_statistics()
    generator.save_null_models()
    
    print("\n--- NULL MODEL GENERATION COMPLETE ---")


if __name__ == "__main__":
    main()

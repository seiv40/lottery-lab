"""
MODULE 9 - ADVANCED NETWORK ANALYSIS
Lottery Lab: Network Analysis of Lottery Structure

Graph-theoretic techniques:
1. Spectral Analysis (Laplacian eigenvalues, spectral gap, Fiedler vector)
2. Persistent Homology / Topological Data Analysis (Betti numbers, persistence diagrams)
3. Graph Embeddings (Node2Vec, DeepWalk, GraphSAGE concepts)
4. Network Reconstruction
5. Algebraic Connectivity

These methods test deeper structural properties that may not
be captured by standard centrality or community metrics.

Read more: start with https://networksciencebook.com/ (it is free)

"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh, eigs
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
import warnings
warnings.filterwarnings('ignore')

# Try importing optional advanced libraries
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: gudhi not available (persistent homology disabled)")

try:
    from karateclub import Node2Vec, DeepWalk, HOPE, LaplacianEigenmaps
    KARATECLUB_AVAILABLE = True
except ImportError:
    KARATECLUB_AVAILABLE = False
    print("Warning: karateclub not available (advanced embeddings disabled)")


class AdvancedNetworkAnalyzer:
    """
    Advanced network analysis for lottery graphs.
    """
    
    def __init__(self, lottery_name: str, graphs_dir: Path, output_dir: Path):
        """
        Initialize advanced analyzer.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        graphs_dir : Path
            Directory containing graph objects
        output_dir : Path
            Directory for saving analysis results
        """
        self.lottery_name = lottery_name
        self.graphs_dir = Path(graphs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.graphs = {}
        self.analysis_results = {}
        
        print(f"Initialized advanced network analyzer for {lottery_name}")
        print(f"  GUDHI available: {GUDHI_AVAILABLE}")
        print(f"  KarateClub available: {KARATECLUB_AVAILABLE}")
    
    def load_graphs(self):
        """Load lottery graphs."""
        print("\nLoading graphs...")
        
        # Load main graphs
        main_graphs = [
            f"{self.lottery_name}_cooccurrence_graph.pkl",
            f"{self.lottery_name}_transition_graph.pkl",
            f"{self.lottery_name}_knn_draws_k10_euclidean.pkl"
        ]
        
        for filename in main_graphs:
            filepath = self.graphs_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    graph_name = filename.replace(f"{self.lottery_name}_", "").replace(".pkl", "")
                    self.graphs[graph_name] = pickle.load(f)
                print(f"  Loaded: {graph_name}")
    
    def run_all_analyses(self):
        """Run all advanced analyses."""
        print("\n" + "="*60)
        print("RUNNING ADVANCED NETWORK ANALYSES")
        print("="*60)
        
        for graph_name, G in self.graphs.items():
            print(f"\n[{graph_name.upper()}]")
            
            self.analysis_results[graph_name] = {}
            
            # Spectral analysis
            print("  Running spectral analysis...")
            spectral_results = self.run_spectral_analysis(G, graph_name)
            self.analysis_results[graph_name]['spectral'] = spectral_results
            
            # Persistent homology (if undirected and not too large)
            if not G.is_directed() and G.number_of_nodes() <= 500 and GUDHI_AVAILABLE:
                print("  Running persistent homology...")
                try:
                    tda_results = self.run_persistent_homology(G, graph_name)
                    self.analysis_results[graph_name]['tda'] = tda_results
                except Exception as e:
                    print(f"    Persistent homology failed: {e}")
            
            # Graph embeddings
            if KARATECLUB_AVAILABLE and G.number_of_nodes() >= 10:
                print("  Computing graph embeddings...")
                try:
                    embedding_results = self.compute_graph_embeddings(G, graph_name)
                    self.analysis_results[graph_name]['embeddings'] = embedding_results
                except Exception as e:
                    print(f"    Graph embeddings failed: {e}")
            
            # Algebraic connectivity and robustness
            if not G.is_directed():
                print("  Computing algebraic connectivity...")
                try:
                    algebraic_results = self.compute_algebraic_connectivity(G, graph_name)
                    self.analysis_results[graph_name]['algebraic'] = algebraic_results
                except Exception as e:
                    print(f"    Algebraic connectivity failed: {e}")
    
    def run_spectral_analysis(self, G: nx.Graph, graph_name: str) -> Dict:
        """
        Comprehensive spectral analysis of graph.
        
        Computes:
        - Adjacency matrix eigenvalues
        - Laplacian matrix eigenvalues
        - Normalized Laplacian eigenvalues
        - Spectral gap
        - Fiedler value (algebraic connectivity)
        - Eigenvalue distribution analysis
        
        Returns
        -------
        results : dict
            All spectral analysis results
        """
        results = {}
        
        # Get largest connected component
        if G.is_directed():
            Gcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        else:
            Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if Gcc.number_of_nodes() < 3:
            return results
        
        n = Gcc.number_of_nodes()
        results['n_nodes'] = n
        results['n_edges'] = Gcc.number_of_edges()
        
        # Adjacency matrix eigenvalues
        try:
            if n <= 1000:
                A = nx.to_numpy_array(Gcc)
                adj_eigenvalues = linalg.eigvals(A)
                
                # Sort by magnitude
                adj_eigenvalues = np.sort(np.abs(adj_eigenvalues))[::-1]
                
                results['adj_max_eigenvalue'] = adj_eigenvalues[0].real
                results['adj_spectral_radius'] = np.max(np.abs(adj_eigenvalues)).real
                results['adj_eigenvalue_mean'] = np.mean(adj_eigenvalues).real
                results['adj_eigenvalue_std'] = np.std(adj_eigenvalues).real
                
                # Spectral gap (difference between two largest eigenvalues)
                if len(adj_eigenvalues) >= 2:
                    results['adj_spectral_gap'] = (adj_eigenvalues[0] - adj_eigenvalues[1]).real
                
                # Eigenvalue concentration
                results['adj_eigenvalue_entropy'] = self._compute_eigenvalue_entropy(adj_eigenvalues)
            else:
                print("    Adjacency matrix too large, using sparse methods")
        except Exception as e:
            print(f"    Adjacency eigenvalues failed: {e}")
        
        # Laplacian eigenvalues (for undirected graphs)
        if not G.is_directed():
            try:
                if n <= 1000:
                    L = nx.laplacian_matrix(Gcc).toarray()
                    lap_eigenvalues = linalg.eigvalsh(L)  # Real symmetric
                    lap_eigenvalues = np.sort(lap_eigenvalues)
                    
                    # Fiedler value (2nd smallest eigenvalue, algebraic connectivity)
                    results['fiedler_value'] = lap_eigenvalues[1]
                    results['lap_max_eigenvalue'] = lap_eigenvalues[-1]
                    results['lap_eigenvalue_mean'] = np.mean(lap_eigenvalues)
                    results['lap_eigenvalue_std'] = np.std(lap_eigenvalues)
                    
                    # Spectral gap (Fiedler value - 0)
                    results['lap_spectral_gap'] = lap_eigenvalues[1] - lap_eigenvalues[0]
                    
                    # Number of zero eigenvalues (should equal number of connected components)
                    results['lap_n_zero_eigenvalues'] = np.sum(lap_eigenvalues < 1e-8)
                else:
                    # Use sparse methods for large graphs
                    L = nx.laplacian_matrix(Gcc)
                    
                    # Compute smallest few eigenvalues
                    eigenvalues, _ = eigsh(L, k=min(10, n-2), which='SM')
                    eigenvalues = np.sort(eigenvalues)
                    
                    results['fiedler_value'] = eigenvalues[1] if len(eigenvalues) > 1 else np.nan
                    results['lap_spectral_gap'] = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else np.nan
            except Exception as e:
                print(f"    Laplacian eigenvalues failed: {e}")
            
            # Normalized Laplacian
            try:
                if n <= 1000:
                    L_norm = nx.normalized_laplacian_matrix(Gcc).toarray()
                    norm_lap_eigenvalues = linalg.eigvalsh(L_norm)
                    norm_lap_eigenvalues = np.sort(norm_lap_eigenvalues)
                    
                    results['norm_lap_min_eigenvalue'] = norm_lap_eigenvalues[0]
                    results['norm_lap_max_eigenvalue'] = norm_lap_eigenvalues[-1]
                    results['norm_lap_eigenvalue_mean'] = np.mean(norm_lap_eigenvalues)
                    results['norm_lap_eigenvalue_std'] = np.std(norm_lap_eigenvalues)
                    
                    # Spectral gap
                    if len(norm_lap_eigenvalues) >= 2:
                        results['norm_lap_spectral_gap'] = norm_lap_eigenvalues[1] - norm_lap_eigenvalues[0]
            except Exception as e:
                print(f"    Normalized Laplacian failed: {e}")
        
        # Fiedler vector (2nd eigenvector of Laplacian)
        if not G.is_directed() and n <= 500:
            try:
                fiedler_vector = nx.fiedler_vector(Gcc)
                
                # Analyze Fiedler vector distribution
                results['fiedler_vector_mean'] = np.mean(fiedler_vector)
                results['fiedler_vector_std'] = np.std(fiedler_vector)
                results['fiedler_vector_min'] = np.min(fiedler_vector)
                results['fiedler_vector_max'] = np.max(fiedler_vector)
                
                # Use Fiedler vector for partitioning
                partition = fiedler_vector >= np.median(fiedler_vector)
                results['fiedler_partition_balance'] = np.mean(partition)  # Should be ~0.5 if balanced
            except Exception as e:
                print(f"    Fiedler vector failed: {e}")
        
        return results
    
    def _compute_eigenvalue_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Compute entropy of eigenvalue distribution.
        
        Measures concentration vs spread of eigenvalues.
        High entropy = many similar eigenvalues (regular structure)
        Low entropy = few dominant eigenvalues (hub structure)
        """
        # Normalize eigenvalues to probabilities
        abs_eigenvalues = np.abs(eigenvalues)
        total = np.sum(abs_eigenvalues)
        
        if total < 1e-10:
            return 0.0
        
        probs = abs_eigenvalues / total
        probs = probs[probs > 1e-10]  # Remove near-zero
        
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def run_persistent_homology(self, G: nx.Graph, graph_name: str) -> Dict:
        """
        Run persistent homology / topological data analysis.
        
        Computes Betti numbers and persistence diagrams using Vietoris-Rips
        complex on graph geodesic distances.
        
        Returns
        -------
        results : dict
            TDA results including Betti numbers and persistence
        """
        if not GUDHI_AVAILABLE:
            return {}
        
        results = {}
        
        # Get largest connected component
        Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if Gcc.number_of_nodes() < 5:
            return results
        
        # Compute all-pairs shortest paths (distance matrix)
        try:
            distances = dict(nx.all_pairs_shortest_path_length(Gcc))
            
            # Convert to distance matrix
            nodes = list(Gcc.nodes())
            n = len(nodes)
            dist_matrix = np.zeros((n, n))
            
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if node_j in distances[node_i]:
                        dist_matrix[i, j] = distances[node_i][node_j]
                    else:
                        dist_matrix[i, j] = np.inf
            
            # Build Rips complex
            max_edge_length = min(5, np.percentile(dist_matrix[dist_matrix < np.inf], 90))
            
            rips_complex = gudhi.RipsComplex(distance_matrix=dist_matrix, 
                                             max_edge_length=max_edge_length)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            
            # Compute persistence
            simplex_tree.persistence()
            
            # Get Betti numbers
            betti_0 = simplex_tree.betti_numbers()[0] if len(simplex_tree.betti_numbers()) > 0 else 0
            betti_1 = simplex_tree.betti_numbers()[1] if len(simplex_tree.betti_numbers()) > 1 else 0
            betti_2 = simplex_tree.betti_numbers()[2] if len(simplex_tree.betti_numbers()) > 2 else 0
            
            results['betti_0'] = betti_0  # Connected components
            results['betti_1'] = betti_1  # Loops/cycles
            results['betti_2'] = betti_2  # Voids/cavities
            
            # Get persistence diagrams
            persistence_pairs = simplex_tree.persistence_pairs()
            
            # Count finite persistence features
            n_finite_0 = 0
            n_finite_1 = 0
            
            for dim, pairs in enumerate(persistence_pairs):
                if dim == 0:
                    n_finite_0 = len([p for p in pairs if p[1] < np.inf])
                elif dim == 1:
                    n_finite_1 = len([p for p in pairs if p[1] < np.inf])
            
            results['persistence_finite_0'] = n_finite_0
            results['persistence_finite_1'] = n_finite_1
            
            # Compute total persistence
            persistence_intervals = simplex_tree.persistence_intervals_in_dimension(1)
            if len(persistence_intervals) > 0:
                lifespans = persistence_intervals[:, 1] - persistence_intervals[:, 0]
                lifespans = lifespans[np.isfinite(lifespans)]
                
                if len(lifespans) > 0:
                    results['persistence_mean_lifespan'] = np.mean(lifespans)
                    results['persistence_max_lifespan'] = np.max(lifespans)
                    results['persistence_total'] = np.sum(lifespans)
            
            print(f"    Betti numbers: β₀={betti_0}, β₁={betti_1}, β₂={betti_2}")
            
        except Exception as e:
            print(f"    Persistent homology failed: {e}")
        
        return results
    
    def compute_graph_embeddings(self, G: nx.Graph, graph_name: str) -> Dict:
        """
        Compute graph node embeddings using multiple methods.
        
        Methods:
        - Node2Vec (random walk + skip-gram)
        - DeepWalk (uniform random walk)
        - HOPE (higher-order proximity)
        - Laplacian Eigenmaps (spectral)
        
        Returns
        -------
        results : dict
            Embedding quality metrics and dimensionality measures
        """
        if not KARATECLUB_AVAILABLE:
            return {}
        
        results = {}
        
        # Get largest connected component
        if G.is_directed():
            Gcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        else:
            Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        # Relabel nodes to integers (required by karateclub)
        Gcc = nx.convert_node_labels_to_integers(Gcc, first_label=0)
        
        n = Gcc.number_of_nodes()
        
        if n < 10:
            return results
        
        # Embedding dimension
        dim = min(64, n // 4)
        
        # Node2Vec
        try:
            model = Node2Vec(dimensions=dim, walk_length=10, num_walks=20,
                            p=1, q=1, workers=1)
            model.fit(Gcc)
            embedding = model.get_embedding()
            
            # Analyze embedding
            results['node2vec_embedding_dim'] = embedding.shape[1]
            results['node2vec_embedding_mean'] = np.mean(embedding)
            results['node2vec_embedding_std'] = np.std(embedding)
            
            # Compute pairwise distances in embedding space
            from scipy.spatial.distance import pdist
            distances = pdist(embedding, metric='euclidean')
            results['node2vec_avg_distance'] = np.mean(distances)
            results['node2vec_std_distance'] = np.std(distances)
            
            # Dimensionality of embedding (PCA explained variance)
            pca = PCA(n_components=min(10, dim))
            pca.fit(embedding)
            results['node2vec_pca_explained_var_top3'] = np.sum(pca.explained_variance_ratio_[:3])
            
            print(f"    Node2Vec: {embedding.shape}, explained var: {results['node2vec_pca_explained_var_top3']:.3f}")
        except Exception as e:
            print(f"    Node2Vec failed: {e}")
        
        # DeepWalk
        try:
            model = DeepWalk(dimensions=dim, walk_length=10, num_walks=20, workers=1)
            model.fit(Gcc)
            embedding = model.get_embedding()
            
            results['deepwalk_embedding_dim'] = embedding.shape[1]
            results['deepwalk_embedding_mean'] = np.mean(embedding)
            results['deepwalk_embedding_std'] = np.std(embedding)
            
            pca = PCA(n_components=min(10, dim))
            pca.fit(embedding)
            results['deepwalk_pca_explained_var_top3'] = np.sum(pca.explained_variance_ratio_[:3])
            
            print(f"    DeepWalk: {embedding.shape}, explained var: {results['deepwalk_pca_explained_var_top3']:.3f}")
        except Exception as e:
            print(f"    DeepWalk failed: {e}")
        
        # Laplacian Eigenmaps
        if not G.is_directed() and n <= 500:
            try:
                model = LaplacianEigenmaps(dimensions=min(32, n-2))
                model.fit(Gcc)
                embedding = model.get_embedding()
                
                results['laplacian_eigenmaps_dim'] = embedding.shape[1]
                results['laplacian_eigenmaps_mean'] = np.mean(embedding)
                results['laplacian_eigenmaps_std'] = np.std(embedding)
                
                print(f"    Laplacian Eigenmaps: {embedding.shape}")
            except Exception as e:
                print(f"    Laplacian Eigenmaps failed: {e}")
        
        return results
    
    def compute_algebraic_connectivity(self, G: nx.Graph, graph_name: str) -> Dict:
        """
        Compute algebraic connectivity and related measures.
        
        Algebraic connectivity (Fiedler value) measures how well-connected
        a graph is. Higher values = more robust to edge removal.
        
        Returns
        -------
        results : dict
            Algebraic connectivity and robustness metrics
        """
        results = {}
        
        if G.is_directed():
            return results
        
        # Get largest connected component
        Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if Gcc.number_of_nodes() < 3:
            return results
        
        # Algebraic connectivity
        try:
            alg_conn = nx.algebraic_connectivity(Gcc, method='lanczos')
            results['algebraic_connectivity'] = alg_conn
            
            print(f"    Algebraic connectivity: {alg_conn:.6f}")
        except Exception as e:
            print(f"    Algebraic connectivity failed: {e}")
        
        # Node/edge connectivity
        if Gcc.number_of_nodes() <= 200:
            try:
                node_connectivity = nx.node_connectivity(Gcc)
                results['node_connectivity'] = node_connectivity
                
                edge_connectivity = nx.edge_connectivity(Gcc)
                results['edge_connectivity'] = edge_connectivity
                
                print(f"    Node connectivity: {node_connectivity}")
                print(f"    Edge connectivity: {edge_connectivity}")
            except Exception as e:
                print(f"    Connectivity computation failed: {e}")
        
        return results
    
    def save_results(self):
        """Save all analysis results."""
        print("\n--- SAVING ADVANCED ANALYSIS RESULTS ---")
        
        all_results = []
        
        for graph_name, analyses in self.analysis_results.items():
            result_row = {'graph_name': graph_name}
            
            # Flatten all analysis results
            for analysis_type, analysis_data in analyses.items():
                for key, value in analysis_data.items():
                    result_row[f"{analysis_type}_{key}"] = value
            
            all_results.append(result_row)
        
        if len(all_results) > 0:
            df = pd.DataFrame(all_results)
            filepath = self.output_dir / f"{self.lottery_name}_advanced_analysis.csv"
            df.to_csv(filepath, index=False)
            print(f"  Saved: {filepath.name}")
        else:
            print("  No results to save")


def main():
    """Main execution function."""
    import sys
    BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
    
    if len(sys.argv) < 2:
        print("Usage: python 05_advanced_analysis.py <lottery_name>")
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
    
    # Run advanced analyses
    analyzer = AdvancedNetworkAnalyzer(lottery_name, graphs_dir, output_dir)
    analyzer.load_graphs()
    analyzer.run_all_analyses()
    analyzer.save_results()
    
    print("\n--- ADVANCED NETWORK ANALYSIS COMPLETE ---")


if __name__ == "__main__":
    main()

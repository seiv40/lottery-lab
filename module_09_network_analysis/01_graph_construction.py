"""
MODULE 9 - GRAPH CONSTRUCTION
Lottery Lab: Network Analysis of Lottery Structure

This script constructs 7 distinct graph representations of lottery data:
1. Ball Co-Occurrence Graph (undirected, weighted)
2. Temporal Transition Graph (directed, weighted)
3. Position Correlation Network (undirected, thresholded)
4. K-Nearest Neighbor Draw Similarity Graph (undirected, weighted)
5. Bipartite Draw-Ball Network (bipartite)
6. Position-Specific Networks (one per position)
7. Temporal Evolution Network (time-windowed snapshots)

More info: read https://networksciencebook.com/ (it is free)

"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class LotteryGraphConstructor:
    """
    Comprehensive graph construction framework for lottery data.
    Implements 7 distinct graph types with full parameterization.
    """
    
    def __init__(self, lottery_name: str, features_df: pd.DataFrame, 
                 draws_df: pd.DataFrame, output_dir: Path):
        """
        Initialize graph constructor.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        features_df : pd.DataFrame
            Feature-engineered data from Module 3
        draws_df : pd.DataFrame
            Raw draw data with ball values
        output_dir : Path
            Directory for saving graph objects
        """
        self.lottery_name = lottery_name
        self.features = features_df.copy()
        self.draws = draws_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure consistent datetime type for merge key
        for df in (self.features, self.draws):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Determine ball ranges
        if 'powerball' in lottery_name.lower():
            self.white_ball_range = (1, 69)
            self.special_ball_range = (1, 26)
            self.special_col = 'powerball'
            self.white_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5']
        else:  # Mega Millions
            self.white_ball_range = (1, 70)
            self.special_ball_range = (1, 25)
            self.special_col = 'megaball'
            self.white_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5']
        
        # Merge features with draws for comprehensive analysis
        self.full_data = pd.merge(
            self.features,
            self.draws,
            on='date',
            how='inner'
        ).sort_values('date').reset_index(drop=True)
        
        print(f"Initialized {lottery_name} graph constructor")
        print(f"  Data shape: {self.full_data.shape}")
        print(f"  Date range: {self.full_data['date'].min()} to {self.full_data['date'].max()}")
        print(f"  White ball range: {self.white_ball_range}")
        print(f"  Special ball range: {self.special_ball_range}")
        
        self.graphs = {}
        
    def construct_all_graphs(self):
        """Build all 7 graph types."""
        print("\n--- CONSTRUCTING ALL GRAPH TYPES ---")
        
        self.graphs['cooccurrence'] = self.build_cooccurrence_graph()
        self.graphs['transition'] = self.build_transition_graph()
        self.graphs['correlation'] = self.build_correlation_network()
        self.graphs['knn_draws'] = self.build_knn_draw_graph()
        self.graphs['bipartite'] = self.build_bipartite_graph()
        self.graphs['position_specific'] = self.build_position_specific_graphs()
        self.graphs['temporal_snapshots'] = self.build_temporal_evolution_graph()
        
        return self.graphs
    
    def build_cooccurrence_graph(self) -> nx.Graph:
        """
        Build Ball Co-Occurrence Graph.
        
        Nodes: Individual ball numbers (1-69/70)
        Edges: Connect balls that appear together in same draw
        Edge weights: Co-occurrence frequency
        
        Null hypothesis: Uniform edge weight distribution (all pairs equally likely)
        """
        print("\n[1/7] Building Ball Co-Occurrence Graph...")
        
        G = nx.Graph()
        
        # Add all possible ball nodes
        for ball in range(self.white_ball_range[0], self.white_ball_range[1] + 1):
            G.add_node(ball, node_type='white_ball')
        
        # Count co-occurrences
        cooccurrence_matrix = np.zeros((self.white_ball_range[1] + 1, 
                                       self.white_ball_range[1] + 1))
        
        for idx, row in self.draws.iterrows():
            balls = [int(row[col]) for col in self.white_cols]
            
            # All pairwise combinations within draw
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    ball_i, ball_j = balls[i], balls[j]
                    cooccurrence_matrix[ball_i, ball_j] += 1
                    cooccurrence_matrix[ball_j, ball_i] += 1
        
        # Add edges with weights
        edge_count = 0
        for i in range(self.white_ball_range[0], self.white_ball_range[1] + 1):
            for j in range(i + 1, self.white_ball_range[1] + 1):
                weight = cooccurrence_matrix[i, j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
                    edge_count += 1
        
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.4f}")
        print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
        
        # Calculate edge weight statistics
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        print(f"  Edge weight stats: mean={np.mean(weights):.2f}, "
              f"std={np.std(weights):.2f}, "
              f"min={np.min(weights):.0f}, max={np.max(weights):.0f}")
        
        # Save graph
        self._save_graph(G, 'cooccurrence_graph')
        return G
    
    def build_transition_graph(self) -> nx.DiGraph:
        """
        Build Temporal Transition Graph.
        
        Nodes: Ball numbers
        Directed edges: Ball at position i in draw t → ball at position i in draw t+1
        Edge weights: Transition frequency
        
        Null hypothesis: Uniform transition matrix (all transitions equally probable)
        """
        print("\n[2/7] Building Temporal Transition Graph...")
        
        G = nx.DiGraph()
        
        # Add nodes
        for ball in range(self.white_ball_range[0], self.white_ball_range[1] + 1):
            G.add_node(ball, node_type='white_ball')
        
        # Track position-wise transitions
        for pos_idx, col in enumerate(self.white_cols):
            for t in range(len(self.draws) - 1):
                curr_ball = int(self.draws.iloc[t][col])
                next_ball = int(self.draws.iloc[t + 1][col])
                
                if G.has_edge(curr_ball, next_ball):
                    G[curr_ball][next_ball]['weight'] += 1
                    G[curr_ball][next_ball][f'position_{pos_idx+1}_count'] = \
                        G[curr_ball][next_ball].get(f'position_{pos_idx+1}_count', 0) + 1
                else:
                    G.add_edge(curr_ball, next_ball, weight=1)
                    G[curr_ball][next_ball][f'position_{pos_idx+1}_count'] = 1
        
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Average in-degree: {np.mean([d for n, d in G.in_degree()]):.2f}")
        print(f"  Average out-degree: {np.mean([d for n, d in G.out_degree()]):.2f}")
        
        # Calculate transition probabilities
        for node in G.nodes():
            out_edges = G.out_edges(node, data=True)
            total_weight = sum([data['weight'] for _, _, data in out_edges])
            if total_weight > 0:
                for _, target, data in out_edges:
                    data['probability'] = data['weight'] / total_weight
        
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        print(f"  Edge weight stats: mean={np.mean(weights):.2f}, "
              f"std={np.std(weights):.2f}, "
              f"min={np.min(weights):.0f}, max={np.max(weights):.0f}")
        
        self._save_graph(G, 'transition_graph')
        return G
    
    def build_correlation_network(self, threshold: float = 0.3, 
                                  method: str = 'pearson') -> nx.Graph:
        """
        Build Feature Correlation Network.
        
        Nodes: Features from Module 3
        Edges: Significant correlations (|rho| > threshold)
        Edge weights: Correlation coefficient
        
        Null hypothesis: Sparse graph after multiple testing correction
        """
        print("\n[3/7] Building Feature Correlation Network...")
        
        # Select numeric features (exclude dates, game, drawing_index)
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['drawing_index', 'year', 'month', 'day_of_week', 
                       'day_of_month', 'day_of_year', 'quarter', 'week_of_year']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"  Analyzing {len(feature_cols)} features...")
        
        # Compute correlation matrix
        feature_data = self.features[feature_cols].dropna()
        
        if method == 'pearson':
            corr_matrix = feature_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = feature_data.corr(method='spearman')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Build graph with thresholded correlations
        G = nx.Graph()
        
        # Add nodes
        for col in feature_cols:
            G.add_node(col, node_type='feature')
        
        # Add edges above threshold
        edge_count = 0
        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols[i+1:], start=i+1):
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) >= threshold:
                    G.add_edge(col1, col2, weight=abs(corr), 
                              correlation=corr, abs_corr=abs(corr))
                    edge_count += 1
        
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges (|rho| >= {threshold}): {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.4f}")
        
        if G.number_of_edges() > 0:
            weights = [G[u][v]['abs_corr'] for u, v in G.edges()]
            print(f"  Correlation stats: mean={np.mean(weights):.3f}, "
                  f"std={np.std(weights):.3f}, "
                  f"min={np.min(weights):.3f}, max={np.max(weights):.3f}")
        
        self._save_graph(G, 'correlation_network')
        return G
    
    def build_knn_draw_graph(self, k: int = 10, metric: str = 'euclidean') -> nx.Graph:
        """
        Build K-Nearest Neighbor Draw Similarity Graph.
        
        Nodes: Individual draws (each draw as a 6D point)
        Edges: Connect each draw to k nearest neighbors
        Distance metric: Euclidean, cosine, or correlation
        
        Null hypothesis: Uniform density, no clustering
        """
        print(f"\n[4/7] Building KNN Draw Similarity Graph (k={k}, metric={metric})...")
        
        # Use raw ball values as features (5 white + 1 special)
        draw_features = self.full_data[self.white_cols + [self.special_col]].values
        
        print(f"  Feature matrix shape: {draw_features.shape}")
        
        # Fit KNN
        if metric == 'cosine':
            # For cosine, use precomputed distance
            dist_matrix = cosine_distances(draw_features)
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
            nbrs.fit(dist_matrix)
            distances, indices = nbrs.kneighbors(dist_matrix)
        else:
            nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric)
            nbrs.fit(draw_features)
            distances, indices = nbrs.kneighbors(draw_features)
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes (draws)
        for i in range(len(draw_features)):
            G.add_node(i, 
                      draw_index=i,
                      date=str(self.full_data.iloc[i]['date']))
        
        # Add edges to k nearest neighbors
        for i in range(len(draw_features)):
            for j_idx, dist in zip(indices[i][1:], distances[i][1:]):  # Skip self
                j = int(j_idx)
                if i < j:  # Avoid duplicate edges
                    # Weight is inverse distance (higher weight = more similar)
                    weight = 1.0 / (dist + 1e-10)
                    G.add_edge(i, j, weight=weight, distance=dist)
        
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
        
        distances_all = [G[u][v]['distance'] for u, v in G.edges()]
        print(f"  Distance stats: mean={np.mean(distances_all):.2f}, "
              f"std={np.std(distances_all):.2f}, "
              f"min={np.min(distances_all):.2f}, max={np.max(distances_all):.2f}")
        
        self._save_graph(G, f'knn_draws_k{k}_{metric}')
        return G
    
    def build_bipartite_graph(self) -> nx.Graph:
        """
        Build Bipartite Draw-Ball Network.
        
        Two node types: Draws (left) and Balls (right)
        Edges: Draw d contains ball b
        
        Null hypothesis: Uniform degree distribution for balls
        """
        print("\n[5/7] Building Bipartite Draw-Ball Network...")
        
        G = nx.Graph()
        
        # Add draw nodes
        for i in range(len(self.draws)):
            G.add_node(f"draw_{i}", bipartite=0, draw_index=i,
                      date=str(self.draws.iloc[i]['date']))
        
        # Add ball nodes
        for ball in range(self.white_ball_range[0], self.white_ball_range[1] + 1):
            G.add_node(f"ball_{ball}", bipartite=1, ball_number=ball)
        
        # Add edges (draw contains ball)
        for i, row in self.draws.iterrows():
            draw_node = f"draw_{i}"
            for col in self.white_cols:
                ball = int(row[col])
                ball_node = f"ball_{ball}"
                G.add_edge(draw_node, ball_node, edge_type='contains')
        
        print(f"  Total nodes: {G.number_of_nodes()}")
        print(f"    Draw nodes: {len([n for n, d in G.nodes(data=True) if d['bipartite'] == 0])}")
        print(f"    Ball nodes: {len([n for n, d in G.nodes(data=True) if d['bipartite'] == 1])}")
        print(f"  Edges: {G.number_of_edges()}")
        
        # Analyze ball degree distribution
        ball_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
        ball_degrees = [G.degree(n) for n in ball_nodes]
        print(f"  Ball degree stats: mean={np.mean(ball_degrees):.2f}, "
              f"std={np.std(ball_degrees):.2f}, "
              f"min={np.min(ball_degrees):.0f}, max={np.max(ball_degrees):.0f}")
        
        self._save_graph(G, 'bipartite_draw_ball')
        return G
    
    def build_position_specific_graphs(self) -> Dict[int, nx.Graph]:
        """
        Build Position-Specific Networks (one per position).
        
        Test whether different positions exhibit different network properties.
        Expected: All positions indistinguishable (random).
        """
        print("\n[6/7] Building Position-Specific Networks...")
        
        position_graphs = {}
        
        for pos_idx, col in enumerate(self.white_cols, 1):
            print(f"  Position {pos_idx}...")
            
            G = nx.Graph()
            
            # Add nodes for this position's ball range
            for ball in range(self.white_ball_range[0], self.white_ball_range[1] + 1):
                G.add_node(ball, position=pos_idx)
            
            # Build temporal transition edges for this position only
            for t in range(len(self.draws) - 1):
                curr_ball = int(self.draws.iloc[t][col])
                next_ball = int(self.draws.iloc[t + 1][col])
                
                if G.has_edge(curr_ball, next_ball):
                    G[curr_ball][next_ball]['weight'] += 1
                else:
                    G.add_edge(curr_ball, next_ball, weight=1)
            
            position_graphs[pos_idx] = G
            print(f"    Edges: {G.number_of_edges()}, "
                  f"Avg degree: {np.mean([d for n, d in G.degree()]):.2f}")
        
        # Save all position graphs
        for pos, G in position_graphs.items():
            self._save_graph(G, f'position_{pos}_graph')
        
        return position_graphs
    
    def build_temporal_evolution_graph(self, window_size: int = 100, 
                                      step_size: int = 50) -> List[nx.Graph]:
        """
        Build Temporal Evolution Network (time-windowed snapshots).
        
        Creates rolling window snapshots of co-occurrence graphs.
        Test: Do network metrics change over time?
        Expected: Stationarity; no evolution.
        """
        print(f"\n[7/7] Building Temporal Evolution Network "
              f"(window={window_size}, step={step_size})...")
        
        snapshots = []
        n_draws = len(self.draws)
        
        for start_idx in range(0, n_draws - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_draws = self.draws.iloc[start_idx:end_idx]
            
            # Build co-occurrence graph for this window
            G = nx.Graph()
            G.graph['window_start'] = start_idx
            G.graph['window_end'] = end_idx
            G.graph['date_start'] = str(window_draws.iloc[0]['date'])
            G.graph['date_end'] = str(window_draws.iloc[-1]['date'])
            
            for ball in range(self.white_ball_range[0], self.white_ball_range[1] + 1):
                G.add_node(ball)
            
            for idx, row in window_draws.iterrows():
                balls = [int(row[col]) for col in self.white_cols]
                for i in range(len(balls)):
                    for j in range(i + 1, len(balls)):
                        if G.has_edge(balls[i], balls[j]):
                            G[balls[i]][balls[j]]['weight'] += 1
                        else:
                            G.add_edge(balls[i], balls[j], weight=1)
            
            snapshots.append(G)
        
        print(f"  Created {len(snapshots)} temporal snapshots")
        print(f"  Snapshot indices: {[G.graph['window_start'] for G in snapshots[:5]]}...")
        
        # Save snapshots
        self._save_graph(snapshots, 'temporal_snapshots')
        return snapshots
    
    def _save_graph(self, graph, name: str):
        """Save graph object to pickle file."""
        filepath = self.output_dir / f"{self.lottery_name}_{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        print(f"  Saved: {filepath.name}")
    
    def save_graph_summary(self):
        """Save summary statistics for all graphs."""
        summary = {
            'lottery': self.lottery_name,
            'n_draws': len(self.draws),
            'date_range': f"{self.full_data['date'].min()} to {self.full_data['date'].max()}",
            'graphs_constructed': list(self.graphs.keys())
        }
        
        filepath = self.output_dir / f"{self.lottery_name}_graph_summary.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(summary, f)
        print(f"\nSaved graph summary: {filepath}")


def main():
    """Main execution function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 01_graph_construction.py <lottery_name>")
        print("  lottery_name: 'powerball' or 'megamillions'")
        sys.exit(1)

    lottery_name = sys.argv[1].lower()

    # define paths
    BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
    processed_dir = BASE_DIR / "data" / "processed" / "features"
    raw_dir = BASE_DIR / "data" / "raw"
    output_dir = BASE_DIR / "modules" / "module_09_network_analysis" / "outputs"
    
    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    if lottery_name == 'powerball':
        features = pd.read_parquet(processed_dir / "features_powerball.parquet")
        draws = pd.read_parquet(raw_dir / "powerball_draws.parquet")
    elif lottery_name == 'megamillions':
        features = pd.read_parquet(processed_dir / "features_megamillions.parquet")
        draws = pd.read_parquet(raw_dir / "megamillions_draws.parquet")
    else:
        print(f"Unknown lottery: {lottery_name}")
        sys.exit(1)

    # construct all graphs
    constructor = LotteryGraphConstructor(lottery_name, features, draws, output_dir)
    graphs = constructor.construct_all_graphs()
    constructor.save_graph_summary()

    print("\n--- GRAPH CONSTRUCTION COMPLETE ---")
    print(f"Total graphs constructed: {len(graphs)}")
    print(f"Output directory: {output_dir}")



if __name__ == "__main__":
    main()

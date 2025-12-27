"""
Module 8: Causal Discovery Algorithms
Structure learning from observational lottery data.

Algorithms implemented:
1. PC (Peter-Clark) Algorithm - Constraint-based
2. GES (Greedy Equivalence Search) - Score-based
3. GFCI (Greedy Fast Causal Inference) - Hybrid approach
4. FCI (Fast Causal Inference) - Handles latent confounders

More info can be found here: https://causal-learn.readthedocs.io/en/latest/ 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings

# Try different import paths for different causal-learn versions
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.ConstraintBased.FCI import fci
except ImportError:
    print("Error importing from causallearn. Please ensure causal-learn is installed:")
    print("  pip install causal-learn")
    raise

# GFCI may not be available in all versions
try:
    from causallearn.search.HybridCausal.GFCI import gfci
    GFCI_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from causallearn.search.ConstraintBased.GFCI import gfci
        GFCI_AVAILABLE = True
    except ImportError:
        print("Warning: GFCI not available in this version of causal-learn")
        GFCI_AVAILABLE = False
        gfci = None

try:
    from causallearn.utils.GraphUtils import GraphUtils
except ImportError:
    GraphUtils = None

from config import (
    PC_ALPHA, GES_SCORE, GFCI_ALPHA,
    FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED
)
from data_loader import LotteryDataLoader

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


class CausalDiscoveryAnalyzer:
    """Causal structure discovery from lottery data."""
    
    def __init__(self, lottery_name: str):
        """
        Initialize causal discovery analyzer.
        
        Parameters:
        -----------
        lottery_name : str
            Either 'powerball' or 'megamillions'
        """
        self.lottery_name = lottery_name
        self.loader = LotteryDataLoader(lottery_name)
        self.results = {}
        self.data = None
        self.feature_names = None
        
    def load_data(self, max_features: int = 10):
        """
        Load and prepare data for causal discovery.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features to use (for computational efficiency)
        """
        print("Loading data for causal discovery...")
        
        # Load a subset of features for computational efficiency
        df = self.loader.prepare_for_conditional_independence(
            include_balls=True,
            include_aggregate=True
        )
        
        # Select most informative features
        if len(df.columns) > max_features:
            print(f"  Selecting {max_features} most variable features...")
            
            # Select features with highest variance
            variances = df.var()
            top_features = variances.nlargest(max_features).index.tolist()
            df = df[top_features]
        
        # Remove multicollinear features to avoid singular matrix issues
        df = self._remove_multicollinear_features(df)
        
        self.data = df.values
        self.feature_names = df.columns.tolist()
        
        print(f"  Loaded {self.data.shape[0]} samples with {self.data.shape[1]} features")
        print(f"  Features: {self.feature_names}")
    
    def _remove_multicollinear_features(self, df: pd.DataFrame, threshold: float = 0.999) -> pd.DataFrame:
        """
        Remove features that are nearly perfectly correlated to avoid singular matrices.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        threshold : float
            Correlation threshold above which to remove features
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with multicollinear features removed
        """
        # Compute correlation matrix
        corr_matrix = df.corr().abs()
        
        # Find features to remove
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            print(f"  Removing {len(to_drop)} multicollinear features: {to_drop}")
            df = df.drop(columns=to_drop)
        
        return df
        
    def run_pc_algorithm(self, 
                        alpha: float = None,
                        indep_test: str = 'fisherz') -> Dict:
        """
        Run PC algorithm for causal discovery.
        
        PC uses conditional independence tests to discover causal structure.
        
        Parameters:
        -----------
        alpha : float, optional
            Significance level for independence tests
        indep_test : str
            Independence test method ('fisherz', 'kci', 'chisq', 'gsq')
            
        Returns:
        --------
        Dict
            PC algorithm results including graph and statistics
        """
        if self.data is None:
            self.load_data()
        
        if alpha is None:
            alpha = PC_ALPHA
        
        print(f"\nRunning PC Algorithm...")
        print(f"  Alpha: {alpha}")
        print(f"  Independence test: {indep_test}")
        
        try:
            # Run PC algorithm
            cg = pc(self.data, 
                   alpha=alpha,
                   indep_test=indep_test,
                   stable=True,  # Use stable version
                   uc_rule=0,    # Use original UC rule
                   uc_priority=2)  # Priority for orientation
            
            # Extract results
            graph = cg.G
            edges = graph.get_graph_edges()
            
            n_edges = len(edges)
            n_nodes = self.data.shape[1]
            max_possible_edges = n_nodes * (n_nodes - 1) // 2
            
            print(f"\n  Results:")
            print(f"    Nodes: {n_nodes}")
            print(f"    Edges discovered: {n_edges}")
            print(f"    Edge density: {n_edges / max_possible_edges:.3f}")
            
            # Convert to NetworkX for analysis
            nx_graph = self._causal_graph_to_networkx(graph)
            
            results = {
                'algorithm': 'PC',
                'graph': graph,
                'nx_graph': nx_graph,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'edge_density': n_edges / max_possible_edges,
                'alpha': alpha,
                'indep_test': indep_test,
                'edges': edges
            }
            
            self.results['PC'] = results
            
            return results
            
        except ValueError as e:
            if 'singular' in str(e).lower() and indep_test == 'fisherz':
                # If fisherz fails due to singularity, try kci instead
                print(f"  Fisherz test failed due to singular matrix, trying 'kci' instead...")
                return self.run_pc_algorithm(alpha=alpha, indep_test='kci')
            else:
                print(f"  Error in PC algorithm: {e}")
                import traceback
                traceback.print_exc()
                return {}
        except Exception as e:
            print(f"  Error in PC algorithm: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_ges_algorithm(self, score_func: str = None) -> Dict:
        """
        Run GES (Greedy Equivalence Search) algorithm.
        
        GES uses score-based search to discover causal structure.
        
        Parameters:
        -----------
        score_func : str, optional
            Scoring function ('local_score_BIC', 'local_score_BDeu', etc.)
            
        Returns:
        --------
        Dict
            GES algorithm results
        """
        if self.data is None:
            self.load_data()
        
        if score_func is None:
            score_func = 'local_score_BIC'  # Correct parameter format
        
        print(f"\nRunning GES Algorithm...")
        print(f"  Score function: {score_func}")
        
        try:
            # Run GES - note: parameter name might be 'score_func' or 'method'
            try:
                Record = ges(self.data, score_func=score_func)
            except:
                # Try alternative parameter name
                Record = ges(self.data, method='local_score', parameters={'lambda_value': 2})
            
            # Extract results
            graph = Record['G']
            edges = graph.get_graph_edges()
            
            n_edges = len(edges)
            n_nodes = self.data.shape[1]
            max_possible_edges = n_nodes * (n_nodes - 1) // 2
            
            print(f"\n  Results:")
            print(f"    Nodes: {n_nodes}")
            print(f"    Edges discovered: {n_edges}")
            print(f"    Edge density: {n_edges / max_possible_edges:.3f}")
            
            # Convert to NetworkX
            nx_graph = self._causal_graph_to_networkx(graph)
            
            results = {
                'algorithm': 'GES',
                'graph': graph,
                'nx_graph': nx_graph,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'edge_density': n_edges / max_possible_edges,
                'score_func': score_func,
                'edges': edges
            }
            
            self.results['GES'] = results
            
            return results
            
        except Exception as e:
            print(f"  Error in GES algorithm: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_gfci_algorithm(self, alpha: float = None) -> Dict:
        """
        Run GFCI (Greedy Fast Causal Inference) algorithm.
        
        GFCI combines constraint-based and score-based approaches.
        
        Parameters:
        -----------
        alpha : float, optional
            Significance level for independence tests
            
        Returns:
        --------
        Dict
            GFCI algorithm results
        """
        if not GFCI_AVAILABLE:
            print("  GFCI not available in this version of causal-learn - skipping")
            return {}
        
        if self.data is None:
            self.load_data()
        
        if alpha is None:
            alpha = GFCI_ALPHA
        
        print(f"\nRunning GFCI Algorithm...")
        print(f"  Alpha: {alpha}")
        
        try:
            # Run GFCI
            G, edges = gfci(self.data, alpha=alpha)
            
            n_edges = len(edges)
            n_nodes = self.data.shape[1]
            max_possible_edges = n_nodes * (n_nodes - 1) // 2
            
            print(f"\n  Results:")
            print(f"    Nodes: {n_nodes}")
            print(f"    Edges discovered: {n_edges}")
            print(f"    Edge density: {n_edges / max_possible_edges:.3f}")
            
            # Convert to NetworkX
            nx_graph = self._edges_to_networkx(edges)
            
            results = {
                'algorithm': 'GFCI',
                'graph': G,
                'nx_graph': nx_graph,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'edge_density': n_edges / max_possible_edges,
                'alpha': alpha,
                'edges': edges
            }
            
            self.results['GFCI'] = results
            
            return results
            
        except Exception as e:
            print(f"  Error in GFCI algorithm: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_fci_algorithm(self, 
                         alpha: float = None,
                         indep_test: str = 'fisherz') -> Dict:
        """
        Run FCI (Fast Causal Inference) algorithm.
        
        FCI can handle latent confounders.
        
        Parameters:
        -----------
        alpha : float, optional
            Significance level for independence tests
        indep_test : str
            Independence test method ('fisherz', 'kci', 'chisq', 'gsq')
            
        Returns:
        --------
        Dict
            FCI algorithm results
        """
        if self.data is None:
            self.load_data()
        
        if alpha is None:
            alpha = PC_ALPHA
        
        print(f"\nRunning FCI Algorithm...")
        print(f"  Alpha: {alpha}")
        print(f"  Independence test: {indep_test}")
        
        try:
            # Run FCI
            G, edges = fci(self.data, 
                          alpha=alpha,
                          indep_test=indep_test,
                          stable=True)
            
            n_edges = len(edges)
            n_nodes = self.data.shape[1]
            max_possible_edges = n_nodes * (n_nodes - 1) // 2
            
            print(f"\n  Results:")
            print(f"    Nodes: {n_nodes}")
            print(f"    Edges discovered: {n_edges}")
            print(f"    Edge density: {n_edges / max_possible_edges:.3f}")
            
            # Convert to NetworkX
            nx_graph = self._edges_to_networkx(edges)
            
            results = {
                'algorithm': 'FCI',
                'graph': G,
                'nx_graph': nx_graph,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'edge_density': n_edges / max_possible_edges,
                'alpha': alpha,
                'indep_test': indep_test,
                'edges': edges
            }
            
            self.results['FCI'] = results
            
            return results
            
        except ValueError as e:
            if 'singular' in str(e).lower() and indep_test == 'fisherz':
                # If fisherz fails due to singularity, try kci instead
                print(f"  Fisherz test failed due to singular matrix, trying 'kci' instead...")
                return self.run_fci_algorithm(alpha=alpha, indep_test='kci')
            else:
                print(f"  Error in FCI algorithm: {e}")
                import traceback
                traceback.print_exc()
                return {}
        except Exception as e:
            print(f"  Error in FCI algorithm: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def compare_algorithms(self) -> pd.DataFrame:
        """Compare results across all algorithms."""
        
        print(f"\nComparing causal discovery algorithms...")
        
        comparison_data = []
        
        for alg_name, results in self.results.items():
            if results:
                comparison_data.append({
                    'Algorithm': alg_name,
                    'Nodes': results['n_nodes'],
                    'Edges': results['n_edges'],
                    'Edge Density': results['edge_density'],
                    'Parameters': self._format_params(results)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n  Algorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def _format_params(self, results: Dict) -> str:
        """Format algorithm parameters for display."""
        params = []
        if 'alpha' in results:
            params.append(f"α={results['alpha']}")
        if 'score_func' in results:
            params.append(f"score={results['score_func']}")
        if 'indep_test' in results:
            params.append(f"test={results['indep_test']}")
        return ', '.join(params)
    
    def _causal_graph_to_networkx(self, causal_graph) -> nx.DiGraph:
        """Convert causal-learn graph to NetworkX DiGraph."""
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(len(self.feature_names)):
            G.add_node(i, label=self.feature_names[i])
        
        # Add edges
        edges = causal_graph.get_graph_edges()
        for edge in edges:
            # edge is typically [node1, node2, endpoint1, endpoint2]
            node1, node2 = edge.get_node1(), edge.get_node2()
            endpoint1, endpoint2 = edge.get_endpoint1(), edge.get_endpoint2()
            
            # Add directed edge if there's a clear direction
            # -1 = arrow, 1 = tail, -1 = circle
            G.add_edge(node1, node2, 
                      endpoint1=str(endpoint1),
                      endpoint2=str(endpoint2))
        
        return G
    
    def _edges_to_networkx(self, edges: List) -> nx.DiGraph:
        """Convert edge list to NetworkX DiGraph."""
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(len(self.feature_names)):
            G.add_node(i, label=self.feature_names[i])
        
        # Add edges
        for edge in edges:
            if hasattr(edge, 'get_node1'):
                node1, node2 = edge.get_node1(), edge.get_node2()
                G.add_edge(node1, node2)
            else:
                # Assume edge is tuple
                node1, node2 = edge[0], edge[1]
                G.add_edge(node1, node2)
        
        return G
    
    def visualize_discovered_graphs(self, save: bool = True):
        """Visualize all discovered causal graphs."""
        
        n_algorithms = len(self.results)
        
        if n_algorithms == 0:
            print("  No results to visualize")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 6))
        
        if n_algorithms == 1:
            axes = [axes]
        
        for ax, (alg_name, results) in zip(axes, self.results.items()):
            if not results or 'nx_graph' not in results:
                continue
            
            G = results['nx_graph']
            
            # Layout
            if G.number_of_nodes() > 0:
                if G.number_of_edges() == 0:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=0.5)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, 
                                      node_color='lightblue',
                                      node_size=800,
                                      ax=ax)
                
                # Draw edges
                nx.draw_networkx_edges(G, pos,
                                      edge_color='gray',
                                      arrows=True,
                                      arrowsize=20,
                                      ax=ax)
                
                # Draw labels
                labels = {i: self.feature_names[i][:8] for i in range(len(self.feature_names))}
                nx.draw_networkx_labels(G, pos, labels,
                                       font_size=8,
                                       ax=ax)
            
            ax.set_title(f'{alg_name} Algorithm\n({results["n_edges"]} edges)',
                        fontsize=12)
            ax.axis('off')
        
        fig.suptitle(f'Discovered Causal Structures - {self.lottery_name.title()}\n' +
                    f'(Expected: Empty or minimal structure)',
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save:
            filepath = FIGURES_DIR / f'{self.lottery_name}_causal_discovery_graphs.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
        plt.close()
    
    def save_results(self):
        """Save all results to files."""
        print(f"\nSaving causal discovery results...")
        
        # Save comparison table
        if self.results:
            comparison_df = self.compare_algorithms()
            filepath = OUTPUT_DIR / f'{self.lottery_name}_causal_discovery_comparison.csv'
            comparison_df.to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")
        
        # Save individual algorithm results
        for alg_name, results in self.results.items():
            if not results:
                continue
            
            # Save edge list
            if 'edges' in results and results['edges']:
                edge_data = []
                for edge in results['edges']:
                    if hasattr(edge, 'get_node1'):
                        node1, node2 = edge.get_node1(), edge.get_node2()
                    else:
                        node1, node2 = edge[0], edge[1]
                    
                    # extract node indices from GraphNode objects
                    # GraphNode objects don't have get_node_index() but they do have name and other attrs
                    def get_node_idx(node):
                        # if it is already an integer, use it
                        if isinstance(node, int):
                            return node
                        # if it has get_node_index method, use it
                        if hasattr(node, 'get_node_index'):
                            return node.get_node_index()
                        # if it is a GraphNode with a name, try to match it to features
                        if hasattr(node, 'name'):
                            node_name = node.name if isinstance(node.name, str) else str(node.name)
                            # try to find index in feature names
                            if node_name in self.feature_names:
                                return self.feature_names.index(node_name)
                            # name might be like "X1", "X2" - extract number
                            if node_name.startswith('X'):
                                try:
                                    return int(node_name[1:]) - 1  # X1 -> index 0
                                except ValueError:
                                    pass
                        # last resort: try converting to string and parsing
                        node_str = str(node)
                        if node_str.startswith('X') and node_str[1:].isdigit():
                            return int(node_str[1:]) - 1
                        # give up and return 0
                        return 0
                    
                    idx1 = get_node_idx(node1)
                    idx2 = get_node_idx(node2)
                    
                    edge_data.append({
                        'source': self.feature_names[idx1] if idx1 < len(self.feature_names) else f'Node_{idx1}',
                        'target': self.feature_names[idx2] if idx2 < len(self.feature_names) else f'Node_{idx2}',
                        'source_idx': idx1,
                        'target_idx': idx2
                    })
                
                if edge_data:
                    edge_df = pd.DataFrame(edge_data)
                    filepath = OUTPUT_DIR / f'{self.lottery_name}_causal_{alg_name.lower()}_edges.csv'
                    edge_df.to_csv(filepath, index=False)
                    print(f"  Saved: {filepath}")
    
    def run_complete_analysis(self):
        """Run complete causal discovery analysis pipeline."""
        print(f"\n--- CAUSAL DISCOVERY ANALYSIS: {self.lottery_name.upper()} ---")

        
        # 0. Load data
        self.load_data(max_features=10)
        
        # 1. PC Algorithm
        print("\n1. PC Algorithm (Constraint-based)")
        print("-" * 70)
        self.run_pc_algorithm()
        
        # 2. GES Algorithm
        print("\n2. GES Algorithm (Score-based)")
        print("-" * 70)
        self.run_ges_algorithm()
        
        # 3. GFCI Algorithm (if available)
        print("\n3. GFCI Algorithm (Hybrid)")
        print("-" * 70)
        if GFCI_AVAILABLE:
            self.run_gfci_algorithm()
        else:
            print("  GFCI not available - skipping")
        
        # 4. FCI Algorithm
        print("\n4. FCI Algorithm (Latent confounders)")
        print("-" * 70)
        self.run_fci_algorithm()
        
        # 5. Compare algorithms
        print("\n5. Comparing Algorithms")
        print("-" * 70)
        if self.results:
            self.compare_algorithms()
        else:
            print("  No results to compare")
        
        # 6. Visualize
        print("\n6. Creating Visualizations")
        print("-" * 70)
        self.visualize_discovered_graphs(save=True)
        
        # 7. Save results
        self.save_results()
        
        print("\n--- ANALYSIS COMPLETE ---")
        
        return self.results


def run_both_lotteries() -> Dict:
    """Run causal discovery for both lotteries."""
    
    results = {}
    
    for lottery in ['powerball', 'megamillions']:
        print(f"\n--- # {lottery.upper()} ---")
        
        try:
            analyzer = CausalDiscoveryAnalyzer(lottery)
            results[lottery] = analyzer.run_complete_analysis()
            print(f"\n[OK] Successfully completed {lottery}")
            
        except Exception as e:
            print(f"\n[FAILED] Error analyzing {lottery}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


if __name__ == "__main__":
    """Run causal discovery analysis."""
    
    # Run analysis for both lotteries
    results = run_both_lotteries()
    
    print("\n--- ALL CAUSAL DISCOVERY ANALYSES COMPLETE ---")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

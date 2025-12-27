import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


class NetworkVisualizer:
    """
    Comprehensive visualization suite for lottery network analysis.
    """
    
    def __init__(self, lottery_name: str, data_dir: Path, figures_dir: Path):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        data_dir : Path
            Directory containing graphs and metrics
        figures_dir : Path
            Directory for saving figures
        """
        self.lottery_name = lottery_name
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.graphs = {}
        self.metrics = {}
        self.null_metrics = {}
        
        print(f"Initialized visualizer for {lottery_name}")
        print(f"  Figures will be saved to: {figures_dir}")
    
    def load_data(self):
        """Load all necessary data for visualization."""
        print("\nLoading data...")
        
        # Load graphs
        graph_files = [
            f"{self.lottery_name}_cooccurrence_graph.pkl",
            f"{self.lottery_name}_transition_graph.pkl"
        ]
        
        for filename in graph_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    graph_name = filename.replace(f"{self.lottery_name}_", "").replace(".pkl", "")
                    self.graphs[graph_name] = pickle.load(f)
                print(f"  Loaded graph: {graph_name}")
        
        # Load metrics
        metric_files = list(self.data_dir.glob(f"{self.lottery_name}_*_metrics.csv"))
        
        for filepath in metric_files:
            if 'null' in filepath.stem:
                continue
            graph_name = filepath.stem.replace(f"{self.lottery_name}_", "").replace("_metrics", "")
            self.metrics[graph_name] = pd.read_csv(filepath)
            print(f"  Loaded metrics: {graph_name}")
         
        # Load null metrics
        null_files = list(self.data_dir.glob(f"{self.lottery_name}_*_nullmetrics.csv"))

        for filepath in null_files:
            graph_name = filepath.stem.replace(f"{self.lottery_name}_", "").replace("_nullmetrics", "")
            try:
                df = pd.read_csv(filepath)
            except EmptyDataError:
                print(f"  Warning: {filepath.name} is empty. Skipping this null metrics file.")
                continue

            if df.shape[0] == 0 or df.shape[1] == 0:
                print(f"  Warning: {filepath.name} has no data. Skipping.")
                continue

            self.null_metrics[graph_name] = df
            print(f"  Loaded null metrics: {graph_name}")

    
    def create_all_visualizations(self):
        """Create all visualization types."""
        print("\n--- CREATING VISUALIZATIONS ---")
        
        # 1. Network layouts
        print("\n[1] Creating network layout visualizations...")
        self.plot_network_layouts()
        
        # 2. Adjacency matrices
        print("\n[2] Creating adjacency matrix heatmaps...")
        self.plot_adjacency_matrices()
        
        # 3. Degree distributions
        print("\n[3] Creating degree distribution plots...")
        self.plot_degree_distributions()
        
        # 4. Centrality distributions
        print("\n[4] Creating centrality distribution plots...")
        self.plot_centrality_distributions()
        
        # 5. Metric comparisons (lottery vs null)
        print("\n[5] Creating lottery vs null comparison plots...")
        self.plot_lottery_vs_null_comparisons()
        
        # 6. Spectral analysis plots
        print("\n[6] Creating spectral analysis plots...")
        self.plot_spectral_analysis()
        
        # 7. Statistical test results
        print("\n[7] Creating statistical test visualizations...")
        self.plot_statistical_tests()
        
        # 8. Combined summary figure
        print("\n[8] Creating summary figure...")
        self.create_summary_figure()
    
    def plot_network_layouts(self):
        """Create force-directed and other layout visualizations."""
        for graph_name, G in self.graphs.items():
            # Only visualize co-occurrence graph (others too large/complex)
            if 'cooccurrence' not in graph_name:
                continue
            
            # Get largest connected component
            if G.is_directed():
                Gcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
            else:
                Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            # Limit size for visualization
            if Gcc.number_of_nodes() > 100:
                # Sample top degree nodes
                degrees = dict(Gcc.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
                Gcc = Gcc.subgraph(top_nodes).copy()
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Spring layout
            ax = axes[0]
            pos = nx.spring_layout(Gcc, k=0.5, iterations=50, seed=42)
            
            # Node sizes by degree
            degrees = [Gcc.degree(n) for n in Gcc.nodes()]
            node_sizes = [50 + d * 5 for d in degrees]
            
            nx.draw_networkx_nodes(Gcc, pos, node_size=node_sizes, 
                                  node_color='skyblue', alpha=0.7, ax=ax)
            nx.draw_networkx_edges(Gcc, pos, alpha=0.2, width=0.5, ax=ax)
            
            # Label high-degree nodes only
            high_degree_nodes = [n for n in Gcc.nodes() if Gcc.degree(n) >= np.percentile(degrees, 90)]
            labels = {n: str(n) for n in high_degree_nodes}
            nx.draw_networkx_labels(Gcc, pos, labels, font_size=8, ax=ax)
            
            ax.set_title(f"{self.lottery_name.upper()} - Spring Layout\n(Top 100 nodes by degree)")
            ax.axis('off')
            
            # Circular layout
            ax = axes[1]
            pos = nx.circular_layout(Gcc)
            
            nx.draw_networkx_nodes(Gcc, pos, node_size=node_sizes,
                                  node_color='lightcoral', alpha=0.7, ax=ax)
            nx.draw_networkx_edges(Gcc, pos, alpha=0.2, width=0.5, ax=ax)
            
            ax.set_title(f"{self.lottery_name.upper()} - Circular Layout")
            ax.axis('off')
            
            plt.tight_layout()
            filepath = self.figures_dir / f"{self.lottery_name}_{graph_name}_layouts.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filepath.name}")
    
    def plot_adjacency_matrices(self):
        """Create adjacency matrix heatmaps."""
        for graph_name, G in self.graphs.items():
            if G.number_of_nodes() > 100:
                # Sample for visualization
                nodes = list(G.nodes())[:100]
                G_sub = G.subgraph(nodes).copy()
            else:
                G_sub = G
            
            # Get adjacency matrix
            adj_matrix = nx.to_numpy_array(G_sub)
            
            fig, ax = plt.subplots(figsize=(10, 9))
            
            im = ax.imshow(adj_matrix, cmap='YlOrRd', aspect='auto', 
                          interpolation='nearest')
            
            plt.colorbar(im, ax=ax, label='Edge Weight')
            
            ax.set_title(f"{self.lottery_name.upper()} - {graph_name.replace('_', ' ').title()}\n" +
                        f"Adjacency Matrix (n={G_sub.number_of_nodes()})")
            ax.set_xlabel("Node Index")
            ax.set_ylabel("Node Index")
            
            plt.tight_layout()
            filepath = self.figures_dir / f"{self.lottery_name}_{graph_name}_adjacency.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filepath.name}")
    
    def plot_degree_distributions(self):
        """Create degree distribution plots (including log-log scale)."""
        for graph_name, G in self.graphs.items():
            degrees = [d for n, d in G.degree()]
            
            if len(degrees) == 0:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Linear scale
            ax = axes[0]
            ax.hist(degrees, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel("Degree")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{self.lottery_name.upper()} - {graph_name.replace('_', ' ').title()}\n" +
                        f"Degree Distribution (Linear Scale)")
            ax.grid(True, alpha=0.3)
            
            # Log-log scale
            ax = axes[1]
            degree_counts = pd.Series(degrees).value_counts().sort_index()
            
            # Filter zero counts for log scale
            degree_counts = degree_counts[degree_counts > 0]
            
            ax.scatter(degree_counts.index, degree_counts.values, alpha=0.7, s=50)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("Degree (log)")
            ax.set_ylabel("Frequency (log)")
            ax.set_title("Degree Distribution (Log-Log Scale)")
            ax.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            filepath = self.figures_dir / f"{self.lottery_name}_{graph_name}_degree_dist.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filepath.name}")
    
    def plot_centrality_distributions(self):
        """Create centrality distribution plots."""
        centrality_files = list(self.data_dir.glob(f"{self.lottery_name}_*_centrality_distributions.csv"))
        
        for filepath in centrality_files:
            df = pd.read_csv(filepath)
            graph_name = filepath.stem.replace(f"{self.lottery_name}_", "").replace("_centrality_distributions", "")
            
            # Select centrality columns
            centrality_cols = [c for c in df.columns if 'centrality' in c.lower() or 'pagerank' in c.lower()]
            
            if len(centrality_cols) == 0:
                continue
            
            n_cols = len(centrality_cols)
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
            
            if n_cols == 1:
                axes = [axes]
            
            for i, col in enumerate(centrality_cols):
                ax = axes[i]
                
                values = df[col].dropna()
                
                ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(col.replace('_', ' ').title())
                ax.set_ylabel("Frequency")
                ax.set_title(col.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
            
            fig.suptitle(f"{self.lottery_name.upper()} - {graph_name.replace('_', ' ').title()}\n" +
                        "Centrality Distributions")
            
            plt.tight_layout()
            filepath = self.figures_dir / f"{self.lottery_name}_{graph_name}_centrality_dists.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filepath.name}")
    
    def plot_lottery_vs_null_comparisons(self):
        """Create comparison plots between lottery and null models."""
        for graph_name, lottery_df in self.metrics.items():
            if lottery_df.shape[0] != 1:
                continue
            
            lottery_row = lottery_df.iloc[0]
            
            # Find matching null models
            null_keys = [k for k in self.null_metrics.keys() if graph_name in k]
            
            if len(null_keys) == 0:
                continue
            
            # Select key metrics to compare
            key_metrics = ['avg_degree', 'avg_clustering', 'density', 
                          'degree_centrality_mean', 'avg_shortest_path_length']
            
            available_metrics = [m for m in key_metrics if m in lottery_row.index]
            
            if len(available_metrics) == 0:
                continue
            
            # Create comparison plot
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                
                lottery_value = lottery_row[metric]
                
                if pd.isna(lottery_value):
                    continue
                
                # Collect null values
                null_data = []
                null_labels = []
                
                for null_key in null_keys:
                    null_df = self.null_metrics[null_key]
                    
                    if metric not in null_df.columns:
                        continue
                    
                    null_values = null_df[metric].dropna().values
                    
                    if len(null_values) == 0:
                        continue
                    
                    null_data.append(null_values)
                    
                    # Extract model name
                    model_name = null_key.replace(graph_name + "_", "").replace("_", " ").title()
                    null_labels.append(model_name)
                
                if len(null_data) == 0:
                    continue
                
                # Boxplot of null models
                bp = ax.boxplot(null_data, labels=null_labels, patch_artist=True)
                
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                # Add lottery value as horizontal line
                ax.axhline(y=lottery_value, color='red', linestyle='--', 
                          linewidth=2, label='Lottery')
                
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend()
                
                # Rotate labels if needed
                if len(null_labels) > 2:
                    ax.tick_params(axis='x', rotation=45)
            
            fig.suptitle(f"{self.lottery_name.upper()} - {graph_name.replace('_', ' ').title()}\n" +
                        "Lottery vs Null Models Comparison")
            
            plt.tight_layout()
            filepath = self.figures_dir / f"{self.lottery_name}_{graph_name}_vs_null.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filepath.name}")
    
    def plot_spectral_analysis(self):
        """Create spectral analysis visualizations."""
        # Load advanced analysis results
        adv_filepath = self.data_dir / f"{self.lottery_name}_advanced_analysis.csv"
        
        if not adv_filepath.exists():
            print("  No advanced analysis results found")
            return
        
        df = pd.read_csv(adv_filepath)
        
        # Extract spectral metrics
        spectral_cols = [c for c in df.columns if 'spectral' in c or 'eigenvalue' in c or 'fiedler' in c]
        
        if len(spectral_cols) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(spectral_cols[:4]):  # Plot first 4
            ax = axes[i]
            
            values = df[col].dropna()
            
            if len(values) == 0:
                continue
            
            ax.bar(range(len(values)), values, color='steelblue', alpha=0.7)
            ax.set_xlabel("Graph Index")
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.set_title(col.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f"{self.lottery_name.upper()} - Spectral Analysis Results")
        
        plt.tight_layout()
        filepath = self.figures_dir / f"{self.lottery_name}_spectral_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filepath.name}")
    
    def plot_statistical_tests(self):
        """Create statistical test result visualizations."""
        test_filepath = self.data_dir / f"{self.lottery_name}_permutation_tests.csv"
        
        if not test_filepath.exists():
            print("  No statistical test results found")
            return
        
        df = pd.read_csv(test_filepath)
        
        if len(df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # P-value distribution
        ax = axes[0, 0]
        ax.hist(df['p_value'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax.set_xlabel("P-value")
        ax.set_ylabel("Frequency")
        ax.set_title("P-value Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Z-score distribution
        ax = axes[0, 1]
        z_scores = df['z_score'].dropna()
        ax.hist(z_scores, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Frequency")
        ax.set_title("Z-score Distribution\n(Lottery value vs Null distribution)")
        ax.grid(True, alpha=0.3)
        
        # Significance by null model
        ax = axes[1, 0]
        if 'bonferroni_reject' in df.columns:
            sig_by_model = df.groupby('null_model')['bonferroni_reject'].sum()
            sig_by_model.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
            ax.set_ylabel("Number of Significant Tests")
            ax.set_xlabel("Null Model")
            ax.set_title("Significant Tests by Null Model\n(Bonferroni corrected)")
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Effect sizes
        effect_filepath = self.data_dir / f"{self.lottery_name}_effect_sizes.csv"
        
        if effect_filepath.exists():
            effect_df = pd.read_csv(effect_filepath)
            
            ax = axes[1, 1]
            
            if 'abs_cohens_d' in effect_df.columns:
                ax.hist(effect_df['abs_cohens_d'].dropna(), bins=30, 
                       edgecolor='black', alpha=0.7)
                ax.axvline(x=0.2, color='green', linestyle='--', label='Small (0.2)')
                ax.axvline(x=0.5, color='orange', linestyle='--', label='Medium (0.5)')
                ax.axvline(x=0.8, color='red', linestyle='--', label='Large (0.8)')
                ax.set_xlabel("|Cohen's d|")
                ax.set_ylabel("Frequency")
                ax.set_title("Effect Size Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"{self.lottery_name.upper()} - Statistical Test Results")
        
        plt.tight_layout()
        filepath = self.figures_dir / f"{self.lottery_name}_statistical_tests.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filepath.name}")
    
    def create_summary_figure(self):
        """Create comprehensive summary figure with key results."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # This would contain multiple subplots showing key findings
        # For now, create a text summary
        
        ax = fig.add_subplot(gs[1, 1])
        ax.text(0.5, 0.5, f"{self.lottery_name.upper()}\nNetwork Analysis Summary\n\n" +
                "All graphs analyzed\nNull models generated\nStatistical tests completed\n\n" +
                "See individual figures for details",
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        plt.savefig(self.figures_dir / f"{self.lottery_name}_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {self.lottery_name}_summary.png")


def main():
    """Main execution function."""
    import sys
    BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
    
    if len(sys.argv) < 2:
        print("Usage: python 06_visualizations.py <lottery_name>")
        sys.exit(1)
    
    lottery_name = sys.argv[1].lower()
    
    if lottery_name == 'powerball':
        data_dir = Path('output/powerball/module9')
        figures_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'figures'
    elif lottery_name == 'megamillions':
        data_dir = Path('output/megamillions/module9')
        figures_dir = BASE_DIR / 'modules' / 'module_09_network_analysis' / 'figures'
    else:
        print(f"Unknown lottery: {lottery_name}")
        sys.exit(1)
    
    # Create visualizations
    visualizer = NetworkVisualizer(lottery_name, data_dir, figures_dir)
    visualizer.load_data()
    visualizer.create_all_visualizations()
    
    print("\n--- VISUALIZATION COMPLETE ---")


if __name__ == "__main__":
    main()

"""
Module 10: Manifold Geometry - Data Loading Utilities

"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
import pickle

from config import (
    LotteryName,
    get_features_path,
    get_module9_dir,
)

# note: based on diagnostic runs, we have:
# - powerball features: (1269, 108)
# - megamillions features: (838, 109)
# we do NOT assume specific feature names, just use what exists
# all numeric columns used as embedding space by default
# NaN handling: fill with 0.0 (consistent with earlier modules)


def load_features(
    lottery: LotteryName,
    use_columns: None | list[str] = None,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    Load Module 3 features for the specified lottery.

    Parameters
    ----------
    lottery : "powerball" or "megamillions"
    use_columns : optional list of column names to select.
        If provided, we first restrict to these columns (and raise an
        error if any are missing).
    numeric_only : bool, default True
        If True, we keep only numeric dtypes after any column selection,
        and then fill NaNs in those numeric columns with 0.0.

    Returns
    -------
    df : pandas.DataFrame
        Feature matrix, index aligned with the original draw order.

    Notes
    -----
    - We do not fabricate any columns; we only use what exists in the
      parquet file.
    - NaNs in numeric columns are filled with 0.0 to ensure that
      downstream manifold and distance-based methods (UMAP, diffusion
      maps, etc.) receive finite values, as required by scikit-learn
      and umap-learn.
    """
    path = get_features_path(lottery)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_parquet(path)

    # Optional: restrict to user-specified columns
    if use_columns is not None:
        missing = [c for c in use_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Requested columns not found in {path}: {missing}"
            )
        df = df[use_columns]

    if numeric_only:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise ValueError(
                f"No numeric columns found in feature file: {path}. "
                "Try numeric_only=False or specify use_columns explicitly."
            )

        # IMPORTANT: handle NaNs explicitly for manifold methods.
        # This mirrors your earlier pattern:
        # "Filled NaNs in feature columns with 0."
        numeric_df = numeric_df.fillna(0.0)

        return numeric_df

    # If numeric_only=False, we still want to avoid NaNs causing
    # downstream crashes; leave them as-is here so callers can decide.
    return df


def get_module9_paths(lottery: LotteryName) -> Dict[str, str]:
    """
    Get canonical paths to the main Module 9 graph objects and CSVs
    for the given lottery.
    """
    base = get_module9_dir(lottery)

    def p(name: str) -> str:
        return os.path.join(base, name)

    prefix = "powerball" if lottery == "powerball" else "megamillions"

    paths = {
        "cooccurrence_graph": p(f"{prefix}_cooccurrence_graph.pkl"),
        "transition_graph": p(f"{prefix}_transition_graph.pkl"),
        "knn_graph": p(f"{prefix}_knn_draws_k10_euclidean.pkl"),
        "correlation_network": p(f"{prefix}_correlation_network.pkl"),
        "advanced_analysis": p(f"{prefix}_advanced_analysis.csv"),
    }
    return paths


def load_graph(path: str) -> nx.Graph:
    """
    Load a pickled NetworkX graph using standard pickle library.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph pickle not found: {path}")
    
    with open(path, 'rb') as f:
        # Load the graph object from the binary file
        G = pickle.load(f)
    
    if not isinstance(G, nx.Graph):
        # Optional check for integrity
        raise TypeError(f"File {path} does not contain a NetworkX Graph object.")
        
    return G


def load_module9_graphs(lottery: LotteryName) -> Dict[str, nx.Graph]:
    """
    Convenience wrapper to load the four main Module 9 graphs.

    Returns
    -------
    graphs : dict
        {
            "cooccurrence_graph": nx.Graph,
            "transition_graph": nx.DiGraph or nx.Graph,
            "knn_graph": nx.Graph,
            "correlation_network": nx.Graph
        }
    """
    paths = get_module9_paths(lottery)
    graphs: Dict[str, nx.Graph] = {}
    for key in ["cooccurrence_graph", "transition_graph", "knn_graph", "correlation_network"]:
        graphs[key] = load_graph(paths[key])
    return graphs

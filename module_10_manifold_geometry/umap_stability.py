"""
Module 10: UMAP Embedding Stability
Tests robustness of UMAP embeddings across hyperparameters and seeds.

"""

import argparse
import os
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
    get_module10_figures_dir,
)
from data import load_features


def compute_neighbor_overlap(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    n_neighbors: int,
) -> float:
    """
    Compute average Jaccard overlap between k-NN neighborhoods in two
    embeddings of the same data.
    """
    from sklearn.neighbors import NearestNeighbors

    n = embedding_a.shape[0]
    k = min(n_neighbors, n - 1)

    nbrs_a = NearestNeighbors(n_neighbors=k + 1).fit(embedding_a)
    nbrs_b = NearestNeighbors(n_neighbors=k + 1).fit(embedding_b)

    inds_a = nbrs_a.kneighbors(return_distance=False)[:, 1:]
    inds_b = nbrs_b.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for i in range(n):
        set_a = set(inds_a[i])
        set_b = set(inds_b[i])
        if not set_a and not set_b:
            overlaps.append(1.0)
        else:
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            overlaps.append(inter / union if union > 0 else 0.0)

    return float(np.mean(overlaps))


def run_umap_stability(
    lottery: LotteryName,
    n_neighbors_grid=(10, 30, 50),  # Reduced from (5, 10, 20, 50, 100)
    min_dist_grid=(0.1, 0.5),  # Reduced from (0.0, 0.1, 0.3, 0.5, 0.9)
    n_components: int = 2,
    n_seeds: int = 3,
    max_samples: int = 500,  # Reduced from 600
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    """
    UMAP stability analysis with reduced grid for performance.
    """
    df = load_features(lottery, numeric_only=True)
    X_full = df.to_numpy(dtype=float)

    # Downsample for speed if needed
    if X_full.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_full.shape[0], size=max_samples, replace=False)
        X = X_full[idx]
        print(
            f"[Module10][UMAP] {lottery}: downsampled from "
            f"{X_full.shape[0]} to {X.shape[0]} samples for stability analysis."
        )
    else:
        X = X_full
        print(
            f"[Module10][UMAP] {lottery}: using all {X.shape[0]} samples "
            f"for stability analysis."
        )

    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)

    # Try to import UMAP. If missing, record "skipped" and exit cleanly.
    try:
        import umap  # type: ignore
    except ImportError:
        print(
            f"[Module10][UMAP] umap-learn not installed. "
            f"Skipping UMAP stability for {lottery}."
        )

        skipped_csv = os.path.join(
            output_dir,
            f"{lottery}_umap_stability.csv",
        )
        pd.DataFrame(
            [
                {
                    "lottery": lottery,
                    "status": "skipped_missing_umap_learn",
                    "n_neighbors": np.nan,
                    "min_dist": np.nan,
                    "n_components": n_components,
                    "n_seeds": n_seeds,
                    "max_samples": max_samples,
                    "stability_mean": np.nan,
                    "stability_std": np.nan,
                }
            ]
        ).to_csv(skipped_csv, index=False)
        return

    results = []
    total_jobs = len(n_neighbors_grid) * len(min_dist_grid)
    job_idx = 0

    # Set up a warning filter context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        for n_neighbors in n_neighbors_grid:
            for min_dist in min_dist_grid:
                job_idx += 1
                print(
                    f"[Module10][UMAP] {lottery}: job {job_idx}/{total_jobs} "
                    f"(n_neighbors={n_neighbors}, min_dist={min_dist}, "
                    f"n_seeds={n_seeds})"
                )

                embeddings = []
                for seed_idx in range(n_seeds):
                    seed = random_state + seed_idx
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        random_state=seed,
                        metric="euclidean",
                    )
                    emb = reducer.fit_transform(X)
                    embeddings.append(emb)

                # Compute pairwise stability scores across all seed pairs
                pair_scores = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        score = compute_neighbor_overlap(
                            embeddings[i],
                            embeddings[j],
                            n_neighbors=n_neighbors,
                        )
                        pair_scores.append(score)

                stability_mean = float(np.mean(pair_scores))
                stability_std = (
                    float(np.std(pair_scores, ddof=1))
                    if len(pair_scores) > 1
                    else 0.0
                )

                results.append(
                    {
                        "lottery": lottery,
                        "status": "ok",
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "n_components": n_components,
                        "n_seeds": n_seeds,
                        "max_samples": max_samples,
                        "stability_mean": stability_mean,
                        "stability_std": stability_std,
                    }
                )

    out_csv = os.path.join(
        output_dir,
        f"{lottery}_umap_stability.csv",
    )
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_csv, index=False)
    print(f"[Module10][UMAP] {lottery}: saved stability results to {out_csv}")

    # Simple heatmap of stability_mean vs (n_neighbors, min_dist)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        pivot = df_res.pivot(
            index="n_neighbors",
            columns="min_dist",
            values="stability_mean",
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cbar=True,
        )
        plt.title(f"{lottery.capitalize()} - UMAP Stability (Mean Overlap)")
        plt.ylabel("n_neighbors")
        plt.xlabel("min_dist")
        fig_path = os.path.join(
            fig_dir,
            f"{lottery}_umap_stability_grid.png",
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[Module10][UMAP] {lottery}: saved figure to {fig_path}")
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - UMAP stability analysis"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=3,
        help="Number of random seeds per hyperparameter setting.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to use for UMAP stability.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE_DEFAULT,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_umap_stability(
        lottery=args.lottery,  # type: ignore[arg-type]
        n_seeds=args.n_seeds,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
"""
Module 10: Distance Concentration Analysis
Tests for curse of dimensionality via pairwise distance distributions.

In high dimensions, distances between random points concentrate around
their mean - this is a key signature of high-dimensional geometry.
We compare real lottery data vs randomized baselines.

"""

import argparse
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
    get_module10_figures_dir,
)
from data import load_features


def _sample_pairwise_distances(
    X: np.ndarray,
    n_pairs: int,
    random_state: int,
) -> np.ndarray:
    """sample random pairs and compute pairwise distances."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples to compute distances")

    # sample random pairs without replacement within each pair
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)

    # ensure i != j for all pairs
    same_mask = i_idx == j_idx
    while same_mask.any():
        i_idx[same_mask] = rng.integers(0, n, size=same_mask.sum())
        j_idx[same_mask] = rng.integers(0, n, size=same_mask.sum())
        same_mask = i_idx == j_idx

    # compute euclidean distances
    diff = X[i_idx] - X[j_idx]
    return np.linalg.norm(diff, axis=1)


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    n = len(values)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(values[idx])
    lower = np.percentile(stats, 100 * (alpha / 2.0))
    upper = np.percentile(stats, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _summarize_distances(
    distances: np.ndarray,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> Dict[str, float]:
    distances = np.asarray(distances)
    mean = float(np.mean(distances))
    std = float(np.std(distances, ddof=1))
    cv = std / mean if mean > 0 else np.nan

    mean_ci = _bootstrap_ci(distances, np.mean, random_state=random_state)
    std_ci = _bootstrap_ci(
        distances,
        lambda x: np.std(x, ddof=1),
        random_state=random_state + 1,
    )
    cv_ci = _bootstrap_ci(
        distances,
        lambda x: np.std(x, ddof=1) / np.mean(x),
        random_state=random_state + 2,
    )

    return {
        "mean": mean,
        "mean_ci_lower": mean_ci[0],
        "mean_ci_upper": mean_ci[1],
        "std": std,
        "std_ci_lower": std_ci[0],
        "std_ci_upper": std_ci[1],
        "cv": cv,
        "cv_ci_lower": cv_ci[0],
        "cv_ci_upper": cv_ci[1],
    }


def run_distance_concentration(
    lottery: LotteryName,
    n_pairs: int = 100_000,
    n_boot: int = 1000,
    random_state: int = RANDOM_STATE_DEFAULT,
    feature_columns: None | list[str] = None,
) -> None:
    df = load_features(
        lottery,
        use_columns=feature_columns,
        numeric_only=True,
    )
    X = df.to_numpy(dtype=float)

    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)

    real_distances = _sample_pairwise_distances(
        X, n_pairs=n_pairs, random_state=random_state
    )
    real_summary = _summarize_distances(real_distances, random_state)

    rng = np.random.default_rng(random_state)
    X_rand = X.copy()
    for j in range(X_rand.shape[1]):
        rng.shuffle(X_rand[:, j])

    rand_distances = _sample_pairwise_distances(
        X_rand, n_pairs=n_pairs, random_state=random_state + 10
    )
    rand_summary = _summarize_distances(rand_distances, random_state + 10)

    rows = []
    for label, summary in [
        ("real", real_summary),
        ("randomized_baseline", rand_summary),
    ]:
        row = {"dataset": label}
        row.update(summary)
        rows.append(row)

    out_csv = os.path.join(
        output_dir,
        f"{lottery}_distance_concentration.csv",
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.hist(
            real_distances, bins=60, density=True, alpha=0.5, label="real"
        )
        plt.hist(
            rand_distances,
            bins=60,
            density=True,
            alpha=0.5,
            label="randomized",
        )
        plt.xlabel("Pairwise distance")
        plt.ylabel("Density")
        plt.title(f"{lottery.capitalize()} - Distance Concentration")
        plt.legend()
        fig_path = os.path.join(
            fig_dir,
            f"{lottery}_distance_concentration.png",
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Distance Concentration Analysis"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE_DEFAULT,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_distance_concentration(
        lottery=args.lottery,  # type: ignore[arg-type]
        n_pairs=args.n_pairs,
        n_boot=args.n_boot,
        random_state=args.random_state,
        feature_columns=None,
    )

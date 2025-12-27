"""
Module 10: Random Manifold Null Tests
Compares lottery data to synthetic random manifolds.

Generates null distributions from:
- Uniform hypercube
- Isotropic Gaussian cloud
- Hypersphere shell
- Column-shuffled data

"""

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
)
from data import load_features
from distance_concentration import _summarize_distances, _sample_pairwise_distances


def generate_random_baselines(
    n_samples: int,
    n_features: int,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)

    # Uniform hypercube [0, 1]^d
    cube = rng.random((n_samples, n_features))

    # Isotropic Gaussian N(0, I)
    gaussian = rng.normal(size=(n_samples, n_features))

    # Hypersphere shell (radius 1)
    gaussian_sphere = rng.normal(size=(n_samples, n_features))
    norms = np.linalg.norm(gaussian_sphere, axis=1, keepdims=True)
    hypersphere = gaussian_sphere / norms

    return {
        "uniform_hypercube": cube,
        "gaussian_cloud": gaussian,
        "hypersphere_shell": hypersphere,
    }


def run_random_manifold_tests(
    lottery: LotteryName,
    n_pairs: int = 50_000,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    df = load_features(lottery, numeric_only=True)
    X = df.to_numpy(dtype=float)

    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)

    # Shuffled draws: permute each column independently
    X_shuffled = X.copy()
    for j in range(n_features):
        rng.shuffle(X_shuffled[:, j])

    baselines = generate_random_baselines(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
    baselines["shuffled_draws"] = X_shuffled
    baselines["real_lottery"] = X

    rows = []
    for name, Xb in baselines.items():
        distances = _sample_pairwise_distances(
            Xb, n_pairs=n_pairs, random_state=random_state + hash(name) % 10000
        )
        summary = _summarize_distances(
            distances, random_state=random_state + hash(name) % 10000
        )
        row = {"dataset": name, "lottery": lottery}
        row.update(summary)
        rows.append(row)

    output_dir = get_module10_output_dir(lottery)
    out_csv = os.path.join(
        output_dir,
        f"{lottery}_random_manifold_tests.csv",
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Synthetic Random Manifold Comparisons"
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
        default=50_000,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_random_manifold_tests(
        lottery=args.lottery,  # type: ignore[arg-type]
        n_pairs=args.n_pairs,
    )

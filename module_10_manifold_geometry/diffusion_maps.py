"""
Module 10: Diffusion Maps Eigenspectrum
Spectral analysis of diffusion operator on data manifold.

"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_kernels

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
    get_module10_figures_dir,
)
from data import load_features


def compute_diffusion_eigenvalues(
    X: np.ndarray,
    n_eigvals: int = 50,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Simple diffusion maps eigenvalue computation using an RBF kernel:

        K_ij = exp(-gamma * ||x_i - x_j||^2)

    Then we row-normalize K to get a Markov matrix P and compute its
    top eigenvalues.
    """
    # Compute kernel matrix (this is O(n^2); your n is small enough)
    K = pairwise_kernels(X, metric="rbf", gamma=gamma)
    row_sums = K.sum(axis=1, keepdims=True)
    P = K / row_sums

    # Compute top n_eigvals eigenvalues of P (symmetric approximation)
    # Use numpy.linalg.eig; for small n this is fine.
    vals, _ = np.linalg.eig(P)
    vals = np.real(vals)
    vals_sorted = np.sort(vals)[::-1]
    return vals_sorted[:n_eigvals]


def run_diffusion_maps(
    lottery: LotteryName,
    n_eigvals: int = 50,
    gamma: float = 1.0,
    max_samples: int = 500,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    df = load_features(lottery, numeric_only=True)
    X_full = df.to_numpy(dtype=float)
    
    # Downsample if needed to avoid O(n^2) memory explosion
    if X_full.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_full.shape[0], size=max_samples, replace=False)
        X = X_full[idx]
        print(
            f"[Module10][DiffusionMaps] {lottery}: downsampled from "
            f"{X_full.shape[0]} to {X.shape[0]} samples."
        )
    else:
        X = X_full
        print(
            f"[Module10][DiffusionMaps] {lottery}: using all {X.shape[0]} samples."
        )

    eigvals = compute_diffusion_eigenvalues(X, n_eigvals=n_eigvals, gamma=gamma)

    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)

    out_csv = os.path.join(
        output_dir,
        f"{lottery}_diffusion_eigenvalues.csv",
    )
    pd.DataFrame(
        {
            "index": np.arange(len(eigvals)),
            "eigenvalue": eigvals,
        }
    ).to_csv(out_csv, index=False)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(np.arange(len(eigvals)), eigvals, marker="o")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.title(f"{lottery.capitalize()} - Diffusion Map Eigenspectrum")
        fig_path = os.path.join(
            fig_dir,
            f"{lottery}_diffusion_eigenspectrum.png",
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[Module10][DiffusionMaps] {lottery}: saved figure to {fig_path}")
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Diffusion Maps Eigenspectrum"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument(
        "--n_eigvals",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE_DEFAULT,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diffusion_maps(
        lottery=args.lottery,  # type: ignore[arg-type]
        n_eigvals=args.n_eigvals,
        gamma=args.gamma,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
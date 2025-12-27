"""
Module 10: Intrinsic Dimensionality Estimation
MLE-based estimation of manifold intrinsic dimension.

Uses k-NN distance ratios to estimate local dimension.
Lower than ambient dimension suggests manifold structure.

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
    get_module10_figures_dir,
)
from data import load_features


def estimate_id_mle(X: np.ndarray, k: int = 10) -> float:
    """
    Simple MLE estimator for intrinsic dimension based on k-NN distances.

    This is a simplified version for illustration; you can replace it
    with scikit-dimension's implementation if available.
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distances[:, 0] is self-distance (0); ignore
    distances = distances[:, 1:]

    # Avoid zeros
    distances = np.maximum(distances, 1e-12)

    logs = np.log(distances[:, -1][:, None] / distances[:, :-1])
    mle = (1.0 / np.mean(np.mean(logs, axis=1)))
    return float(mle)


def run_intrinsic_dimensionality(
    lottery: LotteryName,
    k_values=(5, 10, 20),
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    df = load_features(lottery, numeric_only=True)
    X = df.to_numpy(dtype=float)

    results: list[Dict[str, float | int | str]] = []

    for k in k_values:
        id_hat = estimate_id_mle(X, k=k)
        results.append(
            {
                "lottery": lottery,
                "method": "MLE_knn",
                "k": k,
                "id_hat": id_hat,
            }
        )

    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)

    out_csv = os.path.join(
        output_dir,
        f"{lottery}_intrinsic_dimensionality.csv",
    )
    pd.DataFrame(results).to_csv(out_csv, index=False)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        df_res = pd.DataFrame(results)
        plt.figure()
        plt.plot(df_res["k"], df_res["id_hat"], marker="o")
        plt.xlabel("k (neighbors)")
        plt.ylabel("Estimated Intrinsic Dimension")
        plt.title(f"{lottery.capitalize()} - Intrinsic Dimensionality (MLE)")
        fig_path = os.path.join(
            fig_dir,
            f"{lottery}_intrinsic_dimensionality.png",
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Intrinsic Dimensionality Estimation"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_intrinsic_dimensionality(
        lottery=args.lottery,  # type: ignore[arg-type]
    )

"""
Module 10: Topological Data Analysis (TDA)
Persistent homology and cohomology using GUDHI.

"""

import argparse
import os
import traceback
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
)
from data import load_features


def _summarize_diagram(diagram: np.ndarray, dim: int) -> Dict[str, Any]:
    if diagram.size == 0:
        return {
            "dim": dim,
            "num_features": 0,
            "mean_persistence": 0.0,
            "max_persistence": 0.0,
        }

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    pers = deaths - births
    mask = pers > 0

    if not np.any(mask):
        return {
            "dim": dim,
            "num_features": 0,
            "mean_persistence": 0.0,
            "max_persistence": 0.0,
        }

    pers = pers[mask]
    return {
        "dim": dim,
        "num_features": int(pers.shape[0]),
        "mean_persistence": float(np.mean(pers)),
        "max_persistence": float(np.max(pers)),
    }


def _tda_with_gudhi(X: np.ndarray, maxdim: int) -> List[Dict[str, Any]]:
    import gudhi as gd  # type: ignore

    rips = gd.RipsComplex(points=X)
    st = rips.create_simplex_tree(max_dimension=maxdim + 1)

    st.compute_persistence()

    summaries = []
    for dim in range(0, maxdim + 1):
        intervals = st.persistence_intervals_in_dimension(dim)
        dgm_arr = np.asarray(intervals, dtype=float)
        summaries.append(_summarize_diagram(dgm_arr, dim=dim))
    return summaries


def run_tda_cohomology(
    lottery: LotteryName,
    maxdim: int = 2,  # Reduced from 3
    max_samples: int = 200,  # Reduced from 500 - CRITICAL FOR MEMORY
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    """
    Run TDA analysis with GUDHI.
    
    WARNING: This is EXTREMELY memory-intensive. Even 200 samples can use 
    several GB of RAM. If your system has <16GB RAM, consider skipping this.
    
    Rips complex has exponential complexity: O(n^(maxdim+2))
    """
    
    print(f"\n[Module10][TDA] Starting for {lottery}...")
    print(f"  WARNING: TDA is memory-intensive. Using max {max_samples} samples.")

    output_dir = get_module10_output_dir(lottery)
    os.makedirs(output_dir, exist_ok=True)

    out_csv = os.path.join(output_dir, f"{lottery}_cohomology_results.csv")

    df = load_features(lottery, numeric_only=True)
    X_full = df.to_numpy(dtype=float)

    # Aggressive downsampling for safety
    if X_full.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_full.shape[0], size=max_samples, replace=False)
        X = X_full[idx]
        print(f"  Downsampled from {X_full.shape[0]} to {X.shape[0]} samples.")
    else:
        X = X_full
        print(f"  Using all {X.shape[0]} samples.")

    # Normalize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # Try GUDHI with safety checks
    try:
        import gudhi  # noqa: F401
        
        # Additional safety: warn if sample count is still high
        if X.shape[0] > 150:
            print(f"  WARNING: {X.shape[0]} samples may still crash. Recommend max_samples=150.")
        
        print(f"  Running GUDHI persistent homology (maxdim={maxdim})...")
        summaries = _tda_with_gudhi(X_norm, maxdim=maxdim)
        backend = "gudhi"

        rows = []
        for s in summaries:
            rows.append({
                "lottery": lottery,
                "backend": backend,
                "status": "ok",
                "dim": s["dim"],
                "num_features": s["num_features"],
                "mean_persistence": s["mean_persistence"],
                "max_persistence": s["max_persistence"],
                "maxdim": maxdim,
                "max_samples": max_samples,
            })

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  Wrote GUDHI cohomology summary to {out_csv}")
        return

    except Exception:
        print(f"  GUDHI failed or crashed. Skipping TDA.")
        traceback.print_exc()

        pd.DataFrame([
            {
                "lottery": lottery,
                "backend": "none",
                "status": "skipped_runtime_error",
                "dim": np.nan,
                "num_features": np.nan,
                "mean_persistence": np.nan,
                "max_persistence": np.nan,
                "maxdim": maxdim,
                "max_samples": max_samples,
            }
        ]).to_csv(out_csv, index=False)
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 10 - TDA analysis (GUDHI only)")
    parser.add_argument("--lottery", type=str, choices=["powerball", "megamillions"], required=True)
    parser.add_argument("--maxdim", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tda_cohomology(
        lottery=args.lottery,  # type: ignore
        maxdim=args.maxdim,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
"""
Module 10: Manifold Geometry - Master Pipeline
Orchestrates all geometric and topological analyses.

Analyses performed:
1. Distance concentration (curse of dimensionality)
2. UMAP stability (embedding consistency)
3. Diffusion maps (spectral geometry)
4. Intrinsic dimensionality (MLE estimation)
5. TDA/persistent cohomology (topological features)
6. Random manifold comparisons (null distributions)
7. GNN embeddings (graph neural network)
8. Contrastive embeddings (SimCLR-style)
9. Curvature estimation (Ricci curvature)
10. Geodesic entropy (neighborhood structure)

"""

import argparse
import gc

from config import LotteryName
from distance_concentration import run_distance_concentration
from umap_stability import run_umap_stability
from diffusion_maps import run_diffusion_maps
from intrinsic_dimensionality import run_intrinsic_dimensionality
from tda_cohomology import run_tda_cohomology
from random_manifold_tests import run_random_manifold_tests
from gnn_embeddings import run_gnn_embeddings
from contrastive_embeddings import run_contrastive_embeddings
from curvature_estimation import run_curvature_estimation
from geodesic_entropy import run_geodesic_entropy


def run_module10_for_lottery(lottery: LotteryName, skip_tda: bool = False) -> None:
    """Run all Module 10 analyses for a single lottery."""
    
    print(f"STARTING MODULE 10 PIPELINE FOR {lottery.upper()}")
    if skip_tda:
        print(f"(TDA analysis will be SKIPPED)")
    
    # 1. Distance concentration
    print(f"[1/10] Running distance concentration...")
    run_distance_concentration(lottery=lottery)
    gc.collect()
    
    # 2. UMAP stability
    print(f"[2/10] Running UMAP stability...")
    run_umap_stability(lottery=lottery)
    gc.collect()
    
    # 3. Diffusion maps eigenspectrum
    print(f"[3/10] Running diffusion maps...")
    run_diffusion_maps(lottery=lottery)
    gc.collect()
    
    # 4. Intrinsic dimensionality
    print(f"[4/10] Running intrinsic dimensionality...")
    run_intrinsic_dimensionality(lottery=lottery)
    gc.collect()
    
    # 5. Persistent cohomology / TDA
    if skip_tda:
        print(f"[5/10] SKIPPING TDA (as requested)...")
    else:
        print(f"[5/10] Running TDA/cohomology (WARNING: memory-intensive)...")
        try:
            run_tda_cohomology(lottery=lottery, maxdim=2, max_samples=200)
        except Exception as e:
            print(f"  TDA crashed or failed: {e}")
            print(f"  Continuing with remaining analyses...")
    gc.collect()
    
    # 6. Synthetic random manifold comparisons
    print(f"[6/10] Running random manifold tests...")
    run_random_manifold_tests(lottery=lottery)
    gc.collect()
    
    # 7. GNN embeddings on kNN graph
    print(f"[7/10] Running GNN embeddings...")
    run_gnn_embeddings(lottery=lottery)
    gc.collect()
    
    # 8. Contrastive embeddings (SimCLR-style)
    print(f"[8/10] Running contrastive embeddings...")
    run_contrastive_embeddings(lottery=lottery)
    gc.collect()
    
    # 9. Manifold curvature estimation
    print(f"[9/10] Running curvature estimation...")
    run_curvature_estimation(lottery=lottery)
    gc.collect()
    
    # 10. Geodesic neighborhood entropy
    print(f"[10/10] Running geodesic entropy...")
    run_geodesic_entropy(lottery=lottery)
    gc.collect()
    
    print(f"COMPLETED MODULE 10 PIPELINE FOR {lottery.upper()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Master pipeline for geometric and manifold analyses"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions", "both"],
        required=True,
    )
    parser.add_argument(
        "--skip_tda",
        action="store_true",
        help="Skip TDA analysis (recommended if you have <16GB RAM)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.lottery == "both":
        for lot in ["powerball", "megamillions"]:
            run_module10_for_lottery(lottery=lot, skip_tda=args.skip_tda)  # type: ignore[arg-type]
    else:
        run_module10_for_lottery(lottery=args.lottery, skip_tda=args.skip_tda)  # type: ignore[arg-type]
"""
Module 10: Manifold Geometry & Intrinsic Structure - Configuration

"""

import os
from typing import Literal

LotteryName = Literal["powerball", "megamillions"]

# base paths - lottery-lab structure

BASE_DIR = r"C:\jackpotmath\lottery-lab"

# module 9 outputs (network analysis)
MODULE9_OUTPUT_DIR = os.path.join(
    BASE_DIR, "modules", "module_09_network_analysis", "outputs"
)

# module 10 base directory
MODULE10_DIR = os.path.join(BASE_DIR, "modules", "module_10_manifold_geometry")

# feature data directory
FEATURES_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features",
)

FEATURE_FILES = {
    "powerball": "features_powerball.parquet",
    "megamillions": "features_megamillions.parquet",
}

# module 10 outputs and figures - no lottery subdirs
MODULE10_OUTPUT_DIR = os.path.join(MODULE10_DIR, "outputs")
MODULE10_FIGURES_DIR = os.path.join(MODULE10_DIR, "figures")

RANDOM_STATE_DEFAULT = 42


def get_module9_dir(lottery: LotteryName) -> str:
    """get module 9 outputs directory (same for both lotteries)."""
    return MODULE9_OUTPUT_DIR


def get_module10_output_dir(lottery: LotteryName) -> str:
    """get module 10 outputs directory and create if needed."""
    os.makedirs(MODULE10_OUTPUT_DIR, exist_ok=True)
    return MODULE10_OUTPUT_DIR


def get_module10_figures_dir(lottery: LotteryName) -> str:
    """get module 10 figures directory and create if needed."""
    os.makedirs(MODULE10_FIGURES_DIR, exist_ok=True)
    return MODULE10_FIGURES_DIR


def get_features_path(lottery: LotteryName) -> str:
    """get full path to features parquet file for specified lottery."""
    fname = FEATURE_FILES[lottery]
    return os.path.join(FEATURES_DIR, fname)

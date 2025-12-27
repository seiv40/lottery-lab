#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 5 Zoo - Unified Deep Learning Model Launcher

Usage:
  python module5_zoo.py --model vae --lottery powerball
  python module5_zoo.py --model transformer --lottery megamillions --epochs 100

"""

import argparse
import json
from pathlib import Path
import sys
import traceback

# local utilities and workers
from utils_data import (
    LotteryFeatureDataModule,
    LotterySequenceDataModule,
    OUTPUT_DIR,
)
import utils_seed

import vae_model
import transformer_model
import transformer_bayeshead_model
import bnn_model
import flow_model
import deepset_model
import deepset_v2_model

# Model Registry

REGISTRY = {
    "vae": {
        "module": vae_model,
        "data": "flat",  # single-timestep features
        "desc": "Variational Autoencoder (latent map)",
    },
    "transformer": {
        "module": transformer_model,
        "data": "seq",   # windowed sequences
        "desc": "Encoder-only Transformer",
    },
    "transformer_bayeshead": {
        "module": transformer_bayeshead_model,
        "data": "seq",
        "desc": "Transformer + Bayesian head",
    },
    "bnn": {
        "module": bnn_model,
        "data": "flat",  # MLP or TCN inside the worker remaps window
        "desc": "Bayesian Neural Network (MC Dropout)",
    },
    "flow": {
        "module": flow_model,
        "data": "flat",
        "desc": "Normalizing Flow (MAF/RealNVP)",
    },
    "deepsets": {
        "module": deepset_model,
        "data": "flat",
        "desc": "Deep Sets (exchangeability test)",
    },
    "deepsets_v2": {
        "module": deepset_v2_model,
        "data": "flat",
        "desc": "Deep Sets v2 (residual + multi-agg + heteroscedastic)",
    },
}


def parse_args() -> argparse.Namespace:
    """parse command line arguments for the zoo launcher"""
    p = argparse.ArgumentParser(
        description="Module 5: Deep Learning Model Zoo (unified launcher)"
    )

    # common arguments
    p.add_argument(
        "--model",
        required=True,
        choices=list(REGISTRY.keys()),
        help="which model to run",
    )
    p.add_argument(
        "--lottery",
        default="powerball",
        choices=["powerball", "megamillions"],
        help="dataset to use",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)

    # optimizer settings
    p.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="optimizer to use (default: adamw)",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay (L2 or AdamW decoupled) for regularization",
    )

    # sequence model config (transformers, BNN with seq body, etc.)
    p.add_argument("--window_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # transformer extras
    p.add_argument("--blr_lambda", type=float, default=0.0, 
                   help="laplace/regularizer lambda (if used)")
    p.add_argument("--export_attn_samples", type=int, default=32)
    
    # bayesian head options (for transformer_bayeshead)
    p.add_argument("--hetero_var", action="store_true",
                   help="use heteroscedastic variance head (per-sample uncertainty)")
    p.add_argument("--heavy_tail", action="store_true",
                   help="use Student-t distribution instead of Gaussian")

    # VAE hyperparameters
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl_warmup_epochs", type=int, default=10)

    # BNN hyperparameters
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--kl_weight", type=float, default=0.01)
    p.add_argument("--mc_samples", type=int, default=20)
    p.add_argument("--bnn_arch", type=str, default="mlp", choices=["mlp", "tcn"])
    p.add_argument("--bnn_seq_len", type=int, default=1)
    p.add_argument("--tcn_channels", type=str, default="64,64")
    p.add_argument("--tcn_kernel", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # flow hyperparameters
    p.add_argument("--flow_layers", type=int, default=5)
    p.add_argument("--flow_hidden", type=int, default=128)
    p.add_argument("--flow_act", type=str, default="tanh", 
                   choices=["tanh", "softsign", "relu"])
    p.add_argument("--flow_type", type=str, default="maf", 
                   choices=["maf", "realnvp"])
    p.add_argument("--flow_weight_decay", type=float, default=1e-4)

    # deep sets hyperparameters
    p.add_argument("--ds_phi_hidden", type=int, default=128)
    p.add_argument("--ds_rho_hidden", type=int, default=128)

    # baselines / conformal / e-process (if workers use them)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--conformal_mode", type=str, default="pooled", 
                   choices=["pooled", "per_dim", "bonferroni"])
    p.add_argument("--eprocess_mode", type=str, default="standard", 
                   choices=["standard", "safe"])
    p.add_argument("--safe_e_gamma", type=float, default=0.4)
    p.add_argument("--baseline", type=str, default="copy_last", 
                   choices=["copy_last", "lag7"])

    return p.parse_args()


def _append_summary(lottery: str, record: dict):
    """
    append results to dl_performance_summary.json
    
    this creates a running log of all model results so we can easily
    compare performance across different architectures
    """
    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dl_performance_summary.json"

    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data.append(record)
            else:
                data = [data, record]
        except Exception:
            data = [record]
    else:
        data = [record]

    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    """
    main entry point for the zoo launcher
    
    flow:
    1. parse arguments
    2. look up model in registry
    3. create appropriate data module (flat vs sequential)
    4. call the model's train_and_export() function
    5. append results to summary file
    """
    args = parse_args()
    print(f"[Zoo] Model: {args.model} | Lottery: {args.lottery}")
    print(f"[Zoo] Optimizer: {args.optimizer} | Weight decay: {args.weight_decay}")

    # seed for reproducibility
    utils_seed.set_seed(42)

    # look up model configuration
    config = REGISTRY[args.model]
    worker = config["module"]

    # create appropriate data module based on what the model needs
    if config["data"] == "seq":
        # sequence models need windowed data (transformer, etc.)
        dm = LotterySequenceDataModule(
            lottery=args.lottery,
            window_len=args.window_len,
            batch_size=args.batch_size,
        )
    else:
        # flat models use single timestep features (VAE, Flow, DeepSets, etc.)
        dm = LotteryFeatureDataModule(
            lottery=args.lottery,
            batch_size=args.batch_size,
        )

    # load and prepare data
    dm.setup()

    # run the model's training function
    result = {}
    try:
        # create a run tag for output files (prevents overwriting different variants)
        # example: transformer_bayeshead__hetero_heavy vs transformer_bayeshead__baseline
        tag_parts = []
        if getattr(args, "hetero_var", False):
            tag_parts.append("hetero")
        if getattr(args, "heavy_tail", False):
            tag_parts.append("heavy")
        tag_suffix = "__" + "_".join(tag_parts) if tag_parts else "__baseline"
        args.run_tag = tag_suffix  # pass to worker for file naming
        
        result = worker.train_and_export(dm, args)
        
        if not isinstance(result, dict):
            result = {"ok": True, "note": "worker returned non-dict"}
    except Exception as e:
        print("[Zoo] Worker crashed:")
        traceback.print_exc()
        result = {"ok": False, "error": str(e)}

    # add common fields to the summary record
    result.update({
        "model": args.model,
        "lottery": args.lottery,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })

    # append to summary file
    _append_summary(args.lottery, result)
    print("[Zoo] Summary appended.")
    print("[Zoo] Done.")


if __name__ == "__main__":
    main()

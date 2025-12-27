#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 5 Visualization Script - Deep Learning Results Analysis

After training 7 different deep learning models on lottery data, we need to
visualize and compare results. This script generates all the plots and tables
for understanding model performance:

1. Performance Summary Tables
   - RMSE for all models (lower = better)
   - Coverage statistics (should be ~95% for calibrated models)
   - Interval widths (narrower = more confident predictions)

2. Comparison Plots
   - Bar charts comparing all models on each metric
   - Baseline comparisons (can models beat simple copy-last?)
   - Residual analysis (are errors random or systematic?)

3. Bayesian Head Analysis
   - Calibration curves (do 95% intervals contain 95% of data?)
   - Uncertainty quantification (heteroscedastic vs fixed variance)
   - Student-t vs Gaussian comparisons

4. Time Series Plots
   - Actual vs predicted trajectories
   - Small multiples for high-variance dimensions
   - Mean band plots with confidence intervals

This is like a "results dashboard" that answers: Did deep learning find patterns,
or do all models perform similarly (confirming randomness)?

Usage:
  python visualize_results.py

Outputs: PNG files in lottery-lab/output/visualizations/

"""

from pathlib import Path
from datetime import datetime
import json
import os
import re
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set reasonable default figure size
plt.rcParams['figure.figsize'] = (8, 4)


# Path Configuration

def infer_output_dir():
    """
    find the output directory
    
    tries multiple common locations to handle different machine setups
    """
    candidates = [
        Path(r"C:/jackpotmath/jackpotmath/lottery-lab/output"),  # windows
        Path.cwd() / "lottery-lab" / "output",  # relative to current dir
        Path.cwd() / "output",  # flat structure
    ]
    
    for p in candidates:
        if p.exists():
            return p
    
    # fallback: create in current directory
    fallback = Path.cwd() / "output"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


OUTPUT_DIR = infer_output_dir()
POWERBALL_DIR = OUTPUT_DIR / "powerball"
MEGAM_DIR = OUTPUT_DIR / "megamillions"
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Viz] Output directory: {OUTPUT_DIR}")
print(f"[Viz] Visualization output: {VIZ_DIR}")


# Data Loading Utilities

def load_json(path):
    """safely load JSON file"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _read_json(p: Path):
    """read JSON with error handling"""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read: {p} -> {e}")
        return None


def load_perf(perf_path: str):
    """load performance JSON file"""
    try:
        with open(perf_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# Load All Summary Files

def load_all_summaries():
    """
    recursively find and load all dl_performance_summary.json files
    
    aggregates results from all models and lotteries into single dataframe
    """
    summary_files = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == "dl_performance_summary.json":
                summary_files.append(Path(root) / f)
    
    records = []
    for p in summary_files:
        js = load_json(p)
        if isinstance(js, list):
            records.extend(js)
        elif isinstance(js, dict):
            records.append(js)
    
    df = pd.DataFrame(records).dropna(how="all")
    
    # parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT
    
    # keep only successful runs (error is NaN)
    if not df.empty:
        df_success = df[df.get("error").isna()] if "error" in df.columns else df.copy()
        # take the latest run for each (model, lottery) pair
        df_final = (
            df_success.sort_values("timestamp", ascending=False)
                      .drop_duplicates(subset=["model", "lottery"], keep="first")
                      .reset_index(drop=True)
        )
    else:
        df_final = pd.DataFrame()
    
    print(f"[Viz] Found {len(df_final)} final runs from {len(summary_files)} summary file(s)")
    return df_final


# Bayesian Head Analysis

def _variant_from_filename(name: str) -> str:
    """extract variant type from filename"""
    # expected suffix: __baseline / __hetero / __heavy / __hetero_heavy
    m = re.search(r"transformer_bayeshead_perf__(.+?)\.json$", name)
    return m.group(1) if m else "latest"


def load_bayeshead_runs(lotteries=("powerball", "megamillions")):
    """
    load all transformer+bayeshead variant results
    
    we trained 4 variants (baseline, hetero, heavy, hetero_heavy)
    this function finds and loads all of them for comparison
    """
    rows = []
    pred_map = {}  # (lottery, variant) -> predictive_path or None
    
    for lot in lotteries:
        out = OUTPUT_DIR / lot
        if not out.exists():
            continue
        
        for perf_path in out.glob("transformer_bayeshead_perf*.json"):
            perf = _read_json(perf_path)
            if perf is None:
                continue
            
            variant = _variant_from_filename(perf_path.name)
            
            # extract flags from JSON (more reliable than filename parsing)
            hv = bool(perf.get("hetero_var", False))
            ht = bool(perf.get("heavy_tail", False))
            
            # derive variant from flags if filename did not have it
            if variant == "latest":
                variant = ("hetero_heavy" if hv and ht else
                           "hetero" if hv else
                           "heavy" if ht else
                           "baseline")
            
            # extract metrics
            rmse = perf.get("transformer_bayeshead_rmse_test", np.nan)
            cov = perf.get("coverage_95", np.nan)
            
            rows.append({
                "lottery": lot,
                "variant": variant,
                "hetero_var": hv,
                "heavy_tail": ht,
                "rmse": rmse,
                "coverage_95": cov,
                "perf_path": str(perf_path),
            })
            
            # find matching predictive file for calibration analysis
            pred_name = perf_path.name.replace("perf", "predictive")
            pred_path = perf_path.with_name(pred_name)
            pred_map[(lot, variant)] = str(pred_path) if pred_path.exists() else None
    
    if not rows:
        print("[Viz] No BayesHead perf files found")
        return pd.DataFrame(), pred_map
    
    df = pd.DataFrame(rows).sort_values(["lottery", "variant"]).reset_index(drop=True)
    return df, pred_map


# Model Zoo Performance Loading

def load_model_zoo_perf(base_output=OUTPUT_DIR):
    """
    load performance metrics for ALL models (VAE, Transformer, BNN, Flow, DeepSets, etc.)
    
    reads all *_perf*.json files and extracts key metrics for comparison
    """
    lotteries = ["powerball", "megamillions"]
    records = []
    
    for lot in lotteries:
        lot_dir = base_output / lot
        if not lot_dir.exists():
            continue
        
        for path in lot_dir.glob("*_perf*.json"):
            obj = _read_json(path)
            
            # skip list-style JSONs (summary files, not individual perf files)
            if isinstance(obj, list):
                continue
            
            if obj is None:
                continue
            
            # extract model name
            model_name = obj.get("model_name")
            if not model_name:
                # fallback: extract from filename
                stem = path.stem
                if "_perf" in stem:
                    model_name = stem.replace("_perf", "")
                else:
                    model_name = stem
            
            # extract interval width (can be dict or scalar)
            iw = obj.get("interval_width") or {}
            if isinstance(iw, dict):
                iw_mean = iw.get("mean", np.nan)
            else:
                iw_mean = np.nan
            
            records.append({
                "lottery": obj.get("lottery", lot),
                "model": model_name,
                "rmse": obj.get("rmse", np.nan),
                "coverage_95": obj.get("coverage_95", np.nan),
                "interval_width_mean": iw_mean,
                "runtime_sec": obj.get("runtime_sec", np.nan),
                "source_path": str(path),
            })
    
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(["lottery", "model"]).reset_index(drop=True)
    
    print(f"[Viz] Loaded {len(df)} model performance records")
    return df


# Baseline Comparison

def rmse(y, yhat):
    """compute root mean squared error"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size == 0 or yhat.size == 0 or y.size != yhat.size:
        return np.nan
    return float(np.sqrt(np.mean((y - yhat)**2)))


def naive_baseline(pred_target_series, mode="copy_last"):
    """
    simple baseline predictions
    
    copy_last: predict next value as copy of current value
    rolling_mean: predict next value as mean of last 3 values
    
    these are "null models" - if deep learning cannot beat them, it is not learning
    """
    y_true = np.asarray(pred_target_series.get("y_true", []), dtype=float)
    if y_true.size < 2:
        return np.array([]), np.array([])
    
    if mode == "copy_last":
        yhat = np.roll(y_true, 1)[1:]  # shift by 1
        y = y_true[1:]
        return y, yhat
    elif mode == "rolling_mean":
        yhat = []
        for i in range(len(y_true)):
            if i < 3:
                yhat.append(np.nan)
            else:
                yhat.append(np.mean(y_true[i-3:i]))
        yhat = np.array(yhat)[3:]
        y = y_true[3:]
        return y, yhat
    
    return np.array([]), np.array([])


# Visualization Functions

def plot_zoo_metric(df, lottery, metric, ylabel, ref=None, save_path=None):
    """
    bar chart comparing all models on a single metric
    
    ref: optional reference line (e.g., 0.95 for coverage)
    """
    d = df[(df["lottery"] == lottery) & (~df[metric].isna())].copy()
    if d.empty:
        print(f"[Viz] No {metric} entries for {lottery}")
        return
    
    d = d.sort_values("model")
    x = np.arange(len(d))
    
    plt.figure(figsize=(10, 4))
    plt.bar(x, d[metric], alpha=0.7)
    if ref is not None:
        plt.axhline(ref, linestyle="--", linewidth=1, color='red', 
                   label=f"Target: {ref}")
    plt.xticks(x, d["model"], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{lottery.capitalize()}: {ylabel} by Model")
    if ref is not None:
        plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.show()
    plt.close()


def plot_residuals(perf_path, last_n=200, save_path=None):
    """
    residual analysis plots
    
    left: actual vs predicted time series
    right: histogram of residuals (should be centered at 0 for unbiased model)
    """
    if not perf_path or not Path(perf_path).exists():
        print(f"[Viz] No perf_path for residuals: {perf_path}")
        return
    
    with open(perf_path, "r") as f:
        data = json.load(f)
    
    y_true, y_pred = [], []
    if isinstance(data, dict):
        y_true = data.get("y_true", [])
        y_pred = data.get("y_pred", [])
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "y_true" in data[0] and "y_pred" in data[0]:
            y_true = [d["y_true"] for d in data]
            y_pred = [d["y_pred"] for d in data]
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if y_true.size == 0 or y_pred.size == 0:
        print(f"[Viz] No y_true/y_pred fields in: {perf_path}")
        return
    
    n = min(last_n, y_true.size)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # time series plot
    ax[0].plot(y_true[-n:], label="Actual", alpha=0.7)
    ax[0].plot(y_pred[-n:], label="Predicted", alpha=0.7)
    ax[0].set_title(f"Last {n} Steps: Actual vs Predicted")
    ax[0].set_xlabel("Time Step")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    
    # residual histogram
    resid = y_true - y_pred
    ax[1].hist(resid, bins=30, alpha=0.7, edgecolor='black')
    ax[1].axvline(0, color='red', linestyle='--', linewidth=1, label='Zero')
    ax[1].set_title("Residuals Histogram")
    ax[1].set_xlabel("Residual (Actual - Predicted)")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.show()
    plt.close()


def plot_small_multiples(y_true, y_pred, last=200, k=6, save_path=None):
    """
    small multiples showing top variance dimensions
    
    displays the k dimensions with highest variance (most interesting to predict)
    """
    def top_variance_dims(y, k=6):
        var = np.var(y, axis=0)
        return np.argsort(var)[-k:][::-1]
    
    idx = top_variance_dims(y_true, k)
    T = slice(-last, None)
    
    cols = 3
    rows = int(np.ceil(k / cols))
    
    plt.figure(figsize=(cols * 4, rows * 2.6))
    for i, d in enumerate(idx, 1):
        plt.subplot(rows, cols, i)
        plt.plot(y_true[T, d], label="Actual", alpha=0.7)
        plt.plot(y_pred[T, d], "--", label="Predicted", alpha=0.7)
        plt.title(f"Dimension {d}")
        if i == 1:
            plt.legend()
        plt.grid(alpha=0.3)
    
    plt.suptitle("Last Steps: Actual vs Predicted (Top-Variance Dimensions)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.show()
    plt.close()


def plot_mean_band(y_true, y_pred, last=200, q=0.25, save_path=None):
    """
    mean trajectory with IQR bands
    
    shows average behavior across all dimensions with uncertainty bands
    helps see if model systematically over/underpredicts
    """
    T = slice(-last, None)
    
    # compute means and quantiles across dimensions
    mu_t = y_true[T].mean(axis=1)
    mu_p = y_pred[T].mean(axis=1)
    lo_t = np.quantile(y_true[T], q, axis=1)
    hi_t = np.quantile(y_true[T], 1 - q, axis=1)
    lo_p = np.quantile(y_pred[T], q, axis=1)
    hi_p = np.quantile(y_pred[T], 1 - q, axis=1)
    
    x = np.arange(len(mu_t))
    
    plt.figure(figsize=(10, 4))
    plt.fill_between(x, lo_t, hi_t, alpha=0.15, label="Actual IQR")
    plt.fill_between(x, lo_p, hi_p, alpha=0.15, label="Predicted IQR")
    plt.plot(x, mu_t, label="Actual Mean", linewidth=2)
    plt.plot(x, mu_p, "--", label="Predicted Mean", linewidth=2)
    plt.title("Mean Trajectory with IQR Bands Across Dimensions")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    plt.show()
    plt.close()


# Main Execution

def main():
    """generate all visualization outputs"""
    
    print("\n" + "="*70)
    print("MODULE 5: DEEP LEARNING RESULTS VISUALIZATION")
    print("="*70 + "\n")
    
    # load all performance data
    df_summary = load_all_summaries()
    df_zoo = load_model_zoo_perf()
    df_bayeshead, pred_paths = load_bayeshead_runs()
    
    # display summary tables
    if not df_summary.empty:
        print("\n[Viz] Summary of All Runs:")
        cols = ["timestamp", "lottery", "model", "rmse", "coverage_95", "runtime_sec"]
        print(df_summary[[c for c in cols if c in df_summary.columns]].to_string())
    
    if not df_bayeshead.empty:
        print("\n[Viz] Transformer+BayesHead Variants:")
        print(df_bayeshead.to_string())
    
    # generate comparison plots for each lottery
    print("\n" + "-"*70)
    print("GENERATING COMPARISON PLOTS")
    print("-"*70 + "\n")
    
    for lot in ["powerball", "megamillions"]:
        print(f"\n[Viz] Processing {lot.capitalize()}...")
        
        # rmse comparison
        plot_zoo_metric(
            df_zoo, lot, "rmse", "RMSE (Lower = Better)",
            save_path=VIZ_DIR / f"{lot}_rmse_comparison.png"
        )
        
        # coverage comparison
        plot_zoo_metric(
            df_zoo, lot, "coverage_95", "Empirical 95% Coverage",
            ref=0.95,
            save_path=VIZ_DIR / f"{lot}_coverage_comparison.png"
        )
        
        # interval width comparison
        plot_zoo_metric(
            df_zoo, lot, "interval_width_mean", "Mean 95% Interval Width",
            save_path=VIZ_DIR / f"{lot}_interval_width_comparison.png"
        )
    
    # generate residual plots
    print("\n" + "-"*70)
    print("GENERATING RESIDUAL PLOTS")
    print("-"*70 + "\n")
    
    if not df_summary.empty:
        for _, row in df_summary.iterrows():
            lot = row.get('lottery')
            model = row.get('model')
            perf_path = row.get('perf_path')
            
            # skip if perf_path is NaN or invalid
            if pd.isna(perf_path) or not perf_path:
                continue
            
            perf_path = str(perf_path)  # ensure it's a string
            
            if Path(perf_path).exists():
                print(f"[Viz] Residuals for {lot} / {model}")
                safe_name = f"{lot}_{model}_residuals.png".replace("/", "_")
                plot_residuals(
                    perf_path,
                    save_path=VIZ_DIR / safe_name
                )
    
    print("\n" + "="*70)
    print(f"VISUALIZATION COMPLETE - Outputs saved to: {VIZ_DIR}")
    print("="*70 + "\n")
    
    print("[Viz] Generated plots:")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()

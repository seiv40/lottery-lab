#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics Utilities for Module 5: Deep Learning Model Zoo

Why we need unified metrics logging:

When training 7 different models, you need a central place to compare results.
Without this, you'd have to:
- Hunt through multiple JSON files
- Manually compare metrics across models
- Risk losing track of which hyperparameters were used

This module provides write_summary() which appends all results to a single
dl_performance_summary.json file. This makes it easy to:
1. Compare all models in one place
2. Track which hyperparameters were used
3. See temporal progression (timestamps)
4. Generate comparison plots and tables

The summary file grows as you train more models, creating a complete log
of all experiments.

"""

from pathlib import Path
import json
import time

from utils_data import OUTPUT_DIR


def write_summary(model_name: str, lottery: str, metrics: dict, args=None):
    """
    append model results to unified summary file
    
    this creates a running log of all model performance for easy comparison
    
    args:
        model_name: which model (vae, transformer, bnn, etc.)
        lottery: powerball or megamillions
        metrics: dict with performance metrics (rmse, loss, etc.)
        args: optional argparse namespace with hyperparameters
    
    returns:
        path to summary file (for logging)
    """
    # ensure output directory exists
    out_dir = OUTPUT_DIR / lottery
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'dl_performance_summary.json'
    
    # create record with timestamp
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    record = {
        'timestamp': now,
        'model': model_name,
        **metrics
    }
    
    # extract relevant hyperparameters if provided
    if args is not None:
        keep = ['epochs', 'batch_size', 'learning_rate', 'weight_decay', 
                'window_len', 'latent_dim', 'bnn_seq_len',
                'alpha', 'conformal_mode', 'eprocess_mode', 
                'safe_e_gamma', 'baseline']
        record['args'] = {k: getattr(args, k) for k in keep if hasattr(args, k)}
    
    # load existing summary or create new one
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                data = [data]  # convert single record to list
        except Exception:
            data = []  # corrupted file, start fresh
    else:
        data = []
    
    # append new record
    data.append(record)
    
    # save updated summary
    summary_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    
    return str(summary_path)

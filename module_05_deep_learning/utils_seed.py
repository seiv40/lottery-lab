#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seed Utilities for Module 5: Deep Learning Model Zoo

Why we need deterministic seeding:

Deep learning is inherently stochastic:
- Random weight initialization
- Random data shuffling
- Random dropout masks
- GPU non-deterministic operations

Without fixing seeds, you get different results every run. This makes it
impossible to:
1. Debug issues (can't reproduce crashes)
2. Compare models fairly (different random seeds = different results)
3. Verify that code changes improve performance

We set seeds for ALL random number generators:
- Python's random module
- NumPy
- PyTorch CPU
- PyTorch CUDA
- PyTorch Lightning

We also enable deterministic algorithms where possible. This makes PyTorch
choose slower but reproducible implementations. Slight speed hit for much
better reproducibility.

"""

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl


def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    set all random seeds for reproducibility
    
    call this at the start of every script to ensure reproducible results
    
    args:
        seed: the random seed (default 42, because why not)
        deterministic: if True, use slower but deterministic algorithms
    
    returns:
        seed value (for logging)
    """
    # pytorch lightning's built-in seeding (covers most bases)
    pl.seed_everything(seed, workers=True)
    
    # explicit seeding for extra safety
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPU devices
    
    if deterministic:
        # use deterministic algorithms (slower but reproducible)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # configure CUDA for determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed


# alias for compatibility with Module 5 launcher
set_seed = set_global_seed

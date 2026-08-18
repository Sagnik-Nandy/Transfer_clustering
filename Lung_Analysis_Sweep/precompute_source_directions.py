#!/usr/bin/env python
"""Stage 1/2: precompute and cache each batch's Theta_hat (d, K) mean-matrix
estimate ONCE, at full sample size.

`estimate_source_means`'s output for a given batch depends only on that
batch's own (full, uncentered-then-centered) data -- never on which target
or target-subsample it will later be projected against. The main sweep
(run_sweep.py) needs this same Theta_hat repeatedly (4 targets x 4 k-values
x 30 seeds each use every OTHER batch's Theta_hat), so recomputing it inside
every sweep task would repeat the same expensive RelaxedKMeans solve
hundreds of times for no reason. Instead: 4 tasks here (one per batch, this
is the one-time expensive part, full batch size ~2000-3200), cached to
cache/theta_hat_<batch>.npy; run_sweep.py just loads these.

Usage:
    python precompute_source_directions.py <task_id>   # task_id in [0, 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiments_Script"))

import common
from transfer_clustering.multi_cluster import estimate_source_means

H5AD_PATH = os.path.join(_THIS_DIR, "..", "lung_atlas_analysis", "data", "lung_atlas_hvg_lognorm.h5ad")
CACHE_DIR = os.path.join(_THIS_DIR, "cache")
RELAXED_KMEANS_KWARGS = {"solver": "ADMM", "random_state": common.RANDOM_STATE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()
    batch = common.BATCHES[args.task_id]

    batch_data = common.load_full_batches(H5AD_PATH)
    X, _y = batch_data[batch]
    X = common.center_data(X)
    print(f"Computing Theta_hat for {batch}: n={X.shape[0]}, d={X.shape[1]}, K={common.K}")

    theta_hat = estimate_source_means(X, common.K, RELAXED_KMEANS_KWARGS)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"theta_hat_{batch}.npy")
    np.save(out_path, theta_hat)
    print(f"Saved {out_path}  shape={theta_hat.shape}")


if __name__ == "__main__":
    main()

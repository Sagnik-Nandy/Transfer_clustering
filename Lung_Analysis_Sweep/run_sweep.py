#!/usr/bin/env python
"""Stage 2/2: the target-downsampling sweep itself -- one (target, k, seed)
combination per invocation/Slurm array task.

For each task: draw one proportional subsample of the target (size k, seed),
then evaluate ALL 5 methods on that SAME subsample (paired comparison, same
convention as intro_illustration/<contrast>/'s notebooks):

  - target_only         -- RelaxedKMeans directly on the subsample (the only
                            part of this task that is genuinely expensive --
                            see precompute_source_directions.py for why the
                            other 4 methods are cheap here).
  - Source: <batch>      -- one row per one of the 3 non-target batches,
                            projecting the subsample onto that batch's CACHED
                            Theta_hat (precompute_source_directions.py) and
                            clustering with TSClust -- the K>2 analogue of
                            intro_illustration/<contrast>/'s
                            `source_based_estimate(X_sub, [one_source])`.
  - Multi-source (3 pooled) -- same, but projecting onto the joint span of
                            all 3 cached Theta_hats -- the K>2 analogue of
                            `source_based_estimate(X_sub, all_three_sources)`.

Requires cache/theta_hat_<batch>.npy for every batch (run
precompute_source_directions.py first, all 4 task_ids).

Usage:
    python run_sweep.py <task_id>

task_id indexes task_grid(): outer loop over target batch, middle loop over
k (common.k_grid_for(target's own full n) -- always exactly 4 values: 50,
200, 800, and the target's own full n), inner loop over subsample seed
(0..29). Writes 5 CSV rows (one per method) to
results/raw/<target>__k<k>__seed<seed>.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiments_Script"))

import numpy as np

import common
from transfer_clustering.multi_cluster import (
    _target_branch,
    source_subspace_projector,
    onehot_to_labels,
)
from transfer_clustering.ts_clust import ts_clust

H5AD_PATH = os.path.join(_THIS_DIR, "..", "lung_atlas_analysis", "data", "lung_atlas_hvg_lognorm.h5ad")
CACHE_DIR = os.path.join(_THIS_DIR, "cache")
RESULTS_DIR = os.path.join(_THIS_DIR, "results", "raw")
N_SUBSAMPLE = 30
RELAXED_KMEANS_KWARGS = {"solver": "ADMM"}
FIELDNAMES = ["target", "k", "seed", "method", "elapsed_sec", "misclustering", "ari"]


def task_grid():
    grid = []
    for target in common.BATCHES:
        for k in common.k_grid_for(common.BATCH_N_FULL[target]):
            for seed in range(N_SUBSAMPLE):
                grid.append((target, k, seed))
    return grid


def source_projected_labels(X_sub: np.ndarray, theta_hats: list, seed: int) -> np.ndarray:
    """Project X_sub onto span(theta_hats) and cluster with TSClust --
    mirrors `multi_cluster._source_branch`'s projection+clustering step
    exactly, but taking already-estimated theta_hats instead of raw source
    datasets (so it doesn't re-run estimate_source_means)."""
    Q_S = source_subspace_projector(theta_hats)
    X_hat = X_sub @ Q_S
    n_iter = int(round(2 * np.log(X_sub.shape[0])))
    labels = ts_clust(X_hat, common.K, n_iter=n_iter, random_state=seed)
    return labels


def run_one(task_id: int) -> list:
    grid = task_grid()
    if not (0 <= task_id < len(grid)):
        raise ValueError(f"task_id must be in [0, {len(grid) - 1}], got {task_id}")
    target, k, seed = grid[task_id]
    other_batches = [b for b in common.BATCHES if b != target]

    batch_data = common.load_full_batches(H5AD_PATH)
    X_full, y_full = batch_data[target]
    X_sub, y_sub = common.subsample_target(X_full, y_full, k, seed)
    print(f"target={target}, k={k}, seed={seed}, n_sub={X_sub.shape[0]}")

    theta_hats = {}
    for b in other_batches:
        theta_hats[b] = np.load(os.path.join(CACHE_DIR, f"theta_hat_{b}.npy"))

    rows = []

    # target_only -- the one genuinely expensive call in this task.
    t0 = time.time()
    Z = _target_branch(X_sub, common.K, {**RELAXED_KMEANS_KWARGS, "random_state": seed})
    z_hat = onehot_to_labels(Z)
    elapsed = time.time() - t0
    metrics = common.compute_metrics(z_hat, y_sub)
    rows.append(dict(target=target, k=k, seed=seed, method="target_only",
                      elapsed_sec=elapsed, **metrics))

    # single source: one row per other batch, using its cached theta_hat.
    for b in other_batches:
        t0 = time.time()
        z_hat = source_projected_labels(X_sub, [theta_hats[b]], seed)
        elapsed = time.time() - t0
        metrics = common.compute_metrics(z_hat, y_sub)
        rows.append(dict(target=target, k=k, seed=seed, method=f"source_{b}",
                          elapsed_sec=elapsed, **metrics))

    # multi-source: joint span of all 3 cached theta_hats.
    t0 = time.time()
    z_hat = source_projected_labels(X_sub, list(theta_hats.values()), seed)
    elapsed = time.time() - t0
    metrics = common.compute_metrics(z_hat, y_sub)
    rows.append(dict(target=target, k=k, seed=seed, method="multi_source_pooled",
                      elapsed_sec=elapsed, **metrics))

    for r in rows:
        print(f"  {r['method']}: misclustering={r['misclustering']:.3f} ari={r['ari']:.3f} "
              f"({r['elapsed_sec']:.1f}s)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()

    rows = run_one(args.task_id)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{rows[0]['target']}__k{rows[0]['k']}__seed{rows[0]['seed']}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

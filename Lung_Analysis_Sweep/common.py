"""Shared infrastructure for the all-cell-types intro-illustration sweep:
config constants, data loading, stratified target subsampling, and metrics.

Generalizes `intro_illustration/<contrast>/`'s NK-cell-vs-T-cell downsampling
sweep (K=2, `two_community.py`) to the full 13-cell-type lung atlas (K=13,
`multi_cluster.py`) -- same idea (does transfer help as target sample size k
shrinks?), same 4 Dropseq batches, but every cell type instead of one binary
contrast, and every batch gets a turn as target (4 panels) instead of a
single fixed target/contrast pair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

K = 13  # number of cell types (verified against the atlas's own obs['cell_type'], matches common.py elsewhere in this repo)
BATCHES = ["Dropseq_1", "Dropseq_2", "Dropseq_3", "Dropseq_4"]
RANDOM_STATE = 0

# Full (all-cell-type) per-batch sample size, verified directly from the
# atlas's own obs['batch'].value_counts() -- hardcoded rather than re-read
# from the h5ad in every one of run_sweep.py's ~480 tasks just to build
# task_grid()'s k values.
BATCH_N_FULL = {"Dropseq_1": 2098, "Dropseq_2": 3183, "Dropseq_3": 2387, "Dropseq_4": 2273}

# Subsampling grid: 3 log-spaced points plus each target's own full sample
# size as the 4th (proportional subsampling, so k can run all the way to
# n_full -- same convention as intro_illustration/<contrast>/'s notebooks).
K_GRID_BASE = [50, 200, 800]


def load_full_batches(h5ad_path: str) -> dict:
    """Returns {batch_name: (X (n, d) dense float64 uncentered, cell_type (n,) str array)}
    -- every cell, every cell type, unlike intro_illustration/<contrast>/'s
    build_contrast which filters to 2 classes."""
    import scanpy as sc
    import scipy.sparse as sp

    adata = sc.read_h5ad(h5ad_path)
    out = {}
    for batch in BATCHES:
        sub = adata[adata.obs["batch"] == batch]
        X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X, dtype=float)
        y = sub.obs["cell_type"].astype(str).values
        out[batch] = (X.astype(np.float64), y)
    return out


def center_data(X: np.ndarray) -> np.ndarray:
    """Subtract X's own per-gene mean (unsupervised) -- required by the
    hollowed-Gram/subspace-projection spectral methods, see common.py's
    docstring in the other Final_*_Analysis directories for why."""
    return X - X.mean(axis=0)


def k_grid_for(n_full: int) -> list[int]:
    """3 fixed log-spaced points (50, 200, 800) plus n_full itself as the
    4th, sorted and de-duplicated -- so the sweep always reaches the
    target's true full sample size, matching intro_illustration/<contrast>/'s
    proportional-subsampling convention."""
    grid = sorted(set([k for k in K_GRID_BASE if k < n_full] + [n_full]))
    return grid


def subsample_target(X_full: np.ndarray, y_full: np.ndarray, k_sub: int, seed: int):
    """Stratified proportional subsample of size k_sub, preserving the
    target's own per-cell-type proportions (generalizes
    intro_illustration/<contrast>/'s two-class subsample_target to K=13
    classes). Re-centers the subsample by its own (label-free) mean.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_full, return_counts=True)
    n_full = len(y_full)

    # Largest-remainder rounding so per-class subsample counts sum to
    # exactly k_sub while staying proportional to the class's share of n_full.
    raw = counts * (k_sub / n_full)
    base = np.floor(raw).astype(int)
    base = np.minimum(base, counts)  # never ask for more than a class has
    remainder = k_sub - base.sum()
    frac = raw - base
    order = np.argsort(-frac)
    i = 0
    while remainder > 0 and i < len(order):
        c = order[i]
        if base[c] < counts[c]:
            base[c] += 1
            remainder -= 1
        i += 1

    sel = []
    for c, n_c in zip(classes, base):
        if n_c <= 0:
            continue
        idx_c = np.where(y_full == c)[0]
        sel.append(rng.choice(idx_c, n_c, replace=False))
    sel = np.concatenate(sel)
    X_sub = X_full[sel]
    X_sub = X_sub - X_sub.mean(axis=0)
    y_sub = y_full[sel]
    return X_sub, y_sub


def misclustering_error(z_hat: np.ndarray, y_true_str: np.ndarray) -> float:
    """Best-permutation misclustering error via the Hungarian algorithm on
    the confusion matrix (same construction as the other Final_*_Analysis
    common.py's)."""
    true_labels, true_uniques = pd.factorize(y_true_str)
    R_true = len(true_uniques)
    R_hat = int(z_hat.max()) + 1
    R = max(R_true, R_hat)
    confusion = np.zeros((R, R), dtype=int)
    for i in range(len(z_hat)):
        confusion[int(z_hat[i]), true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    correct = confusion[row_ind, col_ind].sum()
    return float(1.0 - correct / len(z_hat))


def compute_metrics(z_hat: np.ndarray, y_true_str: np.ndarray) -> dict:
    true_labels, _ = pd.factorize(y_true_str)
    return dict(
        misclustering=misclustering_error(z_hat, y_true_str),
        ari=adjusted_rand_score(true_labels, z_hat),
    )

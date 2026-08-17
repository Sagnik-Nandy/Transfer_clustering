"""Shared infrastructure for the lung atlas (GSE130148) method comparison:
config constants, data loading, and metrics. Kept deliberately small and
separate from each method's own script (method_ours.py, method_tlgmm.py,
method_scrna.py, method_gdec.py) -- these three things (what K/batches we
use, how we load data, how we score a result) must be IDENTICAL across
every method for the comparison to be fair, so they live here once rather
than being duplicated (and risking drift) in each method script.

The LaTeX table generator is intentionally NOT here -- it's driver/
presentation logic and lives directly in run_lung_atlas_comparison.ipynb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, v_measure_score

K = 13  # number of cell types / clusters
BATCHES = ["Dropseq_1", "Dropseq_2", "Dropseq_3", "Dropseq_4"]
RANDOM_STATE = 0

METHOD_LABELS = {
    "target_only": "Target-only",
    "multi_source_pooled": "Multi-source pooled",
    "pooled_concat": "Pooled (concat)",
    "adaptive_multi_source": "Adaptive multi-source",
    "tlgmm": "TL-GMM",
    "scrna": "NMF",
    "gdec_gcnfree": "GDEC",
}


def load_batches(h5ad_path: str) -> dict:
    """Returns {batch_name: (X (n, d) dense float64, cell_type (n,) str array)}.

    X is raw/uncentered -- scrna's NMF requires non-negative input, so
    centering happens per-method (see center_data), not here."""
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
    """Subtract X's own per-gene mean (unsupervised). Used by
    method_ours.py/method_tlgmm.py, not by scrna/gdec -- the two-community
    spectral estimators' hollowed-Gram-matrix method assumes no shared
    non-signal offset, which real uncentered expression data violates
    (verified empirically: every method collapsed to an identical
    near-degenerate prediction without this)."""
    return X - X.mean(axis=0)


def misclustering_error(z_hat: np.ndarray, z_true_str: np.ndarray) -> float:
    """Best-permutation misclustering error for general K, via the Hungarian
    algorithm on the confusion matrix (exact; generalizes the K=2 sign-flip
    and K=3 brute-force-permutation metrics from Experiments 1-3, which
    don't scale to K=13)."""
    true_labels, true_uniques = pd.factorize(z_true_str)
    R_true = len(true_uniques)
    R_hat = int(z_hat.max()) + 1
    R = max(R_true, R_hat)
    confusion = np.zeros((R, R), dtype=int)
    for i in range(len(z_hat)):
        confusion[int(z_hat[i]), true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)  # maximize matches
    correct = confusion[row_ind, col_ind].sum()
    return float(1.0 - correct / len(z_hat))


def compute_metrics(z_hat: np.ndarray, z_true_str: np.ndarray) -> dict:
    true_labels, _ = pd.factorize(z_true_str)
    return dict(
        misclustering=misclustering_error(z_hat, z_true_str),
        ari=adjusted_rand_score(true_labels, z_hat),
        v_measure=v_measure_score(true_labels, z_hat),
    )

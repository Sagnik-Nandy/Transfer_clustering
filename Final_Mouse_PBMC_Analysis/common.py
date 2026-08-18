"""Shared infrastructure for the mouse PBMC (Han et al. 2018 Mouse Cell
Atlas, peripheral blood subset) method comparison: config constants, data
loading, and metrics. Mirrors Final_Lung_Atlas_Analysis/common.py's
structure and conventions exactly -- see preprocess.py for how
data/mouse_pbmc_hvg_lognorm.h5ad was produced from the raw MoDaH_Data
source (log-normalized top-2000-HVG expression, no PCA reduction, unlike
the MoDaH paper's own pipeline for this dataset).

Batch sizes here are far more extreme than the lung atlas's (135 to 3201,
a 24x spread vs. the lung atlas's ~1.5x), and several cell types are
entirely absent from some batches -- see preprocess.py's printed
batch x cell_type table. This was the paper's own stated reason for
including this dataset: "sample sizes across different batches vary
drastically and certain cell types are missing in some batches."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, v_measure_score

K = 9  # number of cell types (B cell, Basophil, Dendritic cell, Erythroblast,
       # Macrophage, Monocyte, NK cell, Neutrophil, T cell)
BATCHES = [
    "PeripheralBlood_1", "PeripheralBlood_2", "PeripheralBlood_3",
    "PeripheralBlood_4", "PeripheralBlood_5", "PeripheralBlood_6",
]
RANDOM_STATE = 0

METHOD_LABELS = {
    "target_only": "Target-only",
    "multi_source_pooled": "Multi-source pooled",
    "pooled_concat": "Pooled (concat)",
    "target_source_pooled": "Target+source pooled",
    "adaptive_multi_source": "Adaptive multi-source",
    "tlgmm": "TL-GMM",
    "scrna": "NMF",
    "gdec_gcnfree": "GDEC",
}


def load_batches(h5ad_path: str) -> dict:
    """Returns {batch_name: (X (n, d) dense float64, cell_type (n,) str array)}.

    X is log-normalized (see preprocess.py) but NOT centered -- scrna's NMF
    requires non-negative input, so centering happens per-method (see
    center_data), not here. Matches Final_Lung_Atlas_Analysis/common.py's
    convention exactly."""
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
    """Subtract X's own per-gene mean (unsupervised)."""
    return X - X.mean(axis=0)


def misclustering_error(z_hat: np.ndarray, z_true_str: np.ndarray) -> float:
    """Best-permutation misclustering error via the Hungarian algorithm."""
    true_labels, true_uniques = pd.factorize(z_true_str)
    R_true = len(true_uniques)
    R_hat = int(z_hat.max()) + 1
    R = max(R_true, R_hat)
    confusion = np.zeros((R, R), dtype=int)
    for i in range(len(z_hat)):
        confusion[int(z_hat[i]), true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    correct = confusion[row_ind, col_ind].sum()
    return float(1.0 - correct / len(z_hat))


def compute_metrics(z_hat: np.ndarray, z_true_str: np.ndarray) -> dict:
    true_labels, _ = pd.factorize(z_true_str)
    return dict(
        misclustering=misclustering_error(z_hat, z_true_str),
        ari=adjusted_rand_score(true_labels, z_hat),
        v_measure=v_measure_score(true_labels, z_hat),
    )

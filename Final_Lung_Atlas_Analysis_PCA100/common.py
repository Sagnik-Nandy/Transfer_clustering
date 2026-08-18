"""Shared infrastructure for the lung atlas (GSE130148) method comparison,
PCA-100 variant: config constants, data loading, PCA projection, and
metrics. Kept deliberately small and separate from each method's own script
(method_ours.py, method_tlgmm.py, method_gdec.py) -- these things (what
K/batches we use, how we load and project data, how we score a result)
must be IDENTICAL across every method for the comparison to be fair, so
they live here once rather than being duplicated (and risking drift) in
each method script.

Unlike Final_Lung_Atlas_Analysis/ (which clusters on the raw d=5000 HVG
matrix), every method here runs on a shared 100-dim PCA representation --
approximating a typical Seurat-style pipeline (normalize -> HVG -> PCA ->
cluster on top PCs), rather than clustering on the full gene space
directly. The PCA basis is target-specific: for a given target batch, the
top 100 principal components are fit on that target's own (centered) data,
and every source batch is projected onto that same basis (see
`project_to_target_pcs`) -- one shared 100-dim space per target, rather
than each batch getting its own independently-rotated basis (which would
make cross-batch comparisons/pooling meaningless).

scrna (NMF) is dropped from this comparison: NMF requires non-negative
input, which PCA-projected data (mean-centered, real-valued) violates.

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
N_PCA = 100  # number of leading target-fit principal components every method clusters on

METHOD_LABELS = {
    "target_only": "Target-only",
    "multi_source_pooled": "Multi-source pooled",
    "pooled_concat": "Pooled (concat)",
    "target_source_pooled": "Target+source pooled",
    "adaptive_multi_source": "Adaptive multi-source",
    "tlgmm": "TL-GMM",
    "gdec_gcnfree": "GDEC",
}


def load_batches(h5ad_path: str) -> dict:
    """Returns {batch_name: (X (n, d) dense float64, cell_type (n,) str array)}.

    X is raw/uncentered gene expression (d=5000 HVGs) -- see
    `project_to_target_pcs` for the PCA-100 reduction applied before any
    method runs."""
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
    """Subtract X's own per-column mean (unsupervised). Used by
    method_ours.py/method_tlgmm.py, not by gdec -- the two-community
    spectral estimators' hollowed-Gram-matrix method assumes no shared
    non-signal offset. Applied both here (to raw gene expression, inside
    `project_to_target_pcs`) and again downstream (to the already-centered
    PCA scores, where it is a no-op) by each method's own wrapper."""
    return X - X.mean(axis=0)


def project_to_target_pcs(X_target: np.ndarray, X_others: list[np.ndarray],
                           n_components: int = N_PCA) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit a PCA basis (top `n_components` right singular vectors) on the
    target's own centered data, then project the target and every other
    (source) dataset onto that same basis -- one shared n_components-dim
    space per target, so cross-batch methods (pooled_concat,
    multi_source_pooled, target_source_pooled) compare/combine vectors in a
    common coordinate system rather than each batch's own independently-
    rotated PCs.

    Each dataset is centered by its OWN mean before projecting (matching
    `center_data`'s existing per-dataset convention elsewhere in this
    pipeline), not by the target's mean -- only the projection directions
    (not the centering offset) are shared. Both the returned target and
    source arrays are therefore already exactly zero-mean in every column
    (mean of a centered matrix times any fixed basis is exactly zero), so
    downstream `center_data` calls on them are harmless no-ops.
    """
    Xc_target = center_data(X_target)
    n_target = Xc_target.shape[0]
    if n_components > n_target:
        raise ValueError(f"n_components={n_components} exceeds target n={n_target}")
    _, _, Vt = np.linalg.svd(Xc_target, full_matrices=False)
    V = Vt[:n_components].T  # (d, n_components)

    X_target_reduced = Xc_target @ V
    X_others_reduced = [center_data(X_other) @ V for X_other in X_others]
    return X_target_reduced, X_others_reduced


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

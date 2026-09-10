"""scRNA (nicococo/scRNA, cloned into External_Methods/scRNA): NMF-based
transfer clustering. Fits NMF on the (pooled) source data to get a gene
dictionary, then uses `DaNmfClustering` to reconstruct/transfer the target
data through that dictionary (mixed with the target's own raw data via
`mix`) and clusters the result.

No native multi-source mechanism in the original method (it's 1-source +
1-target) -- we pool all 5 other batches into a single combined source
dataset, the same simplification used for GDEC.

Requires `cvxopt` and the scRNA package's own dependencies in the
`transfer_clustering` conda env (not installed by default):
    pip install cvxopt
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "External_Methods", "scRNA"))

from common import K

SCRNA_MIX = 0.5  # weight on the source-reconstructed target data vs. raw target data

# NMF hyperparameters, taken directly from the paper's own published
# real-data experiment (Mieth et al., External_Methods/scRNA/scripts/
# experiments/main_wrapper_hockley.py: nmf_alpha/nmf_l1/nmf_max_iter),
# rather than NmfClustering.apply/DaNmfClustering.apply's own much weaker
# defaults (alpha=1.0, max_iter=100): at max_iter=100 the Lee-Seung
# multiplicative-update NMF is likely nowhere near converged on data with
# thousands of genes, risking degenerate (ari=0.0) fits. mix itself stays
# fixed at SCRNA_MIX (0.5) -- the paper's own mix-sweep-plus-unsupervised-
# KTA-selection glue code (experiments_utils.py) isn't in the public
# scRNA repo, so it isn't reproduced here.
SCRNA_NMF_ALPHA = 10.0
SCRNA_NMF_L1 = 0.75
SCRNA_NMF_MAX_ITER = 4000


def method_scrna(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    from scRNA.nmf_clustering import NmfClustering, DaNmfClustering

    X_S_pooled = np.vstack(sources)
    gene_ids = np.array([f"g{i}" for i in range(X_T.shape[1])])

    src = NmfClustering(X_S_pooled.T, gene_ids=gene_ids, num_cluster=K, labels=[])
    src.apply(k=K, alpha=SCRNA_NMF_ALPHA, l1=SCRNA_NMF_L1, max_iter=SCRNA_NMF_MAX_ITER)

    trg = DaNmfClustering(src, X_T.T, gene_ids, num_cluster=K)
    trg.apply(k=K, mix=SCRNA_MIX, alpha=SCRNA_NMF_ALPHA, l1=SCRNA_NMF_L1, max_iter=SCRNA_NMF_MAX_ITER)
    return np.asarray(trg.cluster_labels)

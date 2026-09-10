"""Our four methods for the Table 1 comparison (Section 6), all already
implemented in the transfer_clustering package (multi_cluster.py) -- this
file is just thin wrappers giving each one the uniform
(X_T, sources, seed) -> labels signature the other method scripts share,
for a uniform driver in the notebook.

  1. target_only               -- relaxed-K-means target-only estimator
                                   (Algorithm 3's target branch); Table 1's
                                   "Target-only" column.
  2. multi_source_pooled       -- source-projection estimator (all 5 other
                                   batches' combined subspace), Algorithm
                                   3's source branch; Table 1's
                                   "Multi-source pooled" column.
  3. target_source_pooled_capped -- target+source pooled-subspace estimator
                             (`target_source_pooled_subspace_estimate` with
                             `restrict_basis_rank=True`, matching the
                             paper's Algorithm 4): the target's own
                             estimated mean matrix is pooled into the
                             projection subspace alongside the sources',
                             instead of being reserved for a separate
                             target-only branch, and the pooled subspace's
                             rank is capped at K instead of the uncapped
                             (m+1)*K, so a weakly-estimated source's mean
                             matrix can't add pure-noise directions to the
                             space TSClust clusters in. Table 1's "Pooled"
                             column.
  4. adaptive_multi_source -- bootstrap-adaptive target/source switch
                               (Algorithm 3); Table 1's "Adaptive
                               multi-source" column.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from transfer_clustering.multi_cluster import (
    AdaptiveProjectedClustering,
    onehot_to_labels,
    _target_branch,
    target_source_pooled_subspace_estimate,
)

from common import K, center_data

ADAPTIVE_N_BOOT = 30
ADAPTIVE_ALPHA = 0.5
ADAPTIVE_BOUNDARY_SCALE = 1.0

# All methods route through RelaxedKMeans somewhere (target_only directly;
# the others via estimate_source_means, since every Dropseq batch has
# n < d=5000, hitting RelaxedKMeans's n_S < d branch). ADMM (Mixon, Villar
# & Ward 2017 -- see relaxed_kmeans.py) is opted into HERE, per-call,
# matching RelaxedKMeans's own default, so this stays explicit even if that
# shared default changes later: it needs no external SDP solver and is
# cheap at these batch sizes (n_T up to ~3183, K=13).
RELAXED_KMEANS_KWARGS = {"solver": "ADMM"}


def method_target_only(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    # seed threaded into RelaxedKMeans's K-means rounding step so this is
    # reproducible and (crucially) numerically matches adaptive_multi_source
    # whenever it picks the "target" branch on the same data/seed -- see
    # AdaptiveProjectedClustering.fit_predict's rk_kwargs handling.
    Z = _target_branch(center_data(X_T), K, {**RELAXED_KMEANS_KWARGS, "random_state": seed})
    return onehot_to_labels(Z)


def method_multi_source_pooled(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    sources_c = [center_data(X_S) for X_S in sources]
    model = AdaptiveProjectedClustering(K=K, selection="manual", manual_choice="source",
                                         relaxed_kmeans_kwargs=RELAXED_KMEANS_KWARGS,
                                         random_state=seed)
    Z, _branch = model.fit_predict(center_data(X_T), sources_c)
    return onehot_to_labels(Z)


def method_target_source_pooled_capped(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    sources_c = [center_data(X_S) for X_S in sources]
    Z = target_source_pooled_subspace_estimate(
        center_data(X_T), K, sources_c,
        relaxed_kmeans_kwargs=RELAXED_KMEANS_KWARGS, random_state=seed,
        restrict_basis_rank=True,
    )
    return onehot_to_labels(Z)


def method_adaptive_multi_source(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    sources_c = [center_data(X_S) for X_S in sources]
    model = AdaptiveProjectedClustering(
        K=K, selection="bootstrap", alpha=ADAPTIVE_ALPHA, n_boot=ADAPTIVE_N_BOOT,
        boundary_scale=ADAPTIVE_BOUNDARY_SCALE, random_state=seed,
        relaxed_kmeans_kwargs=RELAXED_KMEANS_KWARGS,
    )
    Z, _branch = model.fit_predict(center_data(X_T), sources_c)
    return onehot_to_labels(Z)


OURS_METHODS = {
    "target_only": method_target_only,
    "multi_source_pooled": method_multi_source_pooled,
    "target_source_pooled_capped": method_target_source_pooled_capped,
    "adaptive_multi_source": method_adaptive_multi_source,
}

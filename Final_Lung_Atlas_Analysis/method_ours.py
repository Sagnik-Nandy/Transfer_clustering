"""Our five methods, all already implemented in the transfer_clustering
package (Python_Scripts/transfer_clustering/multi_cluster.py) -- this file
is just thin wrappers giving each one the uniform (X_T, sources, seed) ->
labels signature the other method scripts share, for a uniform driver in
the notebook.

  1. target_only         -- relaxed-K-means target-only estimator
  2. multi_source_pooled -- source-projection estimator (all 3 sources'
                             combined subspace), "manual" branch
  3. pooled_concat       -- pooled estimator (Experiment 1's
                             `pooled_subspace_estimate`): target+all
                             sources' raw data concatenated into one
                             matrix, clustered directly with relaxed
                             K-means, keeping only the target rows'
                             labels
  4. target_source_pooled -- target+source pooled-subspace estimator
                             (`target_source_pooled_subspace_estimate`):
                             like multi_source_pooled, but the target's own
                             estimated mean matrix is pooled into the
                             projection subspace alongside the sources',
                             instead of being reserved for a separate
                             target-only branch. Not part of the discussion
                             draft; a user-specified extension.
  5. adaptive_multi_source -- bootstrap-adaptive target/source switch

NOTE on cost: pooled_concat feeds RelaxedKMeans an SDP over ~9,900
concatenated rows (vs. ~3,183 for target_only alone), and the SDP's cubic
cost in n makes this the single most expensive call in the whole
comparison -- included here at explicit request despite that cost; see
run_lung_atlas_comparison.sh's time budget.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Experiments_Script"))

from transfer_clustering.multi_cluster import (
    AdaptiveProjectedClustering,
    onehot_to_labels,
    _target_branch,
    pooled_subspace_estimate,
    target_source_pooled_subspace_estimate,
)

from common import K, center_data

ADAPTIVE_N_BOOT = 30
ADAPTIVE_ALPHA = 0.5
ADAPTIVE_BOUNDARY_SCALE = 1.0

# All four methods route through RelaxedKMeans somewhere (target_only
# directly; the other three via estimate_source_means, since every
# Dropseq batch has n < d=5000, hitting RelaxedKMeans's n_S < d branch).
# ADMM (Mixon, Villar & Ward 2017 -- see relaxed_kmeans.py) is opted into
# HERE, per-call, matching RelaxedKMeans's own new default, so this stays
# explicit even if that shared default changes later. Previously pinned to
# MOSEK (this machine's license, unavailable on the OSU cluster); ADMM
# needs no external SDP solver at all and is dramatically cheaper at these
# batch sizes (n_T up to ~3183, K=13).
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


def method_pooled_concat(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    sources_c = [center_data(X_S) for X_S in sources]
    Z = pooled_subspace_estimate(
        center_data(X_T), K, sources_c,
        relaxed_kmeans_kwargs={**RELAXED_KMEANS_KWARGS, "random_state": seed},
    )
    return onehot_to_labels(Z)


def method_target_source_pooled(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    sources_c = [center_data(X_S) for X_S in sources]
    Z = target_source_pooled_subspace_estimate(
        center_data(X_T), K, sources_c,
        relaxed_kmeans_kwargs=RELAXED_KMEANS_KWARGS, random_state=seed,
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
    "pooled_concat": method_pooled_concat,
    "target_source_pooled": method_target_source_pooled,
    "adaptive_multi_source": method_adaptive_multi_source,
}

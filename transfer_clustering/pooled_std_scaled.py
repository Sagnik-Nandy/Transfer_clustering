"""Std-scaled pooled-subspace estimators.

An alternative to `two_community.target_source_pooled_subspace_estimate`
(K=2) and `multi_cluster.target_source_pooled_subspace_estimate` (K>2)
that rescales each candidate direction -- target's own estimated
direction/mean-matrix and every source's -- by the *observed standard
deviation of that direction's own projection of X_T*, instead of by its
raw magnitude (the existing default) or by forcing it to unit norm (the
existing `normalize=True`).

Motivation. theta_hat_T is estimated from X_T itself, so X_T @
theta_hat_T_unit has an inflated in-sample variance purely from this
self-reference (theta_hat_T is partly "measuring" X_T's own noise
realization), not from real signal -- and that inflation is not the same
thing as theta_hat_T's raw vector norm, so unit-norming it does not
remove it. A source's direction, estimated from an independent dataset,
has no such inflation: its projection variance on X_T reflects only
genuine cross-correlation. Dividing each candidate by the realized
standard deviation of its own X_T-projection puts every candidate on a
directly comparable, empirically-measured scale before they are pooled
into one subspace via the existing SVD-based pooling step -- rather than
a theoretical noise-floor correction (d/n, a BBP inversion, ...), which
would need a different derivation for every estimation branch.

Applied identically regardless of regime, aspect ratio, or number of
sources -- there is no per-regime branching here, and no assumption that
sources share weight evenly: the actual combination weights are whatever
the existing SVD-based pooling (`source_subspace_basis` /
`source_subspace_projector`) derives from the (now comparably-scaled)
column correlations, exactly as in the original estimators.

This module only ADDS new estimators. It imports from `two_community.py`
and `multi_cluster.py` but does not modify them, and no existing
pipeline, script, or result is touched by adding this file.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .two_community import (
    estimate_source_direction, source_subspace_basis, _leading_eigvec_direction,
)
from .multi_cluster import (
    estimate_source_means, source_subspace_projector, labels_to_onehot,
)
from .ts_clust import ts_clust


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------


def std_scaled_direction(X_T: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Rescale w (a single direction estimate -- target's or a source's) so
    that X_T @ (w / ||w||) -- w's own projection of the target data -- has
    unit standard deviation across X_T's n_T rows.

    Returns w's unit direction divided by that observed standard
    deviation, i.e. a vector pointing the same way as w but whose length
    now encodes "how much of X_T's own spread this direction accounts
    for" rather than w's raw estimation-formula magnitude. Falls back to
    the plain unit vector if the projection is (numerically) constant.
    """
    w_unit = w / np.linalg.norm(w)
    s = np.std(X_T @ w_unit)
    if s < 1e-12:
        return w_unit
    return w_unit / s


# ---------------------------------------------------------------------------
# K=2 (two-community) case -- experiments 1 and 2
# ---------------------------------------------------------------------------


def target_source_pooled_stdscaled_estimate(
    X_T: np.ndarray, source_datasets: Sequence[np.ndarray], restrict_basis_rank: bool = False,
) -> np.ndarray:
    """K=2 std-scaled pooled-subspace estimator.

    Same skeleton as `two_community.target_source_pooled_subspace_estimate`:

        1. estimate theta_hat_T from X_T and theta_hat_S_i from each
           source, via the existing (unmodified) `estimate_source_direction`,
        2. rescale EVERY one of these m+1 directions via
           `std_scaled_direction` (divide by the observed std of its own
           projection of X_T) -- applied uniformly, the same way for the
           target and for every source, regardless of which estimation
           branch produced it,
        3. pool the rescaled directions into a joint subspace Q via the
           existing `source_subspace_basis` (with `normalize=False`,
           since the rescaling above already put every column on a
           comparable scale -- letting `source_subspace_basis` additionally
           force unit norm would discard that rescaling),
        4. project X_T onto Q and cluster by the sign of the leading
           eigenvector of the projected Gram matrix, exactly as in the
           original estimator.

    If `restrict_basis_rank` is True, Q is capped at a single leading
    direction (as in the original estimator); which raw direction
    dominates that cap is now decided by the std-rescaling instead of raw
    magnitude or forced unit-norm.
    """
    theta_hat_T = estimate_source_direction(X_T)
    directions = [theta_hat_T] + [estimate_source_direction(X_S) for X_S in source_datasets]
    scaled_directions = [std_scaled_direction(X_T, w) for w in directions]

    max_rank = 1 if restrict_basis_rank else None
    Q = source_subspace_basis(scaled_directions, max_rank=max_rank, normalize=False)

    X_hat_T = X_T @ Q
    v_hat = _leading_eigvec_direction(X_hat_T)
    scores = X_hat_T @ v_hat
    labels = np.sign(scores)
    labels[labels == 0] = 1.0
    return labels


# ---------------------------------------------------------------------------
# K>2 (multi-cluster) case -- experiment 3
# ---------------------------------------------------------------------------


def std_scaled_mean_matrix(X_T: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Column-wise analogue of `std_scaled_direction`: rescales each of
    Theta's K columns independently (mirrors `source_subspace_projector`'s
    own per-column, not per-matrix, normalization convention), so that
    X_T @ (column / ||column||) has unit standard deviation for every
    column of every mean matrix.
    """
    d, K = Theta.shape
    out = np.empty_like(Theta, dtype=float)
    for k in range(K):
        out[:, k] = std_scaled_direction(X_T, Theta[:, k])
    return out


def target_source_pooled_stdscaled_estimate_multi(
    X_T: np.ndarray,
    K: int,
    source_datasets: Sequence[np.ndarray],
    relaxed_kmeans_kwargs: Optional[Dict] = None,
    random_state: Optional[int] = None,
    restrict_basis_rank: bool = False,
) -> np.ndarray:
    """K>2 std-scaled pooled-subspace estimator, analogue of
    `multi_cluster.target_source_pooled_subspace_estimate` for K>2:

        1. estimate Theta_hat_T (d, K) from X_T and Theta_hat_S_i from
           each source, via the existing (unmodified)
           `estimate_source_means`,
        2. rescale EVERY column of EVERY one of these m+1 mean matrices
           via `std_scaled_mean_matrix` -- applied uniformly, the same
           way for the target and for every source,
        3. pool the rescaled mean matrices into a joint subspace R via
           the existing `source_subspace_projector` (with
           `normalize=False`, for the same reason as in the K=2 case),
        4. project X_T onto R and cluster with TSClust, exactly as in
           the original estimator.

    If `restrict_basis_rank` is True, R is capped at its K leading
    singular directions (as in the original estimator); the std-rescaling
    -- not raw magnitude or forced unit-column-norm -- decides which
    columns dominate that cap.
    """
    rk_kwargs = dict(relaxed_kmeans_kwargs or {})
    rk_kwargs.setdefault("random_state", random_state)
    n_T = X_T.shape[0]
    theta_hat_T = estimate_source_means(X_T, K, rk_kwargs)
    theta_hats = [theta_hat_T] + [
        estimate_source_means(X_S, K, rk_kwargs) for X_S in source_datasets
    ]
    scaled_theta_hats = [std_scaled_mean_matrix(X_T, th) for th in theta_hats]

    max_rank = K if restrict_basis_rank else None
    Q_R = source_subspace_projector(scaled_theta_hats, max_rank=max_rank, normalize=False)

    X_hat_T = X_T @ Q_R
    n_iter = int(round(2 * np.log(n_T)))
    labels = ts_clust(X_hat_T, K, n_iter=n_iter, random_state=random_state)
    return labels_to_onehot(labels, K)

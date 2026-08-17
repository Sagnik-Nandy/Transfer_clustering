"""Algorithm 3 (TSClust): Spectral Initialization Followed by Lloyd Refinement.

From the transfer-learning-for-clustering discussion draft, Section 4.2.
Used as the clustering module applied to the source-projected target
observations in Algorithm 2.

Steps (matching Algorithm 3 exactly):
    1. Form the data matrix, take its top-K singular directions.
    2. Project each point onto those directions to get spectral
       coordinates M_hat_j in R^K.
    3. Initialize via a (1+eps)-approximate K-means solution on the
       spectral coordinates (K-means++, eq. in Algorithm 3 line 5).
    4. Run T Lloyd iterations (recompute centers in the *original*
       feature space, reassign to nearest center) -- lines 6-10.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _spectral_coordinates(X: np.ndarray, K: int) -> np.ndarray:
    """Top-K singular-direction projection: the 'U_hat' of Algorithm 3.

    X is (n, p) with rows as data points. The top-K left singular vectors
    of the paper's (p, n) data matrix correspond exactly to the top-K
    right singular vectors of our (n, p) row-convention matrix.
    """
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U_hat = Vt[:K].T          # (p, K)
    return X @ U_hat           # spectral coordinates, (n, K)


def _kmeanspp_init(M: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ seeding (Arthur & Vassilvitskii, 2007): pick the first
    center uniformly, then each subsequent center with probability
    proportional to its squared distance to the nearest center chosen so
    far. Returns the (K, dim) initial centers."""
    n = M.shape[0]
    centers = np.empty((K, M.shape[1]))
    centers[0] = M[rng.integers(n)]
    closest_sq = np.sum((M - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        probs = closest_sq / closest_sq.sum()
        centers[k] = M[rng.choice(n, p=probs)]
        new_sq = np.sum((M - centers[k]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def _lloyd(M: np.ndarray, centers: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Plain Lloyd's algorithm from a fixed set of initial centers."""
    K = centers.shape[0]
    labels = np.argmin(
        np.linalg.norm(M[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    for _ in range(n_iter):
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = M[mask].mean(axis=0)
        labels = np.argmin(
            np.linalg.norm(M[:, None, :] - centers[None, :, :], axis=2), axis=1
        )
    return labels


def _kmeanspp_labels_numpy(
    M: np.ndarray, K: int, n_init: int = 10, random_state: Optional[int] = None
) -> np.ndarray:
    """Pure-numpy K-means++ fallback (used only if sklearn isn't installed,
    see `_kmeanspp_labels`): `n_init` restarts of k-means++ seeding followed
    by Lloyd's algorithm, keeping the restart with lowest inertia."""
    rng = np.random.default_rng(random_state)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = _kmeanspp_init(M, K, rng)
        labels = _lloyd(M, centers)
        inertia = sum(
            np.sum((M[labels == k] - M[labels == k].mean(axis=0)) ** 2)
            for k in range(K) if np.any(labels == k)
        )
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def _kmeanspp_labels(
    M: np.ndarray, K: int, n_init: int = 10, random_state: Optional[int] = None
) -> np.ndarray:
    """K-means++ + Lloyd on M. Prefers sklearn's (tested, optimized)
    KMeans; falls back to `_kmeanspp_labels_numpy` only if sklearn isn't
    installed."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return _kmeanspp_labels_numpy(M, K, n_init=n_init, random_state=random_state)

    model = KMeans(n_clusters=K, n_init=n_init, random_state=random_state)
    return model.fit_predict(M)


def ts_clust(
    X: np.ndarray,
    K: int,
    n_iter: int,
    n_init: int = 10,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Algorithm 3: spectral initialization (K-means++ on the top-K
    spectral coordinates) followed by `n_iter` Lloyd refinement steps in
    the original feature space.

    Parameters
    ----------
    X : (n, p) array
        Rows are the data points to cluster.
    K : int
        Number of clusters.
    n_iter : int
        Number of Lloyd refinement iterations (T in Algorithm 3; the
        callers in Algorithm 2 use T = round(2 log n_T)).
    """
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    M = _spectral_coordinates(X, K)
    labels = _kmeanspp_labels(M, K, n_init=n_init, random_state=random_state)

    for _ in range(n_iter):
        centers = np.zeros((K, X.shape[1]))
        for a in range(K):
            mask = labels == a
            if mask.any():
                centers[a] = X[mask].mean(axis=0)
            else:
                centers[a] = X[rng.integers(n)]
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

    return labels

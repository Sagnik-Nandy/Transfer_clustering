"""Ndaoud (2018) hollowed-Gram-matrix spectral clustering.

Reference: M. Ndaoud, "Sharp optimal recovery in the two component
Gaussian mixture model," arXiv:1812.08078. The estimator below is
eq. (12) (spectral initializer) composed with the sign-power-iteration
sequence of eq. (15), run for floor(3 log n) steps as in Theorem 4.

This is exactly the routine used for the target-based estimator in
Section 2.2 and in lines 2-4 of Algorithm 1 of Chakraborty & Nandy,
"Transfer Learning in High-Dimensional Clustering: Minimax Thresholds and
Applications in Single-Cell Data" (where it is called GOOD-CLUSTERER).
"""
from __future__ import annotations

import numpy as np


def hollow(gram: np.ndarray) -> np.ndarray:
    """The operator H(M) = M - diag(M): zero out the diagonal of a square matrix."""
    out = np.array(gram, copy=True)
    np.fill_diagonal(out, 0.0)
    return out


def top_eigenvector(sym_matrix: np.ndarray) -> np.ndarray:
    """Unit eigenvector associated with the largest eigenvalue of a symmetric matrix."""
    eigvals, eigvecs = np.linalg.eigh(sym_matrix)
    return eigvecs[:, np.argmax(eigvals)]


def _sign_no_zero(v: np.ndarray) -> np.ndarray:
    s = np.sign(v)
    s[s == 0] = 1.0
    return s


def spectral_sign_clustering(X: np.ndarray, n_iter: int | None = None) -> np.ndarray:
    """Ndaoud's (2018) rate-optimal two-community clustering routine.

    Given n observations X (n x d) following X_j = eta_j * theta + noise_j,
    returns eta_hat in {-1, 1}^n (identifiable up to a global sign flip):

        1. hollowed Gram matrix  B = H(X X^T)
        2. spectral init         eta^0 = sign(leading eigenvector of B)
        3. sign power iteration  eta^{k+1} = sign(B eta^k),
           for k = floor(3 log n) steps (default; matches Theorem 4).

    Parameters
    ----------
    X : (n, d) array
        Rows are the observations.
    n_iter : int, optional
        Number of sign-power-iteration steps. Defaults to floor(3 log n).
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations to cluster.")
    if n_iter is None:
        n_iter = int(np.floor(3 * np.log(n)))
    n_iter = max(n_iter, 0)

    gram = X @ X.T
    B = hollow(gram)

    eta = _sign_no_zero(top_eigenvector(B))
    for _ in range(n_iter):
        eta = _sign_no_zero(B @ eta)
    return eta

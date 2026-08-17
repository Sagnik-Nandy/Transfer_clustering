"""Relaxed K-means clustering via the SDP of Peng & Wei.

Reference: C. Giraud and N. Verzelen, "Partial recovery bounds for
clustering with the relaxed K-means," arXiv:1807.07547 (Section 2).

Peng and Wei showed that the (NP-hard) K-means criterion can be written
as maximizing <X X^T, B> over the set of partition matrices

    P = {B in R^{n x n} : symmetric, B^2 = B, Tr(B) = K, B 1 = 1, B >= 0},

and relaxed this to the convex set

    C = {B in R^{n x n} : PSD, Tr(B) = K, B 1 = 1, B >= 0}       (eq. 6)

giving the tractable "relaxed K-means" SDP

    B_hat in argmax_{B in C} <X X^T, B>.

Solving this SDP requires a general-purpose SDP solver; this module uses
`cvxpy` (not part of the base numpy/scipy stack) and raises a clear error
if it is not installed.

The exact rounding step in the paper (a rho-approximate K-medoids solution
on the rows of B_hat, via the polynomial-time algorithm of Charikar,
Guha, Tardos & Shmoys 2002, eq. 7) is a nontrivial combinatorial
procedure in its own right. As a practical surrogate, we round by running
K-means on the rows of B_hat directly -- this is the simplification most
practical implementations of this SDP use, but it is *not* the exact
CGTS02 guarantee from the paper.

Three ways to obtain a target-branch clustering are provided, selected via
`RelaxedKMeans.solver`:

  - "ADMM" (default): a purpose-built alternating-projection ADMM for
    exactly this constraint set, following Mixon, Villar & Ward (2017,
    "Clustering subgaussian mixtures by semidefinite programming,"
    Information and Inference: A Journal of the IMA, arXiv:1602.06612).
    Each iteration alternates a projection onto {PSD, Tr(B)=K} (eigen-
    decompose, then project the eigenvalue vector onto the radius-K
    simplex) with a projection onto {B >= 0, B @ 1 = 1} (project each row
    onto the probability simplex). Both simplex projections use the
    O(n log n) algorithm of Duchi, Shalev-Shwartz, Singer & Chandra (2008,
    "Efficient Projections onto the L1-Ball for Learning in High
    Dimensional Data," ICML). This solves the *same* SDP as the cvxpy
    path below, just without a generic conic solver's canonicalization/
    embedding overhead -- each iteration is a handful of BLAS/LAPACK
    calls instead of a full interior-point or operator-splitting step
    over a lifted cone program, which is what makes it dramatically
    cheaper at the problem sizes these experiments use.
  - any cvxpy solver name (e.g. "SCS", "MOSEK"): solves the SDP exactly
    as originally, via cvxpy -- kept as a slower but independent
    cross-check of the ADMM path.
  - "ts_clust": skips the SDP entirely and clusters X directly with
    `ts_clust` (Algorithm 3 -- spectral initialization on X's own top-K
    singular directions, then Lloyd refinement), exactly the routine
    `_source_branch` in multi_cluster.py already applies to the
    source-projected target observations. This is not a rounding of the
    SDP relaxation at all, so it is a genuinely different (much cheaper,
    but not Peng-Wei-SDP-backed) estimator, offered as a practical
    fallback rather than a faithful reimplementation of Algorithm 2's
    target branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ADMM solver (Mixon, Villar & Ward, 2017) -- see module docstring.
# ---------------------------------------------------------------------------


def _project_row_simplex(X: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Euclidean projection of each row of X onto the simplex
    {x : x >= 0, sum(x) = radius} (Duchi, Shalev-Shwartz, Singer & Chandra,
    2008, Algorithm 1). Used both for the row projection (radius=1) and,
    with X reshaped to a single row, for the eigenvalue projection
    (radius=K) in `_project_psd_trace`."""
    n = X.shape[1]
    U = np.sort(X, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1)
    idx = np.arange(1, n + 1)
    cond = U - (css - radius) / idx > 0
    rho = cond.sum(axis=1) - 1
    theta = (css[np.arange(X.shape[0]), rho] - radius) / (rho + 1)
    return np.maximum(X - theta[:, None], 0.0)


def _project_psd_trace(X: np.ndarray, K: int) -> np.ndarray:
    """Euclidean projection of a symmetric matrix onto {B : B PSD, Tr(B) = K}:
    eigendecompose, project the eigenvalue vector onto the radius-K simplex
    (via `_project_row_simplex`), and reassemble."""
    X_sym = (X + X.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(X_sym)
    eigvals_proj = _project_row_simplex(eigvals[None, :], radius=float(K))[0]
    return (eigvecs * eigvals_proj) @ eigvecs.T


def _solve_relaxed_kmeans_admm(
    gram: np.ndarray, K: int, rho: float = 1.0, max_iter: int = 500, tol: float = 1e-4,
) -> np.ndarray:
    """ADMM solver for the relaxed K-means SDP (eq. 6):

        max_B <gram, B>   s.t.   B PSD, Tr(B) = K, B @ 1 = 1, B >= 0.

    Splits B (kept PSD with Tr(B)=K) from a copy C (kept row-stochastic and
    nonnegative), consensus-linked by a scaled dual variable U:

        B <- Proj_{PSD, Tr=K}(C - U + gram_scaled / rho)
        C <- Proj_{rows sum to 1, >= 0}(B + U)
        U <- U + B - C

    (Mixon, Villar & Ward 2017; see module docstring.) gram is rescaled by
    1/max|gram| first, for the same conditioning reason as the cvxpy path
    (see `_solve_relaxed_kmeans_sdp`'s docstring) -- it does not change the
    argmax. Stops early once both the primal residual ||B-C|| and the
    (rho-scaled) dual residual ||C - C_prev|| fall below tol*n.
    """
    n = gram.shape[0]
    scale = 1.0 / max(np.abs(gram).max(), 1e-12)
    A = gram * scale

    B = np.eye(n) * (K / n)
    C = np.full((n, n), 1.0 / n)
    U = np.zeros((n, n))

    thresh = tol * n
    for _ in range(max_iter):
        B = _project_psd_trace(C - U + A / rho, K)
        C_new = _project_row_simplex(B + U, radius=1.0)
        U = U + B - C_new

        primal_res = np.linalg.norm(B - C_new)
        dual_res = rho * np.linalg.norm(C_new - C)
        C = C_new
        if primal_res < thresh and dual_res < thresh:
            break

    return B


def _solve_relaxed_kmeans_sdp(gram: np.ndarray, K: int, solver: Optional[str] = None,
                               solver_opts: Optional[dict] = None) -> np.ndarray:
    """Solve the relaxed K-means SDP (eq. 6):

        max_B <gram, B>   s.t.   B PSD, Tr(B) = K, B @ 1 = 1, B >= 0.

    NOTE: gram is rescaled by a positive constant (1/max|gram|) before
    solving. This does NOT change the argmax B_hat -- scaling a linear
    objective by a positive constant leaves its maximizer unchanged, so
    this is purely a numerical preconditioning step. Without it, real
    single-cell data (raw log-expression Gram matrices routinely have
    entries in the thousands) causes SCS's default tolerances to
    misreport this provably compact, bounded feasible region (B >= 0
    with row-sums-to-1 already forces every entry of B into [0,1]) as
    "unbounded" -- a solver-scaling artifact, not genuine infeasibility.
    For the same reason (larger, harder-to-precondition problems at real
    data scale), SCS's iteration budget is also raised from its default.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "RelaxedKMeans requires the 'cvxpy' package to solve the SDP "
            "relaxation of Peng & Wei (eq. 6 of Giraud & Verzelen, 2019). "
            "Install it with `pip install cvxpy` (and an SDP-capable "
            "solver backend, e.g. SCS, which ships with cvxpy by default)."
        ) from exc

    n = gram.shape[0]
    scale = 1.0 / max(np.abs(gram).max(), 1e-12)
    gram_scaled = gram * scale

    B = cp.Variable((n, n), symmetric=True)
    constraints = [
        B >> 0,
        cp.trace(B) == K,
        B @ np.ones(n) == np.ones(n),
        B >= 0,
    ]
    objective = cp.Maximize(cp.trace(gram_scaled @ B))
    problem = cp.Problem(objective, constraints)

    opts = dict(solver_opts) if solver_opts else {}
    if solver == "SCS":
        opts.setdefault("max_iters", 20000)
    problem.solve(solver=solver, **opts)

    if B.value is None:
        raise RuntimeError(
            f"Relaxed K-means SDP solver failed to converge (status={problem.status})."
        )
    return np.asarray(B.value)


def _round_via_kmeans(
    B_hat: np.ndarray, K: int, n_init: int = 10, random_state: Optional[int] = None
) -> np.ndarray:
    """Practical rounding surrogate for the exact rho-approximate K-medoids
    step (eq. 7): cluster the rows of the SDP solution B_hat with K-means,
    via `ts_clust._kmeanspp_labels` (prefers sklearn's KMeans; falls back
    to a pure-numpy implementation only if sklearn isn't installed).
    """
    from .ts_clust import _kmeanspp_labels
    return _kmeanspp_labels(B_hat, K, n_init=n_init, random_state=random_state)


@dataclass
class RelaxedKMeans:
    """Peng & Wei's relaxed K-means SDP, as analyzed by Giraud & Verzelen
    (2019, Section 2): solve the SDP relaxation of the K-means criterion,
    then round the solution to a hard partition.
    """

    K: int
    # Default to the ADMM solver (Mixon, Villar & Ward 2017 -- see module
    # docstring): same SDP, no generic conic-solver overhead, which is what
    # made the cvxpy/SCS path prohibitively slow for repeated calls (e.g.
    # `calibrate_D0_bootstrap_mult`'s n_boot re-solves). Set solver to any
    # cvxpy solver name (e.g. "SCS", "MOSEK") to go through cvxpy instead
    # (a slower independent cross-check of the ADMM path), or to "ts_clust"
    # to bypass the SDP entirely (see module docstring).
    solver: Optional[str] = "ADMM"
    # ADMM-only knobs (ignored otherwise).
    admm_rho: float = 1.0
    admm_max_iter: int = 500
    admm_tol: float = 1e-4
    # Extra kwargs forwarded to cvxpy's problem.solve() (e.g. {"max_iters": 50000}
    # for even larger problems than the raised default in _solve_relaxed_kmeans_sdp).
    # Ignored when solver == "ADMM" or "ts_clust".
    solver_opts: Optional[dict] = None
    n_init: int = 10
    random_state: Optional[int] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if self.solver == "ts_clust":
            from .ts_clust import ts_clust
            n_iter = int(round(2 * np.log(X.shape[0])))
            return ts_clust(X, self.K, n_iter=n_iter, n_init=self.n_init, random_state=self.random_state)

        gram = X @ X.T
        if self.solver == "ADMM":
            B_hat = _solve_relaxed_kmeans_admm(
                gram, self.K, rho=self.admm_rho, max_iter=self.admm_max_iter, tol=self.admm_tol,
            )
        else:
            B_hat = _solve_relaxed_kmeans_sdp(gram, self.K, solver=self.solver, solver_opts=self.solver_opts)
        return _round_via_kmeans(B_hat, self.K, n_init=self.n_init, random_state=self.random_state)

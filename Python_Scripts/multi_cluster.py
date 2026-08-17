"""Algorithm 2 (Adaptive Transfer-Assisted Projected Clustering).

From the transfer-learning-for-clustering discussion draft, Section 4.2:
multi-cluster (K > 2) Gaussian mixture models with multiple source
datasets. Builds on:

  - `relaxed_kmeans.RelaxedKMeans`   (Giraud & Verzelen, 2019 relaxed K-means SDP)
  - `ts_clust.ts_clust`              (Algorithm 3)

Model (Section 4.2):
    X_j^(T) = Theta_T Z_j^(T) + noise_j^(T),   Z_j^(T) in {0,1}^K one-hot,
    X_j^(Si) = Theta_Si Z_j^(Si) + noise_j^(Si), for each source i.

Algorithm 2 proceeds:
    1. Target-based estimator via relaxed K-means directly on X_T.
    2. For each source i, estimate the source mean matrix Theta_hat_Si
       (eq. 51 if n_Si << d, eq. 52 if 4 <= d <~ n_Si).
    3. Form the source subspace R = col(Theta_hat_S1) + ... + col(Theta_hat_Sm)
       (eq. 45) and project the target observations onto it (eq. 46).
    4. Cluster the projected target observations with TSClust (Algorithm 3)
       to get the source-based estimator.
    5. Adaptively choose between the target- and source-based estimators
       using the validation statistic S_hat(Z) (eq. 47-48) and threshold
       t_n (eq. 49), as in eq. (50).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .relaxed_kmeans import RelaxedKMeans
from .ts_clust import ts_clust
from .selection import bootstrap_null_quantile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def labels_to_onehot(labels: np.ndarray, K: int) -> np.ndarray:
    """Integer label vector (n,) -> one-hot label matrix Z in {0,1}^{n x K}."""
    n = labels.shape[0]
    Z = np.zeros((n, K))
    Z[np.arange(n), labels] = 1.0
    return Z


def onehot_to_labels(Z: np.ndarray) -> np.ndarray:
    """One-hot label matrix Z in {0,1}^{n x K} -> integer label vector (n,)."""
    return np.argmax(Z, axis=1)


def cluster_means(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Empirical cluster means: columns are theta_hat_k = sum_j Z_jk X_j / sum_j Z_jk."""
    counts = Z.sum(axis=0)
    counts = np.where(counts == 0, 1.0, counts)
    return (X.T @ Z) / counts  # (d, K)


# ---------------------------------------------------------------------------
# Source mean / subspace estimation (eq. 45, 51, 52)
# ---------------------------------------------------------------------------


def estimate_source_means(
    X_S: np.ndarray, K: int, relaxed_kmeans_kwargs: Optional[Dict] = None
) -> np.ndarray:
    """Estimate the source mean matrix Theta_hat_S (d, K).

    - n_S < d ("n_Si << d" regime): first estimate the source label matrix
      via relaxed K-means, then set columns to the empirical cluster
      means (eq. 51).
    - n_S >= d ("4 <= d <~ n_Si" regime): estimate the source signal
      subspace directly as the top-K right singular vectors of X_S (eq. 52).
    """
    n_S, d = X_S.shape
    if n_S < d:
        rk = RelaxedKMeans(K=K, **(relaxed_kmeans_kwargs or {}))
        labels = rk.fit_predict(X_S)
        Z_hat = labels_to_onehot(labels, K)
        return cluster_means(X_S, Z_hat)              # eq. (51)
    _, _, Vt = np.linalg.svd(X_S, full_matrices=False)
    return Vt[:K].T                                     # eq. (52), (d, K)


def source_subspace_projector(theta_hats: Sequence[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis U_hat_S of R = col(Theta_hat_S1) + ... + col(Theta_hat_Sm) (eq. 45).

    Rank is determined via SVD so that (near-)collinear columns across
    sources collapse correctly instead of padding the basis with spurious
    orthonormal columns unpivoted QR would produce.
    """
    M = np.column_stack(list(theta_hats))
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(S > tol * S[0]))
    return U[:, :r]


# ---------------------------------------------------------------------------
# Validation statistic and threshold (eq. 47-49)
# ---------------------------------------------------------------------------


def validation_statistic_mult(Z: np.ndarray, X_T: np.ndarray, sigma_T2: float) -> float:
    """Validation statistic S_hat(Z), eq. (48): a normalized sum of squared
    pairwise gaps between estimated cluster means, de-biased by the noise
    level."""
    n_T, K = Z.shape
    d = X_T.shape[1]
    n_a = Z.sum(axis=0)
    theta_hat = cluster_means(X_T, Z)  # (d, K)

    total = 0.0
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            gap2 = float(np.sum((theta_hat[:, a] - theta_hat[:, b]) ** 2))
            total += (n_a[a] * n_a[b]) / (2 * n_T**2) * gap2
    total -= sigma_T2 * d * (K - 1) / n_T
    return float(total)


def validation_threshold_mult(sigma_T2: float, d: int, K: int, n_T: int, D0: float = 1.0) -> float:
    """Validation threshold t_n, eq. (49): t_n = D0 K sigma_T^2 (sqrt(dK/n_T) + 1)."""
    return D0 * K * sigma_T2 * (np.sqrt(d * K / n_T) + 1.0)


def estimate_noise_variance_rankK(X_T: np.ndarray, K: int) -> float:
    """Plug-in estimator of sigma_T^2 via the best rank-K approximation of
    X_T (Algorithm 2 remark, analogous to eq. 20-21 for K=2)."""
    n_T, d = X_T.shape
    U, S, Vt = np.linalg.svd(X_T, full_matrices=False)
    X_hat_K = (U[:, :K] * S[:K]) @ Vt[:K]
    resid = X_T - X_hat_K
    return float(np.sum(resid**2) / (n_T * d))


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------


def _regular_simplex_points(K: int) -> np.ndarray:
    """K equidistant points e_k - mean(e), re-expressed in an orthonormal
    basis of the (K-1)-dimensional hyperplane they live in. Returns a
    (K-1, K) array of coordinates with every pairwise squared distance
    exactly 2 (matches ||e_a - e_b||^2 for the standard basis of R^K)."""
    P = np.eye(K) - np.ones((K, K)) / K
    U, _, _ = np.linalg.svd(P)
    basis = U[:, :K - 1]          # (K, K-1) orthonormal basis of the hyperplane
    return basis.T @ P            # (K-1, K)


def calibrate_D0_bootstrap_mult(
    n_T: int,
    d: int,
    K: int,
    sigma_T2: float,
    alpha: float = 0.05,
    n_boot: int = 20,
    boundary_scale: float = 1.0,
    relaxed_kmeans_kwargs: Optional[Dict] = None,
    random_state: Optional[int] = None,
) -> float:
    """Parametric-bootstrap calibration of the constant D0 in
    `validation_threshold_mult` (t_n = D0 * K * sigma_T^2 * (sqrt(dK/n_T)
    + 1), eq. 49) -- K>2 analogue of `two_community.calibrate_C0_bootstrap`.

    The paper leaves D0 an unspecified absolute constant. The bootstrap's
    job is to estimate it: simulate data at a known signal level, see how
    the statistic S_hat behaves relative to the threshold's shape (the
    formula evaluated at D0=1), and read off D0 as a quantile of that
    ratio. "formula" and "bootstrap" selection modes therefore differ only
    in *how* D0 is obtained -- both end up computing
    `t_n = validation_threshold_mult(sigma_T2, d, K, n_T, D0)`.

    Like the K=2 version, this simulates replicates at the recovery
    *boundary* rather than at zero signal (a zero-signal null answers "is
    there any structure at all", not "is the target signal strong enough
    to trust alone" -- see `two_community.calibrate_C0_bootstrap`'s
    docstring for the detection-vs-estimation-gap argument). The K-cluster
    boundary separation is the natural generalization of
    Delta_boundary = max(1, (d/n_T)^(1/4)) whose square matches
    `validation_threshold`'s unit-sigma shape in the K=2 case:

        gap^2 = boundary_scale * sigma_T^2 * K * max(1, sqrt(dK/n_T))

    Each bootstrap replicate places K cluster means at the vertices of a
    regular (K-1)-simplex (embedded via a random orthonormal (d, K-1)
    basis, requiring d >= K-1) scaled so every pair is exactly `gap^2`
    apart -- the K-cluster analogue of the K=2 "+theta/-theta at the
    recovery boundary, random direction" construction. It then divides the
    real S_hat by

        q_hat = sigma_T^2 * K * (sqrt(dK/n_T) + 1)

    (`validation_threshold_mult` evaluated at D0=1) and returns the
    empirical alpha-quantile of that ratio as D0_hat.

    Note: each bootstrap replicate re-solves the relaxed K-means SDP, which
    is much more expensive than the K=2 hollowed-Gram spectral method; keep
    `n_boot` modest (the default of 20 is deliberately small).
    """
    sigma = float(np.sqrt(sigma_T2))
    rk = RelaxedKMeans(K=K, **(relaxed_kmeans_kwargs or {}))

    gap2_unit = K * max(1.0, np.sqrt(d * K / n_T))
    gap2 = boundary_scale * gap2_unit * sigma_T2
    simplex_coords = _regular_simplex_points(K)  # (K-1, K), pairwise sq-dist 2
    shape = validation_threshold_mult(sigma_T2, d, K, n_T, D0=1.0)  # q_hat, i.e. t_n at D0=1

    def simulate_null(rng: np.random.Generator) -> np.ndarray:
        Q, _ = np.linalg.qr(rng.normal(size=(d, K - 1)))  # (d, K-1) orthonormal
        theta = (Q @ simplex_coords) * np.sqrt(gap2 / 2.0)  # (d, K), pairwise sq-dist = gap2
        labels = rng.integers(0, K, size=n_T)
        Z = labels_to_onehot(labels, K)
        noise = rng.normal(scale=sigma, size=(n_T, d))
        return Z @ theta.T + noise

    def estimator_fn(X: np.ndarray) -> np.ndarray:
        return labels_to_onehot(rk.fit_predict(X), K)

    def stat_fn(Z: np.ndarray, X: np.ndarray) -> float:
        return validation_statistic_mult(Z, X, sigma_T2) / shape

    return bootstrap_null_quantile(
        simulate_null, estimator_fn, stat_fn, alpha=alpha, n_boot=n_boot,
        random_state=random_state,
    )


def _target_branch(X_T: np.ndarray, K: int, relaxed_kmeans_kwargs: Dict) -> np.ndarray:
    """Line 1: target-based estimator via relaxed K-means, as a one-hot Z."""
    rk = RelaxedKMeans(K=K, **relaxed_kmeans_kwargs)
    return labels_to_onehot(rk.fit_predict(X_T), K)


def _source_branch(
    X_T: np.ndarray, K: int, source_datasets: Sequence[np.ndarray], relaxed_kmeans_kwargs: Dict,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lines 2-9: source-based estimator via subspace projection + TSClust.
    Returns (Z_S one-hot, X_hat_T projected observations)."""
    n_T = X_T.shape[0]
    theta_hats = [
        estimate_source_means(X_S, K, relaxed_kmeans_kwargs) for X_S in source_datasets
    ]
    Q_S = source_subspace_projector(theta_hats)
    X_hat_T = X_T @ Q_S                                   # eq. (46)
    n_iter = int(round(2 * np.log(n_T)))
    labels_S = ts_clust(X_hat_T, K, n_iter=n_iter, random_state=random_state)
    return labels_to_onehot(labels_S, K), X_hat_T


def pooled_subspace_estimate(
    X_T: np.ndarray,
    K: int,
    source_datasets: Sequence[np.ndarray],
    relaxed_kmeans_kwargs: Optional[Dict] = None,
) -> np.ndarray:
    """Pooled estimator, K>2 analogue of `two_community.pooled_subspace_estimate`:
    an alternative to `AdaptiveProjectedClustering`'s hard target/source
    switch. Instead of using the validation statistic S_hat to select
    *either* the target-only estimate *or* the source-projection estimate,
    this simply pools all the raw data together and clusters it with the
    target-only routine:

        1. row-wise concatenate X_T with every source dataset,
        2. apply relaxed K-means directly to the pooled matrix (as
           `_target_branch` does for X_T alone),
        3. return the one-hot labels for the rows belonging to X_T.

    With no source datasets, this reduces exactly to `_target_branch(X_T, ...)`.
    """
    n_T = X_T.shape[0]
    rk_kwargs = relaxed_kmeans_kwargs or {}
    X_pooled = np.vstack([X_T, *source_datasets]) if source_datasets else X_T
    rk = RelaxedKMeans(K=K, **rk_kwargs)
    labels_pooled = rk.fit_predict(X_pooled)
    return labels_to_onehot(labels_pooled[:n_T], K)


@dataclass
class AdaptiveProjectedClustering:
    """Algorithm 2: Adaptive Transfer-Assisted Projected Clustering.

    No cross-validation / sample-splitting is used anywhere in this class.

    Parameters
    ----------
    K : int
        Number of clusters (K > 2; use `two_community` for K = 2).
    selection : {"formula", "bootstrap", "manual"}
        - "formula": the literal eq. (49) threshold with a
          user-specified constant `D0`. The paper does not specify a
          numeric value for D0; this lets you set it directly.
        - "bootstrap" (default): calibrate D0 via
          `calibrate_D0_bootstrap_mult` (boundary-calibrated, not
          zero-signal -- see `two_community.calibrate_C0_bootstrap`'s
          docstring for why that distinction matters). `alpha` is the
          quantile used as D0_hat; `boundary_scale` rescales the assumed
          recovery-boundary separation (default 1.0). Both "bootstrap" and
          "formula" ultimately compute
          `t_n = validation_threshold_mult(sigma_T2, d, K, n_T, D0)` --
          they only differ in how D0 is obtained.
        - "manual": skip all statistics; directly return the target-based
          or source-based estimate named by `manual_choice`
          ("target"/"source") -- i.e. you specify the oracle branch
          yourself, computing only what that branch needs.
    D0 : float
        Constant in the validation threshold t_n (eq. 49); used when
        `selection == "formula"`.
    manual_choice : {"target", "source"}, optional
        Required when `selection == "manual"`.
    sigma_T2 : float, optional
        Target noise variance. If None, estimated via
        `estimate_noise_variance_rankK`.
    relaxed_kmeans_kwargs : dict, optional
        Extra keyword arguments forwarded to `RelaxedKMeans` (e.g. `solver`).

    After `fit_predict`, `D0_used_` holds the D0 that was actually plugged
    into `validation_threshold_mult` (the calibrated `D0_hat` under
    "bootstrap", or the user-supplied `D0` under "formula"; `None` under
    "manual").

    No cross-validation / sample-splitting is used anywhere in this class.
    """

    K: int
    selection: str = "bootstrap"
    D0: float = 1.0
    manual_choice: Optional[str] = None
    alpha: float = 0.05
    n_boot: int = 20
    boundary_scale: float = 1.0
    sigma_T2: Optional[float] = None
    relaxed_kmeans_kwargs: Dict = field(default_factory=dict)
    random_state: Optional[int] = None
    D0_used_: Optional[float] = field(default=None, init=False, repr=False)

    def fit_predict(
        self, X_T: np.ndarray, source_datasets: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, str]:
        n_T, d = X_T.shape

        # Thread random_state into every RelaxedKMeans/ts_clust call below
        # (via a local copy of relaxed_kmeans_kwargs, so the caller's dict
        # isn't mutated). Previously only calibrate_D0_bootstrap_mult's own
        # bootstrap draws respected self.random_state -- _target_branch/
        # _source_branch's underlying K-means rounding step silently fell
        # back to sklearn's unseeded default (random_state=None), making
        # every RelaxedKMeans-based result non-reproducible across separate
        # calls/processes, and preventing an adaptively-chosen branch from
        # exactly matching a separately-run target-only/source-only call
        # even on identical data.
        rk_kwargs = dict(self.relaxed_kmeans_kwargs)
        rk_kwargs.setdefault("random_state", self.random_state)

        if self.selection == "manual":
            self.D0_used_ = None
            if self.manual_choice == "target":
                return _target_branch(X_T, self.K, rk_kwargs), "target"
            if self.manual_choice == "source":
                Z_S, _X_hat_T = _source_branch(
                    X_T, self.K, source_datasets, rk_kwargs, random_state=self.random_state
                )
                return Z_S, "source"
            raise ValueError(
                f"selection='manual' requires manual_choice in {{'target','source'}}, "
                f"got {self.manual_choice!r}"
            )

        sigma_T2 = (
            self.sigma_T2
            if self.sigma_T2 is not None
            else estimate_noise_variance_rankK(X_T, self.K)
        )

        Z_T = _target_branch(X_T, self.K, rk_kwargs)
        Z_S, _X_hat_T = _source_branch(
            X_T, self.K, source_datasets, rk_kwargs, random_state=self.random_state
        )
        stat = validation_statistic_mult(Z_T, X_T, sigma_T2)

        # Final adaptive choice (eq. 47-50).
        if self.selection == "bootstrap":
            D0_hat = calibrate_D0_bootstrap_mult(
                n_T, d, self.K, sigma_T2, alpha=self.alpha, n_boot=self.n_boot,
                boundary_scale=self.boundary_scale,
                relaxed_kmeans_kwargs=rk_kwargs, random_state=self.random_state,
            )
        elif self.selection == "formula":
            D0_hat = self.D0
        else:
            raise ValueError(
                f"selection must be 'formula', 'bootstrap', or 'manual', got {self.selection!r}"
            )

        self.D0_used_ = D0_hat
        t_n = validation_threshold_mult(sigma_T2, d, self.K, n_T, D0_hat)

        if stat > t_n:
            return Z_T, "target"
        return Z_S, "source"

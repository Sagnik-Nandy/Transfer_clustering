"""Two-community (K=2) transfer-assisted clustering.

Implements Section 2.2, Section 2.3, and Algorithm 1 ("Oracle
Transfer-Assisted Clustering") of the transfer-learning-for-clustering
discussion draft, for the two-component Gaussian mixture model

    X_j^(T) = z_j^(T) theta_T + noise_j^(T),   z_j^(T) in {-1, 1},
    X_j^(S) = z_j^(S) theta_S  + noise_j^(S),  z_j^(S) in {-1, 1},

with one or more source datasets S_1, ..., S_m. Labels are identifiable
only up to a global sign flip.

Equation numbers below refer to the discussion draft unless stated
otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .spectral import spectral_sign_clustering
from .selection import bootstrap_null_quantile


# ---------------------------------------------------------------------------
# Section 2.2: target-based and source-based estimators
# ---------------------------------------------------------------------------


def target_based_estimate(X_T: np.ndarray) -> np.ndarray:
    """Target-only estimator, eq. (9)-(10): Ndaoud's spectral method applied
    directly to the target data. Used when the target signal-to-noise ratio
    is strong enough on its own (condition (8))."""
    return spectral_sign_clustering(X_T)


def estimate_source_direction(X_S: np.ndarray) -> np.ndarray:
    """Estimate the source direction theta_S (up to scale and sign).

    Implements eq. (13)-(14): the construction depends on the relative
    magnitudes of the feature dimension d and the source sample size n_S.

      - d > n_S ("d >> n_S" regime): estimating theta_S directly is hard,
        so we first recover the source labels via the spectral method and
        then average (eq. 13):  theta_hat_S = (1/n_S) sum_j z_hat_S_j X_S_j.
      - d <= n_S ("d <~ n_S" regime): theta_S can be estimated directly as
        the leading (normalized) right singular vector of X_S (eq. 14).
    """
    n_S, d = X_S.shape
    if d > n_S:
        z_hat_S = spectral_sign_clustering(X_S)
        theta_hat = (X_S.T @ z_hat_S) / n_S
    else:
        _, _, Vt = np.linalg.svd(X_S, full_matrices=False)
        theta_hat = Vt[0]
    return theta_hat


def source_subspace_basis(source_directions: Sequence[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis Q_S of span{theta_hat_S_1, ..., theta_hat_S_m}.

    Reduces to a single unit direction when only one source is supplied,
    matching eq. (34)'s span construction. Rank is determined via SVD so
    that (near-)collinear source directions collapse correctly instead of
    padding Q with spurious orthonormal columns unpivoted QR would produce.
    """
    M = np.column_stack(list(source_directions))
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(S > tol * S[0]))
    return U[:, :r]


def _leading_eigvec_direction(X_hat: np.ndarray) -> np.ndarray:
    """Leading eigenvector of the (reduced-dimension) Gram matrix of X_hat,
    i.e. v_hat in eq. (35)."""
    n = X_hat.shape[0]
    Sigma_hat = (X_hat.T @ X_hat) / n
    eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
    return eigvecs[:, np.argmax(eigvals)]


def source_based_estimate(X_T: np.ndarray, source_datasets: Sequence[np.ndarray]) -> np.ndarray:
    """Source-based (projection) estimator, generalizing eq. (12) to m
    sources as in Algorithm 1, lines 6-8:

        1. estimate each source direction theta_hat_S_i,
        2. project target observations onto Q_S := orthonormal basis of
           their span (eq. 34),
        3. cluster the projected observations by the sign of the leading
           eigenvector of their (reduced-dimension) Gram matrix (eq. 35).

    With a single source this is exactly eq. (12): sign(<theta_hat_S, X_j> / ||theta_hat_S||).
    """
    if not source_datasets:
        raise ValueError("source_based_estimate requires at least one source dataset.")
    directions = [estimate_source_direction(X_S) for X_S in source_datasets]
    Q_S = source_subspace_basis(directions)

    X_hat_T = X_T @ Q_S                              # eq. (34)
    v_hat = _leading_eigvec_direction(X_hat_T)

    scores = X_hat_T @ v_hat                           # eq. (35)
    labels = np.sign(scores)
    labels[labels == 0] = 1.0
    return labels


def target_source_pooled_subspace_estimate(
    X_T: np.ndarray, source_datasets: Sequence[np.ndarray]
) -> np.ndarray:
    """Target+source pooled-subspace estimator: like `source_based_estimate`,
    but the target's own estimated direction is pooled into the projection
    subspace alongside the sources', rather than being reserved for a
    separate target-only branch.

        1. estimate theta_hat_T from X_T itself, using the same
           regime-dependent construction as `estimate_source_direction`
           (spectral-clustering-then-average if d > n_T, leading right
           singular vector otherwise) -- i.e. treat the target exactly like
           one more "source" for the purpose of direction estimation only,
        2. estimate theta_hat_S_i from each source dataset the same way,
        3. pool ALL m+1 directions {theta_hat_T, theta_hat_S_1, ...,
           theta_hat_S_m} into one joint subspace Q (orthonormal basis of
           their span, via `source_subspace_basis`),
        4. project the target observations onto Q: X_hat_T = X_T @ Q,
        5. cluster X_hat_T by the sign of the leading eigenvector of its
           Gram matrix, exactly as in `source_based_estimate`.

    Unlike `AdaptiveTransferClustering`, there is no hard target-vs-source
    switch and no validation statistic/threshold -- the target's direction
    is simply one more column pooled into the shared subspace before
    projecting and re-clustering the target data through it. With no
    source datasets, this reduces to projecting X_T onto its own estimated
    direction, i.e. `source_based_estimate(X_T, [X_T])`'s single-direction
    special case (not `target_based_estimate`, which clusters directly on
    the hollowed Gram matrix rather than a 1-D projection).
    """
    theta_hat_T = estimate_source_direction(X_T)
    directions = [theta_hat_T] + [estimate_source_direction(X_S) for X_S in source_datasets]
    Q = source_subspace_basis(directions)

    X_hat_T = X_T @ Q
    v_hat = _leading_eigvec_direction(X_hat_T)

    scores = X_hat_T @ v_hat
    labels = np.sign(scores)
    labels[labels == 0] = 1.0
    return labels


def pooled_subspace_estimate(X_T: np.ndarray, source_datasets: Sequence[np.ndarray]) -> np.ndarray:
    """Pooled estimator: an alternative to `AdaptiveTransferClustering`'s
    hard target/source switch. Instead of using the validation statistic to
    select *either* the target-only estimate *or* the source-projection
    estimate, this simply pools all the raw data together and clusters it
    with the target-only routine:

        1. row-wise concatenate X_T with every source dataset,
        2. apply `target_based_estimate` (Ndaoud's spectral method) to the
           pooled matrix,
        3. return the labels for the rows belonging to X_T.

    With no source datasets, this reduces exactly to `target_based_estimate(X_T)`.
    """
    n_T = X_T.shape[0]
    X_pooled = np.vstack([X_T, *source_datasets]) if source_datasets else X_T
    labels_pooled = target_based_estimate(X_pooled)
    return labels_pooled[:n_T]


# ---------------------------------------------------------------------------
# Algorithm 1: Oracle Transfer-Assisted Clustering
# ---------------------------------------------------------------------------


@dataclass
class OracleTransferClustering:
    """Algorithm 1 ("Oracle Transfer-Assisted Clustering"), two-community case.

    `oracle` indicates which regime is assumed to hold:
      - "target": condition (8) holds -> use the target-based estimator.
      - "source": condition (11) holds -> use the (multi-source) projection
        estimator.

    This is an *oracle* procedure: it requires knowing in advance which
    branch is appropriate (Theorem 1). See `AdaptiveTransferClustering`
    below for the data-driven (oracle-free) version of Section 2.3.
    """

    oracle: str = "target"

    def fit_predict(
        self,
        X_T: np.ndarray,
        source_datasets: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        if self.oracle == "target":
            return target_based_estimate(X_T)
        if self.oracle == "source":
            if not source_datasets:
                raise ValueError("oracle='source' requires at least one source dataset.")
            return source_based_estimate(X_T, source_datasets)
        raise ValueError(f"oracle must be 'target' or 'source', got {self.oracle!r}")


# ---------------------------------------------------------------------------
# Section 2.3: adaptive (oracle-free) selection
# ---------------------------------------------------------------------------


def validation_statistic(z_tilde: np.ndarray, X_T: np.ndarray, sigma_T2: float) -> float:
    """Validation statistic T_hat(z_tilde), eq. (16):

        T_hat(z_tilde) = || (1/n_T) sum_j z_tilde_j X_j^(T) ||_2^2 - (d/n_T) sigma_T^2.
    """
    n_T, d = X_T.shape
    v = (z_tilde @ X_T) / n_T
    return float(np.sum(v**2) - (d / n_T) * sigma_T2)


def validation_threshold(sigma_T2: float, d: int, n_T: int, C0: float = 1.0) -> float:
    """Validation threshold tau_n, eq. (17): tau_n = C0 sigma_T^2 (1 + sqrt(d/n_T))."""
    return C0 * sigma_T2 * (1.0 + np.sqrt(d / n_T))


def estimate_noise_variance_rank1(X_T: np.ndarray) -> float:
    """Plug-in estimator of sigma_T^2 via the best rank-1 approximation of
    X_T (eq. 20-21, Remark 1), for use when sigma_T^2 is not known exactly."""
    n_T, d = X_T.shape
    U, S, Vt = np.linalg.svd(X_T, full_matrices=False)
    X_hat = S[0] * np.outer(U[:, 0], Vt[0])
    resid = X_T - X_hat
    return float(np.sum(resid**2) / (n_T * d))


def calibrate_C0_bootstrap(
    n_T: int,
    d: int,
    sigma_T2: float,
    alpha: float = 0.5,
    n_boot: int = 200,
    boundary_scale: float = 1.0,
    random_state: Optional[int] = None,
) -> float:
    """Parametric-bootstrap calibration of the constant C0 in
    `validation_threshold` (tau_n = C0 * sigma_T^2 * (1 + sqrt(d/n_T)), eq. 17).

    The paper leaves C0 an unspecified absolute constant. The bootstrap's
    entire job is to estimate it: simulate data at a *known* signal level,
    see how the statistic T_hat behaves relative to the threshold's shape
    (i.e. the formula evaluated at C0=1), and read off C0 as a quantile of
    that ratio. "formula" and "bootstrap" selection modes therefore differ
    only in *how* C0 is obtained -- both end up computing
    `tau_n = validation_threshold(sigma_T2, d, n_T, C0)`.

    Important subtlety: calibrating against a *zero-signal* null answers
    "is there any detectable structure at all?", which is not the
    question condition (8) actually asks. Condition (8) asks whether
    Delta_T is large enough for *reliable recovery*, i.e. above the
    threshold max{1, (d/n_T)^(1/4)} -- and a weak-but-nonzero Delta_T can
    be statistically detectable (pushing |T_hat| above a zero-signal
    null's quantiles) while still being far too small for the target-based
    estimator to be trustworthy for clustering (a classic
    detection-vs-estimation gap). So instead of simulating pure noise, this
    bootstrap simulates replicates *at the condition-(8) recovery
    threshold itself*: Delta_T_boundary = boundary_scale * max(1,
    (d/n_T)^(1/4)), with a uniformly random direction each replicate (the
    direction is immaterial by rotational invariance). It runs the same
    `target_based_estimate` + `validation_statistic` pipeline on each
    boundary-strength replicate, divides by the unit-C0 threshold shape,
    and returns the empirical alpha-quantile of |T_hat|/shape as C0_hat.
    With the default alpha=0.5: if the real data's signal is noticeably
    stronger than the boundary, its statistic will exceed
    `C0_hat * shape` most of the time (correctly selecting "target"); if
    noticeably weaker, it will fall short most of the time (correctly
    selecting "source").
    """
    sigma = float(np.sqrt(sigma_T2))
    Delta_boundary = boundary_scale * max(1.0, (d / n_T) ** 0.25)
    shape = validation_threshold(sigma_T2, d, n_T, C0=1.0)  # unit-C0 shape, fixed for this (n_T, d, sigma_T2)

    def simulate_null(rng: np.random.Generator) -> np.ndarray:
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        theta = direction * Delta_boundary * sigma
        z = rng.choice([-1, 1], size=n_T)
        noise = rng.normal(scale=sigma, size=(n_T, d))
        return z[:, None] * theta[None, :] + noise

    def stat_fn(z: np.ndarray, X: np.ndarray) -> float:
        return validation_statistic(z, X, sigma_T2) / shape

    return bootstrap_null_quantile(
        simulate_null, target_based_estimate, stat_fn, alpha=alpha, n_boot=n_boot,
        random_state=random_state,
    )


@dataclass
class AdaptiveTransferClustering:
    """Section 2.3's data-driven selection between the target- and
    source-based estimators, without oracle knowledge of condition (8)
    vs (11) (eq. 16-18), generalized to multiple sources via eq. (37).

    If `sigma_T2` is not supplied, it is estimated from the target data
    via `estimate_noise_variance_rank1` (Remark 1). No cross-validation /
    sample-splitting is used anywhere in this class.

    Parameters
    ----------
    selection : {"formula", "bootstrap", "manual"}
        - "formula": the literal eq. (16)-(18) threshold with a
          caller-specified constant `C0`. The paper does not specify a
          numeric value for C0; this lets you set it directly.
        - "bootstrap" (default): calibrate C0 via `calibrate_C0_bootstrap`,
          simulating replicates at the condition-(8) recovery threshold
          (scaled by `boundary_scale`) rather than at zero signal (see
          that function's docstring for why zero-signal calibration
          answers the wrong question). `alpha` is the quantile of the
          boundary-strength null's |T_hat|/shape ratio used as C0_hat
          (default 0.5: a coin flip right at the threshold). Both
          "bootstrap" and "formula" ultimately compute
          `tau_n = validation_threshold(sigma_T2, d, n_T, C0)` -- they
          only differ in how C0 is obtained.
        - "manual": skip all statistics; directly return the target-based
          or source-based estimate named by `manual_choice`
          ("target"/"source") -- i.e. you specify the oracle branch
          yourself. Equivalent to `OracleTransferClustering` but exposed
          through this class's interface.

    After `fit_predict`, `C0_used_` holds the C0 that was actually plugged
    into `validation_threshold` (the calibrated `C0_hat` under "bootstrap",
    or the caller-supplied `C0` under "formula"; `None` under "manual").

    No cross-validation / sample-splitting is used anywhere in this class.
    """

    selection: str = "bootstrap"
    C0: float = 1.0
    manual_choice: Optional[str] = None
    alpha: float = 0.5
    n_boot: int = 200
    boundary_scale: float = 1.0
    sigma_T2: Optional[float] = None
    random_state: Optional[int] = None
    C0_used_: Optional[float] = field(default=None, init=False, repr=False)

    def fit_predict(
        self,
        X_T: np.ndarray,
        source_datasets: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, str]:
        n_T, d = X_T.shape

        if self.selection == "manual":
            self.C0_used_ = None
            if self.manual_choice == "target":
                return target_based_estimate(X_T), "target"
            if self.manual_choice == "source":
                return source_based_estimate(X_T, source_datasets), "source"
            raise ValueError(
                f"selection='manual' requires manual_choice in {{'target','source'}}, "
                f"got {self.manual_choice!r}"
            )

        sigma_T2 = (
            self.sigma_T2
            if self.sigma_T2 is not None
            else estimate_noise_variance_rank1(X_T)
        )
        z_trg = target_based_estimate(X_T)
        stat = validation_statistic(z_trg, X_T, sigma_T2)

        if self.selection == "bootstrap":
            C0_hat = calibrate_C0_bootstrap(
                n_T, d, sigma_T2, alpha=self.alpha, n_boot=self.n_boot,
                boundary_scale=self.boundary_scale, random_state=self.random_state,
            )
        elif self.selection == "formula":
            C0_hat = self.C0
        else:
            raise ValueError(
                f"selection must be 'formula', 'bootstrap', or 'manual', got {self.selection!r}"
            )

        self.C0_used_ = C0_hat
        tau_n = validation_threshold(sigma_T2, d, n_T, C0_hat)

        if abs(stat) > tau_n:
            return z_trg, "target"

        z_src = source_based_estimate(X_T, source_datasets)
        return z_src, "source"

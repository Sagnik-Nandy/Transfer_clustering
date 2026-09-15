#!/usr/bin/env python
"""Experiment 4: "Comparison with benchmark methods" (Section 5.3 of the
paper) -- comparison against two published comparator methods, across all
three of Experiment 1's aspect-ratio regimes (R1, R2, R3).

    (1) Transfer Gaussian Mixture Model (TGMM) -- Wang, Zhou et al. (2019).
    (2) TL-GMM -- Tian, Weng, Xia & Feng (arXiv:2209.15224), Algorithm 7
        (their transfer-learning generalization of MTL-GMM, supplementary
        Section S.4), with the source "center" beta_bar obtained via their
        Algorithm 1 collapsed to a single source task (no fusion needed
        with only one task).

Usage (matches Slurm_Scripts/experiment4_comparators/run_experiment4.sh):

    python run_experiment4.py <regime> <seed>

One call = one Monte Carlo repetition for one regime, sweeping the full
mu grid internally (Delta_T=0.8, Delta_S=8.0 fixed, mu varying), comparing
our 3 methods (target-only, source-only, adaptive) against the 2
comparators above -- 5 methods total -- in whichever of R1/R2/R3 is passed.

IMPORTANT MODELING NOTE: both comparators are genuinely Gaussian-Mixture-
Model/EM-based and need an estimable covariance matrix. In Regimes R1/R2,
n_T=200 << d=2000, so a full d x d target covariance is classically
unidentifiable (rank <= 200) -- even the TL-GMM paper's own theory
requires n_k >~ p (their Assumption 1(ii)), i.e. it isn't designed for
these regimes either. To make both runnable and comparable, tgmm_fit and
tlgmm_fit below ASSUME an ISOTROPIC covariance for each domain separately
-- Sigma_T = sigma_T2 * I for the target, Sigma_S = sigma_S2 * I for the
source -- rather than attempting to estimate a general covariance. This
isotropic assumption is applied uniformly across all three regimes (not
just R1/R2) so every regime gets the identical fair-comparison treatment,
even though in R3 (n_T=2500 > d=2000) a general covariance would actually
be identifiable. sigma_T2 and sigma_S2 are kept as two genuinely separate
quantities throughout (never assumed equal by the formulas themselves) and
are ESTIMATED per repetition via estimate_noise_variance_rank1 (the same
rank-1-residual estimator our own AdaptiveTransferClustering already uses
internally when its sigma_T2 isn't supplied), applied to X_T and X_S
respectively -- not fixed to their true simulated values. All three
methods (ours and both comparators) are given the exact same estimated
sigma_T2/sigma_S2 per repetition, for a fair comparison under identical
information. This is the only simplification made relative to the
published methods -- see the module-level NOTE comments on
tgmm_fit/tlgmm_fit for how the isotropic assumption changes (and
simplifies) their published update formulas.

TGMM's lambda genuinely has no data-driven selection rule in that paper --
it's hand-tuned there too. TL-GMM's C_lambda0 is different: the paper
(supplementary Sections S.5.1.7-S.5.1.8) DOES give a data-driven procedure
-- find the smallest C_lambda0 that collapses beta_hat entirely to
beta_bar (call it C'_max), build a log-spaced grid from C'_max/50 to
2*C'_max, and pick via 10-fold cross-validation; kappa0 is fixed at 1/3
(not tuned, but shown empirically insensitive to that choice). We only
implement the fixed-kappa0 half of that (TLGMM_KAPPA0 = 1/3, matching
their convention); TLGMM_C_LAMBDA0 is fixed directly on the paper's own
scale (1.7, between their reported "small C_lambda" grid points 1.29 and
2.15, Figure S.19) rather than cross-validated -- a simplification on our
part for this comparator, not a limitation of the published method.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from transfer_clustering.two_community import (
    target_based_estimate, source_based_estimate, AdaptiveTransferClustering,
    estimate_noise_variance_rank1,
)

# ---------------------------------------------------------------------------
# Config -- edit these directly. If you change N_SEEDS or REGIME_ORDER,
# update the matching arithmetic in
# Slurm_Scripts/experiment4_comparators/run_experiment4.sh to stay in sync.
# ---------------------------------------------------------------------------

# Same three aspect-ratio regimes as Experiment 1 (Section 5.1), duplicated
# here rather than imported -- each experiment script here is deliberately
# self-contained rather than split into shared config/utils modules.
REGIMES = {
    "R1": dict(d=2000, n_T=200, n_S=200),
    "R2": dict(d=2000, n_T=200, n_S=2500),
    "R3": dict(d=2000, n_T=2500, n_S=2500),
}
REGIME_ORDER = ["R1", "R2", "R3"]  # fixes the SLURM array's regime_idx order

DELTA_T = 0.8                        # fixed target signal (as in Experiment 1)
DELTA_S = 8.0                        # fixed source signal (decided for Experiments 1-3),
                                       # same value for all three regimes here (not
                                       # regime-dependent, unlike Experiment 1's current
                                       # DELTA_S_BY_REGIME)
MU_GRID = [0.0, 0.1, 0.2, 0.4, 0.8]   # alignment grid (as in Experiment 1)
# sigma_T2, sigma_S2 are NOT fixed here -- see run_one(), which estimates
# both per repetition via estimate_noise_variance_rank1 and threads the
# same estimates into all five methods (see module docstring).

N_SEEDS = 100  # Monte Carlo repetitions per regime

# Switch, matching run_experiment1.py's FIXED_UV/UV_SEED convention: False
# (default) resamples (u,v) fresh each Monte Carlo repetition. True pins
# (u,v) to a single draw from UV_SEED, reused for every regime/seed/mu, so
# theta_T is identical throughout the whole experiment and only the
# noise/label draws (rng seeded by `seed`) vary.
FIXED_UV = False
UV_SEED = 20260711

# Our adaptive method's knobs (matches Experiment 1's defaults, same across
# all three regimes there too).
ADAPTIVE_N_BOOT = 50
ADAPTIVE_ALPHA = 0.5
ADAPTIVE_BOUNDARY_SCALE = 1.0

# Comparator hyperparameters. Neither has a data-driven selection rule in
# the original papers -- fixed by hand, as the papers themselves do.
EM_N_ITER = 50          # EM iterations for TGMM/TL-GMM (source fit and target fit)
TGMM_LAMBDA = 0.5        # TGMM's fixed source-to-target mean-shrinkage weight
TLGMM_KAPPA0 = 1.0 / 3.0   # matches the paper's own fixed convention (S.5.1.8):
                            # kappa0 = 1/3, not cross-validated, but shown
                            # empirically insensitive to this choice there
TLGMM_C_LAMBDA0 = 1.7      # C_lambda0 in Step 2's schedule (lambda_0^[t] =
                            # kappa0*lambda_0^[t-1] + C_lambda0*sqrt(d + log K)).
                            # Fixed directly on the paper's own scale: Figure
                            # S.19 (Sec S.5.1.7) reports 1.29 and 2.15 as its
                            # "small C_lambda" grid points (mild pooling,
                            # close to single-task-GMM) -- 1.7 sits between
                            # them, rather than deriving C_lambda0 from a
                            # separately hand-picked steady-state effect size.

RESULTS_DIR = "../../Results_simulation/experiment4/raw"  # relative to this file's directory
FIELDNAMES = ["regime", "seed", "mu", "method", "error", "branch", "C0_used",
              "sigma_T2_hat", "sigma_S2_hat"]


# ---------------------------------------------------------------------------
# Simulation building blocks (two-community Gaussian mixture model, Section
# 2.1) -- duplicated from run_experiment1.py rather than shared, per this
# project's convention of one self-contained script per experiment.
# ---------------------------------------------------------------------------


def coordinate_free_uv(d: int, rng: np.random.Generator):
    g = rng.normal(size=d)
    u = g / np.linalg.norm(g)
    h = rng.normal(size=d)
    h_tilde = h - (h @ u) * u
    v = h_tilde / np.linalg.norm(h_tilde)
    return u, v


def make_theta_S(Delta_S: float, mu: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return Delta_S * (mu * u + np.sqrt(1.0 - mu**2) * v)


def simulate_two_community(n: int, theta: np.ndarray, rng: np.random.Generator):
    d = theta.shape[0]
    half = n // 2
    z = np.concatenate([np.ones(half), -np.ones(n - half)])
    rng.shuffle(z)
    noise = rng.normal(size=(n, d))
    X = z[:, None] * theta[None, :] + noise
    return X, z


def misclustering_error(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    err_plus = np.mean(z_hat != z_true)
    err_minus = np.mean(z_hat != -z_true)
    return float(min(err_plus, err_minus))


# ---------------------------------------------------------------------------
# Shared comparator helpers
# ---------------------------------------------------------------------------


def align_sign(direction: np.ndarray, X_T: np.ndarray) -> np.ndarray:
    """Fix the sign ambiguity of a source-derived direction against a cheap
    target-only reference direction, analogous in spirit to the alignment
    step both the TL-GMM paper (Sec 2.4) and TGMM implicitly need before
    fusing a source estimate with the target -- simplified here to a single
    sign check since we only ever have one source."""
    z_ref = target_based_estimate(X_T)
    ref_dir = (z_ref[:, None] * X_T).mean(axis=0)
    return direction if np.dot(direction, ref_dir) >= 0 else -direction


def _responsibility_isotropic(X, mu0, mu1, pi0, pi1, sigma2):
    """P(component=1 | x) for an isotropic-covariance (sigma2 * I) 2-component
    GMM. sigma2 here is whichever domain's own (known) noise variance the
    caller passes in -- this function has no notion of target vs source."""
    d0 = np.sum((X - mu0) ** 2, axis=1)
    d1 = np.sum((X - mu1) ** 2, axis=1)
    logits = (np.log(pi1) - d1 / (2 * sigma2)) - (np.log(pi0) - d0 / (2 * sigma2))
    return 1.0 / (1.0 + np.exp(-logits))


def spherical_em(X: np.ndarray, sigma2: float, n_iter: int, rng: np.random.Generator):
    """EM for a 2-component GMM with known isotropic covariance sigma2 * I
    (the only covariance structure identifiable when n << d; see module
    docstring). sigma2 is whichever domain's own noise variance the caller
    passes in (target's or source's) -- this function is domain-agnostic.
    Returns (mu0, mu1, pi0, pi1)."""
    n = X.shape[0]
    idx = rng.choice(n, size=2, replace=False)
    mu0, mu1 = X[idx[0]].copy(), X[idx[1]].copy()
    pi0, pi1 = 0.5, 0.5
    for _ in range(n_iter):
        gamma1 = _responsibility_isotropic(X, mu0, mu1, pi0, pi1, sigma2)
        gamma0 = 1.0 - gamma1
        N0, N1 = gamma0.sum(), gamma1.sum()
        mu0 = (gamma0[:, None] * X).sum(axis=0) / max(N0, 1e-8)
        mu1 = (gamma1[:, None] * X).sum(axis=0) / max(N1, 1e-8)
        pi0, pi1 = N0 / n, N1 / n
    return mu0, mu1, pi0, pi1


# ---------------------------------------------------------------------------
# Comparator 1: TGMM (Wang, Zhou et al. 2019)
# ---------------------------------------------------------------------------


def tgmm_fit(X_T: np.ndarray, X_S: np.ndarray, sigma_T2: float, sigma_S2: float,
             lam: float, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Fit the source GMM by EM (its own known noise variance sigma_S2),
    then fit the target GMM by EM (its own known noise variance sigma_T2)
    whose mean update is precision-weighted-shrunk toward the
    (sign-aligned) source means (Sec III, eq. 12 of the TGMM paper).

    eq. 12's mean update is, in general (N_k = total responsibility mass
    for component k, which cancels out of this ratio regardless of sigma):
        mu_k = [xbar_k/sigma_T2 + lam*mu_tilde_k/sigma_S2]
               / [1/sigma_T2 + lam/sigma_S2]
    This does NOT collapse further unless sigma_T2 == sigma_S2 -- the two
    are kept as separate quantities below rather than assumed equal.
    """
    mu0_S, mu1_S, _, _ = spherical_em(X_S, sigma_S2, n_iter, rng)
    beta_S = align_sign((mu1_S - mu0_S) / sigma_S2, X_T)
    center = (mu0_S + mu1_S) / 2.0
    mu1_S = center + beta_S * sigma_S2 / 2.0
    mu0_S = center - beta_S * sigma_S2 / 2.0

    w_T = 1.0 / sigma_T2
    w_S = lam / sigma_S2

    n_T = X_T.shape[0]
    idx = rng.choice(n_T, size=2, replace=False)
    mu0, mu1 = X_T[idx[0]].copy(), X_T[idx[1]].copy()
    pi0, pi1 = 0.5, 0.5
    for _ in range(n_iter):
        gamma1 = _responsibility_isotropic(X_T, mu0, mu1, pi0, pi1, sigma_T2)
        gamma0 = 1.0 - gamma1
        N0, N1 = gamma0.sum(), gamma1.sum()
        xbar0 = (gamma0[:, None] * X_T).sum(axis=0) / max(N0, 1e-8)
        xbar1 = (gamma1[:, None] * X_T).sum(axis=0) / max(N1, 1e-8)
        mu0 = (w_T * xbar0 + w_S * mu0_S) / (w_T + w_S)
        mu1 = (w_T * xbar1 + w_S * mu1_S) / (w_T + w_S)
        pi0, pi1 = N0 / n_T, N1 / n_T

    gamma1_final = _responsibility_isotropic(X_T, mu0, mu1, pi0, pi1, sigma_T2)
    return np.where(gamma1_final > 0.5, 1.0, -1.0)


# ---------------------------------------------------------------------------
# Comparator 2: TL-GMM (Tian, Weng, Xia & Feng, arXiv:2209.15224, Alg. 7)
# ---------------------------------------------------------------------------


def tlgmm_fit(X_T: np.ndarray, X_S: np.ndarray, sigma_T2: float, sigma_S2: float,
              n_iter: int, kappa0: float, C_lambda0: float,
              rng: np.random.Generator) -> np.ndarray:
    """Algorithm 7 (TL-GMM): means/covariance updated by ordinary target-only
    EM each round (using the target's own known noise variance sigma_T2),
    but the discriminant direction beta is re-derived each round via a
    penalized quadratic minimization that shrinks it toward a
    source-derived center beta_bar (Step 6):

        beta_hat = argmin_beta { 0.5*beta^T Sigma_hat_T beta - beta^T(mu1-mu0)
                                  + (lambda_0^[t]/sqrt(n_T)) * ||beta - beta_bar||_2 }

    where Sigma_hat_T = sigma_T2 * I is the TARGET's own covariance (the
    loss in Step 6 only involves target data), and lambda_0^[t] is the
    schedule from Step 2: lambda_0^[t] = kappa0*lambda_0^[t-1] +
    C_lambda0*sqrt(d + log K) -- with K=1 source, log K = 0, so this is
    C_lambda0*sqrt(d). NOTE the actual Step-6 penalty coefficient is
    lambda_0^[t]/sqrt(n_T), not lambda_0^[t] itself -- easy to miss since
    the schedule (Step 2) and the objective (Step 6) use it differently.

    For a general Sigma_hat_T this needs an iterative proximal solver;
    under our known isotropic Sigma_hat_T it has the closed-form block
    soft-threshold solution derived below: writing
    r = (mu1-mu0) - sigma_T2*beta_bar, rho = ||r||, lambda_t =
    lambda_0^[t]/sqrt(n_T), the minimizer is
        beta_hat = beta_bar + max(0, 1 - lambda_t/rho) * r / sigma_T2.
    The "source center" beta_bar is Algorithm 1 (MTL-GMM) collapsed to a
    single source task -- with only one task there is nothing to fuse
    against, so it reduces to that task's own EM-fitted discriminant
    coefficient, beta_bar = (mu1_S - mu0_S) / sigma_S2 (the SOURCE's own
    noise variance, since beta_bar comes entirely from source data).

    CRUCIALLY, Step 3's E-step uses gamma_{theta_hat^[t-1]}, where
    theta_hat^[t-1] = (w_hat^[t-1], beta_hat^[t-1], delta_hat^[t-1]) is the
    PENALIZED beta from the previous round (Step 8 stores exactly that
    triple for the next iteration) -- not the raw, unpenalized mean
    difference. So (w, beta, delta) is carried as the loop's state, and
    each round's responsibility is
        gamma(z) = w*exp(beta^T z - delta) / (1 - w + w*exp(beta^T z - delta))
    computed from the PREVIOUS round's shrunk beta/delta. This is what
    actually lets the source information influence the target's own
    cluster assignments each round, not just the final output -- getting
    this wrong (e.g. computing responsibilities from the raw target-only
    mean difference every round) would decouple the EM iterations from the
    shrinkage entirely.

    The final classifier follows eq. (1) exactly (Sec 1.1): threshold at
    log((1-w_hat)/w_hat), not 0 -- only equal to 0 when the fitted mixing
    weight is exactly balanced.
    """
    mu0_S, mu1_S, _, _ = spherical_em(X_S, sigma_S2, n_iter, rng)
    beta_bar = align_sign((mu1_S - mu0_S) / sigma_S2, X_T)

    n_T, d = X_T.shape
    idx = rng.choice(n_T, size=2, replace=False)
    mu0_init, mu1_init = X_T[idx[0]].copy(), X_T[idx[1]].copy()
    # Initial theta_hat^[0] = (w, beta, delta) (Algorithm 7's Input).
    w = 0.5
    beta = (mu1_init - mu0_init) / sigma_T2
    delta = 0.5 * beta @ (mu0_init + mu1_init)

    # lam tracks lambda_0^[t] itself (Step 2), using C_lambda0 directly on
    # the paper's own scale. Initialize at the recursion's steady state
    # (C_lambda0*sqrt(d)/(1-kappa0), with log K = 0 for a single source) to
    # avoid a cold-start transient over the first few iterations.
    lam = C_lambda0 * np.sqrt(d) / (1.0 - kappa0)
    for _ in range(n_iter):
        lam = kappa0 * lam + C_lambda0 * np.sqrt(d)
        lambda_t = lam / np.sqrt(n_T)  # Step 6's actual penalty coefficient

        # Step 3 (E-step): responsibility from the PREVIOUS round's
        # penalized (w, beta, delta), not from raw means.
        logits = np.log(w) - np.log(1.0 - w) + (X_T @ beta - delta)
        gamma1 = 1.0 / (1.0 + np.exp(-logits))
        gamma0 = 1.0 - gamma1

        # Steps 3-4: w, mu updates (standard, unpenalized target-only EM).
        N0, N1 = gamma0.sum(), gamma1.sum()
        mu0 = (gamma0[:, None] * X_T).sum(axis=0) / max(N0, 1e-8)
        mu1 = (gamma1[:, None] * X_T).sum(axis=0) / max(N1, 1e-8)
        w = N1 / n_T

        # Step 6: penalized beta update (closed form under isotropic Sigma_hat_T).
        r = (mu1 - mu0) - sigma_T2 * beta_bar
        rho = np.linalg.norm(r)
        shrink = max(0.0, 1.0 - lambda_t / max(rho, 1e-12))
        beta = beta_bar + shrink * (r / sigma_T2)

        # Step 7.
        delta = 0.5 * beta @ (mu0 + mu1)

    threshold = np.log((1.0 - w) / max(w, 1e-12))
    scores = X_T @ beta - delta
    return np.where(scores > threshold, 1.0, -1.0)


# ---------------------------------------------------------------------------
# One Monte Carlo repetition
# ---------------------------------------------------------------------------


def run_one(regime: str, seed: int) -> list:
    cfg = REGIMES[regime]
    d, n_T, n_S = cfg["d"], cfg["n_T"], cfg["n_S"]

    rng = np.random.default_rng(seed)
    if FIXED_UV:
        # Fixed direction pair, independent of `seed`/`regime`: same (u, v)
        # -- hence the same theta_T -- for every Monte Carlo repetition and
        # every mu in the sweep (see UV_SEED/FIXED_UV above).
        u, v = coordinate_free_uv(d, np.random.default_rng(UV_SEED))
    else:
        u, v = coordinate_free_uv(d, rng)
    theta_T = DELTA_T * u
    X_T, z_T = simulate_two_community(n_T, theta_T, rng)

    z_hat_target = target_based_estimate(X_T)
    err_target = misclustering_error(z_hat_target, z_T)

    # Estimated (not known) noise variances -- same estimator our own
    # AdaptiveTransferClustering already falls back on internally, applied
    # explicitly here so every method gets the identical estimate.
    sigma_T2_hat = estimate_noise_variance_rank1(X_T)

    rows = []
    for mu in MU_GRID:
        theta_S = make_theta_S(DELTA_S, mu, u, v)
        X_S, _z_S = simulate_two_community(n_S, theta_S, rng)
        sigma_S2_hat = estimate_noise_variance_rank1(X_S)

        z_hat_source = source_based_estimate(X_T, [X_S])
        err_source = misclustering_error(z_hat_source, z_T)

        adaptive = AdaptiveTransferClustering(
            selection="bootstrap", alpha=ADAPTIVE_ALPHA, n_boot=ADAPTIVE_N_BOOT,
            boundary_scale=ADAPTIVE_BOUNDARY_SCALE, sigma_T2=sigma_T2_hat, random_state=seed,
        )
        z_hat_adaptive, branch = adaptive.fit_predict(X_T, [X_S])
        err_adaptive = misclustering_error(z_hat_adaptive, z_T)

        z_hat_tgmm = tgmm_fit(X_T, X_S, sigma_T2_hat, sigma_S2_hat, TGMM_LAMBDA, EM_N_ITER, rng)
        err_tgmm = misclustering_error(z_hat_tgmm, z_T)

        z_hat_tlgmm = tlgmm_fit(X_T, X_S, sigma_T2_hat, sigma_S2_hat, EM_N_ITER, TLGMM_KAPPA0,
                                 TLGMM_C_LAMBDA0, rng)
        err_tlgmm = misclustering_error(z_hat_tlgmm, z_T)

        common = dict(regime=regime, seed=seed, mu=mu,
                       sigma_T2_hat=sigma_T2_hat, sigma_S2_hat=sigma_S2_hat)
        rows.append(dict(common, method="target",
                          error=err_target, branch="", C0_used=""))
        rows.append(dict(common, method="source",
                          error=err_source, branch="", C0_used=""))
        rows.append(dict(common, method="adaptive",
                          error=err_adaptive, branch=branch, C0_used=adaptive.C0_used_))
        rows.append(dict(common, method="tgmm",
                          error=err_tgmm, branch="", C0_used=""))
        rows.append(dict(common, method="tlgmm",
                          error=err_tlgmm, branch="", C0_used=""))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("regime", choices=list(REGIMES.keys()))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    rows = run_one(args.regime, args.seed)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.regime}_seed{args.seed}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

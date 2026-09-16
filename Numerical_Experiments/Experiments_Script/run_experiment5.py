#!/usr/bin/env python
"""Experiment 5: success and failure modes of the pooled method (matches
the paper's supplementary Figure 7, "Target misclustering error of the
pooled, target-only, and source-only estimators as a function of the
alignment parameter mu").

Usage (matches Slurm_Scripts/experiment5_pooled_modes/run_experiment5.sh):

    python run_experiment5.py <target_regime> <regime> <seed>

Dedicated study of `target_source_pooled_subspace_estimate` (rank-capped,
UNNORMALIZED -- `restrict_basis_rank=True, normalize=False`, both explicit
below, matching the paper's Algorithm 4 exactly): this estimator's
projection-subspace construction reuses the target's own data in-sample
to build its direction estimate, which can cause it to collapse toward
the target-only estimate when the target's own direction estimate has
larger raw magnitude than the source's, independent of how well-aligned
the source actually is. This experiment isolates and studies that
behavior directly, rather than as a side effect of a broader comparison.
It is structurally `run_experiment1.py` (same REGIMES, MU_GRID, theta
construction) with two differences:

  1. Only 3 methods: target, source, pooled. 
  2. A second swept axis, `target_regime`, fixes Delta_T at one of two
     values instead of the single Delta_T=0.8 Experiment 1 uses, so the
     mu sweep is repeated once with the target below its own target-only
     recovery threshold and once with it comfortably above:

       - "below_bbp": Delta_T = 0.8. Below `max(1, (d/n_T)^(1/4))` for
         every regime here (threshold = 1.778 for R1/R2's n_T=200,
         d=2000; 1.0 for R3's n_T=2500, d=2000, since d/n_T<1 there) --
         matches Experiment 1's setting exactly.
       - "above_bbp": Delta_T = 2.0. Comfortably above the threshold in
         every regime (2.0 vs 1.778 for R1/R2, vs 1.0 for R3), while
         still below Delta_S=3.0 (same DELTA_S_BY_REGIME as Experiment
         1), i.e. target SNR is recoverable on its own but still weaker
         than source SNR. The figure for above BBP not reported in paper
         and did not show any significant difference.

    Both values are threaded through the *same* (u, v) draw per (regime,
    seed) -- only Delta_T's magnitude changes, not theta_T's direction --
    so the two target_regime runs for a given (regime, seed) are a paired
    comparison, not independent draws.

One call = one Monte Carlo repetition for one (target_regime, regime)
cell, sweeping the full mu grid internally. Writes one CSV with 3 methods
x len(MU_GRID) rows to RESULTS_DIR.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from transfer_clustering.two_community import (
    target_based_estimate, source_based_estimate,
    target_source_pooled_subspace_estimate,
)

# ---------------------------------------------------------------------------
# Config -- edit these directly. If you change N_SEEDS, REGIME_ORDER, or
# TARGET_REGIME_ORDER, update the matching arithmetic in
# Slurm_Scripts/experiment5_pooled_modes/run_experiment5.sh to stay in sync.
# ---------------------------------------------------------------------------

# Same three aspect-ratio regimes as Experiment 1 (Section 5.1).
REGIMES = {
    "R1": dict(d=2000, n_T=200, n_S=200),
    "R2": dict(d=2000, n_T=200, n_S=2500),
    "R3": dict(d=2000, n_T=2500, n_S=2500),
}
REGIME_ORDER = ["R1", "R2", "R3"]  # fixes the SLURM array's regime_idx order

# The two target-SNR scenarios this experiment crosses against REGIMES.
# "threshold" here is max(1, (d/n_T)^(1/4)) -- see two_community.py's
# validation_threshold / target-only-condition discussion -- recorded per
# entry just for the docstring/notebook, not read by the code below (which
# recomputes it live from each regime's own (d, n_T) for the CSV).
TARGET_REGIMES = {
    "below_bbp": dict(Delta_T=0.8),  # Experiment 1's value; below threshold in every regime here
    "above_bbp": dict(Delta_T=2.0),  # above threshold in every regime here, still < Delta_S=3.0
}
TARGET_REGIME_ORDER = ["below_bbp", "above_bbp"]  # fixes the SLURM array's target_regime_idx order

MU_GRID = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.4, 0.8]  # alignment grid, finer near mu=0
SIGMA2 = 1.0                          # noise variance, known (Section 2.1's noise model)

# Delta_S per regime, set uniformly to 3.0 across all regimes -- same as
# Experiment 1, so "above_bbp"'s Delta_T=2.0 is guaranteed < Delta_S here.
DELTA_S_BY_REGIME = {
    "R1": 3.0,
    "R2": 3.0,
    "R3": 3.0,
}

N_SEEDS = 100  # Monte Carlo repetitions per (target_regime, regime) cell,
               # matching the paper's Figure 7

# Switch, matching run_experiment1.py's FIXED_UV/UV_SEED convention: False
# (default) resamples (u,v) fresh each Monte Carlo repetition. True pins
# (u,v) to a single draw from UV_SEED, reused for every
# target_regime/regime/seed/mu.
FIXED_UV = False
UV_SEED = 20260711

RESULTS_DIR = "../../Results_simulation/experiment5/raw"  # relative to this file's directory
FIELDNAMES = ["target_regime", "Delta_T", "bbp_threshold", "regime", "seed", "mu", "method", "error"]


# ---------------------------------------------------------------------------
# Simulation building blocks (two-community Gaussian mixture model, Section
# 2.1) -- duplicated from run_experiment1.py rather than shared, per this
# project's convention of one self-contained script per experiment.
# ---------------------------------------------------------------------------


def coordinate_free_uv(d: int, rng: np.random.Generator):
    """Sample a Haar-random orthonormal pair (u, v) in R^d:

        g, h ~ N(0, I_d) iid
        u := g / ||g||_2
        v := (h - <h,u> u) / ||h - <h,u> u||_2
    """
    g = rng.normal(size=d)
    u = g / np.linalg.norm(g)
    h = rng.normal(size=d)
    h_tilde = h - (h @ u) * u
    v = h_tilde / np.linalg.norm(h_tilde)
    return u, v


def make_theta_S(Delta_S: float, mu: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """theta_S := Delta_S * (mu*u + sqrt(1-mu^2)*v), which guarantees
    ||theta_S||_2 = Delta_S and
    |<theta_T,theta_S>| / (||theta_T|| ||theta_S||) = mu for theta_T = Delta_T*u."""
    return Delta_S * (mu * u + np.sqrt(1.0 - mu**2) * v)


def simulate_two_community(n: int, theta: np.ndarray, rng: np.random.Generator):
    """X_j = z_j * theta + eps_j, eps_j ~ N(0, I_d) iid, the two-community
    Gaussian mixture model of Section 2.1, with balanced labels z in
    {-1,+1}^n (up to a discrepancy of one observation when n is odd)."""
    d = theta.shape[0]
    half = n // 2
    z = np.concatenate([np.ones(half), -np.ones(n - half)])
    rng.shuffle(z)
    noise = rng.normal(size=(n, d))
    X = z[:, None] * theta[None, :] + noise
    return X, z


def misclustering_error(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    """L(z_hat, z_true) := min_{s in {-1,+1}} (1/n) sum_j 1{z_hat_j != s*z_true_j},
    the target misclustering-error loss used throughout the paper's
    numerical experiments, invariant to the global sign ambiguity of
    cluster labels."""
    err_plus = np.mean(z_hat != z_true)
    err_minus = np.mean(z_hat != -z_true)
    return float(min(err_plus, err_minus))


def bbp_threshold(d: int, n_T: int, sigma_T2: float = SIGMA2) -> float:
    """max(1, (d/n_T)^(1/4)) * sigma_T -- the target-only recovery
    threshold (see transfer_clustering/two_community.py's
    validation_threshold discussion)."""
    return max(1.0, (d / n_T) ** 0.25) * np.sqrt(sigma_T2)


# ---------------------------------------------------------------------------
# One Monte Carlo repetition
# ---------------------------------------------------------------------------


def run_one(target_regime: str, regime: str, seed: int) -> list:
    cfg = REGIMES[regime]
    d, n_T, n_S = cfg["d"], cfg["n_T"], cfg["n_S"]
    Delta_T = TARGET_REGIMES[target_regime]["Delta_T"]
    Delta_S = DELTA_S_BY_REGIME[regime]
    threshold = bbp_threshold(d, n_T)

    rng = np.random.default_rng(seed)
    if FIXED_UV:
        u, v = coordinate_free_uv(d, np.random.default_rng(UV_SEED))
    else:
        # Generated once, reused for every mu in the sweep -- and, since it
        # only depends on (regime, seed), reused identically across both
        # target_regime calls too, so "below_bbp" and "above_bbp" differ
        # only in Delta_T's magnitude, not theta_T's direction.
        u, v = coordinate_free_uv(d, rng)
    theta_T = Delta_T * u
    X_T, z_T = simulate_two_community(n_T, theta_T, rng)

    z_hat_target = target_based_estimate(X_T)
    err_target = misclustering_error(z_hat_target, z_T)

    rows = []
    for mu in MU_GRID:
        theta_S = make_theta_S(Delta_S, mu, u, v)
        X_S, _z_S = simulate_two_community(n_S, theta_S, rng)

        z_hat_source = source_based_estimate(X_T, [X_S])
        err_source = misclustering_error(z_hat_source, z_T)

        z_hat_pooled = target_source_pooled_subspace_estimate(
            X_T, [X_S], restrict_basis_rank=True, normalize=False,
        )
        err_pooled = misclustering_error(z_hat_pooled, z_T)

        common = dict(target_regime=target_regime, Delta_T=Delta_T, bbp_threshold=threshold,
                      regime=regime, seed=seed, mu=mu)
        rows.append(dict(common, method="target", error=err_target))
        rows.append(dict(common, method="source", error=err_source))
        rows.append(dict(common, method="pooled", error=err_pooled))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_regime", choices=list(TARGET_REGIMES.keys()))
    parser.add_argument("regime", choices=list(REGIMES.keys()))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    rows = run_one(args.target_regime, args.regime, args.seed)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.target_regime}_{args.regime}_seed{args.seed}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

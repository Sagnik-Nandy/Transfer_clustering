#!/usr/bin/env python
"""Experiment 1: "Dependence of performance on alignment strength"
(Section 5.1 of the paper; produces the misclustering-error-vs-mu curves).

Usage (matches Slurm_Scripts/experiment1_alignment/run_experiment1.sh):

    python run_experiment1.py <regime> <seed>

One call = one Monte Carlo repetition for one regime, sweeping the full
mu (alignment) grid internally. The two-community Gaussian mixture model
of Section 2.1 requires the same Haar-random direction pair (u, v) to be
used across the entire parameter sweep within one repetition -- satisfied
here by generating (u, v), and the whole target dataset X_T (which does
not depend on mu, since theta_T does not), once per repetition and
reusing both across the mu loop; only the source dataset X_S is
regenerated at each mu (since theta_S depends on mu).

Set FIXED_UV=True below to instead pin (u, v) to a single draw (UV_SEED)
reused across every regime, every seed, and every mu -- so theta_T is
literally identical throughout the whole experiment, and only the
noise/label draws vary by seed. Default (FIXED_UV=False) resamples (u, v)
fresh each Monte Carlo repetition, matching Section 5.1's construction.

Compares three methods, evaluated at every point of the mu grid: the
target-only estimator, the source-based estimator, and the adaptive
selector between them (Algorithm 2). The target+source pooled-subspace
estimator (Algorithm 4) is not included here -- its success and failure
modes as a function of alignment are the dedicated subject of Experiment
5, which reuses this same (u, v)/theta_T construction.

Writes one CSV with 3 methods x len(MU_GRID) rows to RESULTS_DIR: target,
source, adaptive.
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
)

# ---------------------------------------------------------------------------
# Config -- edit these directly. If you change N_SEEDS or REGIME_ORDER,
# update the matching arithmetic in
# Slurm_Scripts/experiment1_alignment/run_experiment1.sh to stay in sync.
# ---------------------------------------------------------------------------

# Section 5.1, "three aspect-ratio regimes": concrete starting choices.
REGIMES = {
    "R1": dict(d=2000, n_T=200, n_S=200),
    "R2": dict(d=2000, n_T=200, n_S=2500),
    "R3": dict(d=2000, n_T=2500, n_S=2500),
}
REGIME_ORDER = ["R1", "R2", "R3"]  # fixes the SLURM array's regime_idx order

DELTA_T = 0.8                        # fixed target signal (Section 5.1)
MU_GRID = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.4, 0.8]  # alignment grid, finer near mu=0
SIGMA2 = 1.0                          # noise variance, known (Section 2.1's noise model)

# Delta_S per regime, set uniformly to 3.0 across all regimes.
DELTA_S_BY_REGIME = {
    "R1": 3.0,
    "R2": 3.0,
    "R3": 3.0,
}

N_SEEDS = 100  # Monte Carlo repetitions per regime, matching the paper's Figure 3

# Switch (see module docstring): False (default) resamples (u,v) fresh each
# Monte Carlo repetition, matching Section 5.1. True pins (u,v) to a single
# draw from UV_SEED, reused for every regime/seed/mu, so theta_T = DELTA_T * u
# is the *same* vector throughout the whole experiment and only the
# noise/label draws (rng seeded by `seed` in run_one) vary. UV_SEED is
# deliberately independent of N_SEEDS/REGIME_ORDER.
FIXED_UV = False
UV_SEED = 20260711

# AdaptiveTransferClustering knobs. ADAPTIVE_N_BOOT is the main cost lever:
# each bootstrap replicate re-runs the spectral estimator on synthetic data
# of the same (n_T, d) as the real problem, so cost scales with both
# ADAPTIVE_N_BOOT and regime size. Test wall-clock time on the cluster with
# a small --array range (e.g. 0-2) before committing to the full
# N_SEEDS * len(REGIMES) sweep.
ADAPTIVE_N_BOOT = 50
ADAPTIVE_ALPHA = 0.5
ADAPTIVE_BOUNDARY_SCALE = 1.0

RESULTS_DIR = "../../Results_simulation/experiment1/raw"  # relative to this file's directory
FIELDNAMES = ["regime", "seed", "mu", "method", "error", "branch", "C0_used"]


# ---------------------------------------------------------------------------
# Simulation building blocks (two-community Gaussian mixture model, Section 2.1)
# ---------------------------------------------------------------------------


def coordinate_free_uv(d: int, rng: np.random.Generator):
    """Sample a Haar-random orthonormal pair (u, v) in R^d, as in Section 5.1:

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
    """theta_S := Delta_S * (mu*u + sqrt(1-mu^2)*v), as in Section 5.1, which
    guarantees ||theta_S||_2 = Delta_S and
    |<theta_T,theta_S>| / (||theta_T|| ||theta_S||) = mu for theta_T = Delta_T*u."""
    return Delta_S * (mu * u + np.sqrt(1.0 - mu**2) * v)


def simulate_two_community(n: int, theta: np.ndarray, rng: np.random.Generator):
    """X_j = z_j * theta + eps_j, eps_j ~ N(0, I_d) iid, the two-community
    Gaussian mixture model of Section 2.1, with balanced labels z in
    {-1,+1}^n (up to a discrepancy of one observation when n is odd, per
    Section 5.1's balanced-label convention)."""
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


# ---------------------------------------------------------------------------
# One Monte Carlo repetition
# ---------------------------------------------------------------------------


def run_one(regime: str, seed: int) -> list:
    cfg = REGIMES[regime]
    d, n_T, n_S = cfg["d"], cfg["n_T"], cfg["n_S"]
    Delta_S = DELTA_S_BY_REGIME[regime]

    rng = np.random.default_rng(seed)
    if FIXED_UV:
        # Fixed direction pair, independent of `seed`/`regime`: same (u, v)
        # -- hence the same theta_T -- for every Monte Carlo repetition and
        # every mu in the sweep (see UV_SEED/FIXED_UV above).
        u, v = coordinate_free_uv(d, np.random.default_rng(UV_SEED))
    else:
        # Generated once, reused for every mu in the sweep (Section 5.1).
        u, v = coordinate_free_uv(d, rng)
    theta_T = DELTA_T * u
    X_T, z_T = simulate_two_community(n_T, theta_T, rng)

    z_hat_target = target_based_estimate(X_T)
    err_target = misclustering_error(z_hat_target, z_T)

    rows = []
    for mu in MU_GRID:
        theta_S = make_theta_S(Delta_S, mu, u, v)
        X_S, _z_S = simulate_two_community(n_S, theta_S, rng)

        z_hat_source = source_based_estimate(X_T, [X_S])
        err_source = misclustering_error(z_hat_source, z_T)

        adaptive = AdaptiveTransferClustering(
            selection="bootstrap",
            alpha=ADAPTIVE_ALPHA,
            n_boot=ADAPTIVE_N_BOOT,
            boundary_scale=ADAPTIVE_BOUNDARY_SCALE,
            sigma_T2=SIGMA2,
            random_state=seed,
        )
        z_hat_adaptive, branch = adaptive.fit_predict(X_T, [X_S])
        err_adaptive = misclustering_error(z_hat_adaptive, z_T)

        rows.append(dict(regime=regime, seed=seed, mu=mu, method="target",
                          error=err_target, branch="", C0_used=""))
        rows.append(dict(regime=regime, seed=seed, mu=mu, method="source",
                          error=err_source, branch="", C0_used=""))
        rows.append(dict(regime=regime, seed=seed, mu=mu, method="adaptive",
                          error=err_adaptive, branch=branch, C0_used=adaptive.C0_used_))

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

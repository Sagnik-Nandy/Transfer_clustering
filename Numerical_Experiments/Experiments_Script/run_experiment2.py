#!/usr/bin/env python
"""Experiment 2: heatmaps of misclustering error over (Delta_T, Delta_S)
at a fixed alignment mu (the paper's supplementary experiment exploring
dependence on target/source signal strength, Appendix A.1).

Usage (matches Slurm_Scripts/experiment2_heatmap/run_experiment2.sh):

    python run_experiment2.py <regime> <seed>

One call = one Monte Carlo repetition for one regime, sweeping the full
(Delta_T, Delta_S) grid internally. As in Experiment 1, the same
Haar-random direction pair (u, v) is used across the entire parameter
sweep within one repetition -- generated once per repetition and reused
for the whole grid. Within that, the target dataset X_T only depends on
Delta_T (not Delta_S), so it -- and the target-only error -- is generated
once per Delta_T row and reused across all Delta_S columns.

Set FIXED_UV=True below to instead pin (u,v) to a single draw (UV_SEED)
reused across every regime, every seed, and the whole grid -- so only the
noise/labels vary by seed. Default (FIXED_UV=False) resamples (u,v) fresh
each Monte Carlo repetition, matching the same construction as Experiment 1.

Compares three methods at every grid point: the target-only estimator,
the source-based estimator, and the adaptive selector between them
(Algorithm 2). Writes one CSV with 3 methods x len(DELTA_T_GRID) x
len(DELTA_S_GRID) rows to RESULTS_DIR: target, source, adaptive.
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
# Slurm_Scripts/experiment2_heatmap/run_experiment2.sh to stay in sync.
# ---------------------------------------------------------------------------

# Same three aspect-ratio regimes as Experiment 1.
REGIMES = {
    "R1": dict(d=2000, n_T=200, n_S=200),
    "R2": dict(d=2000, n_T=200, n_S=2500),
    "R3": dict(d=2000, n_T=2500, n_S=2500),
}
REGIME_ORDER = ["R1", "R2", "R3"]  # fixes the SLURM array's regime_idx order

MU = 0.8  # fixed alignment -- Experiment 2 varies (Delta_T, Delta_S) instead
SIGMA2 = 1.0  # noise variance, known (Section 2.1's noise model)

# 5x5 grid, kept modest since regime R3 (n_T=n_S=2500) is the slowest per task.
DELTA_T_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]
DELTA_S_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]

N_SEEDS = 100  # Monte Carlo repetitions per regime

# Switch (see module docstring), matching run_experiment1.py's
# FIXED_UV/UV_SEED convention: False (default) resamples (u,v) fresh each
# Monte Carlo repetition. True pins (u,v) to a single draw from UV_SEED,
# reused for every regime/seed/grid-point.
FIXED_UV = False
UV_SEED = 20260711

# AdaptiveTransferClustering knobs. Experiment 2's grid has 5*5=25 cells per
# regime, and each adaptive call's cost scales with ADAPTIVE_N_BOOT (each
# bootstrap replicate re-runs the spectral estimator).
ADAPTIVE_N_BOOT = 20
ADAPTIVE_ALPHA = 0.5
ADAPTIVE_BOUNDARY_SCALE = 1.0

RESULTS_DIR = "../../Results_simulation/experiment2/raw"  # relative to this file's directory
FIELDNAMES = ["regime", "seed", "Delta_T", "Delta_S", "method", "error", "branch", "C0_used"]


# ---------------------------------------------------------------------------
# Simulation building blocks (two-community Gaussian mixture model, Section
# 2.1) -- duplicated from run_experiment1.py rather than shared, on purpose:
# each experiment script here is deliberately self-contained rather than
# split into shared config/utils modules.
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


# ---------------------------------------------------------------------------
# One Monte Carlo repetition
# ---------------------------------------------------------------------------


def run_one(regime: str, seed: int) -> list:
    cfg = REGIMES[regime]
    d, n_T, n_S = cfg["d"], cfg["n_T"], cfg["n_S"]

    rng = np.random.default_rng(seed)
    if FIXED_UV:
        # Fixed direction pair, independent of `seed`/`regime`: same (u, v)
        # for every Monte Carlo repetition and the whole (Delta_T, Delta_S)
        # grid (see UV_SEED/FIXED_UV above).
        u, v = coordinate_free_uv(d, np.random.default_rng(UV_SEED))
    else:
        # Generated once, reused for the whole (Delta_T, Delta_S) grid.
        u, v = coordinate_free_uv(d, rng)

    rows = []
    for Delta_T in DELTA_T_GRID:
        # X_T depends only on Delta_T, not Delta_S -- generated once per row
        # and reused across all Delta_S columns.
        theta_T = Delta_T * u
        X_T, z_T = simulate_two_community(n_T, theta_T, rng)

        z_hat_target = target_based_estimate(X_T)
        err_target = misclustering_error(z_hat_target, z_T)

        for Delta_S in DELTA_S_GRID:
            theta_S = make_theta_S(Delta_S, MU, u, v)
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

            rows.append(dict(regime=regime, seed=seed, Delta_T=Delta_T, Delta_S=Delta_S,
                              method="target", error=err_target, branch="", C0_used=""))
            rows.append(dict(regime=regime, seed=seed, Delta_T=Delta_T, Delta_S=Delta_S,
                              method="source", error=err_source, branch="", C0_used=""))
            rows.append(dict(regime=regime, seed=seed, Delta_T=Delta_T, Delta_S=Delta_S,
                              method="adaptive", error=err_adaptive, branch=branch,
                              C0_used=adaptive.C0_used_))

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

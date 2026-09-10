#!/usr/bin/env python
"""Experiment 3: "Performance in multi-source and multi-cluster settings"
(Section 5.2 of the paper).

Usage (matches Slurm_Scripts/experiment3_complementary/run_experiment3.sh):

    python run_experiment3.py <seed>

Unlike Experiments 1-2, this is a single fixed scenario (K=3 target
clusters, 2 source datasets, one concrete parameter set given directly by
the paper) -- there is no regime or grid sweep, just B Monte Carlo
repetitions. One call = one repetition, evaluating all 5 methods.

NOTE: target-only and adaptive clustering route through
RelaxedKMeans (transfer_clustering/relaxed_kmeans.py), which by default
solves the relaxed K-means SDP via its own ADMM solver -- pure numpy/scipy,
no cvxpy needed. cvxpy is only required if RelaxedKMeans is explicitly
configured with a cvxpy solver name (e.g. solver="SCS"), which this script
does not do.

Compares five methods: target-only, source-1-only, source-2-only,
combined-source (both sources pooled into one projection subspace), and
the adaptive selector (Algorithm 3). The target+source pooled-subspace
estimator (Algorithm 4) is not included here -- see Experiment 5 for a
dedicated study of its behavior.

Writes one CSV with 5 rows (one per method) to RESULTS_DIR: target_only,
source1_only, source2_only, combined_sources, adaptive.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from transfer_clustering.multi_cluster import (
    AdaptiveProjectedClustering, onehot_to_labels,
)

# ---------------------------------------------------------------------------
# Config -- edit these directly. If you change N_SEEDS, update the
# --array bound in Slurm_Scripts/experiment3_complementary/run_experiment3.sh.
# ---------------------------------------------------------------------------

K = 3  # target clusters (Section 5.2)

# Section 5.2, "concrete starting parameters" -- given directly by the paper.
D = 1000
N_T = 200
N_S1 = 1500
N_S2 = 1500
SIGMA2 = 1.0  # noise variance, known (sigma_T^2 = sigma_S1^2 = sigma_S2^2 = 1)

A_TARGET = 2.0   # "a": half the (1,2) target separation
B_TARGET = 2.0   # "b": target cluster 3's offset along v
A_SOURCE1 = 2.75  # "A": source-1 separation strength (large)
B_SOURCE2 = 1.6   # "B": source-2 separation strength
ETA = 0.25        # "eta": small leakage of source 1 into the (2,3) contrast

N_SEEDS = 100  # Monte Carlo repetitions; SDP solves are much slower than
               # Experiments 1-2's spectral method, so this starts smaller --
               # test timing on the cluster before raising it.

# Switch, matching run_experiment1.py's FIXED_UV/UV_SEED convention: False
# (default) resamples (u, v, r) fresh each Monte Carlo repetition, matching
# Section 5.2's construction. True pins (u, v, r) to a single draw from
# UV_SEED, reused for every seed, so theta_T, theta_S1, theta_S2 are
# identical throughout the whole experiment and only the noise/label draws
# (rng seeded by `seed`) vary.
FIXED_UV = False
UV_SEED = 20260711

# AdaptiveProjectedClustering (adaptive method) knobs. ADAPTIVE_N_BOOT is the
# main cost lever here: each bootstrap replicate re-solves the relaxed
# K-means SDP, which is much slower than Experiments 1-2's spectral
# estimator, so this starts low. Test wall-clock time on the cluster with
# a small --array range before committing to the full N_SEEDS sweep.
ADAPTIVE_N_BOOT = 10
ADAPTIVE_ALPHA = 0.5
ADAPTIVE_BOUNDARY_SCALE = 1.0

RESULTS_DIR = "../../Results_simulation/experiment3/raw"  # relative to this file's directory
FIELDNAMES = ["seed", "method", "error_overall", "error_12", "error_13", "error_23",
              "branch", "D0_used"]


# ---------------------------------------------------------------------------
# Simulation building blocks (multi-cluster Gaussian mixture model, Section 4;
# concrete construction of Section 5.2)
# ---------------------------------------------------------------------------


def coordinate_free_uvr(d: int, rng: np.random.Generator):
    """Sample a Haar-random orthonormal triple (u, v, r) in R^d via
    Gram-Schmidt of 3 iid N(0, I_d) vectors, matching Section 5.2's
    construction."""
    g1 = rng.normal(size=d)
    u = g1 / np.linalg.norm(g1)
    g2 = rng.normal(size=d)
    g2 = g2 - (g2 @ u) * u
    v = g2 / np.linalg.norm(g2)
    g3 = rng.normal(size=d)
    g3 = g3 - (g3 @ u) * u - (g3 @ v) * v
    r = g3 / np.linalg.norm(g3)
    return u, v, r


def simulate_multi_cluster(n: int, thetas: np.ndarray, rng: np.random.Generator):
    """X_j = thetas[:, z_j] + eps_j, eps_j ~ N(0, I_d) iid, with balanced
    labels z in {0,...,K-1}^n (as even as n allows). thetas has shape (d, K)."""
    d, K = thetas.shape
    base, rem = divmod(n, K)
    counts = [base] * K
    for k in range(rem):
        counts[k] += 1
    z = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    rng.shuffle(z)
    noise = rng.normal(size=(n, d))
    X = thetas[:, z].T + noise
    return X, z


def overall_error_K(z_hat: np.ndarray, z_true: np.ndarray, K: int) -> float:
    """Best label-permutation misclustering error over all K! permutations
    -- the K>2 analogue of the K=2 sign-flip matching used in Experiments
    1-2's misclustering-error loss."""
    best = 1.0
    for perm in itertools.permutations(range(K)):
        mapped = np.array(perm)[z_hat]
        err = float(np.mean(mapped != z_true))
        best = min(best, err)
    return best


def pairwise_contrast_error(z_hat: np.ndarray, z_true: np.ndarray, a: int, b: int) -> float:
    """Restrict to points with true label in {a, b}; compute the best
    two-class misclustering error after best label matching (matching the
    pairwise-contrast metrics L_ab reported alongside L_mult in the
    paper's Figure 4): brute-force over every function from the predicted
    labels present in this subset to {a, b}."""
    mask = np.isin(z_true, [a, b])
    zh, zt = z_hat[mask], z_true[mask]
    labels_present = sorted(set(zh.tolist()))
    best = 1.0
    for assignment in itertools.product([a, b], repeat=len(labels_present)):
        mapping = dict(zip(labels_present, assignment))
        mapped = np.array([mapping[x] for x in zh])
        err = float(np.mean(mapped != zt))
        best = min(best, err)
    return best


# ---------------------------------------------------------------------------
# One Monte Carlo repetition
# ---------------------------------------------------------------------------


def run_one(seed: int) -> list:
    rng = np.random.default_rng(seed)
    if FIXED_UV:
        # Fixed direction triple, independent of `seed`: same (u, v, r) for
        # every Monte Carlo repetition (see UV_SEED/FIXED_UV above).
        u, v, r = coordinate_free_uvr(D, np.random.default_rng(UV_SEED))
    else:
        u, v, r = coordinate_free_uvr(D, rng)

    # Target centers: strong (1,2) contrast, weak (1,3)/(2,3).
    theta_T = np.stack([A_TARGET * u, -A_TARGET * u, B_TARGET * v], axis=1)
    # Source 1: separates (1,2) and (1,3), not (2,3).
    theta_S1 = np.stack([A_SOURCE1 * u, -A_SOURCE1 * u, -A_SOURCE1 * u + ETA * r], axis=1)
    # Source 2: isolates the missing (2,3) contrast.
    theta_S2 = np.stack([np.zeros(D), B_SOURCE2 * v, -B_SOURCE2 * v], axis=1)

    X_T, z_T = simulate_multi_cluster(N_T, theta_T, rng)
    X_S1, _ = simulate_multi_cluster(N_S1, theta_S1, rng)
    X_S2, _ = simulate_multi_cluster(N_S2, theta_S2, rng)

    def eval_method(name, Z_onehot, branch, D0_used=""):
        labels = onehot_to_labels(Z_onehot)
        return dict(
            seed=seed, method=name,
            error_overall=overall_error_K(labels, z_T, K),
            error_12=pairwise_contrast_error(labels, z_T, 0, 1),
            error_13=pairwise_contrast_error(labels, z_T, 0, 2),
            error_23=pairwise_contrast_error(labels, z_T, 1, 2),
            branch=branch, D0_used=D0_used,
        )

    rows = []

    # random_state=seed threaded into every RelaxedKMeans call below (via
    # AdaptiveProjectedClustering.fit_predict's rk_kwargs handling), so every
    # method is reproducible run-to-run for a fixed seed.
    target_model = AdaptiveProjectedClustering(K=K, selection="manual", manual_choice="target",
                                                sigma_T2=SIGMA2, random_state=seed)
    Z_target, branch = target_model.fit_predict(X_T, [])
    rows.append(eval_method("target_only", Z_target, branch))

    source1_model = AdaptiveProjectedClustering(K=K, selection="manual", manual_choice="source",
                                                 sigma_T2=SIGMA2, random_state=seed)
    Z_source1, branch = source1_model.fit_predict(X_T, [X_S1])
    rows.append(eval_method("source1_only", Z_source1, branch))

    source2_model = AdaptiveProjectedClustering(K=K, selection="manual", manual_choice="source",
                                                 sigma_T2=SIGMA2, random_state=seed)
    Z_source2, branch = source2_model.fit_predict(X_T, [X_S2])
    rows.append(eval_method("source2_only", Z_source2, branch))

    combined_model = AdaptiveProjectedClustering(K=K, selection="manual", manual_choice="source",
                                                  sigma_T2=SIGMA2, random_state=seed)
    Z_combined, branch = combined_model.fit_predict(X_T, [X_S1, X_S2])
    rows.append(eval_method("combined_sources", Z_combined, branch))

    adaptive_model = AdaptiveProjectedClustering(
        K=K, selection="bootstrap", alpha=ADAPTIVE_ALPHA, n_boot=ADAPTIVE_N_BOOT,
        boundary_scale=ADAPTIVE_BOUNDARY_SCALE, sigma_T2=SIGMA2, random_state=seed,
    )
    Z_adaptive, branch = adaptive_model.fit_predict(X_T, [X_S1, X_S2])
    rows.append(eval_method("adaptive", Z_adaptive, branch, D0_used=adaptive_model.D0_used_))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    rows = run_one(args.seed)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seed{args.seed}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

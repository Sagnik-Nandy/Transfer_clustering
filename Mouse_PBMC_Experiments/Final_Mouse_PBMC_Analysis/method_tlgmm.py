"""TL-GMM (Tian, Weng, Xia & Feng, arXiv:2209.15224), multi-cluster (K=9),
fused across all 5 other batches (this dataset's source list).

=============================================================================
IMPORTANT MODELING NOTES
=============================================================================

(1) Isotropic covariance, estimated per dataset (via
    `estimate_noise_variance_rankK`, already in the transfer_clustering
    package) -- not a fixed/known value, and not assumed equal across
    datasets (matches Experiment 4's design decision).

(2) ALIGNMENT (Algorithm 6, supplementary Section S.2.2). With R=9
    clusters, naively fusing discriminant coefficients across the 5
    sources (and then the target) requires knowing that "cluster index 3"
    means the same cell type in every dataset -- which an independent
    K-means/EM fit on each dataset does NOT guarantee (GMM component
    labels are only identified up to a permutation). The paper's own
    solution is Algorithm 6 (greedy search): fix task 1's own initial
    labeling as the canonical reference; for each subsequent task, try
    every one of its R clusters as "the reference class" and, for each
    choice, solve the remaining (R-1)-to-(R-1) label matching EXACTLY via
    the Hungarian algorithm (comparing discriminant coefficient vectors
    against every previously-aligned task); keep the best of the R
    choices. The paper's own stated complexity for this is
    O(R! * K * R*K^2) (i.e. they still brute-force each single task's R!
    permutations) which is wasteful for our R=9 (9! ~ 363,000). Our
    `_align_one_task` below solves the IDENTICAL argmin exactly but
    efficiently -- O(R) choices of reference class, each an O(R^3)
    Hungarian solve, so O(R^4) total per task -- by exploiting that for a
    FIXED reference-class choice, the remaining assignment is a genuine
    linear assignment problem. This is a smarter implementation of the
    same specified optimization, not an approximation. As the paper
    recommends, we run the greedy procedure over several random task
    orderings and keep the lowest-total-score result (N_ALIGNMENT_SHUFFLES
    below, fewer than the paper's 200 for practicality). Recovers the true
    correspondence exactly, including under a fully scrambled task
    labeling.

(3) Algorithm 4's Step 12 (per-contrast r=2..R aggregation across sources)
    is a joint minimization over each source's own beta_r^(k) AND a shared
    center beta_bar_r. For fixed beta_bar_r, each source's beta_r^(k) has
    the same closed-form isotropic shrinkage solution derived for TL-GMM
    in Experiment 4 (run_experiment4.py's tlgmm_fit). For fixed
    {beta_r^(k)}, beta_bar_r minimizing a sum of weighted norms
    sum_k w_k * ||beta_r^(k) - beta_bar_r|| is the weighted geometric
    median (Fermat-Weber point), solved by a short Weiszfeld iteration. We
    alternate these two closed-form updates (see `_solve_aggregation`)
    to approximately solve the joint problem -- the paper does not specify
    a numerical solver for this step, only the objective itself.

(4) C_lambda0 (TL-GMM's penalty-schedule scale) is selected via the
    paper's own real procedure (Sec S.5.1.7), NOT fixed by hand as in
    Experiment 4: find C'_max, the smallest C_lambda0 that collapses
    every beta_r entirely to beta_bar_r (full shrinkage), then build a
    log-spaced grid from C'_max/50 to 2*C'_max, and pick the grid point
    maximizing average held-out log-likelihood under a k-fold split of the
    target cells (the paper doesn't specify the CV scoring criterion for
    unsupervised clustering; held-out log-likelihood under the fitted
    isotropic GMM is the standard choice for GMM model selection).
    kappa0 = 1/3, matching the paper's own fixed convention (Sec S.5.1.8).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from transfer_clustering.multi_cluster import estimate_noise_variance_rankK

from common import K, center_data

N_ALIGNMENT_SHUFFLES = 50   # Algorithm 6, task-ordering robustness (paper uses 200)
MTL_EM_ITERS = 30           # Algorithm 4 outer EM iterations (sources)
TLGMM_EM_ITERS = 30         # Algorithm 7-multicluster outer EM iterations (target)
AGGREGATION_INNER_ITERS = 20  # Step 12's beta/beta_bar alternation (per contrast)
TLGMM_KAPPA0 = 1.0 / 3.0    # matches the paper's fixed convention (Sec S.5.1.8)
CV_N_FOLDS = 5
CV_N_GRID = 10              # log-spaced grid size between C'_max/50 and 2*C'_max


# ---------------------------------------------------------------------------
# Algorithm 6: greedy multi-cluster alignment (see module docstring, note 2)
# ---------------------------------------------------------------------------


def _beta_vectors(mu: np.ndarray, sigma2: float, ref_idx: int):
    """mu: (R, d) cluster centers. Returns (idx, betas): idx are the R-1 raw
    indices excluding ref_idx, betas[i] = (mu[idx[i]] - mu[ref_idx]) / sigma2."""
    R = mu.shape[0]
    idx = [r for r in range(R) if r != ref_idx]
    betas = (mu[idx] - mu[ref_idx]) / sigma2
    return idx, betas


def _align_one_task(mu_k: np.ndarray, sigma2_k: float, ref_betas_list) -> tuple:
    """Exact argmin of Algorithm 6's score (eq. S.2.12) over all R!
    permutations of task k's cluster labels, computed efficiently (see
    module docstring, note 2). Returns (perm, cost) where perm[0] is the
    raw index chosen as the reference class and perm[1:] are the raw
    indices for canonical labels 2..R."""
    R = mu_k.shape[0]
    best_cost = np.inf
    best_perm = None
    for j0 in range(R):
        idx, betas = _beta_vectors(mu_k, sigma2_k, j0)  # (R-1,), (R-1, d)
        n = len(idx)
        C = np.zeros((n, n))
        for i in range(n):
            for r in range(n):
                C[i, r] = sum(np.linalg.norm(betas[i] - ref_b[r]) for ref_b in ref_betas_list)
        row_ind, col_ind = linear_sum_assignment(C)
        total_cost = C[row_ind, col_ind].sum()
        if total_cost < best_cost:
            best_cost = total_cost
            perm = np.zeros(R, dtype=int)
            perm[0] = j0
            for i, r in zip(row_ind, col_ind):
                perm[r + 1] = idx[i]
            best_perm = perm
    return best_perm, best_cost


def _total_alignment_score(mus, sigma2s, perms) -> float:
    """Full pairwise score (across all task pairs, not just sequential) for
    a completed alignment, used to compare across random task orderings."""
    betas = []
    for mu, sigma2, perm in zip(mus, sigma2s, perms):
        mu_aligned = mu[perm]
        betas.append((mu_aligned[1:] - mu_aligned[0]) / sigma2)
    K_tasks = len(betas)
    R = betas[0].shape[0] + 1
    total = 0.0
    for r in range(R - 1):
        for k1 in range(K_tasks):
            for k2 in range(k1 + 1, K_tasks):
                total += np.linalg.norm(betas[k1][r] - betas[k2][r])
    return total


def greedy_multitask_alignment(mus: list, sigma2s: list, n_shuffles: int = N_ALIGNMENT_SHUFFLES,
                                rng: np.random.Generator = None) -> list:
    """Algorithm 6, robustified over random task orderings (as the paper
    itself recommends, Sec S.2.1 remark). Returns a list of permutations,
    one per task, in the ORIGINAL task order passed in (mus[0], mus[1], ...)."""
    if rng is None:
        rng = np.random.default_rng(0)
    K_tasks = len(mus)
    best_score = np.inf
    best_perms_original_order = None
    orderings = [list(range(K_tasks))]
    for _ in range(n_shuffles - 1):
        orderings.append(list(rng.permutation(K_tasks)))

    for order in orderings:
        perms_in_order = [np.arange(mus[order[0]].shape[0])]
        aligned_betas = [_beta_vectors(mus[order[0]], sigma2s[order[0]], 0)[1]]
        for k in order[1:]:
            perm_k, _ = _align_one_task(mus[k], sigma2s[k], aligned_betas)
            perms_in_order.append(perm_k)
            mu_k_aligned = mus[k][perm_k]
            aligned_betas.append((mu_k_aligned[1:] - mu_k_aligned[0]) / sigma2s[k])
        perms_original_order = [None] * K_tasks
        for pos, task_idx in enumerate(order):
            perms_original_order[task_idx] = perms_in_order[pos]
        score = _total_alignment_score(mus, sigma2s, perms_original_order)
        if score < best_score:
            best_score = score
            best_perms_original_order = perms_original_order
    return best_perms_original_order


def _initial_kmeans(X: np.ndarray, K: int, random_state: int):
    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return km.cluster_centers_, labels


# ---------------------------------------------------------------------------
# Multinomial-softmax responsibility and the shared aggregation solver
# ---------------------------------------------------------------------------


def _multinomial_responsibility(X: np.ndarray, w: np.ndarray, beta: np.ndarray,
                                 delta: np.ndarray) -> np.ndarray:
    """gamma^(r)(z) = w_r*exp(beta_r^T z - delta_r) / sum_r' w_r'*exp(...),
    with w[0]=w_1 (reference, beta_1=0, delta_1=0 by convention already
    folded into w/beta/delta's r=0 entries being the reference row). Note
    sigma2 doesn't appear here -- it was already used upstream to form
    beta_r = Sigma^-1(mu_r-mu_1) from the raw means; this formula only
    needs the resulting (w, beta, delta).
    beta, delta: (R-1, d)/(R-1,) for the R-1 non-reference classes; w: (R,).
    Returns (n, R) responsibility matrix."""
    n = X.shape[0]
    R = w.shape[0]
    logits = np.zeros((n, R))
    logits[:, 0] = np.log(np.maximum(w[0], 1e-12))
    logits[:, 1:] = np.log(np.maximum(w[1:], 1e-12))[None, :] + (X @ beta.T - delta[None, :])
    logits -= logits.max(axis=1, keepdims=True)
    unnorm = np.exp(logits)
    return unnorm / unnorm.sum(axis=1, keepdims=True)


def _weighted_geometric_median(points: np.ndarray, weights: np.ndarray, n_iter: int = 25) -> np.ndarray:
    """Weiszfeld iteration for the weighted Fermat-Weber point minimizing
    sum_k weights[k] * ||points[k] - x||_2."""
    x = np.average(points, axis=0, weights=weights)
    for _ in range(n_iter):
        dists = np.linalg.norm(points - x[None, :], axis=1)
        dists = np.maximum(dists, 1e-10)
        w = weights / dists
        x = (w[:, None] * points).sum(axis=0) / w.sum()
    return x


def _shrink_closed_form(target_minus_ref: np.ndarray, beta_bar: np.ndarray, sigma2: float,
                         lambda_t: float) -> np.ndarray:
    """Closed-form isotropic shrinkage solution (see run_experiment4.py's
    tlgmm_fit for the full derivation): beta = beta_bar + max(0, 1 -
    lambda_t/rho) * r/sigma2, r = target_minus_ref - sigma2*beta_bar,
    rho = ||r||."""
    r = target_minus_ref - sigma2 * beta_bar
    rho = np.linalg.norm(r)
    shrink = max(0.0, 1.0 - lambda_t / max(rho, 1e-12))
    return beta_bar + shrink * (r / sigma2)


def _solve_aggregation(target_minus_ref_list, sigma2_list, n_list, lambda_t: float,
                        n_inner_iters: int = AGGREGATION_INNER_ITERS):
    """Algorithm 4 Step 12 for a single contrast r: jointly solve for each
    task's beta_r^(k) and the shared center beta_bar_r (see module
    docstring, note 3). Returns (beta_bar, [beta^(k) for each k])."""
    K_tasks = len(target_minus_ref_list)
    weights = np.array([np.sqrt(n_list[k]) * lambda_t for k in range(K_tasks)])
    beta_bar = np.mean(target_minus_ref_list, axis=0)  # init
    betas = [target_minus_ref_list[k] / sigma2_list[k] for k in range(K_tasks)]
    for _ in range(n_inner_iters):
        betas = [
            _shrink_closed_form(target_minus_ref_list[k], beta_bar, sigma2_list[k],
                                 weights[k] / n_list[k])
            for k in range(K_tasks)
        ]
        beta_bar = _weighted_geometric_median(np.stack(betas), weights)
    return beta_bar, betas


# ---------------------------------------------------------------------------
# Algorithm 4: MTL-GMM (multi-cluster), fused across the sources
# ---------------------------------------------------------------------------


def mtl_gmm_multicluster(sources: list, K: int, sigma2s: list, rng: np.random.Generator):
    """Fits Algorithm 4 jointly across `sources` (list of (n_k, d) arrays),
    returning beta_bar (K-1, d): the fused discriminant coefficients for
    each of the R-1=K-1 non-reference-class contrasts, aligned via
    Algorithm 6 first (see module docstring, notes 2-3)."""
    n_sources = len(sources)
    mus0, labels0 = [], []
    for X in sources:
        mu, lab = _initial_kmeans(X, K, random_state=int(rng.integers(1 << 30)))
        mus0.append(mu)
        labels0.append(lab)

    perms = greedy_multitask_alignment(mus0, sigma2s, rng=rng)

    # Initial (w, beta, delta) state per source, in the aligned canonical labeling.
    ws, betas, deltas = [], [], []
    for k in range(n_sources):
        n_k = sources[k].shape[0]
        mu_aligned = mus0[k][perms[k]]
        counts = np.array([(labels0[k] == perms[k][r]).sum() for r in range(K)])
        w = np.maximum(counts / n_k, 1e-3)
        w = w / w.sum()
        beta = (mu_aligned[1:] - mu_aligned[0]) / sigma2s[k]
        delta = 0.5 * np.einsum("rd,d->r", beta, mu_aligned[0]) + 0.5 * np.einsum(
            "rd,rd->r", beta, mu_aligned[1:]
        )
        ws.append(w)
        betas.append(beta)
        deltas.append(delta)

    n_list = [X.shape[0] for X in sources]
    d = sources[0].shape[1]
    lam = 1.0 * np.sqrt(d + np.log(n_sources))  # initial penalty scale (paper: lambda^[0] >= C1*max_k sqrt(n_k))
    C_lambda = 0.3 * np.sqrt(d + np.log(n_sources)) / max(n_sources, 1)

    for _ in range(MTL_EM_ITERS):
        lam = TLGMM_KAPPA0 * lam + C_lambda * np.sqrt(d + np.log(n_sources))
        lambda_t = lam  # this IS the per-Step-9/12 penalty scale here (matches Algorithm 4 exactly, no /sqrt(n) factor -- that rescaling was specific to TL-GMM's single-task Step 6)

        # E-step + local M-step (Steps 4-9) per source.
        mus_r, pis = [], []
        for k in range(n_sources):
            gamma = _multinomial_responsibility(sources[k], ws[k], betas[k], deltas[k])
            counts = gamma.sum(axis=0)
            mu_r = (gamma.T @ sources[k]) / np.maximum(counts, 1e-8)[:, None]
            mus_r.append(mu_r)
            pis.append(counts / sources[k].shape[0])

        # Aggregation (Steps 11-13), independently per contrast r=2..K.
        beta_bar_all = np.zeros((K - 1, d))
        for r in range(K - 1):
            target_minus_ref_list = [mus_r[k][r + 1] - mus_r[k][0] for k in range(n_sources)]
            beta_bar_r, betas_r_k = _solve_aggregation(
                target_minus_ref_list, sigma2s, n_list, lambda_t
            )
            beta_bar_all[r] = beta_bar_r
            for k in range(n_sources):
                betas[k][r] = betas_r_k[k]

        for k in range(n_sources):
            deltas[k] = 0.5 * np.einsum("rd,d->r", betas[k], mus_r[k][0]) + 0.5 * np.einsum(
                "rd,rd->r", betas[k], mus_r[k][1:]
            )
            ws[k] = pis[k]

    return beta_bar_all, mus0, sigma2s, perms


# ---------------------------------------------------------------------------
# TL-GMM (multi-cluster) target-side shrinkage toward beta_bar
# ---------------------------------------------------------------------------


def _tlgmm_target_fit(X_T: np.ndarray, K: int, sigma2_T: float, beta_bar_all: np.ndarray,
                       C_lambda0: float, rng: np.random.Generator, n_iter: int = TLGMM_EM_ITERS):
    """Multi-cluster analogue of Experiment 4's tlgmm_fit: target means/
    weights updated by ordinary EM each round (using the target's own
    estimated sigma2_T), beta_r re-derived each round via the closed-form
    shrinkage toward beta_bar_all[r], independently per contrast."""
    n_T, d = X_T.shape
    mu0, labels0 = _initial_kmeans(X_T, K, random_state=int(rng.integers(1 << 30)))
    perm, _ = _align_one_task(mu0, sigma2_T, [beta_bar_all])
    mu_aligned = mu0[perm]
    counts = np.array([(labels0 == perm[r]).sum() for r in range(K)])
    w = np.maximum(counts / n_T, 1e-3)
    w = w / w.sum()
    beta = (mu_aligned[1:] - mu_aligned[0]) / sigma2_T
    delta = 0.5 * np.einsum("rd,d->r", beta, mu_aligned[0]) + 0.5 * np.einsum(
        "rd,rd->r", beta, mu_aligned[1:]
    )

    lam = C_lambda0 * np.sqrt(d) / (1.0 - TLGMM_KAPPA0)  # start at steady state
    for _ in range(n_iter):
        lam = TLGMM_KAPPA0 * lam + C_lambda0 * np.sqrt(d)
        lambda_t = lam / np.sqrt(n_T)

        gamma = _multinomial_responsibility(X_T, w, beta, delta)
        counts = gamma.sum(axis=0)
        mu_r = (gamma.T @ X_T) / np.maximum(counts, 1e-8)[:, None]
        w = counts / n_T

        for r in range(K - 1):
            target_minus_ref = mu_r[r + 1] - mu_r[0]
            beta[r] = _shrink_closed_form(target_minus_ref, beta_bar_all[r], sigma2_T, lambda_t)
        delta = 0.5 * np.einsum("rd,d->r", beta, mu_r[0]) + 0.5 * np.einsum("rd,rd->r", beta, mu_r[1:])

    return w, beta, delta


def _tlgmm_predict(X: np.ndarray, w: np.ndarray, beta: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Plug-in classifier (eq. S.2.10): argmax_r { (z-(mu1+mu_r)/2)^T beta_r + log(w_r/w_1) }
    = argmax_r { z^T beta_r - delta_r + log(w_r/w_1) }, with beta_1=0, delta_1=0."""
    n = X.shape[0]
    Kc = w.shape[0]
    scores = np.zeros((n, Kc))
    scores[:, 1:] = X @ beta.T - delta[None, :] + np.log(np.maximum(w[1:], 1e-12) / max(w[0], 1e-12))
    return np.argmax(scores, axis=1)


def _held_out_loglik(X_train, X_val, K, sigma2, beta_bar_all, C_lambda0, rng):
    w, beta, delta = _tlgmm_target_fit(X_train, K, sigma2, beta_bar_all, C_lambda0, rng)
    gamma_val = _multinomial_responsibility(X_val, w, beta, delta)
    # Gaussian log-density (isotropic, shared sigma2) contribution, summed
    # via the responsibility-weighted expected complete-data log-likelihood.
    d = X_val.shape[1]
    mu_all = np.zeros((K, d))
    mu_all[1:] = beta * sigma2  # relative to reference; absolute means not needed for the ll shape below
    ll = 0.0
    n_val = X_val.shape[0]
    sqnorm0 = np.sum(X_val**2, axis=1)
    for r in range(K):
        if r == 0:
            sqdiff = sqnorm0
        else:
            sqdiff = np.sum((X_val - mu_all[r][None, :]) ** 2, axis=1)
        log_p = np.log(np.maximum(w[r], 1e-12)) - 0.5 * d * np.log(2 * np.pi * sigma2) - sqdiff / (2 * sigma2)
        ll += np.sum(gamma_val[:, r] * log_p)
    return ll / n_val


def select_C_lambda0_cv(X_T: np.ndarray, K: int, sigma2_T: float, beta_bar_all: np.ndarray,
                         rng: np.random.Generator) -> float:
    """Sec S.5.1.7's real procedure: find C'_max (smallest C_lambda0 fully
    collapsing every beta_r to beta_bar_r), build a log-spaced grid from
    C'_max/50 to 2*C'_max, and k-fold-CV-select via held-out log-likelihood
    (the paper doesn't specify the unsupervised CV criterion; held-out
    GMM log-likelihood is the standard choice)."""
    n_T, d = X_T.shape
    mu0, _labels0 = _initial_kmeans(X_T, K, random_state=int(rng.integers(1 << 30)))
    perm, _ = _align_one_task(mu0, sigma2_T, [beta_bar_all])
    mu_aligned = mu0[perm]
    # rho_r at the very first (unpenalized) iterate, as a proxy for the
    # "worst-case" residual norm used to pick C'_max.
    rhos = [
        np.linalg.norm((mu_aligned[r + 1] - mu_aligned[0]) - sigma2_T * beta_bar_all[r])
        for r in range(K - 1)
    ]
    max_rho = max(rhos) if rhos else 1.0
    # lambda_t = C_lambda0*sqrt(d)/(1-kappa0) at steady state must reach max_rho
    # for full collapse (shrink=0) on the hardest contrast.
    C_prime_max = max_rho * (1.0 - TLGMM_KAPPA0) / np.sqrt(d)
    C_prime_max = max(C_prime_max, 1e-8)
    grid = np.exp(np.linspace(np.log(C_prime_max / 50.0), np.log(2.0 * C_prime_max), CV_N_GRID))

    idx = rng.permutation(n_T)
    folds = np.array_split(idx, CV_N_FOLDS)
    scores = np.zeros(len(grid))
    for g, C_lambda0 in enumerate(grid):
        fold_scores = []
        for f in range(CV_N_FOLDS):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[i] for i in range(CV_N_FOLDS) if i != f])
            if len(train_idx) < K or len(val_idx) < 1:
                continue
            ll = _held_out_loglik(X_T[train_idx], X_T[val_idx], K, sigma2_T, beta_bar_all,
                                   C_lambda0, rng)
            fold_scores.append(ll)
        scores[g] = np.mean(fold_scores) if fold_scores else -np.inf
    return float(grid[np.argmax(scores)])


def method_tlgmm(X_T: np.ndarray, sources: list, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_T = center_data(X_T)
    sources = [center_data(X_S) for X_S in sources]
    sigma2s_src = [estimate_noise_variance_rankK(X, K) for X in sources]
    sigma2_T = estimate_noise_variance_rankK(X_T, K)
    beta_bar_all, _mus0, _sigma2s, _perms = mtl_gmm_multicluster(sources, K, sigma2s_src, rng)
    C_lambda0 = select_C_lambda0_cv(X_T, K, sigma2_T, beta_bar_all, rng)
    w, beta, delta = _tlgmm_target_fit(X_T, K, sigma2_T, beta_bar_all, C_lambda0, rng)
    return _tlgmm_predict(X_T, w, beta, delta)

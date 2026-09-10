"""Shared infrastructure for the mouse PBMC intro-illustration pipeline:
config constants, data loading, per-contrast dataset construction,
proportional target subsampling, and metrics (misclustering loss + ARI).

Produces the underlying sweep data for the paper's Figure 1 (the
introduction's motivating illustration of transfer-assisted clustering on
the mouse peripheral-blood scRNA-seq data of Han et al. 2018, Mouse Cell
Atlas): for two binary (K=2) cell-type contrasts (B cell vs Macrophage,
T cell vs NK cell -- the only two cell-type pairs present in every batch
with non-trivial counts) and every one of the 6 `PeripheralBlood_<i>`
batches as target in turn, sweeps target sample size and records both the
misclustering loss and the Adjusted Rand Index of the target-only,
single-source, and multi-source (all-other-batches-pooled) oracle
estimators (`transfer_clustering.two_community`'s `target_based_estimate`
/ `source_based_estimate`; no adaptive/bootstrap selection).
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import adjusted_rand_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
H5AD_PATH = os.path.join(REPO_ROOT, "Mouse_PBMC_Experiments", "Final_Mouse_PBMC_Analysis",
                          "data", "mouse_pbmc_hvg_lognorm.h5ad")

BATCHES = [
    "PeripheralBlood_1", "PeripheralBlood_2", "PeripheralBlood_3",
    "PeripheralBlood_4", "PeripheralBlood_5", "PeripheralBlood_6",
]

# Display-only mapping used by the R plotting script (kept here too so the
# rule lives in one place): PeripheralBlood_<i> -> Batch_<i>.
BATCH_DISPLAY = {b: f"Batch_{b.split('_')[-1]}" for b in BATCHES}

# The two binary contrasts (K=2), each scoped to every batch as target.
# B cell vs Macrophage and T cell vs NK cell are the only two cell-type
# pairs present in every one of the 6 batches with non-trivial counts
# (Monocyte is ~absent outside PeripheralBlood_2).
CONTRASTS = {
    "B_cell_vs_Macrophage": ("B cell", "Macrophage"),
    "T_cell_vs_NK_cell": ("NK cell", "T cell"),
}

K_MIN = 10           # floor on the smallest target subsample size
N_K_POINTS = 8       # log-spaced points per target, scaled to that target's own
                      # 2-class subset size -- naturally denser at the low end in
                      # absolute terms (constant ratio between consecutive points)
N_SUBSAMPLE = 30
RANDOM_SEED = 42


def load_adata():
    import scanpy as sc
    return sc.read_h5ad(H5AD_PATH)


def build_contrast(adata_all, batch_name: str, contrast_types: tuple[str, str]):
    """All cells of both types from the given batch, centred per gene by
    the POOLED mean over all cells regardless of class (X.mean(axis=0)) --
    fully unsupervised, no label leakage. Returns (X (n, d) float64, z (n,)
    in {+1 (contrast_types[0]), -1 (contrast_types[1])})."""
    sub = adata_all[adata_all.obs["batch"] == batch_name]
    idx_pos = np.where((sub.obs["cell_type"] == contrast_types[0]).values)[0]
    idx_neg = np.where((sub.obs["cell_type"] == contrast_types[1]).values)[0]
    sel = np.sort(np.concatenate([idx_pos, idx_neg]))
    sub_sel = sub[sel]
    z = np.where(sub_sel.obs["cell_type"].values == contrast_types[0], 1, -1)
    X = sub_sel.X.toarray() if hasattr(sub_sel.X, "toarray") else np.array(sub_sel.X, dtype=float)
    X = (X - X.mean(axis=0)).astype(np.float64)
    return X, z


def k_grid_for(n_full: int) -> list[int]:
    """N_K_POINTS log-spaced integers from K_MIN to n_full (inclusive of
    both ends), rounded and de-duplicated."""
    pts = np.geomspace(K_MIN, n_full, N_K_POINTS)
    return sorted(set(int(round(p)) for p in pts))


def subsample_target(X_tgt: np.ndarray, z_tgt: np.ndarray, k_sub: int, seed: int):
    """Proportional subsample of size k_sub preserving the batch's class
    mix, floored at >=1 sample per class regardless of k_sub. Re-centres
    by the subsample's own (label-free) mean."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(z_tgt == 1)[0]
    idx_neg = np.where(z_tgt == -1)[0]
    frac_pos = len(idx_pos) / (len(idx_pos) + len(idx_neg))
    n_pos = int(round(k_sub * frac_pos))
    n_pos = min(max(n_pos, 1), len(idx_pos), k_sub - 1)
    n_neg = k_sub - n_pos
    sel = np.concatenate([
        rng.choice(idx_pos, n_pos, replace=False),
        rng.choice(idx_neg, n_neg, replace=False),
    ])
    X_sub = X_tgt[sel]
    X_sub = X_sub - X_sub.mean(axis=0)
    z_sub = z_tgt[sel]
    return X_sub, z_sub


def misclustering_loss(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """min_{s in {+1,-1}} mean( sign(z_pred) != s * sign(z_true) )."""
    zp = np.sign(z_pred)
    zt = np.sign(z_true)
    return float(min(np.mean(zp != zt), np.mean(zp != -zt)))


def ari_score(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """Adjusted Rand Index between the predicted sign-clustering and the
    true labels -- label-permutation invariant by construction, so (unlike
    misclustering_loss) no explicit sign-flip search is needed."""
    return float(adjusted_rand_score(np.sign(z_pred), z_true))


def run_target_sweep(contrast_name: str, target: str):
    """One target batch's full sweep for one contrast: for every k in
    k_grid_for(n_full) x every seed in range(N_SUBSAMPLE), evaluates
    target_only, each of the 5 other batches as a single source
    (`source_based_estimate(X_sub, [X_src])`), and all 5 pooled as a
    multi-source (`source_based_estimate(X_sub, [X_src1, ..., X_src5])`)
    -- oracle branches only (`transfer_clustering.two_community`), no
    adaptive/bootstrap selection. Called once per target batch (for Slurm
    array parallelism across targets), recording both misclustering loss
    and ARI for each method.

    Returns (ari_rows, misclustering_rows), each a list of dicts with keys
    target, k, method, mean_value, std_value, n_sub, contrast -- one row
    per (k, method) pair, aggregated (mean/std) over the N_SUBSAMPLE seeds.
    """
    import sys

    sys.path.insert(0, REPO_ROOT)
    from transfer_clustering.two_community import target_based_estimate, source_based_estimate

    if target not in BATCHES:
        raise ValueError(f"target must be one of {BATCHES}, got {target!r}")
    contrast_types = CONTRASTS[contrast_name]
    contrast_label = f"{contrast_types[0]} vs {contrast_types[1]}"

    adata = load_adata()
    contrast_data = {b: build_contrast(adata, b, contrast_types) for b in BATCHES}
    del adata

    other_batches = [b for b in BATCHES if b != target]
    X_tgt_full, z_tgt_full = contrast_data[target]
    n_full = len(z_tgt_full)
    k_values = k_grid_for(n_full)
    print(f"target={target} (n={n_full}, contrast={contrast_label}) k values: {k_values}")

    ari_rows, mis_rows = [], []
    for k_sub in k_values:
        to_mis, to_ari = [], []
        multi_mis, multi_ari = [], []
        so_mis = {b: [] for b in other_batches}
        so_ari = {b: [] for b in other_batches}

        for sub_seed in range(N_SUBSAMPLE):
            X_sub, z_sub = subsample_target(X_tgt_full, z_tgt_full, k_sub, seed=sub_seed)

            z_trg = target_based_estimate(X_sub)
            to_mis.append(misclustering_loss(z_trg, z_sub))
            to_ari.append(ari_score(z_trg, z_sub))

            for b in other_batches:
                X_src, _ = contrast_data[b]
                z_src = source_based_estimate(X_sub, [X_src])
                so_mis[b].append(misclustering_loss(z_src, z_sub))
                so_ari[b].append(ari_score(z_src, z_sub))

            z_multi = source_based_estimate(X_sub, [contrast_data[b][0] for b in other_batches])
            multi_mis.append(misclustering_loss(z_multi, z_sub))
            multi_ari.append(ari_score(z_multi, z_sub))

        def _emit(rows, method, values):
            arr = np.array(values)
            rows.append({
                "target": target, "k": k_sub, "method": method,
                "mean_value": float(arr.mean()), "std_value": float(arr.std()),
                "n_sub": N_SUBSAMPLE, "contrast": contrast_label,
            })

        _emit(mis_rows, "Target-only", to_mis)
        _emit(ari_rows, "Target-only", to_ari)
        for b in other_batches:
            _emit(mis_rows, f"Source: {b}", so_mis[b])
            _emit(ari_rows, f"Source: {b}", so_ari[b])
        _emit(mis_rows, "Multi-source (pooled)", multi_mis)
        _emit(ari_rows, "Multi-source (pooled)", multi_ari)

        print(f"  k={k_sub:4d}  Target-only mis={np.mean(to_mis):.3f} ari={np.mean(to_ari):.3f}  "
              + "  ".join(f"{b}={np.mean(so_mis[b]):.3f}" for b in other_batches)
              + f"  Multi={np.mean(multi_mis):.3f}")

    return ari_rows, mis_rows

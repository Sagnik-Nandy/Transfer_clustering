#!/usr/bin/env python
"""One-time preprocessing: raw-counts MoDaH mouse PBMC data (Han et al. 2018
Mouse Cell Atlas, peripheral blood subset) -> log-normalized HVG matrix, in
the same on-disk convention as lung_atlas_analysis/data/lung_atlas_hvg_lognorm.h5ad
(obs['batch'], obs['cell_type'], X = log-normalized top-HVG expression),
so Final_Mouse_PBMC_Analysis/common.py can load it exactly like the other
Final_*_Analysis directories.

Source: MoDaH_Data/real_world/pbmc/data/pbmc_subset.h5ad (raw counts,
obs['dataset']=batch, obs['cell_type_combined']=cell type -- see that
repo's pbmc.ipynb for provenance). Unlike MoDaH's own pipeline (which goes
on to select 1000 HVGs, scale, and reduce to 20 PCs before batch
correction), we stop after HVG selection -- no PCA, no scaling to unit
variance -- matching this repo's Final_*_Analysis convention of clustering
on raw (log-normalized) gene expression directly.

Usage: python preprocess.py
"""
from __future__ import annotations

import os

import scanpy as sc

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(_THIS_DIR, "..", "MoDaH_Data", "real_world", "pbmc", "data", "pbmc_subset.h5ad")
OUT_PATH = os.path.join(_THIS_DIR, "data", "mouse_pbmc_hvg_lognorm.h5ad")
N_HVG = 2000  # smaller than the lung atlas's 5000 -- this dataset has only
              # 11870 genes pre-filtering and 7095 cells total (vs. lung
              # atlas's larger gene pool and 9941 cells), so a proportionally
              # smaller HVG count keeps a comparable selectivity.


def main():
    adata = sc.read_h5ad(SRC_PATH)
    print(f"Loaded raw: {adata.shape}")

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"After filter/normalize/log1p: {adata.shape}")

    # Batch-aware HVG selection (batch_key='dataset') so no single large
    # batch dominates gene selection -- matches MoDaH's own pbmc.ipynb choice.
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key="dataset")
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"After HVG selection: {adata.shape}")

    adata.obs["batch"] = adata.obs["dataset"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type_combined"].astype(str)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    adata.write(OUT_PATH)
    print(f"Wrote {OUT_PATH}")

    print(adata.obs.groupby(["batch", "cell_type"], observed=True).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()

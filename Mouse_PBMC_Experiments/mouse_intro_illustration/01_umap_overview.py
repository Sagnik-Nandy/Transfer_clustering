#!/usr/bin/env python
"""Panel 1 of the paper's Figure 1: full-atlas UMAP of the mouse PBMC
dataset (all 6 `PeripheralBlood_<i>` batches, all 9 cell types), computed
fresh for this pipeline.

Standard scanpy visualization recipe (scale -> PCA(50) -> neighbors(15,
30 PCs) -> UMAP); this is purely for visualization, not the same
preprocessing used to cluster the data in the Table 1 comparison (which
runs directly on the log-normalized HVG matrix, with no scaling or PCA
reduction -- see Final_Mouse_PBMC_Analysis/preprocess.py). Cheap and
quick enough to run directly, no Slurm needed.

Exports panel1_umap.csv (UMAP1, UMAP2, cell_type, batch) for the
R/ggplot2 plotting script (plot_intro_figure.R); no modelling happens
here, this is pure data prep. Shared across all 24 (contrast, target,
metric) figures -- run this once.

Usage:
    python 01_umap_overview.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scanpy as sc
import warnings

warnings.filterwarnings("ignore")

import common

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sc.settings.verbosity = 1

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    adata = sc.read_h5ad(common.H5AD_PATH)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Cell types ({adata.obs['cell_type'].nunique()}): {sorted(adata.obs['cell_type'].unique())}")
    print(f"Batches: {sorted(adata.obs['batch'].unique())}")

    adata_umap = adata.copy()
    sc.pp.scale(adata_umap, max_value=10)
    sc.tl.pca(adata_umap, n_comps=50, random_state=RANDOM_SEED)
    sc.pp.neighbors(adata_umap, n_neighbors=15, n_pcs=30, random_state=RANDOM_SEED)
    sc.tl.umap(adata_umap, random_state=RANDOM_SEED)
    print(f"UMAP done: {adata_umap.n_obs} cells")

    umap_xy = adata_umap.obsm["X_umap"]
    df_out = pd.DataFrame({
        "UMAP1": umap_xy[:, 0],
        "UMAP2": umap_xy[:, 1],
        "cell_type": adata_umap.obs["cell_type"].values,
        "batch": adata_umap.obs["batch"].values,
    })

    csv_path = os.path.join(OUT_DIR, "panel1_umap.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(df_out)} rows)")


if __name__ == "__main__":
    main()

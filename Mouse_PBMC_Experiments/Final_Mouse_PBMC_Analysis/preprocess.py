#!/usr/bin/env python
"""One-time preprocessing: raw-counts mouse PBMC data (Han et al. 2018
Mouse Cell Atlas, peripheral blood subset) -> log-normalized HVG matrix
(obs['batch'], obs['cell_type'], X = log-normalized top-HVG expression),
matching the preprocessing pipeline described in the paper's Appendix
B.1: the raw count matrix was deposited at Gene Expression Omnibus
accession GSE108097 as part of the Mouse Cell Atlas; the PBMC-specific
subset used here was sourced from the Zenodo repository accompanying Cao
& Ma (2025), "MoDaH achieves rate optimal batch correction"
(arXiv:2512.09259). This script filters genes seen in fewer than 10
cells, library-size normalizes to 1e4 counts per cell, applies log1p,
and selects the top N_HVG batch-aware highly-variable genes -- it does
not go on to scale or reduce via PCA, unlike batch-correction pipelines
that use this same source data; the methods compared here cluster
directly on the log-normalized gene expression.

Usage: python preprocess.py
    (EDIT: set SRC_PATH below to your local copy of the raw h5ad file.)
"""
from __future__ import annotations

import os

import scanpy as sc

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# EDIT: point this at your local copy of the raw PBMC subset h5ad (raw
# counts, obs['dataset']=batch, obs['cell_type_combined']=cell type) from
# the Zenodo repository described above.
SRC_PATH = os.path.join(_THIS_DIR, "..", "..", "raw_data", "pbmc_subset.h5ad")
OUT_PATH = os.path.join(_THIS_DIR, "data", "mouse_pbmc_hvg_lognorm.h5ad")
N_HVG = 2000  # matches the paper's Appendix B.1 preprocessing (11,870
              # genes pre-filtering, 7,095 cells across the six batches).


def main():
    adata = sc.read_h5ad(SRC_PATH)
    print(f"Loaded raw: {adata.shape}")

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"After filter/normalize/log1p: {adata.shape}")

    # Batch-aware HVG selection (batch_key='dataset') so no single large
    # batch dominates gene selection, matching the paper's Appendix B.1.
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

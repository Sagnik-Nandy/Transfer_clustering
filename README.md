# Transfer_clustering

Code accompanying **"Transfer Learning in High-Dimensional Clustering: Minimax
Thresholds and Applications in Single-Cell Data"** (Chakraborty & Nandy),
[arXiv:2607.25031](https://arxiv.org/abs/2607.25031).

The paper studies clustering a *target* dataset with the help of related
*source* datasets in the two-community and multi-cluster Gaussian mixture
models, characterizes the minimax phase transition for consistent recovery,
and gives an adaptive procedure that requires no oracle knowledge of whether
the source data is actually useful.

## Scope of this repository

This is a curated subset of the full research codebase, containing:

- the `transfer_clustering` package implementing Algorithms 1–3,
- the simulation experiments from Section 5,
- and the Section 6 real-data analysis (human lung scRNA-seq atlas).

**Not included** (regenerable or out of scope for this release): raw data
files, generated result CSVs, Slurm logs, and additional exploratory
real-data analyses (pancreas, fMRI, metabolomics, and two other lung-atlas
cell-type groupings) that live in the private research repo but aren't part
of the published paper. See `.gitignore` for the exact exclusions.

## Repository structure

```
Python_Scripts/                  the transfer_clustering package (imported via the
                                  Experiments_Script/transfer_clustering symlink)
    spectral.py                  Ndaoud (2018) hollowed-Gram-matrix spectral clustering
    two_community.py             Sec. 2.2-2.3, Algorithm 1 (K=2: target/source/adaptive estimators)
    relaxed_kmeans.py            Giraud & Verzelen (2019) relaxed K-means SDP (K>2 target-based step)
    ts_clust.py                  Algorithm 3 (TSClust): spectral init + Lloyd refinement
    multi_cluster.py             Sec. 4.2, Algorithm 2 (K>2 adaptive projected clustering)
    api.py                       TransferClustering: unified K=2 / K>2 entry point
    selection.py                 bootstrap calibration helpers for the adaptive threshold

Experiments_Script/              Slurm-array entry points for Section 5's simulations
    run_experiment1.py           error vs. alignment (mu) across three (d, n_T, n_S) regimes
    run_experiment2.py           heatmaps over (Delta_T, Delta_S)
    run_experiment3.py           multi-source, K=3 complementary-information experiment
    run_experiment4.py           comparison against TGMM / TL-GMM benchmarks
Slurm_Scripts/                   matching #SBATCH array scripts for each experiment above
Notebooks_simulation/            aggregate-and-plot notebooks producing the paper's Sec. 5 figures

Final_Lung_Atlas_Analysis/       Section 6: transfer-assisted clustering of the human lung
                                  scRNA-seq atlas (Vieira Braga et al., 2019; GSE130148),
                                  K=13 fine-grained cell types, leave-one-patient-out
    common.py                    shared config, data loading, and metrics (K, batches, scoring)
    method_ours.py                target-only / multi-source-pooled / pooled-concat / adaptive
    method_tlgmm.py               TL-GMM comparator (Tian et al., 2026)
    method_scrna.py               NMF-based comparator (Mieth et al., 2019)
    method_gdec.py                GDEC comparator (Wang et al., 2024)
    run_lung_atlas_comparison.py  one (target batch, method) pair per Slurm array task
    run_lung_atlas_comparison.ipynb / aggregate_and_plot*.ipynb   interactive/aggregation versions

External_Methods/GDEC/           vendored comparator implementation used by method_gdec.py
External_Methods/scRNA/          vendored comparator implementation used by method_scrna.py
                                  (both locally modified; see Attribution below)
```

## Installation

Core dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `cvxpy`
(relaxed K-means SDP), `matplotlib`/`seaborn` (plotting), `scanpy`
(`.h5ad` loading for Section 6).

The `scrna` and `gdec_gcnfree` comparator methods in
`Final_Lung_Atlas_Analysis/` additionally require their own vendored
dependencies under `External_Methods/` — notably `torch`, `dgl`,
`tensorboardX`, `click`, `cytoolz` for GDEC. These are only needed if you
run those two specific comparator methods; the core package and the
`target_only` / `multi_source_pooled` / `pooled_concat` / `adaptive_multi_source`
/ `tlgmm` methods don't need them.

No pinned `environment.yml`/`requirements.txt` is included in this release;
install the packages above with `pip`/`conda` as needed for the parts you
want to run.

## Quick start

```python
from transfer_clustering import TransferClustering

model = TransferClustering(K=2)
labels, branch = model.fit_predict(X_target, [X_source_1, X_source_2])
```

`K=2` dispatches to the two-community spectral estimators in
`two_community.py`; `K>2` dispatches to `multi_cluster.py`'s relaxed
K-means / projected-clustering pipeline.

## Reproducing the simulation experiments (Section 5)

Each `run_experimentN.py` is a single Monte Carlo replication driven by a
Slurm array task index (seed, or (regime, seed) — see each script's
docstring). Submit via the matching script in `Slurm_Scripts/`, e.g.:

```bash
cd Slurm_Scripts/experiment1_alignment
sbatch run_experiment1.sh
```

then open the corresponding notebook in `Notebooks_simulation/` to
aggregate the per-task CSV outputs and reproduce the paper's figures.
`#SBATCH` paths/resources in each script are cluster-specific — edit the
`PROJECT_ROOT`, output paths, and resource requests for your own cluster
before submitting.

## Reproducing the lung atlas analysis (Section 6)

The raw data isn't included in this repository. Obtain the Vieira Braga
et al. (2019) human lung scRNA-seq atlas (GEO accession
[GSE130148](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130148)),
preprocess it as described in Supplement B.1 of the paper (QC filtering,
HVG selection to `d=5000`, log-normalization), and place the result at:

```
lung_atlas_analysis/data/lung_atlas_hvg_lognorm.h5ad
```

relative to the repository root (a sibling of `Final_Lung_Atlas_Analysis/`),
with `obs["batch"]` giving the 4 patient batches and `obs["cell_type"]`
the 13 fine-grained cell-type labels used as ground truth. Then either run
the notebook `Final_Lung_Atlas_Analysis/run_lung_atlas_comparison.ipynb`
directly, or submit the Slurm-array version:

```bash
cd Final_Lung_Atlas_Analysis
sbatch run_lung_atlas_comparison.sh   # edit paths/resources for your cluster first
```

and aggregate the resulting `results*/raw/*.csv` files with
`aggregate_and_plot.ipynb` to reproduce Table 1.

## Attribution

`External_Methods/GDEC` and `External_Methods/scRNA` are vendored, locally
modified copies of two external comparator methods used in Section 6:

- **GDEC**: originally from [YuzhiSun/GDEC](https://github.com/YuzhiSun/GDEC)
  (Wang et al., 2024).
- **scRNA**: originally from [nicococo/scRNA](https://github.com/nicococo/scRNA)
  (Mieth et al., 2019), MIT licensed.

Both are included here as plain files (rather than as submodules) because
the versions actually used by `method_gdec.py`/`method_scrna.py` differ
from upstream — see each original repository for full history and
licensing terms.

## Citation

If you use this code, please cite:

```bibtex
@article{chakraborty2026transfer,
  title   = {Transfer Learning in High-Dimensional Clustering: Minimax Thresholds and Applications in Single-Cell Data},
  author  = {Chakraborty, Abhinav and Nandy, Sagnik},
  journal = {arXiv preprint arXiv:2607.25031},
  year    = {2026}
}
```

# Transfer Clustering

Implementation and experiments for Chakraborty, A. and Nandy, S., "Transfer
Learning in High-Dimensional Clustering: Minimax Thresholds and Applications
in Single-Cell Data" (submitted to the Annals of Statistics): a
minimax-optimal transfer-assisted clustering procedure for two-community and
multi-cluster high-dimensional Gaussian mixture models. A target dataset's
own signal is combined with one or more related source datasets by
estimating and projecting onto a shared signal subspace, with a data-driven
selector that adaptively falls back to target-only clustering when the
target signal is already strong enough on its own. Evaluated on synthetic
data (`Numerical_Experiments/`) and a real single-cell RNA-seq application
(`Mouse_PBMC_Experiments/`): recovering cell types across batches of a mouse
peripheral-blood scRNA-seq atlas, benchmarked against TL-GMM, an NMF-based
transfer method, and GDEC.

## Repository layout

| Folder | Contents |
|---|---|
| [`transfer_clustering/`](transfer_clustering/README.md) | The algorithm library: Algorithms 1-6 of the paper (two-community and multi-cluster oracle/adaptive/pooled transfer-assisted clustering). |
| [`Numerical_Experiments/`](Numerical_Experiments/README.md) | Experiments 1-5: synthetic-data Monte Carlo studies reproducing the paper's Figures 3-7. |
| [`Mouse_PBMC_Experiments/`](Mouse_PBMC_Experiments/README.md) | The real-data application (Section 6): the Table 1 seven-method comparison and the Figure 1 introductory illustration, both on the mouse peripheral-blood scRNA-seq dataset. |
| [`External_Methods/`](External_Methods/README.md) | Vendored third-party implementations of two of the real-data comparator methods (GDEC, scRNA/NMF). |

Each folder's own README has the detailed layout, run order, and per-file
breakdown.

## Requirements

Two runtimes: Python for everything, plus R for
`Mouse_PBMC_Experiments/mouse_intro_illustration/`'s plotting script.

### Python — core (`transfer_clustering/`, `Numerical_Experiments/`, `Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/`)

```bash
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `numpy`, `scipy` | Core linear algebra; `scipy.optimize.linear_sum_assignment` for best-permutation misclustering-error matching |
| `scikit-learn` | `KMeans` (TSClust's Lloyd-refinement step), `adjusted_rand_score`/`v_measure_score` (real-data metrics) |
| `pandas` | Loading/aggregating per-task result CSVs |
| `matplotlib` | Result figures (several use `text.usetex=True`, requiring a TeX Live install on `PATH` — not a pip package) |
| `scanpy` | Loading, preprocessing, and computing UMAP embeddings of the mouse PBMC `.h5ad` data |
| `jupyter`, `ipython` | Running the aggregation/analysis notebooks |
| `cvxpy` *(optional)* | Only if `transfer_clustering.RelaxedKMeans` is configured with a cvxpy solver name (e.g. `solver="SCS"`) instead of the default `"ADMM"` solver, which needs no external SDP solver; imported in a `try/except`, so the rest of the pipeline works without it |

### Python — `External_Methods/GDEC` (comparator: `method_gdec.py`)

Not pinned in `requirements.txt` (a separate, heavier environment —
GDEC's own `environment.yml` has the full pinned list):

| Package | Used for |
|---|---|
| `torch` | The GDEC autoencoder (SDAE) + DEC self-training pipeline |
| `tqdm` | Progress bars during SDAE/DEC training (`ptsdae.model`) |
| `cytoolz` | Sliding-window utility used by the SDAE construction (`ptsdae.sdae`) |

### Python — `External_Methods/scRNA` (comparator: `method_scrna.py`)

Not pinned in `requirements.txt` (see `External_Methods/scRNA/requirements.txt`
for the vendored package's own full list):

| Package | Used for |
|---|---|
| `cvxopt` | Used elsewhere in the vendored `scRNA` package (e.g. its clustering utilities) |
| `numba` | JIT-accelerated routines in the vendored `scRNA` package |

### R — `Mouse_PBMC_Experiments/mouse_intro_illustration/`

| Package | Used for |
|---|---|
| `ggplot2` | The 3-panel intro figure |
| `dplyr` | Filtering/relabeling the sweep result CSVs before plotting |
| `patchwork` | Combining the 3 panels into one figure |
| `latex2exp` | Rendering the `$n_T$` axis label |

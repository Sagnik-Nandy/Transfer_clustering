# Final_Mouse_PBMC_Analysis

Reproduces Table 1 of the paper (Section 6): 7 clustering procedures
compared on the mouse peripheral-blood scRNA-seq dataset (Han et al. 2018,
Mouse Cell Atlas, GEO accession GSE108097), using each of its 6
`PeripheralBlood_<i>` batches as target against the other 5 pooled as
source, K=9 cell types. See the root README for install requirements
(the GDEC and scRNA comparators additionally need the packages listed
there under `External_Methods/`).

## Run order

1. **`preprocess.py`** -- one-time raw-counts -> log-normalized, top-2000-HVG
   `.h5ad` conversion (see the script's docstring for the exact filtering/
   normalization steps and data provenance). Point `SRC_PATH` at your own
   local copy of the raw PBMC subset first. Produces
   `data/mouse_pbmc_hvg_lognorm.h5ad`. Alternatively, skip this step and
   [download the preprocessed file directly](https://www.dropbox.com/scl/fi/lyd87h024usuk45tc406a/mouse_pbmc_hvg_lognorm.h5ad?rlkey=6spahvvu77644zi8tcl6v9x6p&st=7l3w5iuf&dl=1)
   to `data/mouse_pbmc_hvg_lognorm.h5ad`.
2. **`run_mouse_pbmc_comparison.py <task_id>`** -- one (target batch,
   method) combination per task (42 tasks total: 6 batches x 7 methods).
   Writes one CSV to `results/raw/<batch>__<method>.csv`. Submit at scale
   via `run_mouse_pbmc_comparison.sh` (edit its placeholder paths and
   conda environment name for your cluster first).
3. **`aggregate_and_plot.ipynb`** -- once all 42 tasks have finished,
   reproduces Table 1 (LaTeX) and the misclustering/ARI bar charts from
   `results/raw/*.csv`.

`check_data_stats.ipynb` is a standalone verification notebook: it
recomputes every number quoted in the paper's Section 6 / Appendix B.1
directly from the raw and preprocessed data, including an exact
cell-by-cell cross-check of the batch x cell-type contingency table
against Table 1 of Cao & Ma (2025), "MoDaH achieves rate optimal batch
correction" (arXiv:2512.09259) -- the paper the raw PBMC subset is drawn
from.

## File roles

| File | Role |
|---|---|
| `common.py` | Config (`K`, `BATCHES`, `METHOD_LABELS`), data loading (`load_batches`), and metrics (`compute_metrics`: misclustering error, ARI, V-measure) |
| `method_ours.py` | The four methods from this paper: `target_only`, `multi_source_pooled`, `target_source_pooled_capped` (Algorithm 4), `adaptive_multi_source` (Algorithm 3) |
| `method_tlgmm.py` | The TL-GMM comparator (Tian, Weng, Xia & Feng, arXiv:2209.15224) |
| `method_scrna.py` | The NMF-based comparator (Mieth et al.), via the vendored `External_Methods/scRNA` package |
| `method_gdec.py` | The GDEC comparator (Wang et al., 2024), via the vendored `External_Methods/GDEC` package |
| `preprocess.py` | Raw counts -> log-normalized, top-HVG `.h5ad`, matching Appendix B.1 |
| `run_mouse_pbmc_comparison.py` | The 42-task cluster-parallel driver |
| `aggregate_and_plot.ipynb` | Table 1 and the misclustering/ARI figures |
| `check_data_stats.ipynb` | Independent sanity check of every data statistic quoted in the paper |

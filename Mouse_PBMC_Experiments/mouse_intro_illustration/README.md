# Mouse PBMC intro illustration

Generates the sweep data and figure underlying the paper's Figure 1 (the
introduction's motivating example of transfer-assisted clustering), for
the mouse PBMC dataset (Han et al. 2018 Mouse Cell Atlas, peripheral
blood subset, 6 `PeripheralBlood_<i>` batches). Every batch is used as
target in turn, for two binary (K=2) contrasts (B cell vs Macrophage,
T cell vs NK cell -- the only two cell-type pairs present in every batch
with non-trivial counts), recording both ARI and misclustering loss over
an 8-point log-spaced target-sample-size grid (`K_MIN=10`). See the root
README for install requirements (Python for the sweep, R for
`plot_intro_figure.R`).

Estimator: oracle branches only (`transfer_clustering.two_community`'s
`target_based_estimate` / `source_based_estimate`), no adaptive/bootstrap
selection -- this pipeline illustrates the *availability* of useful
source information as a function of target sample size, not the
adaptive-selection procedure itself (which is instead the subject of the
Table 1 comparison in `Final_Mouse_PBMC_Analysis/`).

## Layout

```
mouse_intro_illustration/
├── common.py                        # shared config, data prep, metrics, sweep core
├── 01_umap_overview.py              # panel 1 data (run once, no Slurm) -> panel1_umap.csv
├── panel1_umap.csv                  # produced by 01_umap_overview.py
├── plot_intro_figure.R              # parameterized 3-panel ggplot2 figure (EXPERIMENT SWITCH
│                                     #   at top, or positional CLI args)
├── generate_all_figures.sh          # loops plot_intro_figure.R over all 24 combinations
├── logs/                            # Slurm stdout/stderr
├── figures/                         # all 24 figures land here (2 x pdf+png = 48 files)
│   ├── B_cell_vs_Macrophage/        #   <target>_<metric>.{pdf,png} (6 x 2 = 12 combos)
│   └── T_cell_vs_NK_cell/           #   12 combos
├── B_cell_vs_Macrophage/
│   ├── run_sweep.py                 # one Slurm array task per target batch (task_id 0-5)
│   ├── run_sweep.sh                 # sbatch --array=0-5
│   └── results/                     # <target>_ari.csv, <target>_misclustering.csv (6 x 2 = 12 files)
└── T_cell_vs_NK_cell/
    ├── run_sweep.py
    ├── run_sweep.sh
    └── results/                     # 12 files
```

24 result CSVs total (2 contrasts x 6 target batches x 2 metrics), and 24
figures (48 files with both pdf+png) under `figures/`.

## Running

Requires `data/mouse_pbmc_hvg_lognorm.h5ad` to already exist under
`Final_Mouse_PBMC_Analysis/` -- see that directory's `preprocess.py`.

```bash
# 1. Panel 1 data (shared across everything) -- run once, no Slurm needed.
cd mouse_intro_illustration
python 01_umap_overview.py

# 2. The sweeps -- one Slurm array (6 tasks) per contrast.
cd B_cell_vs_Macrophage && sbatch run_sweep.sh && cd ..
cd T_cell_vs_NK_cell    && sbatch run_sweep.sh && cd ..

# 3. Plotting -- generate all 24 at once:
./generate_all_figures.sh
# ...or one at a time, either via the EXPERIMENT SWITCH at the top of
# plot_intro_figure.R, or positional CLI args:
#   Rscript plot_intro_figure.R B_cell_vs_Macrophage PeripheralBlood_1 ari
```

Each run writes `figures/<CONTRAST_DIR>/<TARGET_BATCH>_<METRIC>.{pdf,png}`
(24 distinct combinations -- no overwriting across runs).

## Design notes

- **k grid**: `K_MIN=10`, `N_K_POINTS=8`, log-spaced up to each
  (contrast, target)'s own 2-class subset size (not the target's full
  batch size) -- inherently denser at the low end in absolute terms.
- **Panel 3's single source**: not user-specified per run, so auto-picked
  as the *largest other batch* (by full, un-restricted batch size --
  `PeripheralBlood_3` for any target other than itself, else
  `PeripheralBlood_2`), reproducibly and without manual per-target
  curation. Override via `SINGLE_SOURCE_BATCH` in the R script if a
  different pick is wanted for a given run.
- **Batch display**: `PeripheralBlood_<i>` is shown as `Batch_<i>` in
  every plot label/legend/title (never in file paths, directory names, or
  CSV contents, which keep the full `PeripheralBlood_<i>` names).

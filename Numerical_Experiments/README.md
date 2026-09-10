# Numerical_Experiments

The paper's five synthetic-data Monte Carlo studies (Section 5 and
Appendix A), all built on the `transfer_clustering` library. Each
experiment follows the same chain: a Slurm array job launches many
`(regime, seed)` (or `(target_regime, regime, seed)`) trials of a Python
script, each trial writes one CSV under `Results_simulation/<experiment>/raw/`,
and the matching notebook globs that directory to produce the paper's
figure. See the root README for install requirements.

## Layout

```
Numerical_Experiments/
├── Experiments_Script/     One driver script per experiment (run_experiment{1-5}.py)
├── Slurm_Scripts/           One Slurm array-job launcher per experiment
└── Notebooks_simulation/    Aggregates each experiment's per-task CSVs into
                              the paper's figures
```

| Experiment | Script | Slurm job | Notebook | Paper figure |
|---|---|---|---|---|
| 1: alignment strength | `run_experiment1.py` | `experiment1_alignment/run_experiment1.sh` | `experiment1_01_aggregate_and_plot.ipynb` | Figure 3 (Section 5.1) |
| 2: signal-strength heatmap | `run_experiment2.py` | `experiment2_heatmap/run_experiment2.sh` | `experiment2_00_aggregate_and_plot.ipynb` | Figure 6 (Appendix A.1) |
| 3: multi-source, multi-cluster | `run_experiment3.py` | `experiment3_complementary/run_experiment3.sh` | `experiment3_00_aggregate_and_plot.ipynb` | Figure 4 (Section 5.2) |
| 4: comparison with TGMM/TL-GMM | `run_experiment4.py` | `experiment4_comparators/run_experiment4.sh` | `experiment4_00_aggregate_and_plot.ipynb` | Figure 5 (Section 5.3) |
| 5: pooled-estimator success/failure modes | `run_experiment5.py` | `experiment5_pooled_modes/run_experiment5.sh` | `experiment5_00_aggregate_and_plot.ipynb` | Figure 7 (Appendix A.3) |

Experiments 1, 2, 4, and 5 sweep three aspect-ratio regimes (R1/R2/R3) and
a fixed grid (alignment `mu`, or `(Delta_T, Delta_S)`); Experiment 3 is a
single fixed multi-cluster scenario with just Monte Carlo repetitions
(no regime/grid sweep).

## Running an experiment

Each script takes command-line arguments and writes one CSV per call:

```bash
python Numerical_Experiments/Experiments_Script/run_experiment1.py R1 0
```

At scale, submit the matching Slurm array job (first edit the placeholder
paths and conda environment name in the `.sh` file for your cluster):

```bash
sbatch Numerical_Experiments/Slurm_Scripts/experiment1_alignment/run_experiment1.sh
```

Once all array tasks finish, run the corresponding notebook in
`Numerical_Experiments/Notebooks_simulation/` to aggregate results into the
paper's figure. Each notebook's own `PROJECT_ROOT` variable (top setup
cell) needs editing to your local clone's path.

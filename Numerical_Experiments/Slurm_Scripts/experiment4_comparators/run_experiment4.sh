#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-299                    # 3 regimes x 100 seeds = 300 tasks -- EDIT if REGIMES/N_SEEDS change in run_experiment4.py
# EDIT: the two paths below must point at wherever you upload this project
# on the cluster. SBATCH directives are parsed by slurm itself before the
# script body runs, so these must be literal paths -- they cannot
# reference the PROJECT_ROOT variable set further down.
#SBATCH -o /path/to/Transfer_clustering/Numerical_Experiments/Slurm_Scripts/experiment4_comparators/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /path/to/Transfer_clustering/Numerical_Experiments/Slurm_Scripts/experiment4_comparators/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }
# Unlike Experiment 3, this experiment does NOT need cvxpy (all three
# comparators here are EM- or spectral-based) -- but it still uses the same
# "all_amp_projects" env as Experiments 1-3 for numpy/scipy/scikit-learn.

# Prefer the conda environment's own libs over system ones
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

# Threading controls to match cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Python  : $(which python)"
echo "Conda   : $CONDA_PREFIX"
python -V

# EDIT: must match the literal paths in the #SBATCH -o/-e lines above.
PROJECT_ROOT=/path/to/Transfer_clustering
cd "$PROJECT_ROOT/Numerical_Experiments/Experiments_Script"

# --- array math: map task ID to (regime, seed) ---
# EDIT: keep in sync with REGIME_ORDER / N_SEEDS in run_experiment4.py.
regimes=(R1 R2 R3)
num_seeds=100

idx=$SLURM_ARRAY_TASK_ID
regime_idx=$(( idx / num_seeds ))
seed=$(( idx % num_seeds ))
regime=${regimes[$regime_idx]}

echo "Running regime=$regime; seed=$seed"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Numerical_Experiments/Experiments_Script/run_experiment4.py" \
  "$regime" "$seed"

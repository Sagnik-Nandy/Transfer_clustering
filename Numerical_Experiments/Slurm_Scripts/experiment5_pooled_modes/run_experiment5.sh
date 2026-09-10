#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-599                    # 2 target_regimes x 3 regimes x 100 seeds = 600 tasks -- EDIT if TARGET_REGIME_ORDER/REGIME_ORDER/N_SEEDS change in run_experiment5.py
# EDIT: the two paths below must point at wherever you upload this project
# on the cluster. SBATCH directives are parsed by slurm itself before the
# script body runs, so these must be literal paths -- they cannot
# reference the PROJECT_ROOT variable set further down.
#SBATCH -o /path/to/Transfer_clustering/Numerical_Experiments/Slurm_Scripts/experiment5_pooled_modes/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /path/to/Transfer_clustering/Numerical_Experiments/Slurm_Scripts/experiment5_pooled_modes/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }

# Prefer env's libs (matches run_experiment1.sh convention)
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

# --- array math: map task ID to (target_regime, regime, seed) ---
# EDIT: keep in sync with TARGET_REGIME_ORDER / REGIME_ORDER / N_SEEDS in
# run_experiment5.py. target_regime is the outermost axis, then regime,
# then seed (innermost) -- i.e. idx = target_regime_idx*300 + regime_idx*100 + seed.
target_regimes=(below_bbp above_bbp)
regimes=(R1 R2 R3)
num_seeds=100
num_regimes=${#regimes[@]}

idx=$SLURM_ARRAY_TASK_ID
per_target_regime=$(( num_regimes * num_seeds ))
target_regime_idx=$(( idx / per_target_regime ))
remainder=$(( idx % per_target_regime ))
regime_idx=$(( remainder / num_seeds ))
seed=$(( remainder % num_seeds ))
target_regime=${target_regimes[$target_regime_idx]}
regime=${regimes[$regime_idx]}

echo "Running target_regime=$target_regime; regime=$regime; seed=$seed"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Numerical_Experiments/Experiments_Script/run_experiment5.py" \
  "$target_regime" "$regime" "$seed"

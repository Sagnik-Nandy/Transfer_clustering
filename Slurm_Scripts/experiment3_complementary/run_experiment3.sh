#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-99                     # N_SEEDS=100 -- EDIT if N_SEEDS changes in run_experiment3.py
# EDIT: the two paths below must point at wherever you upload this project
# on the cluster (mirrors the layout of your existing vary_meth.sh under
# Experiments_DAIF/GMM). SBATCH directives are parsed by slurm itself
# before the script body runs, so these must be literal paths -- they
# cannot reference the PROJECT_ROOT variable set further down.
#SBATCH -o /home/nandy.15/Research/Transfer_clustering/Slurm_Scripts/experiment3_complementary/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Transfer_clustering/Slurm_Scripts/experiment3_complementary/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }
# EDIT: unlike Experiments 1-2, this experiment needs cvxpy (for
# RelaxedKMeans, used by target-only, pooled, and the adaptive method's
# target branch). "all_amp_projects" (same env as Experiments 1-2) has
# cvxpy installed -- confirmed by checking its site-packages -- so it
# covers this experiment too.

# Prefer env's libs (matches vary_meth.sh convention)
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
PROJECT_ROOT=/home/nandy.15/Research/Transfer_clustering
cd "$PROJECT_ROOT/Experiments_Script"

# --- array math: this experiment is a single fixed scenario (Sec 5.8), so
# the array index IS the seed directly -- no regime/grid decoding needed.
seed=$SLURM_ARRAY_TASK_ID

echo "Running seed=$seed"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Experiments_Script/run_experiment3.py" \
  "$seed"

#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 2:00:00                       # EDIT: dominant cost per task is ONE RelaxedKMeans (ADMM)
                                          # solve on the subsampled target (n=k, up to ~3183 at the
                                          # largest k); the 4 source-projection methods reuse cached
                                          # Theta_hats (precompute_source_directions.py) and are cheap.
                                          # Generous upper bound, lower once you've seen real wall-clock times.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-479                    # 4 targets x 4 k-values x 30 seeds = 480 tasks -- EDIT if
                                          # common.BATCHES, common.K_GRID_BASE, or run_sweep.N_SUBSAMPLE
                                          # changes. See run_sweep.py's task_grid() for the exact enumeration.
                                          # REQUIRES precompute_source_directions.sh's array to have
                                          # completed first (cache/theta_hat_<batch>.npy for all 4 batches).
#SBATCH -o /home/nandy.15/Research/Transfer_clustering/Lung_Analysis_Sweep/Output_Messages/sweep_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Transfer_clustering/Lung_Analysis_Sweep/Error_Messages/sweep_%A_%a_%x_%j_%t.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Python  : $(which python)"
echo "Conda   : $CONDA_PREFIX"
python -V

PROJECT_ROOT=/home/nandy.15/Research/Transfer_clustering
cd "$PROJECT_ROOT/Lung_Analysis_Sweep"

task_id=$SLURM_ARRAY_TASK_ID
echo "Running task_id=$task_id"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Lung_Analysis_Sweep/run_sweep.py" \
  "$task_id"

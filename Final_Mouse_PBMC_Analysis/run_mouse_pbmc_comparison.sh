#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 4:00:00                       # EDIT: generous upper bound, mirrors run_lung_atlas_comparison.sh --
                                          # lower once real wall-clock times are seen. Batches here are far
                                          # smaller than the lung atlas's (max 3201 vs ~3183 similar, but min
                                          # 135 vs ~2098) so most tasks should be faster, not slower.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-47                     # 6 batches x 8 methods = 48 tasks -- EDIT if common.BATCHES or
                                          # common.METHOD_LABELS changes. See run_mouse_pbmc_comparison.py's
                                          # task_grid() for the exact enumeration.
#SBATCH -o /home/nandy.15/Research/Transfer_clustering/Final_Mouse_PBMC_Analysis/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Transfer_clustering/Final_Mouse_PBMC_Analysis/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

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
cd "$PROJECT_ROOT/Final_Mouse_PBMC_Analysis"

task_id=$SLURM_ARRAY_TASK_ID
echo "Running task_id=$task_id"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Final_Mouse_PBMC_Analysis/run_mouse_pbmc_comparison.py" \
  "$task_id"

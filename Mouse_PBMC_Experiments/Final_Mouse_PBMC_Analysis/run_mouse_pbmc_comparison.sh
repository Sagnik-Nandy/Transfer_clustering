#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 4:00:00                       # EDIT: generous upper bound -- lower once real wall-clock times
                                          # are seen. Batch sizes here range from 135 to 3201 cells, so
                                          # per-task time varies substantially across the array.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-41                     # 6 batches x 7 methods = 42 tasks -- EDIT if common.BATCHES or
                                          # common.METHOD_LABELS changes. See run_mouse_pbmc_comparison.py's
                                          # task_grid() for the exact enumeration.
# EDIT: the two paths below must point at wherever you upload this project
# on the cluster. SBATCH directives are parsed by slurm itself before the
# script body runs, so these must be literal paths -- they cannot
# reference the PROJECT_ROOT variable set further down.
#SBATCH -o /path/to/Transfer_clustering/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /path/to/Transfer_clustering/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

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

PROJECT_ROOT=/path/to/Transfer_clustering
cd "$PROJECT_ROOT/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis"

task_id=$SLURM_ARRAY_TASK_ID
echo "Running task_id=$task_id"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/run_mouse_pbmc_comparison.py" \
  "$task_id"

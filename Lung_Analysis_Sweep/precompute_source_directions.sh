#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 2:00:00                       # EDIT: one RelaxedKMeans (ADMM) solve at full batch size
                                          # (n up to ~3183, K=13, d=5000) per task -- generous upper
                                          # bound, lower once you've seen real wall-clock times.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                # ADMM's per-iteration eigendecomposition benefits from
                                          # multi-threaded BLAS -- see OMP/MKL/OPENBLAS exports below.
#SBATCH --mem=16G
#SBATCH --array=0-3                      # one task per batch (common.BATCHES) -- EDIT if that list changes.
#SBATCH -o /home/nandy.15/Research/Transfer_clustering/Lung_Analysis_Sweep/Output_Messages/precompute_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Transfer_clustering/Lung_Analysis_Sweep/Error_Messages/precompute_%A_%a_%x_%j_%t.err

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
  "$PROJECT_ROOT/Lung_Analysis_Sweep/precompute_source_directions.py" \
  "$task_id"

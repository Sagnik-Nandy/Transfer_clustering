#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 1:00:00                       # cheap: one leading eigenvector per call (oracle
                                          # target_based_estimate/source_based_estimate), not
                                          # RelaxedKMeans/SDP -- 8 k-values x 30 seeds x 7 methods
                                          # per task. Generous upper bound.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-5                      # 6 target batches (common.BATCHES) -- one task per target,
                                          # each covering its own full k x seed grid internally. EDIT
                                          # if common.BATCHES changes.
#SBATCH -o /path/to/Transfer_clustering/Mouse_PBMC_Experiments/mouse_intro_illustration/logs/tcell_nkcell_%A_%a_%x_%j_%t.out
#SBATCH -e /path/to/Transfer_clustering/Mouse_PBMC_Experiments/mouse_intro_illustration/logs/tcell_nkcell_%A_%a_%x_%j_%t.err

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
cd "$PROJECT_ROOT/Mouse_PBMC_Experiments/mouse_intro_illustration/T_cell_vs_NK_cell"

task_id=$SLURM_ARRAY_TASK_ID
echo "Running task_id=$task_id"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Mouse_PBMC_Experiments/mouse_intro_illustration/T_cell_vs_NK_cell/run_sweep.py" \
  "$task_id"

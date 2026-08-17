#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 4:00:00                       # EDIT: generous upper bound -- target_only finishes in
                                          # ~1 min, but multi_source_pooled/adaptive_multi_source
                                          # (3 and ~34 relaxed-K-means SDP solves respectively) and
                                          # the scrna/gdec comparators are untimed at this data scale
                                          # on the cluster; lower once you've seen real wall-clock times.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                # ADMM's per-iteration eigendecomposition (relaxed_kmeans.py)
                                          # benefits from multi-threaded BLAS -- see OMP/MKL/OPENBLAS
                                          # exports below, which must match this.
#SBATCH --mem=16G
#SBATCH --array=0-79                     # 4 batches x 20 (target_only:1 + 4 multi-source methods x
                                          # 4 source-specs each [all + 3 single-source] + 3 comparators:1)
                                          # = 80 tasks -- EDIT if common.BATCHES, common.METHOD_LABELS,
                                          # or MULTI_SOURCE_METHODS in run_lung_atlas_comparison.py changes.
                                          # See that file's task_grid() for the exact enumeration.
# EDIT: the two paths below must point at wherever you upload this project
# on the cluster (mirrors Slurm_Scripts/experiment3_complementary/run_experiment3.sh).
# SBATCH directives are parsed by slurm itself before the script body runs,
# so these must be literal paths -- they cannot reference the PROJECT_ROOT
# variable set further down.
#SBATCH -o /home/nandy.15/Research/Transfer_clustering/Final_Lung_Atlas_Analysis/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Transfer_clustering/Final_Lung_Atlas_Analysis/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }
# EDIT: needs numpy/scipy/pandas/scikit-learn/scanpy (common.py's data
# loading + metrics), plus torch (method_gdec.py's SDAE/DEC pipeline,
# CPU-only is fine at this data scale) and cvxopt (method_scrna.py's NMF
# pipeline) -- see run_lung_atlas_comparison.ipynb's header note. UNLIKE
# Experiments 1-4's cvxpy_env, cvxpy/MOSEK are NOT required: RelaxedKMeans's
# default solver is now ADMM (Mixon, Villar & Ward 2017), a self-contained
# numpy/scipy routine -- see Experiments_Script/transfer_clustering/relaxed_kmeans.py.
# Verified 2026-07-10: all_amp_projects has scanpy/torch/cvxopt/cvxpy all
# importable (torch needs the LD_LIBRARY_PATH export below; scanpy needed a
# typing_extensions>=4.16 upgrade -- anndata's pin -- done in that env).

# Prefer env's libs (matches vary_meth.sh / run_experiment3.sh convention)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

# Threading controls to match cpus-per-task (matters for ADMM's eigh calls)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Python  : $(which python)"
echo "Conda   : $CONDA_PREFIX"
python -V

# EDIT: must match the literal paths in the #SBATCH -o/-e lines above.
PROJECT_ROOT=/home/nandy.15/Research/Transfer_clustering
cd "$PROJECT_ROOT/Final_Lung_Atlas_Analysis"

# --- array math: the array index IS the task_id directly -- run_lung_atlas_
# comparison.py itself maps it to a (target_batch, method) pair via
# itertools.product(common.BATCHES, common.METHOD_LABELS.keys()).
task_id=$SLURM_ARRAY_TASK_ID

echo "Running task_id=$task_id"

"$CONDA_PREFIX/bin/python" \
  "$PROJECT_ROOT/Final_Lung_Atlas_Analysis/run_lung_atlas_comparison.py" \
  "$task_id"

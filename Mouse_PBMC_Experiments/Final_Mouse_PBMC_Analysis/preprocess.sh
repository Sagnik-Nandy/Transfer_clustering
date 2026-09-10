#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# EDIT: the two paths below must point at wherever you upload this project
# on the cluster. SBATCH directives are parsed by slurm itself before the
# script body runs, so these must be literal paths -- they cannot
# reference the PROJECT_ROOT variable set further down.
#SBATCH -o /path/to/Transfer_clustering/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/Output_Messages/preprocess_%j.out
#SBATCH -e /path/to/Transfer_clustering/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/Error_Messages/preprocess_%j.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }
export PYTHONNOUSERSITE=1

PROJECT_ROOT=/path/to/Transfer_clustering
cd "$PROJECT_ROOT/Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis"
"$CONDA_PREFIX/bin/python" preprocess.py

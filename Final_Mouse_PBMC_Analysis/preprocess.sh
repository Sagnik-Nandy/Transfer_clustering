#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o /home/nandy.15/Research/Transfer_clustering/Final_Mouse_PBMC_Analysis/Output_Messages/preprocess_%j.out
#SBATCH -e /home/nandy.15/Research/Transfer_clustering/Final_Mouse_PBMC_Analysis/Error_Messages/preprocess_%j.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }
export PYTHONNOUSERSITE=1

PROJECT_ROOT=/home/nandy.15/Research/Transfer_clustering
cd "$PROJECT_ROOT/Final_Mouse_PBMC_Analysis"
"$CONDA_PREFIX/bin/python" preprocess.py

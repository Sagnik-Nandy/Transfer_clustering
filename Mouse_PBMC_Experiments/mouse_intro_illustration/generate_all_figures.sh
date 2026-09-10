#!/bin/bash
# Generates all 24 intro figures (2 contrasts x 6 targets x 2 metrics) by
# calling plot_intro_figure.R once per combination with positional args
# (CONTRAST_DIR TARGET_BATCH METRIC). Each run writes
# figures/<CONTRAST_DIR>/<TARGET_BATCH>_<METRIC>.{pdf,png}.
#
# Requires: panel1_umap.csv (01_umap_overview.py) and both contrasts'
# results/ CSVs (run_sweep.py / run_sweep.sh) already in place.
#
# Usage:
#   conda activate all_amp_projects   # has r-ggplot2/r-dplyr/r-patchwork installed
#   ./generate_all_figures.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONTRASTS=("B_cell_vs_Macrophage" "T_cell_vs_NK_cell")
TARGETS=("PeripheralBlood_1" "PeripheralBlood_2" "PeripheralBlood_3"
         "PeripheralBlood_4" "PeripheralBlood_5" "PeripheralBlood_6")
METRICS=("ari" "misclustering")

n=0
for contrast in "${CONTRASTS[@]}"; do
  for target in "${TARGETS[@]}"; do
    for metric in "${METRICS[@]}"; do
      n=$((n + 1))
      echo "[$n/24] $contrast  $target  $metric"
      Rscript plot_intro_figure.R "$contrast" "$target" "$metric"
    done
  done
done
echo "Done: $n figures written under figures/<contrast>/."

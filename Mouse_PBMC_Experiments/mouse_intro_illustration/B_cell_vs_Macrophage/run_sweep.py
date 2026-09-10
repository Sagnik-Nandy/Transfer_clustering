#!/usr/bin/env python
"""One Slurm array task per target batch (task_id 0..5 indexes
common.BATCHES) for the **B cell vs Macrophage** contrast. Runs
common.run_target_sweep for that one target and writes its results to two
CSVs (one per metric) in results/:

    results/<target>_ari.csv
    results/<target>_misclustering.csv

6 targets x 2 metrics = 12 CSVs for this contrast (24 total across both
contrast subpackages). See ../common.py for the sweep itself and
../README.md for the full pipeline.

Usage:
    python run_sweep.py <task_id>     # task_id in [0, 5]
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))  # mouse_intro_illustration/ for common.py

import pandas as pd
import common

CONTRAST_NAME = "B_cell_vs_Macrophage"
RESULTS_DIR = os.path.join(_THIS_DIR, "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int, nargs="?",
                         default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    args = parser.parse_args()
    task_id = int(args.task_id)
    if not (0 <= task_id < len(common.BATCHES)):
        raise ValueError(f"task_id must be in [0, {len(common.BATCHES) - 1}], got {task_id}")
    target = common.BATCHES[task_id]

    ari_rows, mis_rows = common.run_target_sweep(CONTRAST_NAME, target)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ari_path = os.path.join(RESULTS_DIR, f"{target}_ari.csv")
    mis_path = os.path.join(RESULTS_DIR, f"{target}_misclustering.csv")
    pd.DataFrame(ari_rows).to_csv(ari_path, index=False, float_format="%.6f")
    pd.DataFrame(mis_rows).to_csv(mis_path, index=False, float_format="%.6f")
    print(f"Saved: {ari_path}")
    print(f"Saved: {mis_path}")


if __name__ == "__main__":
    main()

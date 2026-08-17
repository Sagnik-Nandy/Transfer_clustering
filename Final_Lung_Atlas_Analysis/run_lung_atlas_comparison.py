#!/usr/bin/env python
"""Lung atlas (GSE130148) multi-cluster comparison -- cluster-parallel driver.

Non-interactive counterpart to run_lung_atlas_comparison.ipynb: the notebook
runs all (target batch, method) combinations sequentially in one process,
which is fine for a single sanity-check batch but not for the full 4-batch
comparison (adaptive_multi_source alone re-solves the relaxed K-means SDP
~34 times per batch -- see method_ours.py). This script runs exactly ONE
(target batch, method) combination per invocation, so each can be its own
Slurm array task (Slurm_Scripts convention: matches run_experiment3.py /
run_experiment4.py's "one seed per array task", just indexed over
(batch, method) here instead of seed).

Usage (matches run_lung_atlas_comparison.sh):

    python run_lung_atlas_comparison.py <task_id>

task_id in [0, len(BATCHES) * len(METHOD_LABELS) - 1] = [0, 23] indexes into
itertools.product(common.BATCHES, common.METHOD_LABELS.keys()) -- outer loop
over target batch, inner loop over method, matching the notebook's own
nested loop order. Writes one CSV row to results/raw/<batch>__<method>.csv;
concatenate all of them (see the notebook's aggregation cell) to reproduce
the notebook's `results` DataFrame.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import common
import method_ours
import method_tlgmm
import method_scrna
import method_gdec

H5AD_PATH = os.path.join(_THIS_DIR, "..", "lung_atlas_analysis", "data", "lung_atlas_hvg_lognorm.h5ad")

# Same 6 methods, same order, as common.METHOD_LABELS -- fixes the
# task_id <-> (batch, method) mapping used below.
ALL_METHODS = dict(method_ours.OURS_METHODS)
ALL_METHODS["tlgmm"] = method_tlgmm.method_tlgmm
ALL_METHODS["scrna"] = method_scrna.method_scrna
ALL_METHODS["gdec_gcnfree"] = method_gdec.method_gdec_gcnfree
assert set(ALL_METHODS) == set(common.METHOD_LABELS), "ALL_METHODS must match common.METHOD_LABELS exactly"

RESULTS_DIR = "results_0.5_alpha/raw"  # relative to this file's directory
FIELDNAMES = ["target", "method", "elapsed_sec", "misclustering", "ari", "v_measure"]


def task_grid():
    """The fixed (batch, method) ordering task_id indexes into -- outer
    loop over target batch, inner loop over method (method order taken
    from common.METHOD_LABELS, matching the notebook's own loop)."""
    return list(itertools.product(common.BATCHES, common.METHOD_LABELS.keys()))


def run_one(task_id: int) -> dict:
    grid = task_grid()
    if not (0 <= task_id < len(grid)):
        raise ValueError(f"task_id must be in [0, {len(grid) - 1}], got {task_id}")
    target_batch, method_name = grid[task_id]

    batch_data = common.load_batches(H5AD_PATH)
    X_T, y_T = batch_data[target_batch]
    sources = [batch_data[b][0] for b in common.BATCHES if b != target_batch]
    print(f"target={target_batch}, method={method_name}, n_T={X_T.shape[0]}, "
          f"sources n=({', '.join(str(s.shape[0]) for s in sources)})")

    fn = ALL_METHODS[method_name]
    t0 = time.time()
    z_hat = fn(X_T, sources, common.RANDOM_STATE)
    elapsed = time.time() - t0

    metrics = common.compute_metrics(z_hat, y_T)
    row = dict(target=target_batch, method=method_name, elapsed_sec=elapsed, **metrics)
    print(f"{method_name} on {target_batch}: {elapsed:.1f}s -- {metrics}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()

    row = run_one(args.task_id)

    out_dir = os.path.join(_THIS_DIR, RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{row['target']}__{row['method']}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

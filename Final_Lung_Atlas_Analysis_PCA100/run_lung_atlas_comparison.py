#!/usr/bin/env python
"""Lung atlas (GSE130148) multi-cluster comparison, PCA-100 variant --
cluster-parallel driver.

Same task structure as Final_Lung_Atlas_Analysis/run_lung_atlas_comparison.py
(one (target batch, method, source) combination per invocation/Slurm array
task), but every method here clusters on a shared 100-dim PCA
representation instead of the raw d=5000 HVG matrix -- see
common.project_to_target_pcs: for each task, PCA is fit on the target
batch's own centered data, and the source batch(es) used in that task are
projected onto that same target-fit basis before any method runs. scrna
(NMF) is dropped (needs non-negative input, which PCA scores violate).

The `source` dimension: `target_only` uses no sources at all, and the
comparators (tlgmm/gdec_gcnfree) always use every other batch pooled
together ("all"). Each of the 4 "ours" methods that take a `sources` list
(multi_source_pooled, pooled_concat, target_source_pooled,
adaptive_multi_source) is run BOTH with all 3 other batches pooled ("all")
AND, separately, with each individual other batch as the sole source
("Dropseq_2" etc.) -- so a method's per-source sensitivity can be inspected
directly, not just its pooled-source performance.

Usage (matches run_lung_atlas_comparison.sh):

    python run_lung_atlas_comparison.py <task_id>

task_id in [0, len(task_grid()) - 1] indexes into task_grid() below, which
enumerates (target_batch, method, source_spec) triples -- outer loop over
target batch, inner loop over method, innermost loop over source_spec (only
>1 value for the 4 multi-source "ours" methods). Writes one CSV row to
results_0.5_alpha/raw/<batch>__<method>__<source>.csv; concatenate all of
them (see the notebook's aggregation cell) to reproduce the notebook's
`results` DataFrame.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import common
import method_ours
import method_tlgmm
import method_gdec

H5AD_PATH = os.path.join(_THIS_DIR, "..", "lung_atlas_analysis", "data", "lung_atlas_hvg_lognorm.h5ad")

# Same methods, same order, as common.METHOD_LABELS -- fixes the
# task_id <-> (batch, method, source) mapping used below.
ALL_METHODS = dict(method_ours.OURS_METHODS)
ALL_METHODS["tlgmm"] = method_tlgmm.method_tlgmm
ALL_METHODS["gdec_gcnfree"] = method_gdec.method_gdec_gcnfree
assert set(ALL_METHODS) == set(common.METHOD_LABELS), "ALL_METHODS must match common.METHOD_LABELS exactly"

# The 4 "ours" methods that take a `sources` list and pool/project across it
# -- these are the ones that additionally get a one-source-at-a-time run,
# on top of their all-sources-pooled run. target_only takes no sources at
# all; the comparators (tlgmm/gdec_gcnfree) keep their all-sources-only
# behavior.
MULTI_SOURCE_METHODS = {
    "multi_source_pooled", "pooled_concat", "target_source_pooled", "adaptive_multi_source",
}

RESULTS_DIR = "results_0.5_alpha/raw"  # relative to this file's directory
FIELDNAMES = ["target", "method", "source", "elapsed_sec", "misclustering", "ari", "v_measure"]


def task_grid():
    """The fixed (batch, method, source) ordering task_id indexes into --
    outer loop over target batch, middle loop over method (order taken from
    common.METHOD_LABELS, matching the notebook's own loop), innermost loop
    over source_spec:

      - "target_only": a single row, source_spec=None (no sources used).
      - the 4 MULTI_SOURCE_METHODS: one row with source_spec="all" (every
        other batch pooled together, the original behavior), plus one row
        per other batch with source_spec=<that batch name> (single-source).
      - the 2 comparators (tlgmm, gdec_gcnfree): a single row, source_spec="all".
    """
    grid = []
    for target in common.BATCHES:
        other_batches = [b for b in common.BATCHES if b != target]
        for method in common.METHOD_LABELS:
            if method == "target_only":
                grid.append((target, method, None))
            elif method in MULTI_SOURCE_METHODS:
                grid.append((target, method, "all"))
                for src in other_batches:
                    grid.append((target, method, src))
            else:
                grid.append((target, method, "all"))
    return grid


def run_one(task_id: int) -> dict:
    grid = task_grid()
    if not (0 <= task_id < len(grid)):
        raise ValueError(f"task_id must be in [0, {len(grid) - 1}], got {task_id}")
    target_batch, method_name, source_spec = grid[task_id]

    batch_data = common.load_batches(H5AD_PATH)
    X_T_raw, y_T = batch_data[target_batch]
    if source_spec is None:
        sources_raw = []
    elif source_spec == "all":
        sources_raw = [batch_data[b][0] for b in common.BATCHES if b != target_batch]
    else:
        sources_raw = [batch_data[source_spec][0]]

    # PCA-100 projection: basis fit on the target's own centered data,
    # sources projected onto that same basis -- see common.project_to_target_pcs.
    X_T, sources = common.project_to_target_pcs(X_T_raw, sources_raw)
    print(f"target={target_batch}, method={method_name}, n_T={X_T.shape[0]}, d={X_T.shape[1]}, "
          f"sources n=({', '.join(str(s.shape[0]) for s in sources)})")

    fn = ALL_METHODS[method_name]
    t0 = time.time()
    z_hat = fn(X_T, sources, common.RANDOM_STATE)
    elapsed = time.time() - t0

    metrics = common.compute_metrics(z_hat, y_T)
    source_label = source_spec if source_spec is not None else "none"
    row = dict(target=target_batch, method=method_name, source=source_label,
               elapsed_sec=elapsed, **metrics)
    print(f"{method_name} (source={source_label}) on {target_batch}: {elapsed:.1f}s -- {metrics}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()

    row = run_one(args.task_id)

    out_dir = os.path.join(_THIS_DIR, RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{row['target']}__{row['method']}__{row['source']}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

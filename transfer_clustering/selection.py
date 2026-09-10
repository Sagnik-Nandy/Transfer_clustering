"""Practical strategies for calibrating the oracle-free selection between
target-based and source-based estimators (Section 2.3 for K=2, Section 4
for K>2, of Chakraborty & Nandy's paper).

The paper's validation threshold (tau_n = C0 sigma_T^2 (1+sqrt(d/n_T)) for
K=2, and its K>2 analogue) involves an absolute constant C0 (resp. D0)
that the paper never pins down numerically. `two_community.AdaptiveTransferClustering`
and `multi_cluster.AdaptiveProjectedClustering` expose three selection
strategies (no cross-validation / sample-splitting in any of them):

  - "formula": the literal threshold with a caller-specified C0/D0.
  - "bootstrap": `bootstrap_null_quantile` below -- a parametric bootstrap
    that simulates synthetic replicates at the target-only-condition-style
    recovery threshold (not at zero signal -- see
    `two_community.calibrate_C0_bootstrap`'s docstring for why
    that distinction matters), runs the *same* estimator + statistic
    pipeline on each replicate, and takes an empirical quantile of the
    resulting distribution as the threshold.
  - "manual": the caller directly names which branch (target/source) to
    use, bypassing all statistics.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def bootstrap_null_quantile(
    simulate_null: Callable[[np.random.Generator], np.ndarray],
    estimator_fn: Callable[[np.ndarray], np.ndarray],
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    alpha: float = 0.05,
    n_boot: int = 200,
    random_state: Optional[int] = None,
) -> float:
    """Parametric-bootstrap calibration of a validation threshold.

    Repeatedly draws synthetic replicates via `simulate_null`, applies
    `estimator_fn` to get candidate labels, evaluates
    `statistic_fn(labels, X)`, and returns the empirical (1 - alpha)
    quantile of |statistic| across replicates. Using the *same*
    estimator/statistic pair as in the real pipeline means any
    self-referential inflation (the estimator being derived from -- and
    then re-evaluated on -- the same data) is captured automatically,
    rather than relying on an analytic (and here, uncalibrated) constant.
    """
    rng = np.random.default_rng(random_state)
    values = np.empty(n_boot)
    for b in range(n_boot):
        X_null = simulate_null(rng)
        labels = estimator_fn(X_null)
        values[b] = abs(statistic_fn(labels, X_null))
    return float(np.quantile(values, 1.0 - alpha))

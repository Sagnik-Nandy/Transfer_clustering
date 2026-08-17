"""Unified entry point for transfer-assisted clustering.

Dispatches between the two-community (K=2) pipeline of Section 2 /
Algorithm 1 (Ndaoud's spectral method) and the multi-cluster (K>2)
pipeline of Section 4.2 / Algorithm 2 (Giraud & Verzelen's relaxed
K-means + TSClust), depending on the requested number of clusters K.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .two_community import AdaptiveTransferClustering, OracleTransferClustering
from .multi_cluster import AdaptiveProjectedClustering


class TransferClustering:
    """Transfer-assisted clustering of a target dataset using one or more
    source datasets.

    Parameters
    ----------
    K : int
        Number of clusters.
          - K == 2: uses the two-community machinery of Section 2
            (Ndaoud's hollowed-Gram spectral method, Algorithm 1, and the
            Section 2.3 adaptive selection).
          - K > 2: uses the multi-cluster machinery of Section 4.2
            (Algorithm 2: Giraud & Verzelen's relaxed K-means SDP +
            TSClust / Algorithm 3).
    oracle : {"target", "source", None}
        If given (K == 2 only), bypasses the adaptive selection and runs
        Algorithm 1 directly under the stated oracle assumption (Theorem 1).
        If None (default), uses the oracle-free adaptive selection
        (Section 2.3 for K=2, Algorithm 2 for K>2).
    **kwargs
        Forwarded to the underlying estimator
        (`AdaptiveTransferClustering` / `AdaptiveProjectedClustering`),
        e.g. `sigma_T2`, `C0` / `D0`, `relaxed_kmeans_kwargs`.

    Example
    -------
    >>> model = TransferClustering(K=2)
    >>> labels, branch = model.fit_predict(X_target, [X_source_1, X_source_2])
    """

    def __init__(self, K: int, oracle: Optional[str] = None, **kwargs):
        if K < 2:
            raise ValueError("K must be >= 2.")
        self.K = K
        self.oracle = oracle
        self.kwargs = kwargs

    def fit_predict(
        self, X_T: np.ndarray, source_datasets: Sequence[np.ndarray]
    ) -> Union[np.ndarray, Tuple[np.ndarray, str]]:
        if self.K == 2:
            if self.oracle is not None:
                estimator = OracleTransferClustering(oracle=self.oracle)
                return estimator.fit_predict(X_T, source_datasets), self.oracle
            estimator = AdaptiveTransferClustering(**self.kwargs)
            return estimator.fit_predict(X_T, source_datasets)

        estimator = AdaptiveProjectedClustering(K=self.K, **self.kwargs)
        return estimator.fit_predict(X_T, source_datasets)

"""Transfer-assisted clustering for two-community and multi-cluster
Gaussian mixture models, implementing Section 2, Section 4, and
Algorithms 1-3 of the transfer-learning-for-clustering discussion draft
(Chakraborty & Nandy).

Modules
-------
spectral
    Ndaoud (2018) hollowed-Gram-matrix spectral clustering (arXiv:1812.08078),
    the reusable subroutine behind the two-community target-based estimator
    and Algorithm 1.
two_community
    Section 2.2 (target-based / source-based estimators), Section 2.3
    (adaptive selection), and Algorithm 1 (Oracle Transfer-Assisted
    Clustering) for the K=2 two-community model.
relaxed_kmeans
    Giraud & Verzelen (2019) relaxed K-means SDP (arXiv:1807.07547),
    used as the target-based clustering routine for K>2.
ts_clust
    Algorithm 3 (TSClust): spectral initialization + Lloyd refinement.
multi_cluster
    Section 4.2 and Algorithm 2 (Adaptive Transfer-Assisted Projected
    Clustering) for the K>2 multi-cluster model.
api
    `TransferClustering`: unified entry point dispatching between the
    K=2 and K>2 pipelines.

Quick start
-----------
>>> from transfer_clustering import TransferClustering
>>> model = TransferClustering(K=2)
>>> labels, branch = model.fit_predict(X_target, [X_source])
"""

from .api import TransferClustering
from .spectral import spectral_sign_clustering
from .two_community import (
    AdaptiveTransferClustering,
    OracleTransferClustering,
    target_based_estimate,
    source_based_estimate,
    estimate_source_direction,
)
from .relaxed_kmeans import RelaxedKMeans
from .ts_clust import ts_clust
from .multi_cluster import AdaptiveProjectedClustering

__all__ = [
    "TransferClustering",
    "spectral_sign_clustering",
    "AdaptiveTransferClustering",
    "OracleTransferClustering",
    "target_based_estimate",
    "source_based_estimate",
    "estimate_source_direction",
    "RelaxedKMeans",
    "ts_clust",
    "AdaptiveProjectedClustering",
]

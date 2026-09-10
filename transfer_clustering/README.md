# transfer_clustering

The algorithm library implementing Algorithms 1-6 of the paper: oracle,
adaptive, and pooled transfer-assisted clustering for both the
two-community (K=2) and multi-cluster (K>2) high-dimensional Gaussian
mixture models. Used directly by `Numerical_Experiments/` and
`Mouse_PBMC_Experiments/`. See the root README for install requirements.

## Entry point

```python
from transfer_clustering import TransferClustering

model = TransferClustering(K=2)
labels, branch = model.fit_predict(X_target, [X_source_1, X_source_2])
```

`TransferClustering.__init__(K, oracle=None, **kwargs)` dispatches between
the K=2 and K>2 machinery based on `K`; pass `oracle="target"` or
`oracle="source"` to bypass adaptive selection and run the named branch
directly. Each module below can also be used on its own (e.g. calling
`two_community.target_source_pooled_subspace_estimate` directly for
Algorithm 4, without going through `TransferClustering`).

## Modules

| File | Implements | Role |
|---|---|---|
| `spectral.py` | Ndaoud (2018)'s hollowed-Gram-matrix spectral method | The reusable target-based clustering subroutine behind Algorithm 1's target branch |
| `two_community.py` | Algorithm 1 (Meta Algorithm, one source), Algorithm 2 (Adaptive, two communities), Algorithm 4's K=2 branch (pooled), and the oracle multi-source two-community procedure from the supplement | Target-based / source-based / pooled estimators and the oracle-free adaptive selector, for K=2 |
| `relaxed_kmeans.py` | Giraud & Verzelen (2019)'s relaxed K-means SDP | The target-based clustering routine for K>2 (`RelaxedKMeans`, default solver `"ADMM"`, no external SDP solver required) |
| `ts_clust.py` | Algorithm 5 (TSClust) | Spectral initialization + Lloyd refinement, applied to the source-projected target observations in Algorithm 3 |
| `multi_cluster.py` | Algorithm 3 (Adaptive Transfer-Assisted Projected Clustering) and Algorithm 4's K>2 branch (pooled) | Target-based / source-based / pooled estimators and the oracle-free adaptive selector, for K>2 |
| `pooled_std_scaled.py` | An alternative normalization for Algorithm 4's pooling step (not in the paper) | Rescales each candidate direction by the observed standard deviation of its own projection of the target data, instead of by raw magnitude or forced unit norm, before pooling |
| `selection.py` | The bootstrap calibration shared by both adaptive selectors | `bootstrap_null_quantile`: simulates replicates at the target-only recovery threshold and takes an empirical quantile of the validation statistic as the calibrated threshold constant |
| `api.py` | — | `TransferClustering`: the unified entry point shown above |

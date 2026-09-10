# External_Methods

Vendored, unmodified third-party implementations of two of the real-data
comparator methods used in Table 1 of the paper (Section 6). Each is a
copy of its original GitHub repository, credited and linked in its own
README below. See the root README's Requirements section for the extra
packages each one needs.

| Folder | Comparator | Source |
|---|---|---|
| [`GDEC/`](GDEC/README.md) | GDEC (autoencoder + deep embedded clustering) | [YuzhiSun/GDEC](https://github.com/YuzhiSun/GDEC) |
| [`scRNA/`](scRNA/README.md) | NMF-based transfer clustering | [nicococo/scRNA](https://github.com/nicococo/scRNA) |

Driven from `Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/method_gdec.py`
and `method_scrna.py` respectively, which import directly from these
folders (see that directory's README for how they're wired in).

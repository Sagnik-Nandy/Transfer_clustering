# Mouse_PBMC_Experiments

The paper's real-data application (Section 6): transfer-assisted
clustering of mouse peripheral-blood scRNA-seq data (Han et al. 2018,
Mouse Cell Atlas), using each of the dataset's 6 batches as target in
turn against the other 5 as sources. See the root README for install
requirements.

| Folder | Contents |
|---|---|
| [`Final_Mouse_PBMC_Analysis/`](Final_Mouse_PBMC_Analysis/README.md) | The Table 1 comparison: 7 methods (4 of ours, plus TL-GMM, NMF, and GDEC) x 6 target batches, K=9 cell types. |
| [`mouse_intro_illustration/`](mouse_intro_illustration/README.md) | The Figure 1 introductory illustration: a K=2 target-sample-size sweep for two binary cell-type contrasts, every batch as target. |

Both pipelines load the same preprocessed data,
`Final_Mouse_PBMC_Analysis/data/mouse_pbmc_hvg_lognorm.h5ad` -- produced by
`Final_Mouse_PBMC_Analysis/preprocess.py` -- so run that first regardless
of which pipeline you're using.

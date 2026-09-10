> **Vendored code.** This directory is a copy of [YuzhiSun/GDEC](https://github.com/YuzhiSun/GDEC),
> the implementation accompanying Wang, Y. M., Sun, Y., Wang, B., Wu, Z., He, X. Y. and Zhao, Y. (2024),
> "Transfer learning for clustering single-cell RNA-seq data crossing-species and batch, case on
> uterine fibroids," *Briefings in Bioinformatics* 25, bbad426. It is used here as the GDEC
> comparator method in Table 1 of this repository's paper (see `Mouse_PBMC_Experiments/Final_Mouse_PBMC_Analysis/method_gdec.py`
> for the driver code); all credit for the method and this implementation belongs to its original authors.

# GDEC
Transfer learning for clustering single-cell RNA-seq data crossing-species and batch, case on Uterine fibroids
## Code description
The files starting with util is the processing code for each dataset.   
The files starting with DECtransfer are the training codes for each dataset.  
dglNet.py is the util for construct the GCN.  
dglNet_batch_gene.py is used for constructing embedding features of genes.  
DECpredictValid.py is used for predict the result.  
## Environment
For details environment.yml  
## Data
The data used in paper can be obtained on https://drive.google.com/file/d/1o0JOBVzTp7GdgbCi3YCKU4mWrh8fDiPg/view?usp=drive_link
## Pipline
1. Constructing GCN Network Nodes and Relationships Using dglNet.py .
2. Constructing GCN embedding features with dglNet_batch_gene.py .
3. Create a folder corresponding to the dataset. For example, we should create the file named "mouse"and "mouse_gcn" folder to save the data for model training on the brain dataset.
4. Constructing features for train using the file starting with util.
5. Training the model using the file starting with DECtransfer.

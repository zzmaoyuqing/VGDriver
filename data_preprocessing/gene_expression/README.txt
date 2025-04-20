preprocessing expression data

1.run preprocess_manifest_express.R
The script uses BRCA as an example. This script gets the TPM tumor and normal matrices (gene X sample) for SPECIFIC CANCER, filtering the genes on the condition that they: 1. are duplicated; 2. contain a value of 0
Notice: Before run this R script, make sure to download the manifest file to get samples files by gdc client tool.

Here are the steps to get samples files:
Access the website(https://portal.gdc.cancer.gov/analysis_page?app=CohortBuilder&tab=general), select the BRCA cancer of TCGA program and download the metadata and manifest files of transcriptome profiling, then use the GDC client tool to download them.
Run the command line in a terminal 'gdc-client download -m .\gdc_manifest_express_BRCA.2024-06-03.txt -d .\BRCA\' to get sample files of BRCA.

2.run preprocess_gene_expression.py
The script uses BRCA as an example.This script is to calculate the log2 fold change value for each gene on the paired samples corresponding to that gene.
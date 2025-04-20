preprocessing CNAs

Notice: If you want other cancer types in addition to the ones in folder “gene_summary”, then run the scripts from the first step to the last step, if you don't need other cancer types, you can run the script 'preprocess_copy_number.py' in the fourth step directly.

1. run get_ctype_gene_CNA_matrix.py
Notice: Before run this python script, make sure to download the manifest file to get samples files by gdc client tool.

Here are the steps to get samples files:
Access the website(https://portal.gdc.cancer.gov/analysis_page?app=CohortBuilder&tab=general), select the BRCA cancer of TCGA program and download the metadata and manifest files of copy number variation, then use the GDC client tool to download them.
Run the command line in a terminal 'gdc-client download -m .\gdc_manifest_CNVs_BRCA.2024-06-01.txt -d .\BRCA\' to get sample files of BRCA.

2.run preprocess_CNAs.R
Notice: Before run this R script, make sure to download the makerfile from https://api.gdc.cancer.gov/data/9bd7cbce-80f9-449e-8007-ddc9b1e89dfb

3.GISTIC2.0 analysis
Run the command line in a Linux terminal './run_gistic_BRCA' to get sample files of BRCA.
Notice: Make sure you have installed the GISTIC2.0 software locally before running the script. If you don't want to install GISTIC2.0, you can access the website(https://www.genepattern.org/) to analyze it online.

4.run preprocess_copy_number.py
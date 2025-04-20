preprocessing SNVs

1.run preprocess_manifest_SNVs.R
The script uses BRCA as an example. The script gets two output files which are the input files for the following two python scripts.
Notice: Before run this R script, make sure to download the manifest file to get samples files by gdc client tool.

Here are the steps to get samples files:
Access the website(https://portal.gdc.cancer.gov/analysis_page?app=CohortBuilder&tab=general), select the BRCA cancer of TCGA program and download the manifest files of simple nucleotide variation, then use the GDC client tool to download them.
Run the command line in a terminal 'gdc-client download -m .\gdc_manifest_SNVs_BRCA.2024-05-30.txt -d .\BRCA\' to get sample files of BRCA.

2.run preprocess_mutation_freqs.py
This script filtering the genes on the condition that they: 1. silent mutations; 2. ultra-mutations

Notice: Before run this python script, make sure to download the GFF(general feature format) file from 'https://www.gencodegenes.org/human/'.
Choose Content with Basic gene annotation, Regions with CHR, Download with GFF3 and download the annotation file which named as 'gencode.v46.basic.annotation.gff3'

3.run preprocess_mutation_freqs2.py
The matrix obtained by this script is the number of mutations in the gene.

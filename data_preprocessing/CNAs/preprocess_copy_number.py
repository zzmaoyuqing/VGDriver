import pandas as pd
import numpy as np
import os


special_cancer_type = False  # special_cancer or pan_cancer

cnv_base_dir = 'gene_summary/'

all_matrices = []
ctypes = []
slash = '/'

for file in os.listdir(cnv_base_dir):
    if file.endswith('gene_summary.txt'):
        ctype = file.split('_')[0]
        ctypes.append(ctype)
        gene_CNV_df = pd.read_csv(os.path.join(cnv_base_dir, file), sep='\t', header=0)
        # save columns of 'total' and 'MutatedSamples'
        interested_columns = ['total', 'AlteredSamples']
        gene_CNV_matrix = gene_CNV_df[interested_columns]
        gene_CNV_matrix.index = gene_CNV_df['Hugo_Symbol']
        all_matrices.append(gene_CNV_matrix)



whole_gene_summary_matrix = pd.concat([all_matrices[i] for i in range(len(all_matrices))], axis=1)
whole_gene_summary_matrix = whole_gene_summary_matrix.fillna(0)
odd_columns = whole_gene_summary_matrix.iloc[:, ::2]
even_columns = whole_gene_summary_matrix.iloc[:, 1::2]


if special_cancer_type:
    odd_columns.columns = ['CNAC: ' + ctypes]
    even_columns.columns = ['CNAS: ' + ctypes]
    pancancer_gene_CNACS_matrix = pd.concat([odd_columns, even_columns], axis=1)
    pancancer_gene_CNACS_matrix.to_csv('' + 'CNAC_CNAS_' + ctypes[0] + '_CNACS_matrix.tsv', sep='\t')
else:
    CANCER_TYPES = ctypes
    CANCER_TYPES_MC = ['CNAC: BLCA', 'CNAC: BRCA', 'CNAC: CESC', 'CNAC: COAD', 'CNAC: ESCA', 'CNAC: HNSC', 'CNAC: KIRC', 'CNAC: KIRP', 'CNAC: LIHC', 'CNAC: LUAD', 'CNAC: LUSC', 'CNAC: PRAD', 'CNAC: READ', 'CNAC: STAD',
     'CNAC: THCA', 'CNAC: UCEC']
    CANCER_TYPES_MS = ['CNAS: BLCA', 'CNAS: BRCA', 'CNAS: CESC', 'CNAS: COAD', 'CNAS: ESCA', 'CNAS: HNSC', 'CNAS: KIRC', 'CNAS: KIRP', 'CNAS: LIHC', 'CNAS: LUAD', 'CNAS: LUSC', 'CNAS: PRAD', 'CNAS: READ', 'CNAS: STAD',
     'CNAS: THCA', 'CNAS: UCEC']

    odd_columns.columns = CANCER_TYPES_MC
    even_columns.columns = CANCER_TYPES_MS
    pancancer_gene_CNACS_matrix = pd.concat([odd_columns, even_columns], axis=1)
    pancancer_gene_CNACS_matrix.to_csv(cnv_base_dir + 'pancancer_gene_CNACS_matrix.tsv', sep='\t')
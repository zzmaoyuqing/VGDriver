import pandas as pd
import numpy as np
import os


special_cancer_type = False  # special_cancer or pan_cancer

snv_base_dir = '../../../source_processed_data/pancancer/TCGA/mutation/SNV/'     # the path of the output files of preprocess_manifest_SNVs.R script

all_matrices = []
ctypes = []
slash = '/'
# cname: cancer name              'maf_analysis_BRCA'

for cname in os.listdir(snv_base_dir):
    cancer_name = cname
    cname_ = cname+slash

    ctype_dir = os.path.join(snv_base_dir, cname_)
    if os.path.isdir(ctype_dir):
        ctype = cname.split('_')[2]
        ctypes.append(ctype)
        for file in os.listdir(ctype_dir):
            if file.endswith('geneSummary.txt'):
                file_dir = os.path.join(ctype_dir, file)
                gene_SNV_df = pd.read_csv(file_dir, sep='\t', header=0)
                interested_columns = ['total', 'MutatedSamples']
                gene_SNV_matrix = gene_SNV_df[interested_columns]
                gene_SNV_matrix.index = gene_SNV_df['Hugo_Symbol']
                all_matrices.append(gene_SNV_matrix)


whole_gene_summary_matrix = pd.concat([all_matrices[i] for i in range(len(all_matrices))], axis=1)
whole_gene_summary_matrix = whole_gene_summary_matrix.fillna(0)
odd_columns = whole_gene_summary_matrix.iloc[:, ::2]
even_columns = whole_gene_summary_matrix.iloc[:, 1::2]


if special_cancer_type:
    odd_columns.columns = ['MC: ' + ctypes]
    even_columns.columns = ['MS: ' + ctypes]
    pancancer_gene_mutation_matrix = pd.concat([odd_columns, even_columns], axis=1)
    pancancer_gene_mutation_matrix.to_csv('' + 'MC_MS_' + ctypes[0] + '_mutation_matrix.tsv', sep='\t')
else:
    CANCER_TYPES = ctypes
    CANCER_TYPES_MC = ['MC: BLCA', 'MC: BRCA', 'MC: CESC', 'MC: COAD', 'MC: ESCA', 'MC: HNSC', 'MC: KIRC', 'MC: KIRP', 'MC: LIHC', 'MC: LUAD', 'MC: LUSC', 'MC: PRAD', 'MC: READ', 'MC: STAD',
     'MC: THCA', 'MC: UCEC']
    CANCER_TYPES_MS = ['MS: BLCA', 'MS: BRCA', 'MS: CESC', 'MS: COAD', 'MS: ESCA', 'MS: HNSC', 'MS: KIRC', 'MS: KIRP', 'MS: LIHC', 'MS: LUAD', 'MS: LUSC', 'MS: PRAD', 'MS: READ', 'MS: STAD',
     'MS: THCA', 'MS: UCEC']

    odd_columns.columns = CANCER_TYPES_MC
    even_columns.columns = CANCER_TYPES_MS
    pancancer_gene_mutation_matrix = pd.concat([odd_columns, even_columns], axis=1)
    pancancer_gene_mutation_matrix.to_csv(snv_base_dir + 'pancancer_gene_mutation_matrix.tsv', sep='\t')
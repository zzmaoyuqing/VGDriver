import pandas as pd
import os
import numpy as np


# set input file path and output file path
express_base_dir = '../../../source_processed_data/pancancer/TCGA/gene_expression/'   # the path of the output files of preprocess_manifest_express.R script
out_file_name_pan = 'pancancer_mean_expression_fc_matrix.tsv'
out_file_name_sp = '_mean_expression_fc_matrix.tsv'

special_cancer_type = False                # Whether it is a specific cancer type
USE_PATIENT_NORMAL_IF_AVAILABLE = True    # Whether it is matched tumor and normal files for patients

def normalize_sample(col, normal):
    patients_with_normal = normal.columns.map(get_patient_from_barcode)
    if get_patient_from_barcode(col.name) in patients_with_normal:
        idx_col = patients_with_normal == get_patient_from_barcode(col.name)
        corresponding_patient_normal = normal.iloc[:, idx_col]
        # If you don't use “remove genes with 0” for expression data in R, it's better to add 1 to the denominator to prevent dividing by 0,
        # because in the R script I filtered the genes by removing all genes with 0, so I didn't add 1 to the denominator.
        fc = col / (corresponding_patient_normal.median(axis=1))
    else:
        print("normalizing using average normal expression")
        fc = col / normal.median(axis=1)
    return fc


# function to get the fold changes
def compute_geneexpression_foldchange(tumor_path, normal_path):
    """Parameters:
    tumor_path:            Path to Tumor Sample Data
    normal_path:           Path to Normal Sample Data
    """
    # read tumor and normal data
    tumor_ge = pd.read_csv(tumor_path, index_col='symbol', compression='gzip', sep='\t',)
    normal_ge = pd.read_csv(normal_path, index_col='symbol', compression='gzip', sep='\t')

    assert (np.all(tumor_ge.index == normal_ge.index))
    print(tumor_ge.shape, normal_ge.shape)

    # compute mean expression for tumor and normal. Then, compute log
    if USE_PATIENT_NORMAL_IF_AVAILABLE:
        original_tumor_cols = tumor_ge.columns
        patients_with_normal = normal_ge.columns.map(get_patient_from_barcode)
        tumor_ge.columns = tumor_ge.columns.map(get_patient_from_barcode)
        # now, get only tumor samples for patients that have normals, too
        tumor_normal_matched = tumor_ge.loc[:, tumor_ge.columns.isin(patients_with_normal)]
        fc = tumor_normal_matched.apply(lambda col: normalize_sample(col, normal_ge), axis=0)
        fc.columns = original_tumor_cols[tumor_ge.columns.isin(patients_with_normal)]
        print(fc.shape)

    else:
        fc = tumor_ge.divide(normal_ge.median(axis=1), axis=0)
        print(fc.shape)

    log_fc = np.log2(fc)
    log_fc = log_fc.replace([np.inf, -np.inf], np.nan).dropna(axis=0)  # remove NaN and inf (from division by 0 or 0+eta)
    print("Dropped {} genes because they contained NaNs".format(fc.shape[0] - log_fc.shape[0]))
    return log_fc, tumor_ge, normal_ge


# load tumor and normal files for cancer
tumor_files = []
normal_files = []
for file in os.listdir(express_base_dir):
    if file.endswith('TPM_tumor.txt.gz'):
        tumor_files.append(file)
    elif file.endswith('TPM_normal.txt.gz'):
        normal_files.append(file)


ctypes = []
log_fold_changes = []
mean_fold_changes = []
matched_samples = []
get_patient_from_barcode = lambda x: '-'.join(str(x).split('.')[:3])

for i in range(len(tumor_files)):
    tumor_file = tumor_files[i]
    normal_file = normal_files[i]
    ctype = tumor_file.split('_')[1]

    if special_cancer_type:
        # interested specific cancer types
        if ctype == "BRCA" or ctype == "KIRC" or ctype == "LIHC" or ctype == "LUAD" or ctype == "PRAD" or ctype == "STAD":
            ctypes.append(ctype)

            log_fc, tumor, normal = compute_geneexpression_foldchange(tumor_path=os.path.join(express_base_dir, tumor_file),
                                                                      normal_path=os.path.join(express_base_dir, normal_file))
            print(log_fc.shape)
            log_fc = log_fc.add_prefix(ctype + ':')
            if log_fc.columns[0].__contains__('.'):
                log_fc.columns = log_fc.columns.str.replace('.', '-')
            matched_samples.append(log_fc.columns.values)
            log_fold_changes.append(log_fc)
            mean_fold_changes.append(log_fc.median(axis=1))
    else:
        ctypes.append(ctype)

        log_fc, tumor, normal = compute_geneexpression_foldchange(tumor_path=os.path.join(express_base_dir, tumor_file),
                                                                  normal_path=os.path.join(express_base_dir, normal_file))
        print(log_fc.shape)

        log_fc = log_fc.add_prefix(ctype + ':')
        if log_fc.columns[0].__contains__('.'):
            log_fc.columns = log_fc.columns.str.replace('.', '-')
        matched_samples.append(log_fc.columns.values)
        log_fold_changes.append(log_fc)
        mean_fold_changes.append(log_fc.median(axis=1))


mean_fold_changes_matrix = pd.DataFrame(mean_fold_changes, index=[ctypes[i] for i in range(len(ctypes))]).T
mean_fold_changes_matrix.fillna(0, inplace=True)

print("Paired Samples with gene expression per Cancer Type:")
for i in range(len(mean_fold_changes)):
    print("{}: {} Samples".format(ctypes[i], log_fold_changes[i].shape))

if special_cancer_type:
    for i in range(len(ctypes)):
        mean_fold_changes_matrix[ctypes[i]].columns = ['GE: ' + ctypes[i]]
        mean_fold_changes_matrix[ctypes[i]].to_csv('GE_' + ctypes[i] + out_file_name_sp, sep='\t')
else:
    CANCER_TYPES = ctypes
    CANCER_TYPES_MF = ['GE: BLCA', 'GE: BRCA', 'GE: CESC', 'GE: COAD', 'GE: ESCA', 'GE: HNSC', 'GE: KIRC', 'GE: KIRP', 'GE: LIHC', 'GE: LUAD', 'GE: LUSC', 'GE: PRAD', 'GE: READ', 'GE: STAD',
     'GE: THCA', 'GE: UCEC']
    mean_fold_changes_matrix.columns = CANCER_TYPES_MF
    mean_fold_changes_matrix.to_csv(out_file_name_pan, sep='\t')

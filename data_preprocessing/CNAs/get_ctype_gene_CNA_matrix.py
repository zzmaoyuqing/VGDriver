import pandas as pd
import os

"""Get ctype gene CNA files
    1. download the manifest files of specific cancer from TCGA respectively
    2. use gdc-client tool to download sample files of each cancer to cnv_base_dir
"""

# set input file path and output file path
cnv_base_dir = '../../../source_DATABASE/pancancer/TCGA/mutation/CNA/'
out_dir = '../../../source_processed_data/pancancer/TCGA/mutation/CNA/'


ctypes = []  # Record the number of cancers in the snv_base_dir path.
slash = '/'
# cname: cancer name              'BLCA'
# ctype_dir: cancer的dir           '../../../source_DATABASE/pancancer/TCGA/mutation/CNA/BLCA/'

# sname: sample name               '0096dec0-0348-427b-a27b-ebc6f6352d28'
# csdir_2 :  cancer+sample的dir     '../../../source_DATABASE/pancancer/TCGA/mutation/CNA/BLCA/0096dec0-0348-427b-a27b-ebc6f6352d28'


for cname in os.listdir(cnv_base_dir):
    cancer_name = cname
    cname_ = cname+slash
    ctype_dir = os.path.join(cnv_base_dir, cname_)  # len(os.listdir(ctype_dir)) is the number of samples in the current cancer.

    # Obtain all samples for specific cancer
    ctype_gene_sample_matrix = []
    # Get all the genes contained in all the samples of specific cancer
    ctype_segfile = []
    # Iterate over all samples in each ctype_dir
    for sname in os.listdir(ctype_dir):
        cs_dir_2 = os.path.join(ctype_dir, sname)
        if os.path.isdir(cs_dir_2):
            # Read the seg file of the current sample
            for segfile in os.listdir(cs_dir_2):
                if segfile.endswith('ch38.seg.v2.txt'):
                    seg_dir_3 = os.path.join(cs_dir_2, segfile)
                    seg = pd.read_csv(seg_dir_3, sep='\t', comment='#', header=0)
                    # Extract each seg sample file in specific cancer append to ctype_segfile, each cancer is 6 columns of CNV information.
                    ctype_segfile.append(seg)

    print("{} has {} samples from TCGA".format(cancer_name, len(ctype_segfile)))  # Processed BLCA with 814 samples
    # Get specific cancer all seg sample files in (gene X 6) concat into (all gene X 6) BlCA:(89568,6)
    ctype_segfile_df = pd.concat([ctype_segfile[i] for i in range(len(ctype_segfile))], axis=0)

    # index=false No row names, otherwise the output is 7 columns
    ctype_segfile_df.to_csv(out_dir + cancer_name + '_gene_CNV_matrix.tsv', sep='\t', index=False)
    ctypes.append(cancer_name)

print(ctypes)
print('number of cancer: ', len(ctypes))

# # You can test this with the following code to see if the output shape is correct
# test_path = out_dir + 'BLCA_gene_SNV_matrix.tsv.gz'
# maf = pd.read_csv(test_path, compression='gzip', sep='\t', comment='#', header=0)
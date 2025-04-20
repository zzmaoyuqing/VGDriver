## input files
CNV_file_dir <- "BRCA_gene_CNV_matrix.tsv"
cart_file_dir <- "metadata.cart.CNVs.BRCA.2024-06-01.json"

## out_file
out_file_path <- "BRCA_MaskedCopyNumberSegment.txt"


cnv_df <- read.csv(CNV_file_dir, sep = '\t')
head(cnv_df)

metadata <- jsonlite::fromJSON(cart_file_dir)
library(dplyr)
metadata_id <- metadata %>% 
  dplyr::select(c(file_name,associated_entities)) 

meta_df <- do.call(rbind,metadata$associated_entities)
head(metadata_id,2)


cnv_df$Sample <- meta_df$entity_submitter_id[match(cnv_df$GDC_Aliquot,meta_df$entity_id)]
length(unique(cnv_df$GDC_Aliquot))
length(unique(cnv_df$Sample))
head(cnv_df)

cnv_df$Sample <- substring(cnv_df$Sample,1,16)
head(cnv_df)


cnv_df <- cnv_df[,c('Sample','Chromosome','Start','End','Num_Probes','Segment_Mean')]
head(cnv_df)

cnv_df$Chromosome[cnv_df$Chromosome == "X"] <- 23
dim(cnv_df)
# [1] 255065      6
cnv_df <- cnv_df[grep("01A$",cnv_df$Sample),]
dim(cnv_df)
# [1] 146694      6


write.table(cnv_df,out_file_path,sep="\t",
            quote = F,col.names = F,row.names = F)








### The following codes only need to be run once. Different cancers use the same marker file.
# input file和output file dir。
hg38_marker_file_dir <- "snp6.na35.remap.hg38.subset.txt.gz"
out_file_dir <- "hg38_marker_file.txt"


hg38_marker_file <- read.delim(hg38_marker_file_dir)
table(hg38_marker_file$freqcnv)

# “If you are using Masked Copy Number Segment for GISTIC analysis, please only keep probesets with freqcnv =FALSE”
hg_marker_file <- hg38_marker_file[hg38_marker_file$freqcnv=="FALSE",]
hg_marker_file <- hg_marker_file [,c(1,2,3)]
colnames(hg_marker_file) = c("Marker Name","Chromosome","Marker Position")
head(hg_marker_file)

# save maker file
write.table(hg_marker_file, out_file_dir,sep = "\t",col.names = T,row.names = F)

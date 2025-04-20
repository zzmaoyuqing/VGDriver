library(maftools)
library(dplyr)

gdcfliename <- "BRCA"
path1 <- "BRCA/"
outfilenamegz = "BRCA_gene_SNV_matrix.tsv.gz"
mafbasename <- './maf_analysis_BRCA/input'


count_file <- list.files(gdcfliename,pattern = '*maf.gz$',recursive = TRUE)
count_file_name <- strsplit(count_file,split='/')
count_file_name <- sapply(count_file_name,function(x){x[2]})
mut_file <- data.frame()
for (i in 1:length(count_file_name)){
  path = paste0(path1,count_file[i])
  mut<- read.delim(path, skip = 7, header = T, fill = TRUE, sep = '\t')
  mut_file <- rbind(mut_file, mut)
}
#> dim(mut_file)
#[1] 89568   140 BRCA

# save matrix
write.table(mut_file,file=gzfile(outfilenamegz),row.names = F,quote = F,sep = "\t")

mut_file$Tumor_Sample_Barcode = substr(mut_file$Tumor_Sample_Barcode, 1,12)
all_mut <- read.maf(mut_file)
write.mafSummary(maf = all_mut, basename = mafbasename)


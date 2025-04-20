rm(list = ls())
library(rjson)
library(limma)

metafile="metadata.cart.express.BRCA.2024-06-03.json"
gdcfliename="BRCA"
path1="BRCA/"
outfilenamegz_tumor = "TCGA_BRCA_TPM_tumor.txt.gz"
outfilenamegz_normal = "TCGA_BRCA_TPM_normal.txt.gz"

json = jsonlite::fromJSON(metafile)
id = json$associated_entities[[1]][,1]
sample_id = sapply(json$associated_entities,function(x){x[,1]})
file_sample = data.frame(sample_id,file_name=json$file_name)  

count_file <- list.files(gdcfliename,pattern = '*gene_counts.tsv$',recursive = TRUE)
count_file_name <- strsplit(count_file,split='/')
count_file_name <- sapply(count_file_name,function(x){x[2]})
matrix = data.frame(matrix(nrow=60660,ncol=0))
for (i in 1:length(count_file_name)){
  path = paste0(path1,count_file[i])
  data<- read.delim(path,fill = TRUE,header = FALSE,row.names = 1)
  colnames(data)<-data[2,]
  data <-data[-c(1:6),]
  data <- data[6]
  colnames(data) <- file_sample$sample_id[which(file_sample$file_name==count_file_name[i])]
  matrix <- cbind(matrix,data)
}



sample1 = paste0(path1,count_file[1])  
names=read.delim(sample1,fill = TRUE,header = FALSE,row.names = 1)
colnames(names)<-names[2,]
names <-names[-c(1:6),]
names = names[,1:2]
same=intersect(rownames(matrix),rownames(names))
matrix=matrix[same,]
names=names[same,]
matrix$symbol=names[,1]
matrix=matrix[,c(ncol(matrix),1:(ncol(matrix)-1))]
# Remove duplicate genes from the symbol column and keep the maximum expression result for each gene
matrix0 <- aggregate( . ~ symbol,data=matrix, max)  

matrix0_numeric <- as.data.frame(lapply(matrix0,as.numeric))
matrix0_numeric['symbol'] <- matrix0[,1]

# Removal of 0-containing genes
matrix1 <- matrix0_numeric[!as.logical(rowSums(matrix0_numeric==0)), ]          


#Divided into normal and tumor matrices
sample <- colnames(matrix1)
normal <- c()
tumor <- c()
for (i in 1:length(sample)){
  if((substring(colnames(matrix1)[i],14,15)>10)){
    normal <- append(normal,sample[i])
  } else {
    tumor <- append(tumor,sample[i])
  }
}
tumor_matrix <- matrix1[,tumor]
normal_matrix <- matrix1[,normal]
normal_matrix$symbol=matrix1[,1]
normal_matrix=normal_matrix[,c(ncol(normal_matrix),1:(ncol(normal_matrix)-1))]

# save tumor and normal matrices
write.table(tumor_matrix,file=gzfile(outfilenamegz_tumor),row.names = F,quote = F,sep = "\t")
write.table(normal_matrix,file=gzfile(outfilenamegz_normal),row.names = F,quote = F,sep = "\t") 

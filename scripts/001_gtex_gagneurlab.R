#!/usr/bin/env Rscript

library(data.table)
library(DESeq2)

#source_dir = path.expand("/s/project/gtex_genetic_diagnosis/processed_data/v29/counts")
#target_dir = path.expand("~/tmp/counts/")

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("Usage: Rscript <this_script> <source_dir> <target_dir>", call.=FALSE)
}

source_dir = normalizePath(args[1])
target_dir = normalizePath(args[2])

print(paste0("source_dir: ", source_dir))
print(paste0("target_dir: ", target_dir))

files = file.path(source_dir, list.files(source_dir, pattern="total_counts.Rds", recursive = T, ignore.case = T))

print("count files:")
print(files)

# create list of per-tissue SummarizedExperiments
data = lapply(files, function (x) {
  print(x)
  total_counts = readRDS(x)

  # remove samples with all-zero counts
  total_counts = total_counts[, colSums(assay(total_counts) != 0) > 0]

  # calculate size factors
  #total_counts = estimateSizeFactors(DESeqDataSet(total_counts, design = ~1), type="poscounts")
  colData(total_counts)$sf = estimateSizeFactorsForMatrix(assay(total_counts))
  
  total_counts
})

observations = rbindlist(lapply(data, function(x) as.data.table(colData(x))))
features = as.data.table(rowData(data[[1]]))
counts = rbindlist(lapply(data, function(x) as.data.table(t(assay(x)))))

fwrite(observations, file.path(target_dir, "observations.csv"))
fwrite(features, file.path(target_dir, "features.csv"))
fwrite(counts, file.path(target_dir, "counts.csv"))


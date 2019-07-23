#!/usr/bin/env Rscript

library(data.table)
library(OUTRIDER)
library(stringr)

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

files = file.path(source_dir, list.files(source_dir, pattern="*.Rds", recursive = T, ignore.case = T))
files = files[! endsWith(files, "Kremer_ODS.RDS")]
files = files[! str_detect(basename(files), regex("simulation.*.Rds", ignore_case=T))]


print("OUTRIDER files:")
print(files)

# create list of per-tissue SummarizedExperiments
data = lapply(files, function (x) {
  print(x)

  ods = readRDS(x)
  ods
})

observations = rbindlist(
  lapply(data, function(x) as.data.table(colData(x))),
  use.names=TRUE
)
fwrite(observations, file.path(target_dir, "observations.csv"))
rm(observations)

counts = rbindlist(
  lapply(data, function(x) as.data.table(t(assays(x)$counts))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(counts, file.path(target_dir, "counts.csv"))

l2fc = rbindlist(
  lapply(data, function(x) as.data.table(t(assays(x)$l2fc))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(l2fc, file.path(target_dir, "l2fc.csv"))

for (ods in data){
  w_encoder = metadata(ods)[["E"]]
  w_decoder = metadata(ods)[["D"]]
  fwrite(w_encoder, file.path(target_dir, paste0(colData(ods)$SMTSD[[1]], ".w_encoder.csv"))
  fwrite(w_decoder, file.path(target_dir, paste0(colData(ods)$SMTSD[[1]], ".w_decoder.csv"))
}



gene_ids = colnames(counts)
rm(counts)

features = rbindlist(
  lapply(
    data,
    function(x) {
      f = as.data.table(rowData(x))
      f$gene_id = rownames(x)

      f[, c("gene_id", "gene_symbol", "basepairs")]
    }
  ),
  use.names=TRUE
)
setkey(features, "gene_id")
features = unique(features)
features = features[gene_ids,]

fwrite(features, file.path(target_dir, "features.csv"))
rm(features)


#!/usr/bin/env Rscript

library(data.table)
library(OUTRIDER)
library(stringr)

rep.row<-function(x,n){
   matrix(rep(x,each=n),nrow=n)
}

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
  lapply(data, function(ods) as.data.table(colData(ods))),
  use.names=TRUE
)
fwrite(observations, file.path(target_dir, "observations.csv"))
rm(observations)

counts = rbindlist(
  lapply(data, function(ods) as.data.table(t(assays(ods)$counts))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(counts, file.path(target_dir, "counts.csv"))

l2fc = rbindlist(
  lapply(data, function(ods) as.data.table(t(assays(ods)$l2fc))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(l2fc, file.path(target_dir, "l2fc.csv"))

mu = rbindlist(
  lapply(data, function(ods) as.data.table(t(normalizationFactors(ods)))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(mu, file.path(target_dir, "mu.csv"))

theta = rbindlist(
  lapply(data, function(ods) {
    theta = rep.row(mcols(ods)$theta, length(colnames(ods)))
    colnames(theta) = rownames(ods)
    theta = as.data.table(theta)
  }),
  use.names=TRUE,
  fill=TRUE
)
fwrite(theta, file.path(target_dir, "theta.csv"))

pval = rbindlist(
  lapply(data, function(ods) as.data.table(t(pValue(ods)))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(pval, file.path(target_dir, "pval.csv"))

padj = rbindlist(
  lapply(data, function(ods) as.data.table(t(padj(ods)))),
  use.names=TRUE,
  fill=TRUE
)
fwrite(padj, file.path(target_dir, "padj.csv"))


gene_ids = colnames(counts)
rm(counts)

features = rbindlist(
  lapply(
    data,
    function(ods) {
      f = as.data.table(rowData(ods))
      f$gene_id = rownames(ods)

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


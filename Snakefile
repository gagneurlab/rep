workdir: "data"

configfile: "config.yaml"

include: "snakemk/001a_gtex_gagneurlab.smk"
include: "snakemk/001b_gtex_OUTRIDER.smk"
# include: "snakemk/002_basenji.smk"

rule all:
    input:
#         "processed/gtex/gagneurlab/raw_counts.h5ad",
#         "processed/gtex/OUTRIDER/counts.h5ad",
#         "processed/gtex/OUTRIDER/l2fc.h5ad",
#         "processed/gtex/OUTRIDER/mu.h5ad",
#         "processed/gtex/OUTRIDER/theta.h5ad",
#         "processed/gtex/OUTRIDER/pval.h5ad",
#         "processed/gtex/OUTRIDER/padj.h5ad",
         "processed/gtex/OUTRIDER/gtex_unstacked.zarr",
         "processed/gtex/OUTRIDER_intronic/gtex_unstacked.zarr",
         # basenji
         #"processed/basenji/GRCh38/gtex",
         #"processed/basenji/GRCh38/reference",

workdir: "data"

configfile: "config.yaml"

include: "snakemk/001a_gtex_gagneurlab.smk"
include: "snakemk/001b_gtex_OUTRIDER.smk"


rule all:
    input:
         "processed/gtex/gagneurlab/raw_counts.h5ad",
         "processed/gtex/OUTRIDER/counts.h5ad",
         "processed/gtex/OUTRIDER/l2fc.h5ad",

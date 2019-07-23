import os
import numpy as np
import pandas as pd

SNAKEMAKE_DIR = os.path.dirname(workflow.snakefile)

RAW_DATA_DIR = "raw/gtex/gagneurlab"
CACHE_DATA_DIR = "cache/gtex/gagneurlab"
PROCESSED_DATA_DIR = "processed/gtex/gagneurlab"


rule _001a_gtex_gagneurlab:
    input:
        RAW_DATA_DIR
    output:
        os.path.join(CACHE_DATA_DIR, "observations.csv"),
        os.path.join(CACHE_DATA_DIR, "features.csv"),
        os.path.join(CACHE_DATA_DIR, "counts.csv"),
    params:
        output_dir=CACHE_DATA_DIR
    shell:
        "Rscript %s/scripts/001_gtex_gagneurlab.R {input} {params.output_dir}" % SNAKEMAKE_DIR

rule _001a_convert_to_h5ad:
    input:
        observations=os.path.join(CACHE_DATA_DIR, "observations.csv"),
        features=os.path.join(CACHE_DATA_DIR, "features.csv"),
        counts=os.path.join(CACHE_DATA_DIR, "counts.csv"),
    output:
        os.path.join(PROCESSED_DATA_DIR, "raw_counts.h5ad")
    run:
        import anndata as ad
        import pandas as pd

        print("Reading data...")
        observations = pd.read_csv(input.observations)
        observations.set_index("Sample_Name", inplace=True)
        features = pd.read_csv(input.features)
        features.set_index("gene_id", inplace=True)
        counts = np.asarray(pd.read_csv(input.counts))

        adata = ad.AnnData(counts, obs=observations, var=features)
        
        # add individual-info
        individualIDs = []
        for sampleId in adata.obs.index.values:
            splits = sampleId.split("-")
            individualId = splits[0] + "-" + splits[1]
            individualIDs.append(individualId)
        adata.obs["individual"] = individualIDs

        # add means
        adata.var["feature_mean"] = np.nanmean(adata.X, axis=0)

        print("writing data to '%s'..." % output[0])
        adata.write_h5ad(output[0])


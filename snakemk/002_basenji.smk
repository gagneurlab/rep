import os
import numpy as np
import pandas as pd

SNAKEMAKE_DIR = os.path.dirname(workflow.snakefile)

RAW_DATA_DIR = "raw/basenji"
CACHE_DATA_DIR = "cache/basenji"
PROCESSED_DATA_DIR = "processed/basenji"

BASENJI_GTEX = os.path.join(RAW_DATA_DIR, "gtex/")
BASENJI_REF = os.path.join(RAW_DATA_DIR, "ref_GRCh37/")

def convert_to_h5ad(observations_file, features_file, values_file, output):
    import anndata as ad
    import pandas as pd

    print("Reading data...")
    observations = pd.read_csv(observations_file)
    # rename some columns
    observations.rename(
        columns=dict(
            SAMPID="Sample_Name",
            subjectID="individual",
            SMTS="tissue",
            SMTSD="subtissue",
        ),
        inplace=True
    )
    observations.set_index("Sample_Name", inplace=True)

    features = pd.read_csv(features_file)
    features.set_index("gene_id", inplace=True)

    values = np.asarray(pd.read_csv(values_file))

    adata = ad.AnnData(
        values,
        obs=observations,
        var=features
    )

    print("writing data to '%s'..." % output)
    adata.write_h5ad(output)


rule _001b_gtex_OUTRIDER:
    input:
         RAW_DATA_DIR
    output:
          os.path.join(CACHE_DATA_DIR, "observations.csv"),
          os.path.join(CACHE_DATA_DIR, "features.csv"),
          os.path.join(CACHE_DATA_DIR, "counts.csv"),
          os.path.join(CACHE_DATA_DIR, "l2fc.csv"),
          os.path.join(CACHE_DATA_DIR, "mu.csv"),
          os.path.join(CACHE_DATA_DIR, "theta.csv"),
          os.path.join(CACHE_DATA_DIR, "pval.csv"),
          os.path.join(CACHE_DATA_DIR, "padj.csv"),
    params:
          output_dir=CACHE_DATA_DIR
    shell:
         "Rscript %s/scripts/001_gtex_OUTRIDER.R {input} {params.output_dir}" % SNAKEMAKE_DIR

rule _001b_convert_to_h5ad:
    input:
         observations=os.path.join(CACHE_DATA_DIR, "observations.csv"),
         features=os.path.join(CACHE_DATA_DIR, "features.csv"),
         target=os.path.join(CACHE_DATA_DIR, "{target}.csv"),
         #l2fc = os.path.join(CACHE_DATA_DIR, "l2fc.csv"),
         #mu = os.path.join(CACHE_DATA_DIR, "mu.csv"),
         #theta = os.path.join(CACHE_DATA_DIR, "theta.csv"),
    output:
          os.path.join(PROCESSED_DATA_DIR, "{target}.h5ad")
          #os.path.join(PROCESSED_DATA_DIR, "l2fc.h5ad")
          #os.path.join(PROCESSED_DATA_DIR, "mu.h5ad")
          #os.path.join(PROCESSED_DATA_DIR, "theta.h5ad")
    #     wildcard_constraints:
    #         target="counts|l2fc"
    run:
        convert_to_h5ad(
            input.observations,
            input.features,
            input.target,
            output[0]
        )

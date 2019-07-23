import os
import numpy as np
import pandas as pd

SNAKEMAKE_DIR = os.path.dirname(workflow.snakefile)

RAW_DATA_DIR = "raw/gtex/OUTRIDER"
CACHE_DATA_DIR = "cache/gtex/OUTRIDER"
PROCESSED_DATA_DIR = "processed/gtex/OUTRIDER"

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

    # add means
    adata.var["feature_mean"] = np.nanmean(adata.X, axis=0)

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
        os.path.join(CACHE_DATA_DIR, "tissues.csv"),
    params:
        output_dir=CACHE_DATA_DIR
    shell:
        "Rscript %s/scripts/001_gtex_OUTRIDER.R {input} {params.output_dir}" % SNAKEMAKE_DIR


rule _001b_convert_to_h5ad:
    input:
        observations=os.path.join(CACHE_DATA_DIR, "observations.csv"),
        features=os.path.join(CACHE_DATA_DIR, "features.csv"),
        counts=os.path.join(CACHE_DATA_DIR, "{target}.csv"),
    output:
        os.path.join(PROCESSED_DATA_DIR, "{target}.h5ad")
#     wildcard_constraints:
#         target="counts|l2fc"
    run:
        convert_to_h5ad(
            input.observations,
            input.features,
            input.counts,
            output[0]
        )

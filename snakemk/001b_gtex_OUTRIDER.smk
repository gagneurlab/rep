import os
import numpy as np
import pandas as pd
import dask.dataframe as ddf
import dask.array as da
import xarray as xr

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

    print("writing data to '%s'..." % output)
    adata.write_h5ad(output)


def convert_to_xarray(observations_file, features_file, values_file, output):
    import anndata as ad
    import pandas as pd

    print("Reading data...")
    observations = ddf.read_csv(observations_file)
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

    features = ddf.read_csv(features_file)
    features.set_index("gene_id", inplace=True)

    values = xr.DataArray(ddf.read_csv(values_file, sample=2000000).to_dask_array())

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

rule _001b_convert_to_xarray:
    input:
         observations=os.path.join(CACHE_DATA_DIR, "observations.csv"),
         features=os.path.join(CACHE_DATA_DIR, "features.csv"),
         counts=os.path.join(CACHE_DATA_DIR, "counts.csv"),
         l2fc = os.path.join(CACHE_DATA_DIR, "l2fc.csv"),
         mu = os.path.join(CACHE_DATA_DIR, "mu.csv"),
         theta = os.path.join(CACHE_DATA_DIR, "theta.csv"),
    output:
          directory(os.path.join(PROCESSED_DATA_DIR, "gtex.zarr"))
    #     wildcard_constraints:
    #         target="counts|l2fc"
    run:
        print("Reading data...")

        observations = pd.read_csv(input.observations)
        # rename some columns
        observations.rename(
            columns=dict(
                SAMPID="observations",
                subjectID="individual",
                SMTS="tissue",
                SMTSD="subtissue",
            ),
            inplace=True
        )

        features = pd.read_csv(input.features)
        # rename some columns
        features.rename(
            columns=dict(
                gene_id="genes",
            ),
            inplace=True
        )

        # counts = np.asarray(pd.read_csv(input.counts))
        # l2fc = np.asarray(pd.read_csv(input.l2fc))
        # mu = np.asarray(pd.read_csv(input.mu))
        # theta = np.asarray(pd.read_csv(input.theta))
        counts = ddf.read_csv(input.counts, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        l2fc = ddf.read_csv(input.l2fc, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        mu = ddf.read_csv(input.mu, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        theta = ddf.read_csv(input.theta, sample=2000000, dtype="float32").to_dask_array(lengths=True)

        xrds = xr.Dataset(
            {
                "counts": (["observations", "genes"], counts),
                "l2fc": (["observations", "genes"], l2fc),
                "mu": (["observations", "genes"], mu),
                "theta": (["observations", "genes"], theta),
            },
            coords={
                **{k: (["genes",], v) for k, v in features.items()},
                **{k: (["observations",], v) for k, v in observations.items()},
            }
        )

        import rep.random as rnd
        import scipy.stats as scistats

        ppf_func = da.as_gufunc(signature="()->()", output_dtypes=float, vectorize=True)(scistats.norm.ppf)

        cdf = rnd.NegativeBinomial(mean=xrds.mu, r=xrds.theta).cdf(xrds.counts)
        ppf = xr.apply_ufunc(ppf_func, cdf, dask="allowed")

        xrds["cdf"] = cdf
        xrds["ppf"] = ppf

        xrds = xrds.chunk({"observations": 10, "genes": 1000})

        xrds.to_zarr(output[0])
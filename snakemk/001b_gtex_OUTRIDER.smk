import os
import numpy as np
import pandas as pd
import dask.dataframe as ddf
import dask.array as da
import xarray as xr

SNAKEMAKE_DIR = os.path.dirname(workflow.snakefile)

#RAW_DATA_DIR = "raw/gtex/OUTRIDER"
#CACHE_DATA_DIR = "cache/gtex/OUTRIDER"
#PROCESSED_DATA_DIR = "processed/gtex/OUTRIDER"
RAW_DATA_DIR = "raw"
CACHE_DATA_DIR = "cache"
PROCESSED_DATA_DIR = "processed"


rule _001b_gtex_OUTRIDER:
    input:
         os.path.join(RAW_DATA_DIR, "gtex/{count_type}")
    output:
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "observations.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "features.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "counts.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "l2fc.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "mu.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "theta.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "log_cdf.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "zscore.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "pval.parquet"),
          os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "padj.parquet"),
    params:
          output_dir=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}"),
          alpha_cutoff=0.05
    shell:
         "Rscript %s/scripts/001_gtex_OUTRIDER.R {input} {params.output_dir} {params.alpha_cutoff}" % SNAKEMAKE_DIR


rule _001b_outrider_convert_to_xarray:
    input:
         observations=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "observations.parquet"),
         features=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "features.parquet"),
         counts=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "counts.parquet"),
         l2fc=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "l2fc.parquet"),
         mu=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "mu.parquet"),
         theta=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "theta.parquet"),
         log_cdf=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "log_cdf.parquet"),
         zscore=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "zscore.parquet"),
         pval=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "pval.parquet"),
         padj=os.path.join(CACHE_DATA_DIR, "gtex/{count_type}", "padj.parquet"),
    output:
          directory(os.path.join(PROCESSED_DATA_DIR, "gtex/{count_type}", "gtex_unstacked.zarr"))
          #     wildcard_constraints:
          #         target="counts|l2fc"
    run:
        print("Reading data...")

        observations = pd.read_parquet(input["observations"])
        ## rename some columns
        #observations = observations.rename(
        #    columns=dict(
        #        SAMPID="sample_id",
        #        # subjectID="individual",
        #        SMTS="tissue",
        #        SMTSD="subtissue",
        #    ),
        #)
        observations["individual"] = observations["Sample_Name"].str.split("-").apply(lambda s: "-".join(s[:2]))
        observations = observations.set_index(["subtissue", "individual"])

        features = pd.read_parquet(input["features"])
        # rename some columns
        features = features.rename(
            columns=dict(
                gene_id="gene",
            ),
        ).set_index("gene")

        # dense matrices
        counts = da.from_array(np.asarray(pd.read_parquet(input["counts"])), chunks=(1000, 1000))
        l2fc = da.from_array(np.asarray(pd.read_parquet(input["l2fc"])), chunks=(1000, 1000))
        mu = da.from_array(np.asarray(pd.read_parquet(input["mu"])), chunks=(1000, 1000))
        theta = da.from_array(np.asarray(pd.read_parquet(input["theta"])), chunks=(1000, 1000))
        log_cdf = da.from_array(np.asarray(pd.read_parquet(input["log_cdf"])), chunks=(1000, 1000))
        zscore = da.from_array(np.asarray(pd.read_parquet(input["zscore"])), chunks=(1000, 1000))
        pval = da.from_array(np.asarray(pd.read_parquet(input["pval"])), chunks=(1000, 1000))
        padj = da.from_array(np.asarray(pd.read_parquet(input["padj"])), chunks=(1000, 1000))

        print("reading finished")

        # counts = ddf.read_parquet(input.counts, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        # l2fc = ddf.read_parquet(input.l2fc, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        # mu = ddf.read_parquet(input.mu, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        # theta = ddf.read_parquet(input.theta, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        # cdf = ddf.read_parquet(input.cdf, sample=2000000, dtype="float32").to_dask_array(lengths=True)
        # zscore = ddf.read_parquet(input.zscore, sample=2000000, dtype="float32").to_dask_array(lengths=True)

        def cdf_to_categorical(cdf, alpha=0.05):
            return (cdf > (1 - alpha) / 2).astype("int8") - (cdf < alpha / 2).astype("int8")


        xrds = xr.Dataset(
            {
                "counts": (("observations", "gene"), counts),
                "log_cdf": (("observations", "gene"), log_cdf),
                "l2fc": (("observations", "gene"), l2fc),
                "zscore": (("observations", "gene"), zscore),
                "pval": (("observations", "gene"), pval),
                "padj": (("observations", "gene"), padj),
            },
            coords={
                "observations": observations.index,
                "gene": features.index,
            }
        )
        xrds["cdf"] = np.exp(xrds.log_cdf)

        # xrds = xrds.set_index(observations=["subtissue", "individual"])
        # xrds = xrds.chunk({})

        alpha = 0.05

        xrds = xrds.unstack("observations").chunk({"subtissue": 1})
        xrds = xrds.assign(**observations.to_xarray(), **features.to_xarray())
        xrds = xrds.chunk({"gene": None, "subtissue": 1, "individual": 10})

        xrds = xrds.transpose("subtissue", "individual", "gene")

        with np.errstate(invalid='ignore'):
            xrds = xrds.assign({
                "hilo": cdf_to_categorical(xrds.cdf, alpha=alpha),
                "hilo_padj": ((xrds.padj < alpha).astype("int8") * xr.where(np.exp(xrds.log_cdf) < 0.5, -1, 1)).astype("int8"),
                "missing": np.isnan(xrds.log_cdf),
            })

        print("writing zarr output...")

        xrds.to_zarr(
            output[0],
            encoding={
                'counts': dict(dtype='int32'),
                'hilo': dict(dtype='int8'),
                'hilo_padj': dict(dtype='int8'),
#                'SMCENTER': dict(dtype='str'),
#                'SMPTHNTS': dict(dtype='str'),
#                'tissue': dict(dtype='str'),
#                'SMUBRID': dict(dtype='str'),
#                'SMTSTPTREF': dict(dtype='str'),
#                'SMNABTCH': dict(dtype='str'),
#                'SMNABTCHT': dict(dtype='str'),
#                'SMNABTCHD': dict(dtype='str'),
#                'SMGEBTCH': dict(dtype='str'),
#                'SMGEBTCHD': dict(dtype='str'),
#                'SMGEBTCHT': dict(dtype='str'),
#                'SMAFRZE': dict(dtype='str'),
#                'AGE': dict(dtype='str'),
#                'genes': dict(dtype='str'),
#                'gene_symbol': dict(dtype='str'),
#                'sample_name': dict(dtype='str'),
#                'subtissue': dict(dtype='str'),
#                'individual': dict(dtype='str'),
#                'sampleID'                       object
#                'RNA_ID'                         object
#                'submitted_subject_id'           object
#                subtissue                      object
#                Sample_Name                    object
#                SRA_Sample                     object
#                sex                            object
#                BioSample                      object
#                Experiment                     object
#                tissue                         object
#                RIN                           float64
#                SMAFRZE                        object
#                DNA_ID                         object
#                DNA_ASSAY                      object
#                TISSUE_CLEAN                   object
#                DROP_GROUP                     object
#                COUNT_MODE                     object
#                PAIRED_END                       bool
#                COUNT_OVERLAPS                   bool
#                STRAND                         object
#                RNA_BAM_FILE                   object
#                DNA_VCF_FILE                   object
#                N                               int64
#                expressedGenes                  int64
#                unionExpressedGenes             int64
#                intersectionExpressedGenes      int64
#                passedFilterGenes               int64
#                expressedGenesRank              int64
#                sizeFactor                    float64
#                thetaCorrection                 int64
#
            }
        )

        print("writing zarr output finished")

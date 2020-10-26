import os
import sys

import pandas as pd
import numpy as np
import xarray as xr

import desmi

from cached_property import cached_property

__all__ = [
    "GTExTranscriptProportions",
    "DesmiGTFetcher",
]


# class VariantCSQDataset:
#     def __init__(self, db_conn=None):
#         self.db_conn = desmi.database.get_db_conn(db_conn)
#
#     def sel(self, gene, af_limit=1):
#         # make sure that genes will be a list
#         if isinstance(gene, str):
#             genes = [gene]
#         else:
#             genes = gene
#
#         df_iter = pd.read_sql(
#             #             """
#             #             select
#             #                 -- g."chrom",
#             #                 -- g."start",
#             #                 -- g."end",
#             #                 -- g."ref",
#             #                 -- g."alt",
#             #                 -- g."GT",
#             #                 -- g."AF",
#             #                 -- g."AC",
#             #                 -- g."feature",
#             #                 -- g."feature_type",
#             #                 -- g."gene",
#             #                 -- g."{target_feature}",
#             #             """
#             f"""
#             select
#                 g.*,
#             from hoelzlwi.gtex_features as g
#             where
#                 g."gene" IN ('{"', '".join(genes)}')
#                 and g."AF" <= '{af_limit}'
#             order by g."gene"
#             ;
#             """,
#             # and g."{target_feature}" = '1'
#             con=self.db_conn,
#             chunksize=VARIANTS_CHUNKSIZE,
#         )
#
#         # some type casting
#         for df in df_iter:
#             df = df.astype({
#                 **{f: "boolean" for f in flags},
#                 **{s: "float32" for s in scores_higher_is_deleterious},
#                 **{s: "float32" for s in scores_lower_is_deleterious},
#                 **{s: "float32" for s in scores_absdiff_is_deleterious},
#                 **{c: "float32" for c in [
#                     "mean_transcript_proportions",
#                     "median_transcript_proportions",
#                     "sd_transcript_proportions",
#                     "blood_mean_transcript_proportions",
#                     "blood_median_transcript_proportions",
#                     "blood_sd_transcript_proportions",
#                 ]},
#                 **{c: "str" for c in [
#                     "condel_prediction",
#                     "sift_prediction",
#                 ]},
#             }).fillna({
#                 "mean_transcript_proportions": 0,
#                 "median_transcript_proportions": 0,
#                 "sd_transcript_proportions": 0,
#             })
#
#             # yield the casted df
#             yield df

class DesmiGTFetcher:
    def __init__(self, gt_array: desmi.genotype.Genotype):
        self.gt_array = gt_array

    def get(self, variant: desmi.objects.Variant, variable=["GT", "GQ", "DP", "AC", "AF"]) -> xr.Dataset:
        gt_array = self.gt_array

        retval = xr.Dataset(
            data_vars={
                v: xr.DataArray(
                    gt_array.get(var=variant, path=v),
                    dims=("variant", "sample_id"),
                ) for v in variable if v in {"GT", "GQ", "DP"}
            },
            coords={
                "variant": (("variant",), variant.to_records()),
                #                  "variant": (("variant",), variant.df.index),
                #                  **{c: (("variant", ), v) for c, v in variant.df.items()},
                "sample_id": (("sample_id",), gt_array.sample_anno.sample_id),
                "sample_idx": (("sample_id",), gt_array.sample_anno.sample_idx),
            }
        )

        if "GT" in variable:
            if "AC" in variable:
                retval["AC"] = retval.GT.sum(dim="sample_id")

                if "AF" in variable:
                    retval["AF"] = retval.AC / (retval.dims["sample_id"] * 2)

        return retval

    def to_xarray(self, *args, **kwargs) -> xr.Dataset:
        return self.get(*args, **kwargs)

    def to_dataframe(self, *args, filter_reference_allele=True, **kwargs) -> pd.DataFrame:
        retval = self.get(*args, **kwargs).to_dataframe()
        if filter_reference_allele:
            retval = retval.query("(GT > 0)")

        retval = retval.assign(
            GT=retval["GT"].astype(pd.CategoricalDtype([0, 1, 2])).cat.rename_categories(
                {0: "reference", 1: "heterozygous", 2: "homozygous"})
        )

        return retval


class GTExTranscriptProportions:
    """
    Fetches transcript proportions for list of genes and subtissues.
    Example:
        ```
        tp = GTExTranscriptProportions()
        all_subtissues = tp.subtissues()
        all_subtissues = tp.subtissues()
        tp.get(gene=genes, subtissue=all_subtissues)
        ```
        ```
        <xarray.Dataset>
        Dimensions:                        (subtissue: 30, transcript: 194302)
        Coordinates:
          * subtissue                      (subtissue) object 'Adipose - Subcutaneous...
          * transcript                     (transcript) object 'ENST00000000233' ... ...
        Data variables:
            gene                           (transcript) object 'ENSG00000004059' ... ...
            mean_transcript_proportions    (subtissue, transcript) float32 0.7504935 ...
            median_transcript_proportions  (subtissue, transcript) float32 0.7525773 ...
            sd_transcript_proportions      (subtissue, transcript) float32 0.05871942...
            tissue                         (subtissue) object 'Adipose Tissue' ... 'B...
            transcript_version             (transcript) object '9' '7' '10' ... '1' '1'
            gene_version                   (transcript) object '10' '7' ... '17' '6'
        ```

    """

    def __init__(self, db_conn=None, db_conn_factory=desmi.database.create_engine_from_url):
        self.db_conn = desmi.database.get_db_conn(db_conn)
        self.db_conn_factory = db_conn_factory

    @cached_property
    def genes(self):
        return pd.read_sql(
            """
            select distinct gene from gtex_transcript_proportions order by gene;
            """,
            con=self.db_conn,
        )["gene"]

    @cached_property
    def subtissues(self):
        return pd.read_sql(
            """
            select distinct subtissue from gtex_transcript_proportions order by subtissue;
            """,
            con=self.db_conn,
        )["subtissue"]

    def _get_canonical_transcript(self, gene, subtissue=None):
        """
        Idea:
         1) Check if there is only one transcript in a gene; when true, return this as canonical.
         2) Otherwise, return maximum-expressed transcript per subtissue as canonical.
             If there is a missing value for a subtissue, replace it with the mean proportion across subtissues.

        Args:
            gene:
            subtissue:

        Returns:

        """
        if subtissue is None:
            subtissue = self.subtissues

        val = self.get(gene=gene)  # get all available subtissues
        val = val.set_coords("gene").median_transcript_proportions

        if len(val.transcript) == 1:
            # there is only one transcript
            for subt in subtissue:
                yield gene, subt, val.transcript.item()
        else:
            val_mean = val.mean(dim="subtissue")

            mean_canonical_transcript = val_mean.argmax(dim="transcript")
            mean_canonical_transcript = val.transcript[mean_canonical_transcript].item()

            # Replace NaN's with the mean expression proportion across tissues
            val = val.fillna(val_mean)

            for subt in subtissue:
                if subt in val.subtissue:
                    val_subt = val.sel(subtissue=subt)
                    yield gene, subt, val.transcript[val_subt.argmax(dim="transcript")].item()
                else:
                    yield gene, subt, mean_canonical_transcript

        # max_transcript = xrds.transcript[
        #                      xrds.set_coords("gene").median_transcript_proportions.groupby("gene").apply(
        #                          lambda c: c.argmax(dim="transcript"))
        #                  ].to_dataframe().iloc[:, 0]

    def get_canonical_transcript(self, gene, subtissue=None) -> pd.Series:
        if isinstance(subtissue, str):
            subtissue = [subtissue]
        if isinstance(gene, str):
            retval_df = pd.DataFrame.from_records(
                self._get_canonical_transcript(gene, subtissue),
                columns=["gene", "subtissue", "transcript"]
            ).set_index(["gene", "subtissue"])
            return retval_df["transcript"]
        else:
            retval_df = pd.concat([
                pd.DataFrame.from_records(
                    self._get_canonical_transcript(g, subtissue),
                    columns=["gene", "subtissue", "transcript"]
                ).set_index(["gene", "subtissue"])
                for g in gene
            ], axis=0)
            return retval_df["transcript"]

    def get(self, gene=None, subtissue=None) -> xr.Dataset:
        if gene is None:
            gene = self.genes

        if subtissue is None:
            subtissue = self.subtissues

        # make sure that genes will be a list
        if isinstance(gene, str):
            genes = [gene]
        else:
            genes = gene

        # make sure that subtissues will be a list
        if isinstance(subtissue, str):
            subtissues = [subtissue]
        else:
            subtissues = subtissue

        query = f"""
            select * from gtex_transcript_proportions
            where
                "subtissue" IN ('{"', '".join(subtissues)}')
                and "gene" IN ('{"', '".join(genes)}')
            order by "gene"
            ;
        """
        df = pd.read_sql(
            query,
            con=self.db_conn,
        )

        # some type casting
        df = df.astype({
            **{f: "str" for f in [
                "subtissue",
                "tissue",
                "transcript",
                "transcript_version",
                "gene",
                "gene_version",
            ]},
            **{s: "float32" for s in [
                "mean_transcript_proportions",
                "median_transcript_proportions",
                "sd_transcript_proportions",
            ]},
        }).fillna({
            "mean_transcript_proportions": 0,
            "median_transcript_proportions": 0,
            "sd_transcript_proportions": 0,
        })

        df = df.set_index(["subtissue", "transcript"])

        # convert to 2D xarray dataset
        xrds = df.to_xarray()
        xrds["gene"] = xrds["gene"].isel(subtissue=0)
        xrds["transcript_version"] = xrds["transcript_version"].isel(subtissue=0)
        xrds["gene_version"] = xrds["gene_version"].isel(subtissue=0)
        xrds["tissue"] = xrds["tissue"].isel(transcript=0)

        # make sure that subtissues are aligned as specified
        (xrds,) = xr.align(xrds, indexes={"subtissue": subtissues})

        return xrds

    def __getstate__(self):
        state = self.__dict__.copy()

        state["db_url"] = state['db_conn'].url
        del state['db_conn']
        return state

    def __setstate__(self, state):
        # Restore db connection
        db_conn_factory = state["db_conn_factory"]
        state["db_conn"] = db_conn_factory(state["db_url"])
        del state["db_url"]

        # Restore instance attributes
        self.__dict__.update(state)

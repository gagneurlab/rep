import os
import sys
from typing import Callable, Dict, Union
import warnings

import pandas as pd
import numpy as np
import xarray as xr

from cached_property import cached_property

import desmi
from rep.data.desmi import GTExTranscriptProportions

__all__ = [
    "VEPGeneLevelVariantAggregator",
    "VEPTranscriptLevelVariantAggregator"
]


class VEPTranscriptLevelVariantAggregator:
    functions = {
        'min': 'min',
        'mean': 'mean',
        'median': 'median',
        'max': 'max',
        'count': 'count',
        'sum': 'sum',
        'abs_max': lambda c: c.abs().max(),
        'abs_min': lambda c: c.abs().min(),
        'abs_mean': lambda c: c.abs().mean(),
        'abs_median': lambda c: c.abs().median(),
        'pval_max_significant': lambda c: np.fmin(- np.log(np.nanmin(c.astype("float64"))), 30),
    }

    def __init__(
            self,
            vep_anno,
            gt_fetcher,
            variables=None,
            variable_dtypes=None,
            metadata_transformer: Dict[str, Callable] = None,
            genotype_query="(GQ >= 80) & (DP >= 4) & (AF < 0.01)"
    ):
        self.vep_anno = vep_anno
        self.gt_fetcher = gt_fetcher

        if variables is None:
            self.variables = {
                **{c: ['max', 'sum'] for c in self.vep_anno.consequences},
                'cadd_raw': ['max'],
                'cadd_phred': ['max'],
                'polyphen_score': ['max'],
                'condel_score': ['max'],
                'sift_score': ['pval_max_significant'],
                'maxentscan_diff': ["abs_max"],
            }
        else:
            self.variables = variables

        if variable_dtypes is None:
            self.variable_dtypes = {
                **{c: "bool" for c in self.vep_anno.consequences},
                'cadd_raw': "float32",
                'cadd_phred': "float32",
                'polyphen_score': "float32",
                'condel_score': "float32",
                'sift_score': "float32",
                'maxentscan_diff': "float32",
            }
        else:
            self.variable_dtypes = variable_dtypes

        self.metadata_transformer = metadata_transformer
        self.genotype_query = genotype_query

    def get_vep_csq(self, gene, variants: desmi.objects.Variant = None) -> pd.DataFrame:
        if variants is None:
            # fetch variants annotated by gene
            variants = self.get_variants_for_gene(gene)

        # get variables for provided gene and variants
        cols = [
            "chrom",
            "start",
            "end",
            "ref",
            "alt",
            "feature",
            "gene",
            *self.variables.keys()
        ]

        vep_csq = self.vep_anno.get_columns(
            variants, columns=cols, feature=gene, feature_type="gene", aggregation=None, preserve_all_variants=False,
            escape_columns=True
        )

        # --- now set index to (gene, feature, variant)
        vep_csq["variant"] = pd.DataFrame({
            "variant": desmi.objects.Variant(vep_csq[["chrom", "start", "end", "ref", "alt"]],
                                             sanitize=False).to_records()
        },
            index=vep_csq.index).variant  # directly assigning the variant records does not work due to some bug in Pandas
        vep_csq = vep_csq.drop(columns=[
            "chrom",
            "start",
            "end",
            "ref",
            "alt"
        ])
        vep_csq = vep_csq.set_index([
            "gene",
            "feature",
            "variant"
        ])

        # --- result: (gene, feature, variant) -> (vep features*)
        return vep_csq

    def get_variants_for_gene(self, gene):
        return self.vep_anno.get_variants(feature=gene, feature_type="gene")

    def get_gt_df(self, variants: desmi.objects.Variant):
        """
        get genotype dataframe for variants
        """
        gt_fetcher = self.gt_fetcher
        genotype_query = self.genotype_query

        df: pd.DataFrame = gt_fetcher.to_dataframe(variants)
        df = df.query(genotype_query)

        # result: (sample_id, variant) -> (GT, GQ, DP, AC, AF)
        return df

    @cached_property
    def aggregation_functions(self):
        aggregation_functions = {
            k: [(f, self.functions[f]) for f in v] for k, v in self.variables.items()
        }
        return aggregation_functions

    #     @property
    #     def schema(self):
    #         return {

    #         }

    @cached_property
    def schema(self) -> pd.DataFrame:
        """
        Returns an empty dataframe with the expected schema of the output
        """
        index = [
            # "subtissue",
            "gene",
            "feature",
            "sample_id",
        ]
        columns = [
            "gene",
            "feature",
            "sample_id",
            *[".".join(["heterozygous", key, agg]) for key, value in self.variables.items() for agg in value],
            *[".".join(["homozygous", key, agg]) for key, value in self.variables.items() for agg in value],
        ]
        dtype = {
            "gene": str,
            "feature": str,
            "sample_id": str,
            **{".".join(["heterozygous", key, agg]): "float32"
               for key, value in self.variables.items() for agg in value},
            **{".".join(["homozygous", key, agg]): "float32"
               for key, value in self.variables.items() for agg in value},
        }

        retval = pd.DataFrame(columns=columns)
        retval = retval.astype(dtype)
        retval = retval.set_index(index)

        return retval

    def agg_transcript_level(self, gene, variant_batch_size=10000) -> pd.DataFrame:
        agg_functions = self.aggregation_functions

        variants = self.get_variants_for_gene(gene)
        if variants.df.empty:
            # no known variants in selection; just return empty dataframe
            return self.schema

        partial_gt_df = []
        for _, df in variants.df.groupby(np.arange(len(variants.df)) // variant_batch_size):
            gt_df_batch = self.get_gt_df(desmi.objects.Variant(df, sanitize=False))
            partial_gt_df.append(gt_df_batch)
        gt_df = pd.concat(partial_gt_df, axis=0)

        #         gt_df = self.get_gt_df(variants)
        if gt_df.empty:
            # no variants in selection; just return empty dataframe
            return self.schema

        gt_variants = desmi.objects.Variant.from_records(gt_df.index.unique("variant"))
        vep_csq = self.get_vep_csq(gene=gene, variants=gt_variants)

        # retval = {
        #     "metadata": {
        #     }
        # }

        joined = vep_csq.join(gt_df, how="inner")
        dtypes = {c: self.variable_dtypes[c] for c in joined.columns if c in self.variable_dtypes}
        joined = joined.astype(dtypes)
        grouped = joined.groupby(["GT", "gene", "feature", "sample_id"])

        # # calculate size metadata
        # size = grouped.agg("size").unstack("GT").loc[:, ["heterozygous", "homozygous"]]
        # retval["metadata"]["size"] = size
        #
        # if self.metadata_transformer is not None:
        #     if isinstance(self.metadata_transformer, dict):
        #         for k, v in self.metadata_transformer:
        #             retval["metadata"][k] = v(grouped)
        #     else:
        #         return ValueError("Dictionary of functions expected!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(divide='ignore', invalid='ignore'):
                agg = grouped.agg(agg_functions)

        to_join = {
            'heterozygous': agg.query("GT == 'heterozygous'").droplevel("GT"),
            'homozygous': agg.query("GT == 'homozygous'").droplevel("GT"),
        }
        agg = pd.concat(to_join.values(), axis=1, keys=to_join.keys(), join="outer")

        # # --- old method that does not preserve schema on empty df:
        # agg = agg.unstack("GT")
        # agg = agg.loc[:, agg.columns.to_frame().query("GT != 'reference'").index]
        # # ---

        # flatten multiindex: concatenate levels with '.'
        agg.columns = [".".join(c) for c in agg.columns.to_list()]

        # cast to float
        agg = agg.astype("float32")
        agg = agg.reorder_levels(["gene", "feature", "sample_id"])

        # retval = {
        #     # "gene": [gene],
        #     # "index": agg.index,
        #     "input": agg,
        #     "metadata": {
        #         "size": size,
        #     }
        # }
        # return retval
        return agg

    def align(self, index: pd.MultiIndex, how="left"):
        gene = index.unique("gene")

        retval = [self.agg_transcript_level(gene=g) for g in gene]
        retval = pd.concat(retval, axis=0)

        index_df = pd.DataFrame(index=index)
        retval = index_df.join(retval, how=how)

        return retval

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.agg_transcript_level(gene=index)
        else:
            return pd.concat(
                [self.agg_transcript_level(gene=g) for g in index],
                axis=0
            )


class VEPGeneLevelVariantAggregator:

    def __init__(self, vep_tl_aggr: VEPTranscriptLevelVariantAggregator, gtex_tp: GTExTranscriptProportions = None):
        self.vep_tl_aggr = vep_tl_aggr
        if gtex_tp is None:
            self.gtex_tp = GTExTranscriptProportions()
        else:
            self.gtex_tp = gtex_tp

    @cached_property
    def schema(self) -> pd.DataFrame:
        """
        Returns an empty dataframe with the expected schema of the output
        """
        tl_schema = self.vep_tl_aggr.schema
        index = [
            "subtissue",
            "gene",
            # "feature",
            "sample_id",
        ]
        columns = [
            "subtissue",
            "gene",
            # "feature",
            "sample_id",
            *tl_schema.columns
        ]
        dtype = {
            "subtissue": str,
            "gene": str,
            # "feature": str,
            "sample_id": str,
            **tl_schema.dtypes
        }

        retval = pd.DataFrame(columns=columns)
        retval = retval.astype(dtype)
        retval = retval.set_index(index)

        return retval

    def agg_gene_level(self, gene, subtissue) -> pd.DataFrame:
        if isinstance(subtissue, str):
            subtissue = [subtissue]

        transcript_level_batch = self.vep_tl_aggr[gene]
        # if transcript_level_batch is None:
        #     return None

        max_transcript_df = self.gtex_tp.get_canonical_transcript(gene=gene, subtissue=subtissue)
        max_transcript_df = pd.DataFrame(dict(feature=max_transcript_df)).set_index("feature", append=True)

        # retval = {
        #     # "gene": [gene],
        #     # "subtissue": subtissue,
        #     "metadata": {
        #     }
        # }

        # gene_level_df = max_transcript_df.join(transcript_level_batch["input"], how="inner")
        gene_level_df = max_transcript_df.join(transcript_level_batch, how="inner")
        gene_level_df = gene_level_df.droplevel("feature")
        gene_level_df = gene_level_df.reorder_levels(["subtissue", "gene", "sample_id"])
        # retval["input"] = gene_level_df

        # size = max_transcript_df.join(transcript_level_batch["metadata"]["size"], how="inner")
        # size = size.droplevel("feature")
        # retval["metadata"]["size"] = size

        # retval["index"] = retval["input"].index

        return gene_level_df

    def align(self, index: pd.MultiIndex, how="left"):
        gene = index.unique("gene")
        subtissue = index.unique("subtissue")

        retval = [self.agg_gene_level(gene=g, subtissue=subtissue) for g in gene]
        retval = pd.concat(retval, axis=0)

        index_df = pd.DataFrame(index=index)
        retval = index_df.join(retval, how=how)

        return retval

    def __getitem__(self, selection):
        return self.agg_gene_level(**selection)

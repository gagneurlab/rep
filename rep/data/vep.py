import os
import sys

import pandas as pd
import numpy as np
import xarray as xr

from cached_property import cached_property

import desmi
from rep.data.desmi import GTExTranscriptProportions


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
        'pval_max_significant': lambda c: np.fmin(- np.log(np.nanmin(c)), 30),
    }

    def __init__(self, vep_anno, gt_fetcher, variables=None, genotype_query="(GQ >= 80) & (DP >= 4) & (AF < 0.01)"):
        self.vep_anno = vep_anno
        self.gt_fetcher = gt_fetcher

        self.variables = variables
        self.genotype_query = genotype_query

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
        }).variant  # directly assigning the variant records does not work due to some bug in Pandas
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

    def agg_transcript_level(self, gene, flatten_cols=True):
        agg_functions = self.aggregation_functions

        variants = self.get_variants_for_gene(gene)
        gt_df = self.get_gt_df(variants)
        vep_csq = self.get_vep_csq(gene=gene, variants=variants)

        agg = vep_csq.join(gt_df).groupby(["GT", "gene", "feature", "sample_id"])
        agg = agg.agg(agg_functions)
        agg = agg.unstack("GT")
        agg = agg.loc[:, agg.columns.to_frame().query("GT != 'reference'").index]

        if flatten_cols:
            # flatten multiindex: concatenate levels with '.'
            agg.columns = [".".join(c) for c in agg.columns.to_list()]

        return agg.astype("float32")

    def __getitem__(self, gene):
        if isinstance(gene, str):
            return self.agg_transcript_level(gene=gene)


class VEPGeneLevelVariantAggregator:
    def __init__(self, vep_tl_aggr: VEPTranscriptLevelVariantAggregator, gtex_tp: GTExTranscriptProportions = None):
        self.vep_tl_aggr = vep_tl_aggr
        if gtex_tp is None:
            self.gtex_tp = GTExTranscriptProportions()
        else:
            self.gtex_tp = gtex_tp



    def agg_gene_level(self, gene, subtissue=None):
        transcript_level_df = self.vep_tl_aggr[gene]
        max_transcript_df = self.get_canonical_transcript(gene=gene, subtissue=subtissue)

        gene_level_df = pd.DataFrame(dict(feature=max_transcript_df)).set_index("feature", append=True).join(
            transcript_level_df)
        gene_level_df = gene_level_df.droplevel("feature")

        return gene_level_df

    def __getitem__(self, selection):
        return self.agg_gene_level(**selection)

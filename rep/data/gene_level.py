import collections

import xarray as xr
import pandas as pd

import desmi
from rep.data.desmi import GTExTranscriptProportions, DesmiGTFetcher
from rep.data.expression import GeneExpressionFetcher
from rep.data.vep import VEPTranscriptLevelVariantAggregator, VEPGeneLevelVariantAggregator

__all__ = [
    "_expression_xrds",
    "REPGeneLevelDL",
]

import logging

log = logging.getLogger(__name__)


def _expression_xrds(gene_expression_zarr_path, samples=None):
    xrds = xr.open_zarr(gene_expression_zarr_path)
    xrds["gene"] = xrds["gene"].to_pandas().apply(lambda g: g.split(".")[0])
    if samples is not None:
        xrds = xrds.sel(individual=xrds.individual.isin(samples))
    xrds = xrds.sel(gene=(xrds.hilo_padj != 0).any(dim=["individual", "subtissue"]))

    xrds = xrds.rename(individual="sample_id")
    return xrds


class REPGeneLevelDL:

    def __init__(
            self,
            vep_variables,
            gt_array_path,
            gene_expression_zarr_path,
            target_tissues=None,
            sample_ids=None,
            gene_expression_variables=None,
            gene_expression_subtissues=None,
            expression_query="~ missing",
            # expression_vep_join="left",
            genotype_query: str = "(GQ >= 80) & (DP >= 4) & (AF < 0.01)",
            vep_tl_args=None,
            vep_gl_args=None,
    ):
        # set default values
        if gene_expression_variables is None:
            gene_expression_variables = ["zscore", "missing", "hilo_padj"]
        if gene_expression_subtissues is None:
            gene_expression_subtissues = ["Whole Blood", "Cells - Transformed fibroblasts"]
        if target_tissues is None:
            target_tissues = ["Lung", "Brain"]
        if vep_tl_args is None:
            vep_tl_args = {}
        if vep_gl_args is None:
            vep_gl_args = {}

        self.dataloaders = {}
        self.gene_expression_variables = gene_expression_variables
        self.gene_expression_subtissues = gene_expression_subtissues
        self.vep_tl_args = vep_tl_args
        self.vep_gl_args = vep_gl_args
        self.genotype_query = genotype_query

        self.expression_query = expression_query
        # self.expression_vep_join = expression_vep_join

        # setup genotype
        gt_array = desmi.genotype.Genotype(path=gt_array_path)
        gt_fetcher = DesmiGTFetcher(gt_array=gt_array)
        # get gene expression dataset
        expression_xrds = _expression_xrds(gene_expression_zarr_path, samples=gt_array.samples())

        if sample_ids is not None:
            expression_xrds = expression_xrds.sel(sample_id=sample_ids)
        else:
            sample_ids = pd.Index(gt_array.samples())
            sample_ids = sample_ids.join(expression_xrds.sample_id, how="inner")
            log.info("Keeping %d samples (intersection between gene expression and DNA dataset)", len(sample_ids))

        self.genes = expression_xrds.gene.values
        self.target_tissues = target_tissues
        self.sample_ids = sample_ids
        self.gt_array = gt_array
        self.gt_fetcher = gt_fetcher
        self.expression_xrds = expression_xrds

        vep_anno = desmi.annotations.get("VEP")
        vep_tl_agg = VEPTranscriptLevelVariantAggregator(
            vep_anno=vep_anno,
            gt_fetcher=gt_fetcher,
            variables=vep_variables,
            genotype_query=genotype_query,
            **vep_tl_args,
        )
        vep_gl_agg = VEPGeneLevelVariantAggregator(
            vep_tl_agg,
            **vep_gl_args
        )
        self.dataloaders["vep"] = vep_gl_agg

        gene_expression_fetcher = GeneExpressionFetcher(
            expression_xrds,
            variables=gene_expression_variables
        )
        self.dataloaders["expression"] = gene_expression_fetcher

    # @property
    # def target_tissues(self):
    #     return self.expression_xrds.subtissue.values

    def get(self, gene):
        # this loads a dataframe of multiple target tissues
        vep = self.dataloaders.get("vep")[dict(gene=gene, subtissue=self.target_tissues)]

        # gene expression input is independent of the target tissue
        expression = self.dataloaders.get("expression")[dict(gene=gene, subtissue=self.gene_expression_subtissues)]
        # convert subtissue to columns
        expression = expression.unstack(level="subtissue")
        # flatten column names
        expression.columns = [".".join(c) for c in expression.columns.to_list()]
        # add to batch

        for t in self.target_tissues:
            index = pd.MultiIndex.from_product(
                [[t], [gene], self.sample_ids],
                names=["subtissue", "gene", "sample_id"]
            )
            batch_df = pd.DataFrame(index=index)

            # add gene expression to batch
            batch_df = batch_df.join(expression, how="left")
            # add vep to batch
            batch_df = batch_df.join(vep["input"], how="left")
            batch_df = batch_df.reorder_levels(order=["subtissue", "gene", "sample_id"])

            batch = {
                "gene": [gene],
                "subtissue": [t],
                "sample_id": self.sample_ids,
                "index": index,
                "input": batch_df,
                "metadata": {
                    "vep": vep["metadata"],
                }
            }
            yield batch

    def __getitem__(self, selection):
        return self.get(**selection)

    def iter(self):
        for gene in self.genes:
            for batch in self.get(gene):
                yield batch

    def train_iter(self):
        for gene in self.genes:
            for batch in self.get(gene):
                target = self.dataloaders.get("expression")[dict(gene=batch["gene"], subtissue=batch["subtissue"])]
                # target = expression.query(self.expression_query, engine="python")
                #             expression, vep = expression.align(vep, axis=0, join=self.expression_vep_join)
                #             batch = {
                #                 "expression": expression,
                #                 "vep": vep
                #             }

                # align to batch index
                target = pd.DataFrame(index=batch["index"]).join(target, how="left")

                batch["target"] = target

                yield batch

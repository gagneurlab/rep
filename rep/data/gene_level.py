import collections
from typing import List

import xarray as xr
import pandas as pd

import desmi
from rep.data.desmi import GTExTranscriptProportions, DesmiGTFetcher
from rep.data.expression import GeneExpressionFetcher
from rep.data.vep import VEPTranscriptLevelVariantAggregator, VEPGeneLevelVariantAggregator

from cached_property import cached_property

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
            expression_xrds,
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
        if vep_tl_args is None:
            vep_tl_args = {}
        if vep_gl_args is None:
            vep_gl_args = {}
        if isinstance(expression_xrds, str):
            expression_xrds = _expression_xrds(expression_xrds)

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

        if target_tissues is None:
            target_tissues = expression_xrds.subtissue.values

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

    @cached_property
    def _is_missing(self):
        return self.expression_xrds.missing.all(dim="sample_id").to_dataframe()["missing"]

    def is_expressed(self, gene, subtissue):
        return not self._is_missing.loc[(subtissue, gene)]

    # @property
    # def target_tissues(self):
    #     return self.expression_xrds.subtissue.values

    def get(self, index: pd.MultiIndex):
        gene = index.unique("gene")
        sample_id = index.unique("sample_id")
        subtissue = index.unique("subtissue")

        # this loads a dataframe of multiple target tissues
        # vep = self.dataloaders.get("vep")[dict(gene=gene, subtissue=self.target_tissues)]
        vep = self.dataloaders.get("vep")[dict(gene=gene, subtissue=subtissue)]
        if vep is None:
            return None

        # gene expression input is independent of the target tissue
        expression = self.dataloaders.get("expression")[dict(gene=gene, subtissue=self.gene_expression_subtissues)]
        # convert subtissue to columns
        expression = expression.unstack(level="subtissue")
        # flatten column names
        expression.columns = [".".join(c) for c in expression.columns.to_list()]
        # add to batch

        # for t in subtissue:
        # if not self.is_expressed(gene, t):
        #     continue

        # index = pd.MultiIndex.from_product(
        #     [[t], [gene], self.sample_ids],
        #     names=["subtissue", "gene", "sample_id"]
        # )
        batch_df = pd.DataFrame(index=index)

        # add gene expression to batch
        batch_df = batch_df.join(expression, how="left")
        # add vep to batch
        batch_df = batch_df.join(vep, how="left")
        # batch_df = batch_df.join(vep["input"].assign(vep_missing=-1), how="left")
        # batch_df.loc[:, "vep_missing"] = (batch_df["vep_missing"].fillna(0) + 1).astype(bool)
        batch_df = batch_df.reorder_levels(order=["subtissue", "gene", "sample_id"])

        # batch = {
        #     # "gene": [gene],
        #     # "subtissue": [t],
        #     # "sample_id": self.sample_ids,
        #     # "index": index,
        #     "inputs": batch_df,
        #     "metadata": {
        #         "index": batch_df.index,
        #         "vep": vep["metadata"],
        #         "vep_present": batch_df.index.isin(
        #             vep["input"].index.reorder_levels(order=["subtissue", "gene", "sample_id"])
        #         ),
        #     }
        # }
        # yield batch
        return batch_df

    def __getitem__(self, selection):
        return self.get(**selection)

    def iter(self, genes=None):
        if genes is None:
            genes = self.genes
        for gene in genes:
            for batch in self.get(gene):
                if batch is None:
                    continue
                yield batch

    def train_iter(self, genes: List[str] = None, target_variable="zscore"):
        if genes is None:
            genes = self.genes
        for gene in genes:
            batch_iter = self.get(gene)
            if batch_iter is None:
                continue
            for batch in batch_iter:
                targets = self.expression_xrds[[target_variable, "missing"]].sel(
                    gene=batch["metadata"]["index"].unique("gene"),
                    subtissue=batch["metadata"]["index"].unique("subtissue"),
                    sample_id=batch["metadata"]["index"].unique("sample_id"),
                ).to_dataframe()
                targets = targets.query("~ missing")[target_variable]
                # target = expression.query(self.expression_query, engine="python")
                #             expression, vep = expression.align(vep, axis=0, join=self.expression_vep_join)
                #             batch = {
                #                 "expression": expression,
                #                 "vep": vep
                #             }

                # align to batch index

                targets, inputs = targets.align(batch["inputs"], join="inner", axis=0)
                batch["inputs"] = inputs
                batch["targets"] = targets
                batch["metadata"]["index"] = targets.index

                yield batch

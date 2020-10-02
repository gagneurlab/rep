import xarray as xr
import pandas as pd

import desmi
from rep.data.desmi import GTExTranscriptProportions, DesmiGTFetcher
from rep.data.expression import GeneExpressionFetcher
from rep.data.vep import VEPTranscriptLevelVariantAggregator, VEPGeneLevelVariantAggregator


def expression_xrds(gene_expression_zarr_path, samples=None):
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
            expression_query="~ missing",
            expression_vep_join="left",
            target_tissues=("Lung", "Brain")
    ):
        self.dataloaders = {}

        self.expression_query = expression_query
        self.expression_vep_join = expression_vep_join

        gt_array = desmi.genotype.Genotype(path=gt_array_path)
        gt_fetcher = DesmiGTFetcher(gt_array=gt_array)

        self.expression_xrds = expression_xrds(gene_expression_zarr_path, samples=gt_array.samples())

        vep_anno = desmi.annotations.get("VEP")
        vep_tl_agg = VEPTranscriptLevelVariantAggregator(
            vep_anno=vep_anno,
            gt_fetcher=gt_fetcher,
            variables=vep_variables,
        )
        vep_gl_agg = VEPGeneLevelVariantAggregator(vep_tl_agg)
        self.dataloaders["vep"] = vep_gl_agg

        #         if "expression" in self.features:
        gene_expression_fetcher = GeneExpressionFetcher(xrds, variables=self.features.get("expression", None))
        self.dataloaders["expression"] = gene_expression_fetcher

    @property
    def target_tissues(self):
        return self.expression.subtissue.values

    def get(self, gene):
        vep = self.dataloaders.get("vep")[dict(gene=gene)]
        for t in self.target_tissues:
            expression = self.dataloaders.get("expression")[dict(gene=gene, subtissue=t)]

            expression = expression.query(self.expression_query, engine="python")
            #             expression, vep = expression.align(vep, axis=0, join=self.expression_vep_join)
            #             batch = {
            #                 "expression": expression,
            #                 "vep": vep
            #             }

            batch = expression.join(vep, how=self.expression_vep_join)

            yield batch.index, batch

    def __getitem__(self, selection):
        return self.get(**selection)

    def iter(self):
        for gene in self.xrds.gene:
            for batch in self.get(gene):
                yield batch

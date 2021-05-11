import xarray as xr
import pandas as pd
import pandera as pdt

try:
    from functools import cached_property
except ImportError:
    # python < 3.8
    from backports.cached_property import cached_property

from rep.transformers.dataframe import PandasTransformer, LambdaTransformer

__all__ = [
    "GeneExpressionFetcher"
]


class GeneExpressionFetcher(PandasTransformer):
    def __init__(self, xrds: xr.Dataset, variables=None):
        if variables is None:
            variables = ["zscore", "missing", "hilo_padj"]
        self.xrds: xr.Dataset = xrds[variables]

    @cached_property
    def schema(self) -> pdt.DataFrameSchema:
        return LambdaTransformer(lambda: self.__call__(
            gene=self.xrds.gene[0].values,
            subtissue=self.xrds.subtissue[0].values
        )).schema

    def get(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, gene, subtissue) -> pd.DataFrame:
        if isinstance(gene, str):
            gene = [gene]
        if isinstance(subtissue, str):
            subtissue = [subtissue]

        retval = self.xrds.sel(
            gene=gene,
            subtissue=subtissue,
        )
        retval = retval.to_dataframe()
        # retval = retval.set_index(["gene"], append=True)
        return retval

    def __getitem__(self, selection):
        return self.get(**selection)

import xarray as xr
import pandas as pd

__all__ = [
    "GeneExpressionFetcher"
]


class GeneExpressionFetcher:
    def __init__(self, xrds: xr.Dataset, variables=None):
        if variables is None:
            variables = ["zscore", "missing", "hilo_padj"]
        self.xrds: xr.Dataset = xrds[variables]

    def get(self, gene, subtissue) -> pd.DataFrame:
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

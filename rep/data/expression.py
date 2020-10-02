import xarray as xr
import pandas as pd


class GeneExpressionFetcher:
    def __init__(self, xrds: xr.Dataset, variables=None):
        if variables is None:
            variables = ["zscore", "missing", "hilo_padj"]
        self.xrds = xrds[variables]

    def get(self, gene, subtissue) -> pd.DataFrame:
        retval = self.xrds.sel(
            gene=gene,
            subtissue=subtissue
        )
        retval = retval.to_dataframe()
        retval = retval.set_index(["gene", "subtissue"], append=True)
        return retval

    def __getitem__(self, selection):
        return self.get(**selection)

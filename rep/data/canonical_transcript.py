import pandas as pd
import numpy as np
import xarray as xr

try:
    from functools import cached_property
except ImportError:
    # python < 3.8
    from backports.cached_property import cached_property

__all__ = [
    "GTExTranscriptProportions",
]


class GTExTranscriptProportions:
    """
    Fetches transcript proportions for list of genes and subtissues.
    Example:
        ```
        tp = GTExTranscriptProportions()
        all_subtissues = tp.subtissues()
        all_genes = tp.genes()
        tp.get(gene=all_genes, subtissue=all_subtissues)
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

    def __init__(self, xrds):
        self.xrds = xrds

    @cached_property
    def genes(self):
        return np.unique(self.xrds["gene"].values)

    @property
    def tissues(self):
        return self.xrds["tissue"].values

    @property
    def subtissues(self):
        return self.xrds["subtissue"].values

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

        val = self.get(gene=gene).compute()  # get all available subtissues
        val = val.set_coords("gene").median_transcript_proportions

        if len(val.transcript) == 1:
            # there is only one transcript
            for subt in subtissue:
                yield gene, subt, val.transcript.item()
        else:
            val_mean = val.mean(dim="subtissue")

            mean_canonical_transcript = val_mean.argmax(dim="transcript")
            mean_canonical_transcript = val.transcript[mean_canonical_transcript]

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

        retval = self.xrds.sel(
            transcript=np.isin(self.xrds["gene"], gene),
            subtissue=subtissue,
        )

        return retval


def test_gtex_tp():
    xrds_path = "/s/project/rep/processed/training_results_v3/general/gtex_subtissue_level_pext_scores_withdefault.zarr"
    xrds = xr.open_zarr(xrds_path)
    gtex_tp = GTExTranscriptProportions(xrds)

    assert all(
        gtex_tp.get_canonical_transcript("ENSG00000003096").values == [
            'ENST00000371882', 'ENST00000371882', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000371882',
            'ENST00000371882', 'ENST00000541812', 'ENST00000541812',
            'ENST00000541812', 'ENST00000541812', 'ENST00000541812',
            'ENST00000541812', 'ENST00000541812', 'ENST00000541812',
            'ENST00000541812', 'ENST00000541812', 'ENST00000541812',
            'ENST00000541812', 'ENST00000541812', 'ENST00000371882',
            'ENST00000371882', 'ENST00000541812', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000371882',
            'ENST00000541812', 'ENST00000541812', 'ENST00000541812',
            'ENST00000371882', 'ENST00000371882', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000541812',
            'ENST00000371882', 'ENST00000541812', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000541812',
            'ENST00000371882', 'ENST00000541812', 'ENST00000371882',
            'ENST00000371882', 'ENST00000371882', 'ENST00000541812'
        ])

import os
import sys
from typing import Union, Iterable, List, Callable

import pandas as pd
import numpy as np
import pandera as pdt
import xarray as xr

import desmi

try:
    from functools import cached_property
except ImportError:
    # python < 3.8
    from backports.cached_property import cached_property

from rep.variant_types import VariantArray, VariantDtype, Variant
from rep.data.veff.vep import VEPAnnotation, VEP_VARIABLES, VEP_DTYPES

__all__ = [
    # "GTExTranscriptProportions",
    "DesmiGTFetcher",
]


def _desmi_to_variantarray(desmi_variant: desmi.objects.Variant) -> VariantArray:
    # make 0-based
    df = desmi_variant.df
    df = df.assign(start=df["start"] - 1)
    return VariantArray.from_df(df)


def _variantarray_to_desmi(variants: VariantArray) -> desmi.objects.Variant:
    variants = VariantArray(variants)
    desmi_variants = variants.as_frame()
    # convert 0-based to 1-based
    desmi_variants["start"] = desmi_variants["start"] + 1
    desmi_variants = desmi.objects.Variant(desmi_variants, sanitize=False)

    return desmi_variants


class DesmiGTFetcher:
    def __init__(
            self,
            gt_array: desmi.genotype.Genotype,
            variant_filters: List[Callable[[VariantArray], List[bool]]] = None,
    ):
        self.gt_array = gt_array
        if variant_filters is not None:
            self.variant_filters = variant_filters
        else:
            self.variant_filters = []

    def get(
            self,
            variant: VariantArray,
            variable=("GT", "GQ", "DP", "AC", "AF", "PC", "private"),
    ) -> xr.Dataset:
        gt_array = self.gt_array

        # ensure ArrayType
        variant = VariantArray(variant)

        # filter variants
        for filter in self.variant_filters:
            variant = variant[filter(variant)]

        # convert to desmi
        desmi_variant = _variantarray_to_desmi(variant)

        attrs = [v for v in variable if v in {"GT", "GQ", "DP"}]
        arrays = gt_array.get(var=desmi_variant, attr=attrs)

        retval = xr.Dataset(
            data_vars={
                k: xr.DataArray(
                    v,
                    dims=("variant", "sample_id"),
                ) for k, v in zip(attrs, arrays)
            },
            coords={
                "variant": (("variant",), np.asarray(variant)),
                #                  "variant": (("variant",), variant.df.index),
                #                  **{c: (("variant", ), v) for c, v in variant.df.items()},
                "sample_id": (("sample_id",), gt_array.sample_anno.sample_id),
                "sample_idx": (("sample_id",), gt_array.sample_anno.sample_idx),
            }
        )

        if "GT" in variable:
            if "AC" in variable:
                # sum all heterozygous + 2x homozygous genotypes
                retval["AC"] = ((retval.GT == 1) + 2 * (retval.GT == 2)).sum(dim="sample_id")

                if "AF" in variable:
                    retval["AF"] = retval.AC / (retval.dims["sample_id"] * 2)

            if "PC" in variable:
                retval["PC"] = ((retval.GT == 1) + (retval.GT == 2)).sum(dim="sample_id")
                if "private" in variable:
                    # check if there is at max. one individual that has the variant
                    retval["private"] = retval.PC == 1

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
        # .reset_index()
        # .astype({
        #     "variant": "Variant",
        # })
        # .set_index(["sample_id", "variant"])

        return retval


class DesmiVEPAnnotation(VEPAnnotation):

    def __init__(self, vep_anno=None, vep_variables=None):
        if vep_anno is None:
            import desmi
            self.vep_anno: desmi.annotations.vep.VEPPlugin = desmi.annotations.get("VEP", version="99v2")
        else:
            self.vep_anno = vep_anno

        if vep_variables is None:
            self.vep_variables = VEP_VARIABLES
        else:
            self.vep_variables = vep_variables

    @cached_property
    def schema(self) -> pdt.DataFrameSchema:
        return pdt.DataFrameSchema(
            index={
                "variant": pdt.Column(VariantDtype()),
                "feature": pdt.Column(pdt.String),  # a.k.a. transcript
            },
            columns={var: pdt.Column(VEP_DTYPES[var]) for var in self.vep_variables},
        )

    def get_variants(self, gene: Union[str, Iterable[str]]) -> VariantArray:
        desmi_variant = self.vep_anno.get_variants(feature=gene, feature_type="gene")
        return _desmi_to_variantarray(desmi_variant)

    def __call__(self, variants: VariantArray, gene: Union[str, Iterable[str]]):
        desmi_variants = _variantarray_to_desmi(variants)

        # get variables for provided gene and variants
        cols = [
            "chrom",
            "start",
            "end",
            "ref",
            "alt",
            "feature",
            "gene",
            *self.vep_variables
        ]

        vep_csq = self.vep_anno.get_columns(
            desmi_variants,
            columns=cols,
            feature=gene,
            feature_type="gene",
            aggregation=None,
            preserve_all_variants=False,
            escape_columns=True
        )

        # --- now set index to (gene, feature, variant)
        vep_csq["variant"] = _desmi_to_variantarray(
            desmi.objects.Variant(vep_csq[["chrom", "start", "end", "ref", "alt"]], sanitize=False)
        )
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

        return vep_csq


def test_desmigtfetcher():
    gt_array_path = "/s/project/variantDatabase/fastStorage/gtex/genotypeArray_GTEx_v7-hg19/"
    gt_array = desmi.genotype.Genotype(gt_array_path)

    gt_fetcher = DesmiGTFetcher(gt_array)

    variant = VariantArray.from_vcf_str([
        '1:10468:T>TAA',
        '1:10468:TCGCGG>T',
    ])

    gt_fetcher.get(variant)
    gt_fetcher.to_dataframe(variant)

    # import tiledb
    # fd: tiledb.array.SparseArray = gt_array.open_array()
    # fd.multi_index[:576483232329498624 + 1000000000]
    #
    # gt_array.get_variantkey()


def test_desmigtfetcher_maf_filter():
    gt_array_path = "/s/project/variantDatabase/fastStorage/gtex/genotypeArray_GTEx_v7-hg19/"
    gt_array = desmi.genotype.Genotype(gt_array_path)

    from rep.data.maf import VariantMafDB
    mafdb_path = "/s/project/gtex-processed/mmsplice-scripts/data/processed/maf.db"
    mafdb = VariantMafDB(mafdb_path)

    gt_fetcher = DesmiGTFetcher(
        gt_array,
        variant_filters=[
            lambda variants: [mafdb.get(v.to_vcf_str(), default=0) < 0.001 for v in variants]
        ]
    )

    variant = VariantArray.from_vcf_str([
        '1:10468:T>TAA',
        '1:10468:TCGCGG>T',
        "10:107494853:C>A",
        "10:107494857:C>A",
        "10:107494858:T>C",
        "10:107494873:C>T",
        "10:107494874:G>A",
        "10:107494905:GAGAA>G",
        "10:107494908:A>G",
        "10:107494929:T>C",
        "10:107494933:T>C",
        "10:107494935:G>A",
        "10:107494937:C>G",
        "10:107494941:CTTG>C",
        "10:107494942:T>A",
        "10:107494943:T>C",
        "10:107494960:G>T",
        "10:107494964:C>A",
        "10:107494979:G>A",
        "10:10749497:A>G",
        "10:107494988:T>C",
        "10:107494989:C>T",
        "10:10749498:C>T",
        "10:107494998:T>C",
        "10:10749499:G>A",
        "10:107495002:T>C",
    ])

    expected_variants = variant[[mafdb.get(v.to_vcf_str(), default=0) < 0.001 for v in variant]]

    xrds = gt_fetcher.get(variant)
    assert np.all(np.sort(np.unique(expected_variants)) == np.sort(np.unique(xrds.variant.values)))

    gt_fetcher.to_dataframe(variant)

    # import tiledb
    # fd: tiledb.array.SparseArray = gt_array.open_array()
    # fd.multi_index[:576483232329498624 + 1000000000]
    #
    # gt_array.get_variantkey()

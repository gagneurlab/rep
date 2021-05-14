import pytest

import pandas as pd

import desmi

GENE = "ENSG00000003402"

from rep.data.variantdb import DesmiGTFetcher
from rep.data.canonical_transcript import GTExTranscriptProportions
from rep.data import VEPGeneLevelVariantAggregator, VEPTranscriptLevelVariantAggregator


@pytest.fixture()
def vep_anno():
    from rep.data.variantdb import DesmiVEPAnnotation

    return DesmiVEPAnnotation()


@pytest.fixture()
def gene():
    return GENE


@pytest.fixture()
def variants(vep_anno, gene):
    variants = vep_anno.get_variants(gene)
    return variants


@pytest.fixture()
def gt_array():
    gt = desmi.genotype.Genotype('/s/project/variantDatabase/fastStorage/gtex/genotypeArray_GTEx_v7-hg19/')
    return gt


@pytest.fixture()
def mafdb():
    from rep.data.maf import VariantMafDB

    mafdb_path = "/s/project/gtex-processed/mmsplice-scripts/data/processed/maf.db"
    mafdb = VariantMafDB(mafdb_path)

    return mafdb


def test_variant_mafdb(mafdb):
    from rep.variant_types import VariantArray
    variants = VariantArray.from_vcf_str([
        "chr10:107494853:C>A",
        "chr10:107494857:C>A",
        "chr10:107494858:T>C",
        "chr10:107494873:C>T",
        "chr10:107494874:G>A",
        "chr10:107494905:GAGAA>G",
        "chr10:107494908:A>G",
        "chr10:107494929:T>C",
        "chr10:107494933:T>C",
        "chr10:107494935:G>A",
        "chr10:107494937:C>G",
        "chr10:107494941:CTTG>C",
        "chr10:107494942:T>A",
        "chr10:107494943:T>C",
        "chr10:107494960:G>T",
        "chr10:107494964:C>A",
        "chr10:107494979:G>A",
        "chr10:10749497:A>G",
        "chr10:107494988:T>C",
        "chr10:107494989:C>T",
        "chr10:10749498:C>T",
        "chr10:107494998:T>C",
        "chr10:10749499:G>A",
        "chr10:107495002:T>C",
    ])
    assert all([mafdb.get(v.to_vcf_str()) > 0 for v in variants])


@pytest.fixture()
def gt_fetcher(gt_array, mafdb):
    gtds = DesmiGTFetcher(
        gt_array=gt_array,
        variant_filters=[
            lambda variants: [mafdb.get(v.to_vcf_str(), default=0) < 0.001 for v in variants]
        ]
    )

    return gtds


def test_gt_fetcher(gt_fetcher, variants):
    gt_fetcher.to_dataframe(variants).query("(GQ >= 80) & (DP >= 4) & (AF < 0.01)")


@pytest.fixture()
def vep_tl_agg(vep_anno, gt_fetcher):
    vep_tl_agg = VEPTranscriptLevelVariantAggregator(
        vep_anno=vep_anno,
        gt_fetcher=gt_fetcher,
        variables={'cadd_raw': ['max', 'mean']}
    )
    return vep_tl_agg


def test_VEPTranscriptLevelVariantAggregator(vep_tl_agg, gene):
    agg = vep_tl_agg.agg_transcript_level(gene)
    schema = vep_tl_agg.schema
    assert isinstance(agg, pd.DataFrame)
    assert agg.dtypes.equals(schema.dtypes)
    assert agg.index.names == schema.index.names


@pytest.fixture()
def gtex_tp():
    import xarray as xr

    xrds_path = "/s/project/rep/processed/training_results_v3/general/gtex_subtissue_level_pext_scores_withdefault.zarr"
    xrds = xr.open_zarr(xrds_path)
    gtex_tp = GTExTranscriptProportions(xrds)

    return gtex_tp


def test_gtex_tp(gtex_tp: GTExTranscriptProportions, gene):
    subtissues = gtex_tp.subtissues
    genes = gtex_tp.genes

    tp_xrds = gtex_tp.get(gene=gene)

    canoncial_df = gtex_tp.get_canonical_transcript(gene=gene)
    canoncial_df2 = gtex_tp.get_canonical_transcript(gene=gene, subtissue="asdf")
    canoncial_df3 = gtex_tp.get_canonical_transcript(gene=gene, subtissue=["asdf", "Whole Blood"])


@pytest.fixture()
def vep_gl_agg(vep_tl_agg, gtex_tp):
    vep_gene_level = VEPGeneLevelVariantAggregator(vep_tl_agg, gtex_tp)
    return vep_gene_level


def test_vep_gl_agg(vep_gl_agg, gene):
    agg = vep_gl_agg.agg_gene_level(gene, subtissue="Whole Blood")
    schema = vep_gl_agg.schema
    assert isinstance(agg, pd.DataFrame)
    assert agg.dtypes.equals(schema.dtypes)
    assert agg.index.names == schema.index.names


def test_vep_gl_dl(vep_gl_agg, gene):
    df = vep_gl_agg.agg_gene_level(
        gene=[gene],
        subtissue=['Lung',
                   'Brain - Cerebellum',
                   'Skin - Sun Exposed (Lower leg)',
                   'Artery - Tibial',
                   'Adipose - Subcutaneous']
    )
    assert all(vep_gl_agg.align(df.index[:10]) == df.iloc[:10])

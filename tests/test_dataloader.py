import pytest

import pandas as pd

import desmi

GENE = "ENSG00000003402"

from rep.data.desmi import DesmiGTFetcher, GTExTranscriptProportions
from rep.data.gene_level import VEPGeneLevelVariantAggregator, VEPTranscriptLevelVariantAggregator


@pytest.fixture()
def vep_anno():
    return desmi.annotations.get("VEP")


@pytest.fixture()
def gene():
    return GENE


@pytest.fixture()
def variants(vep_anno, gene):
    variants = vep_anno.get_variants(feature=gene, feature_type="gene")
    return variants


@pytest.fixture()
def gt_array():
    gt = desmi.genotype.Genotype('/s/project/variantDatabase/fastStorage/genotypeArray_GTEx_v7-hg19/')
    return gt


@pytest.fixture()
def gt_fetcher(gt_array):
    gtds = DesmiGTFetcher(gt_array=gt_array)
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
    assert isinstance(agg, pd.DataFrame)


@pytest.fixture()
def gtex_tp():
    gtex_tp = GTExTranscriptProportions()
    return gtex_tp


def test_gtex_tp(gtex_tp: GTExTranscriptProportions, gene):
    subtissues = gtex_tp.subtissues
    genes = gtex_tp.genes

    tp_xrds = gtex_tp.get(gene=gene)

    canoncial_df = gtex_tp.get_canonical_transcript(gene=gene)
    canoncial_df2 = gtex_tp.get_canonical_transcript(gene=gene, subtissue="asdf")
    canoncial_df3 = gtex_tp.get_canonical_transcript(gene=gene, subtissue=["asdf", "Whole Blood"])


@pytest.fixture()
def vep_gl_agg(vep_tl_agg):
    vep_gene_level = VEPGeneLevelVariantAggregator(vep_tl_agg)
    return vep_gene_level


def test_vep_gl_agg(vep_gl_agg, gene):
    agg = vep_gl_agg.agg_gene_level(gene, subtissue="Whole Blood")
    assert isinstance(agg, pd.DataFrame)

import pytest

import pandas as pd

import desmi

GENE = "ENSG00000003402"

from rep.data.desmi import DesmiGTFetcher, GTExTranscriptProportions
from rep.data.gene_level import VEPGeneLevelVariantAggregator, VEPTranscriptLevelVariantAggregator, REPGeneLevelDL


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
    schema = vep_tl_agg.schema()
    assert isinstance(agg, pd.DataFrame)
    assert agg.dtypes.equals(schema.dtypes)
    assert agg.index.names == schema.index.names


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
    schema = vep_gl_agg.schema()
    assert isinstance(agg, pd.DataFrame)
    assert agg.dtypes.equals(schema.dtypes)
    assert agg.index.names == schema.index.names


@pytest.fixture()
def rep_gl_dl():
    variables = {
        'expression': {
            'variables': ['zscore', 'missing', 'hilo_padj'],
            'subtissues': ['Whole Blood', 'Cells - Transformed fibroblasts']
        },
        'vep': {
            '3_prime_UTR_variant': ['sum'],
            '5_prime_UTR_variant': ['sum'],
            'NMD_transcript_variant': ['sum'],
            'coding_sequence_variant': ['sum'],
            'downstream_gene_variant': ['sum'],
            'frameshift_variant': ['sum'],
            'incomplete_terminal_codon_variant': ['sum'],
            'inframe_deletion': ['sum'],
            'inframe_insertion': ['sum'],
            'intergenic_variant': ['sum'],
            'intron_variant': ['sum'],
            'mature_miRNA_variant': ['sum'],
            'missense_variant': ['sum'],
            'non_coding_transcript_exon_variant': ['sum'],
            'non_coding_transcript_variant': ['sum'],
            'protein_altering_variant': ['sum'],
            'splice_acceptor_variant': ['sum'],
            'splice_donor_variant': ['sum'],
            'splice_region_variant': ['sum'],
            'start_lost': ['sum'],
            'start_retained_variant': ['sum'],
            'stop_gained': ['sum'],
            'stop_lost': ['sum'],
            'stop_retained_variant': ['sum'],
            'synonymous_variant': ['sum'],
            'transcript_ablation': ['sum'],
            'upstream_gene_variant': ['sum'],
            'LoF_HC': ['sum'],
            'LoF_LC': ['sum'],
            'LoF_filter_3UTR_SPLICE': ['sum'],
            'LoF_filter_5UTR_SPLICE': ['sum'],
            'LoF_filter_ANC_ALLELE': ['sum'],
            'LoF_filter_END_TRUNC': ['sum'],
            'LoF_filter_EXON_INTRON_UNDEF': ['sum'],
            'LoF_filter_GC_TO_GT_DONOR': ['sum'],
            'LoF_filter_INCOMPLETE_CDS': ['sum'],
            'LoF_filter_NON_ACCEPTOR_DISRUPTING': ['sum'],
            'LoF_filter_NON_DONOR_DISRUPTING': ['sum'],
            'LoF_filter_RESCUE_ACCEPTOR': ['sum'],
            'LoF_filter_RESCUE_DONOR': ['sum'],
            'LoF_filter_SMALL_INTRON': ['sum'],
            'LoF_flag_NAGNAG_SITE': ['sum'],
            'LoF_flag_NON_CAN_SPLICE': ['sum'],
            'LoF_flag_PHYLOCSF_UNLIKELY_ORF': ['sum'],
            'LoF_flag_PHYLOCSF_WEAK': ['sum'],
            'LoF_flag_SINGLE_EXON': ['sum'],
            'cadd_raw': ['max'],
            'cadd_phred': ['max'],
            'condel_score': ['max'],
            'sift_score': ['min']
        }
    }

    rep_gl_dl = REPGeneLevelDL(
        gt_array_path="/s/project/variantDatabase/fastStorage/genotypeArray_GTEx_v7-hg19/",
        expression_xrds='/s/project/rep/processed/gtex/OUTRIDER/gtex_unstacked.zarr',
        vep_variables=variables["vep"],
        gene_expression_variables=variables["expression"]["variables"],
    )
    return rep_gl_dl


def test_rep_gl_dl(rep_gl_dl, gene):
    for b in rep_gl_dl.get(gene):
        pass


def test_rep_gl_dl_iter(rep_gl_dl):
    for b in rep_gl_dl.iter():
        assert isinstance(b, dict)

        # stop after first batch
        break


def test_rep_gl_dl_train_iter(rep_gl_dl):
    for b in rep_gl_dl.train_iter():
        assert isinstance(b, dict)
        assert b["target"]["missing"].notnull().all()

        # stop after first batch
        break


def test_rep_gl_dl_train_iter_gene(rep_gl_dl, gene):
    for b in rep_gl_dl.train_iter(genes=["ENSG00000000003"]):
        assert isinstance(b, dict)
        assert b["target"]["missing"].notnull().all()

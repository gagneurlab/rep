import abc
from typing import Union, Iterable, List

import pandera as pdt
from rep.variant_types import VariantArray, VariantDtype, Variant

try:
    from functools import cached_property
except ImportError:
    # python < 3.8
    from backports.cached_property import cached_property

VEP_CONSEQUENCES = [
    'non_coding_transcript_variant',
    'start_retained_variant',
    'coding_sequence_variant',
    'intergenic_variant',
    '3_prime_UTR_variant',
    'inframe_deletion',
    'NMD_transcript_variant',
    'inframe_insertion',
    'downstream_gene_variant',
    'transcript_ablation',
    'stop_retained_variant',
    'splice_region_variant',
    'splice_donor_variant',
    'missense_variant',
    'stop_gained',
    'splice_acceptor_variant',
    'frameshift_variant',
    'protein_altering_variant',
    'non_coding_transcript_exon_variant',
    'intron_variant',
    'synonymous_variant',
    'incomplete_terminal_codon_variant',
    'stop_lost',
    'mature_miRNA_variant',
    'upstream_gene_variant',
    '5_prime_UTR_variant',
    'start_lost',
]
LOFTEE_FLAGS = [
    'LoF_HC',
    'LoF_LC',
    'LoF_filter_END_TRUNC',
    'LoF_filter_INCOMPLETE_CDS',
    'LoF_filter_EXON_INTRON_UNDEF',
    'LoF_filter_SMALL_INTRON',
    'LoF_filter_ANC_ALLELE',
    'LoF_filter_NON_DONOR_DISRUPTING',
    'LoF_filter_NON_ACCEPTOR_DISRUPTING',
    'LoF_filter_RESCUE_DONOR',
    'LoF_filter_RESCUE_ACCEPTOR',
    'LoF_filter_GC_TO_GT_DONOR',
    'LoF_filter_5UTR_SPLICE',
    'LoF_filter_3UTR_SPLICE',
    'LoF_flag_SINGLE_EXON',
    'LoF_flag_NAGNAG_SITE',
    'LoF_flag_PHYLOCSF_WEAK',
    'LoF_flag_PHYLOCSF_UNLIKELY_ORF',
    'LoF_flag_NON_CAN_SPLICE'
]
VEP_SCORES = [
    'cadd_raw',
    'cadd_phred',
    'polyphen_score',
    'condel_score',
    'sift_score',
    'maxentscan_diff',
]

VEP_VARIABLES = [
    *VEP_CONSEQUENCES,
    *LOFTEE_FLAGS,
    *VEP_SCORES,
]

VEP_DTYPES = {
    **{c: "bool" for c in VEP_CONSEQUENCES},
    **{c: "bool" for c in LOFTEE_FLAGS},
    **{c: "float32" for c in VEP_SCORES},
}

from rep.transformers.dataframe import PandasTransformer


class VEPAnnotation(PandasTransformer, metaclass=abc.ABCMeta):

    @property
    def schema(self) -> pdt.DataFrameSchema:
        return pdt.DataFrameSchema(
            index={
                "gene": pdt.Column(pdt.String),
                "feature": pdt.Column(pdt.String),  # a.k.a. transcript
                "variant": pdt.Column(pdt.Object),
            },
            columns={var: pdt.Column(VEP_DTYPES[var]) for var in VEP_VARIABLES}
        )

    @property
    def consequences(self) -> List[str]:
        return VEP_CONSEQUENCES

    @abc.abstractmethod
    def get_variants(self, gene: Union[str, Iterable[str]]) -> VariantArray:
        """
        Get variants that annotate a certain gene

        :param gene: A (list of) gene(s)
        :return: Variants object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, variants: VariantArray, gene: Union[str, Iterable[str]]):
        """
        Get annotations for a list of variants

        :param variants: Variants object
        :param gene: A (list of) gene(s) to subset for
        :return: Pandas dataframe with the schema as obtainable by self.schema
        """
        raise NotImplementedError

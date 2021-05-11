from __future__ import annotations

import operator
import typing
from typing import (
    TYPE_CHECKING,
    Type,
    Union,
    TypeVar,
    Tuple,
    List,
    NamedTuple,
    Iterable,
    Sequence,
    Dict,
    Mapping
)

import collections
import dataclasses

import re

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    register_extension_dtype,
    ExtensionDtype,
    ExtensionArray,
)
from pandas.core.construction import extract_array

# if TYPE_CHECKING:
import pyarrow as pa

import logging

log = logging.getLogger(__name__)

VariantArrayT = TypeVar("VariantArrayT", bound="VariantArray")

_VCF_SPLIT_PATTERN = re.compile(":|>")
_VARIANT_SPLIT_PATTERN = re.compile(":|-|>")


@dataclasses.dataclass
class Variant:  # (Mapping):
    """
    Dataclass to represent a genomic Variant.

    Attributes:
        - chrom: chromosome name
        - start: variant start (0-based, inclusive)
        - end: variant end (1-based, exclusive)
        - ref: reference sequence
        - alt: alternative sequence

    In addition, can be converted from/to VCF-formatted (chrom, pos, ref, alt) representations.
    """
    __slots__ = "chrom", "start", "end", "ref", "alt"

    chrom: str
    "chromosome name"
    start: int
    "variant start (0-based, inclusive)"
    end: int
    "end: variant end (1-based, exclusive)"
    ref: str
    "reference sequence"
    alt: str
    "alternative sequence"

    def __init__(self, chrom, start, end, ref, alt):
        """
        Create new Variant object.

        :param chrom: chromosome name
        :param start: 0-based, inclusive start position
        :param end: 1-based, exclusive end position
        :param ref: reference sequence
        :param alt: alternative sequence
        """
        self.chrom = str(chrom)
        self.start = int(start) if start is not None else -1
        self.end = int(end) if end is not None else -1
        self.ref = str(ref)
        self.alt = str(alt)

    @classmethod
    def from_vcf(cls, chrom, pos, ref, alt) -> Variant:
        """
        Create new Variant object from 1-based VCF format.

        :param chrom: chromosome name
        :param pos: 1-based position of the variant
        :param ref: reference sequence
        :param alt: alternative sequence
        :return: Variant object
        """
        start = int(pos) - 1
        end = start + len(ref)

        return cls(chrom, start, end, ref, alt)

    @classmethod
    def from_tuple(cls, value: Tuple[str, int, int, str, str]):
        """
        Creates a Variant object from a tuple

        :param value: Tuple
            - chrom: chromosome name
            - start: 0-based, inclusive start of the variant
            - end: 1-based, exclusive end of the variant
            - ref: reference sequence
            - alt: alternative sequence
        """
        return cls(*value)

    @classmethod
    def from_dict(cls, value: Dict[str, Union[str, int]]):
        """
        Creates a Variant object from a dict

        :param value: Dict containing the following keys:
            - chrom: chromosome name
            - start: 0-based, inclusive start of the variant
            - end: 1-based, exclusive end of the variant
            - ref: reference sequence
            - alt: alternative sequence
        """
        return cls(**value)

    def as_tuple(self) -> Tuple[str, int, int, str, str]:
        """
        Return a tuple representation of this object:
            - chrom: chromosome name
            - start: 0-based, inclusive start of the variant
            - end: 1-based, exclusive end of the variant
            - ref: reference sequence
            - alt: alternative sequence
        """
        return (self.chrom, self.start, self.end, self.ref, self.alt)

    def as_vcf_tuple(self) -> Tuple[str, int, str, str]:
        """
        Return a tuple representation of this object in VCF format:
            - chrom: chromosome name
            - pos: 1-based position of the variant
            - ref: reference sequence
            - alt: alternative sequence
        """
        return (self.chrom, self.pos, self.ref, self.alt)

    # def __getitem__(self, key):
    #     return getattr(self, key)
    #
    # def __setitem__(self, key, value):
    #     setattr(self, key, value)
    #
    # def __len__(self):
    #     return len(self.__slots__)
    #
    # def __iter__(self):
    #     for f in self.__slots__:
    #         yield self[f]
    #
    # def keys(self):
    #     return collections.KeysView(self.__slots__)
    #
    # def items(self):
    #     for attribute in self.__slots__:
    #         yield attribute, getattr(self, attribute)

    def __copy__(self):
        return Variant.from_tuple(self.as_tuple())

    # def is_normalized(self) -> bool:
    #
    # def left_normalize(self) -> Variant:

    @property
    def pos(self) -> np.ndarray[np.int32]:
        """
        Return 1-based (i.e. VCF format) position of the variant
        """
        return self.start + 1

    @pos.setter
    def pos(self, value):
        """
        Sets start from an array of 1-based (i.e. VCF format) positions
        :param value: array-like, 1-based variant positions
        """
        self.start = int(value) - 1

    @property
    def length(self) -> int:
        """
        Return the length of this variant.
        """
        return self.end - self.start

    def to_vcf_str(self) -> str:
        """
        Convert variant array to string representation in the form 'chr:pos:ref>alt'
        :return: Pandas series of strings
        """
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"

    @classmethod
    def from_vcf_str(cls, value: str) -> Variant:
        """
        Create variant array from string representation

        :param value: string representation of variant(s) in the form 'chr:pos:ref>alt'
        :return: Variant array
        """
        # split into array of [chr, pos, ref, alt] records
        chrom, pos, ref, alt = re.split(_VCF_SPLIT_PATTERN, value)

        # return new Variant object
        return Variant.from_vcf(chrom, pos, ref, alt)

    def to_str(self) -> str:
        """
        Convert variant array to string representation in the form 'chr:pos:ref>alt'
        :return: Pandas series of strings
        """
        return f"{self.chrom}:{self.start}-{self.end}:{self.ref}>{self.alt}"

    @classmethod
    def from_str(cls, value: str) -> Variant:
        """
        Create variant array from string representation

        :param value: string representation of variant(s) in the form 'chr:pos:ref>alt'
        :return: Variant array
        """
        # split into array of [chr, pos, ref, alt] records
        chrom, start, end, ref, alt = re.split(_VARIANT_SPLIT_PATTERN, value)

        # return new Variant object
        return Variant(chrom, start, end, ref, alt)

    def __repr__(self):
        return f"Variant('{self.to_str()}')"

    def __hash__(self):
        return hash((self.chrom, self.pos, self.ref, self.alt))

    def _cmp_method(self, other: Variant, op):
        return op(self.as_tuple(), other.as_tuple())

    def __eq__(self, other):
        if isinstance(other, Variant):
            return self._cmp_method(other, operator.eq)
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, Variant):
            return self._cmp_method(other, operator.ne)
        else:
            return True

    def __gt__(self, other):
        if isinstance(other, Variant):
            return self._cmp_method(other, operator.gt)
        else:
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")

    def __ge__(self, other):
        if isinstance(other, Variant):
            return self._cmp_method(other, operator.ge)
        else:
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")

    def __lt__(self, other):
        if isinstance(other, Variant):
            return self._cmp_method(other, operator.lt)
        else:
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")

    def __le__(self, other):
        if isinstance(other, Variant):
            return self._cmp_method(other, operator.le)
        else:
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")


@register_extension_dtype
class VariantDtype(ExtensionDtype):
    name = "Variant"

    # TODO: StructDtype.na_value uses None?
    na_value = None
    kind = "O"
    base = np.dtype("O")

    @property
    def type(self) -> Type[Variant]:
        return Variant

    @classmethod
    def construct_array_type(cls) -> Type["VariantArray"]:
        """
        Return the array type associated with this dtype.
        Returns
        -------
        type
        """
        return VariantArray

    def __repr__(self) -> str:
        return "VariantDtype"

    def __from_arrow__(
            self, array: Union["pa.Array", "pa.ChunkedArray"]
    ) -> "VariantArray":
        """
        Construct VariantArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow as pa

        if isinstance(array, pa.Array):
            chunks = [array]
        else:
            # pyarrow.ChunkedArray
            chunks = array.chunks

        results = []
        for arr in chunks:
            chrom = np.asarray(arr.storage.field("chrom"), dtype=object)
            start = np.asarray(arr.storage.field("start"), dtype=np.int32)
            end = np.asarray(arr.storage.field("end"), dtype=np.int32)
            ref = np.asarray(arr.storage.field("ref"), dtype=object)
            alt = np.asarray(arr.storage.field("alt"), dtype=object)

            # set missing values correctly
            mask = ~ np.asarray(arr.is_valid())
            # chrom[mask] = None
            start[mask] = -1
            end[mask] = -1
            # ref[mask] = None
            # alt[mask] = None

            iarr = VariantArray.from_arrays(chrom, start, end, ref, alt)

            results.append(iarr)

        return VariantArray._concat_same_type(results)


class ArrowVariantType(pa.ExtensionType):
    def __init__(self):
        storage_type = pa.struct([
            ('chrom', pa.string()),
            ('start', pa.int32()),
            ('end', pa.int32()),
            ('ref', pa.string()),
            ('alt', pa.string()),
        ])
        pa.ExtensionType.__init__(self, storage_type, "kipoi.variant")

    def __arrow_ext_serialize__(self):
        # metadata = {"subtype": str(self.subtype), "closed": self.closed}
        # return json.dumps(metadata).encode()
        return b'{}'

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        # metadata = json.loads(serialized.decode())
        # subtype = pyarrow.type_for_alias(metadata["subtype"])
        # closed = metadata["closed"]
        return ArrowVariantType()

    def __eq__(self, other):
        if isinstance(other, pa.BaseExtensionType):
            return type(self) == type(other)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def to_pandas_dtype(self):
        return VariantDtype()


# register the type with a dummy instance
_arrow_variant_type = ArrowVariantType()
pa.register_extension_type(_arrow_variant_type)


def _sanitize_chromosome_name(chrom: str):
    if chrom is None:
        return None
    chrom = (re.sub('^chr', '', chrom, flags=re.I)).upper()
    chrom = chrom if not chrom.startswith('M') else 'MT'
    return chrom if re.search('^[0-9]*$|^MT$|^X$|^Y$', chrom) else ''


def _annotate_chr(obj):
    if type(obj) == str:
        return _sanitize_chromosome_name(obj)
    elif type(obj) == int:
        return str(obj) if obj >= 1 and obj <= 22 else ""
    elif type(obj) == list:
        return [_sanitize_chromosome_name(unit) for unit in obj]
    elif type(obj) == np.ndarray:
        return np.array([_sanitize_chromosome_name(unit) for unit in obj])


class VariantArray(ExtensionArray):
    """
    Array representation of a list of Variant objects.

    Internally, stores the variants in a column-based representation.

    Attributes:
        - chrom: array of chromosome names
        - start: array of variant starts (0-based, inclusive)
        - end: array of variant ends (1-based, exclusive)
        - ref: array of reference sequences
        - alt: array of alternative sequences

    In addition, can be converted from/to VCF-formatted (chrom, pos, ref, alt) representations.
    """
    _chrom: np.ndarray[str]
    "chromosome name"
    _start: np.ndarray[np.int32]
    "variant start (0-based, inclusive)"
    _end: np.ndarray[np.int32]
    "end: variant end (1-based, exclusive)"
    _ref: np.ndarray[str]
    "reference sequence"
    _alt: np.ndarray[str]
    "alternative sequence"

    na_value = pd.NA

    def __init__(self, value, verify=True, copy=False):
        if type(value) == type(self):
            # just copy the array references
            self._chrom = value.chrom
            self._start = value.start
            self._end = value.end
            self._ref = value.ref
            self._alt = value.alt
        elif isinstance(value, tuple):
            chrom, start, end, ref, alt = value
            self._chrom = extract_array(chrom)
            self._start = extract_array(start)
            self._end = extract_array(end)
            self._ref = extract_array(ref)
            self._alt = extract_array(alt)
        elif isinstance(value, dict) or isinstance(value, pd.DataFrame):
            self._chrom = extract_array(value["chrom"])
            self._start = extract_array(value["start"])
            self._end = extract_array(value["end"])
            self._ref = extract_array(value["ref"])
            self._alt = extract_array(value["alt"])
        elif self._is_valid_scalar(value):
            chrom, start, end, ref, alt = self._parse_scalar(value)
            self._chrom = np.asanyarray([chrom], dtype=object)
            self._start = np.asanyarray([start], dtype=np.int32)
            self._end = np.asanyarray([end], dtype=np.int32)
            self._ref = np.asanyarray([ref], dtype=object)
            self._alt = np.asanyarray([alt], dtype=object)
        else:
            chrom, start, end, ref, alt = self._parse_listlike(value)
            self._chrom = chrom
            self._start = start
            self._end = end
            self._ref = ref
            self._alt = alt

        if verify:
            try:
                self._validate()
            except Exception as e:
                raise ValueError("Could not validate data") from e

        if copy:
            self._chrom = np.copy(extract_array(self._chrom))
            self._start = np.copy(extract_array(self._start))
            self._end = np.copy(extract_array(self._end))
            self._ref = np.copy(extract_array(self._ref))
            self._alt = np.copy(extract_array(self._alt))

    @staticmethod
    def _validate_dtypes(chrom, start, end, ref, alt) -> Tuple[
        np.ndarray[object],
        np.ndarray[np.int32],
        np.ndarray[np.int32],
        np.ndarray[object],
        np.ndarray[object]
    ]:
        """
        ensure correct array types
        """
        if np.asanyarray(chrom).dtype.kind not in {'O'}:
            chrom = np.asanyarray(chrom, dtype=object)
        if np.asanyarray(start).dtype != np.int32:
            start = np.asanyarray(start, dtype=np.int32)
        if np.asanyarray(end).dtype != np.int32:
            end = np.asanyarray(end, dtype=np.int32)
        if np.asanyarray(ref).dtype.kind not in {'O'}:
            ref = np.asanyarray(ref, dtype=object)
        if np.asanyarray(alt).dtype.kind not in {'O'}:
            alt = np.asanyarray(alt, dtype=object)

        # make sure that all arrays have the same shape
        chrom, start, end, ref, alt = np.broadcast_arrays(
            chrom, start, end, ref, alt)

        if np.shape(chrom) == ():
            chrom = np.expand_dims(chrom, axis=0)
            start = np.expand_dims(start, axis=0)
            end = np.expand_dims(end, axis=0)
            ref = np.expand_dims(ref, axis=0)
            alt = np.expand_dims(alt, axis=0)

        return chrom, start, end, ref, alt

    def _validate(self):
        """
        ensure correct array types for this object
        """
        self._chrom, self._start, self._end, self._ref, self._alt = self._validate_dtypes(
            self._chrom, self._start, self._end, self._ref, self._alt)

    def sanitize(
            self,
            annotate_chromosomes=True,
            remove_inconsistent_variants=True,
            remove_unspecified_ref_or_alt=True,
            inplace=False
    ) -> VariantArrayT:
        """
        check and clean variant input

        :param annotate_chromosomes:
        :param remove_inconsistent_variants:
        :param remove_unspecified_ref_or_alt:
        :param inplace:
        :return:
        """

        if inplace:
            retval = self
        else:
            retval = self.copy()

        if annotate_chromosomes:
            retval.chrom = [_annotate_chr(c) for c in retval.chrom]
        retval.ref = retval.ref.str.upper()
        retval.alt = retval.alt.str.upper()

        # remove Variants on non standard chromosomes. (They are set to an empty string by annotated_chr())
        standard_chr = retval.chrom.str.len() == 0
        if remove_inconsistent_variants:
            if np.any(standard_chr):
                log.warning("%d variants were removed because they had a non standard chr name.", np.sum(standard_chr))
                retval._filter(~ standard_chr, inplace=True)
        else:
            if np.any(standard_chr):
                raise ValueError(
                    "variants have a non standard chr name. "
                    "Please change chr name or turn ignore_inconsistent_variants parameter to true "
                    "to remove variants automatically."
                )

        if remove_unspecified_ref_or_alt:
            # remove variants with unspecified ref or alt field.
            unspecified_alt_or_ref = np.logical_or(np.isin(retval.ref, ["*", "."]), np.isin(retval.alt, ["*", "."]))
            if np.any(unspecified_alt_or_ref):
                log.warning("%d variants were removed because their alt or ref was not specified",
                            np.sum(unspecified_alt_or_ref))
                retval._filter(~ unspecified_alt_or_ref, inplace=True)

        # # remove variants with ref or alt, which contain N.
        # N_alt_or_ref = ~ np.logical_or(df['ref'].str.contains('N'), df['alt'].str.contains('N'))
        # if np.any(N_alt_or_ref == False):
        #     log.warning("%d variants were removed because their alt or ref contain 'N'", np.sum(N_alt_or_ref == False))
        #     df = df[N_alt_or_ref]

        ## remove variants with ref or alt, which contain N.
        # N_alt_or_ref = ~ np.logical_or(df['ref'].apply(len) > 50, df['alt'].apply(len) > 50)
        # if np.any(N_alt_or_ref == False):
        #    warnings.warn(
        #        "{} variants were removed because their alt or ref has more than 50 bp".format(np.sum(N_alt_or_ref == False)))
        #    df = df[N_alt_or_ref]

        # drop any varians where alt == ref
        same_ref_alt = retval.ref == retval.alt
        if np.any(same_ref_alt):
            log.warning("%d variants were removed because their alt or ref was not specified",
                        np.sum(same_ref_alt))
            retval._filter(same_ref_alt, inplace=True)

        return retval

    def _filter(self, cond, inplace=False):
        if inplace:
            retval = self
        else:
            retval = self.copy()

        retval._chrom = retval.chrom[cond]
        retval._start = retval.start[cond]
        retval._end = retval.end[cond]
        retval._ref = retval.ref[cond]
        retval._alt = retval.alt[cond]

        if not inplace and np.shape(retval._chrom) == ():
            if pd.isna(retval._chrom):
                return self.na_value
            else:
                return Variant(retval._chrom, retval._start, retval._end, retval._ref, retval._alt)
        else:
            # ensure correct shapes
            retval._validate()
            return retval

    @classmethod
    def from_vcf_str(cls, value: Union[str, Iterable[str]]) -> VariantArrayT:
        """
        Create variant array from string representation

        :param value: string representation of variant(s) in the form 'chr:pos:ref>alt'
        :return: Variant array
        """
        if np.ndim(value) < 1:
            value = [value]

        # split into array of [chr, pos, ref, alt] records
        split_series = pd.Series(value).str.split(":|>")
        split_series.iloc[pd.isna(split_series)] = pd.Series([(None, 0, None, None)])

        # convert splitted records into VCF-like dataframe
        vcf_df = pd.DataFrame.from_records(
            split_series,
            columns=["chrom", "pos", "ref", "alt"],
        ).astype({
            "pos": int
        })

        # return new Variant object
        return VariantArray.from_vcf_df(vcf_df)

    def to_vcf_str(self) -> pd.Series[str]:
        """
        Convert variant array to string representation in the form 'chr:pos:ref>alt'
        :return: Pandas series of strings
        """
        return (
                self.chrom +
                ":" +
                self.pos.astype(str) +
                ":" +
                self.ref +
                ">" +
                self.alt
        ).fillna(pd.NA)

    @classmethod
    def from_vcf_str(cls, value: Union[str, Iterable[str]]) -> VariantArrayT:
        """
        Create variant array from string representation

        :param value: string representation of variant(s) in the form 'chr:pos:ref>alt'
        :return: Variant array
        """
        if np.ndim(value) < 1:
            value = [value]

        # split into VCF-like dataframe of [chr, pos, ref, alt]
        vcf_df = (
            pd.Series(value, dtype=pd.StringDtype())
                .str.split(_VCF_SPLIT_PATTERN.pattern, expand=True)
        )
        vcf_df.columns = ["chrom", "pos", "ref", "alt"]
        vcf_df["pos"] = vcf_df["pos"].astype(pd.Int32Dtype()).fillna(0)

        # return new Variant object
        return VariantArray.from_vcf_df(vcf_df)

    def to_str(self) -> pd.Series[str]:
        """
        Convert variant array to string representation in the form 'chr:pos:ref>alt'
        :return: Pandas series of strings
        """
        return (
                self.chrom +
                ":" +
                self.start.astype(str) +
                "-" +
                self.end.astype(str) +
                ":" +
                self.ref +
                ">" +
                self.alt
        ).fillna(pd.NA)

    @classmethod
    def from_str(cls, value: Union[str, Iterable[str]]) -> VariantArrayT:
        """
        Create variant array from string representation

        :param value: string representation of variant(s) in the form 'chr:pos:ref>alt'
        :return: Variant array
        """
        if np.ndim(value) < 1:
            value = [value]

        # split into dataframe of [chr, start, end, ref, alt]
        variant_df = (
            pd.Series(value, dtype=pd.StringDtype())
                .str.split(_VARIANT_SPLIT_PATTERN.pattern, expand=True)
        )
        variant_df.columns = ["chrom", "start", "end", "ref", "alt"]
        variant_df["start"] = variant_df["start"].astype(pd.Int32Dtype()).fillna(-1)
        variant_df["end"] = variant_df["end"].astype(pd.Int32Dtype()).fillna(-1)

        # return new Variant object
        return VariantArray.from_df(variant_df)

    @property
    def chrom(self) -> pd.Index[pd.StringDtype]:
        return pd.Index(self._chrom, dtype=pd.StringDtype())

    @chrom.setter
    def chrom(self, value):
        try:
            np.broadcast_to(value, np.shape(self._chrom))
        except ValueError as e:
            raise ValueError("Invalid array: could not broadcast to desired shape") from e
        self._chrom = np.asanyarray(value, dtype=object)

    @property
    def start(self) -> pd.Index[pd.Int32Dtype]:
        """
        Return 0-based position of the variant
        """
        return pd.Index(self._start)

    @start.setter
    def start(self, value):
        """
        Sets 0-based start of variant
        :param value: array-like, 0-based variant start
        """
        try:
            np.broadcast_to(value, np.shape(self._chrom))
        except ValueError as e:
            raise ValueError("Invalid array: could not broadcast to desired shape") from e
        self._start = np.asanyarray(value, dtype="int32")

    @property
    def pos(self) -> pd.Index[pd.Int32Dtype]:
        """
        Return 1-based (i.e. VCF format) position of the variant
        """
        return pd.Index(self._start + 1)

    @pos.setter
    def pos(self, value):
        """
        Sets start from an array of 1-based (i.e. VCF format) positions
        :param value: array-like, 1-based variant positions
        """
        self._start = np.asanyarray(value, dtype="int32") - 1

    @property
    def end(self) -> pd.Index[pd.Int32Dtype]:
        return pd.Index(self._end)

    @end.setter
    def end(self, value):
        try:
            np.broadcast_to(value, np.shape(self._chrom))
        except ValueError as e:
            raise ValueError("Invalid array: could not broadcast to desired shape") from e
        self._end = np.asanyarray(value, dtype="int32")

    @property
    def ref(self) -> pd.Index[pd.StringDtype]:
        return pd.Index(self._ref, dtype=pd.StringDtype())

    @ref.setter
    def ref(self, value):
        try:
            np.broadcast_to(value, np.shape(self._chrom))
        except ValueError as e:
            raise ValueError("Invalid array: could not broadcast to desired shape") from e
        self._ref = np.asanyarray(value, dtype=object)

    @property
    def alt(self) -> pd.Index[pd.StringDtype]:
        return pd.Index(self._alt, dtype=pd.StringDtype())

    @alt.setter
    def alt(self, value):
        try:
            np.broadcast_to(value, np.shape(self._chrom))
        except ValueError as e:
            raise ValueError("Invalid array: could not broadcast to desired shape") from e
        self._alt = np.asanyarray(value, dtype=object)

    @classmethod
    def _parse_listlike(cls, value: Iterable[Variant]):
        # list-like of variants
        value = list(value)
        length = len(value)
        chrom = np.empty(length, dtype=object)
        start = np.empty(length, dtype="int32")
        end = np.empty(length, dtype="int32")
        ref = np.empty(length, dtype=object)
        alt = np.empty(length, dtype=object)

        for idx, val in enumerate(value):
            val = cls._parse_scalar(val)
            chrom[idx] = val[0]
            start[idx] = val[1]
            end[idx] = val[2]
            ref[idx] = val[3]
            alt[idx] = val[4]

        return chrom, start, end, ref, alt

    @classmethod
    def _parse_scalar(cls, value) -> Tuple[str, int, int, str, str]:
        if isinstance(value, Variant):
            return value.as_tuple()
        elif isinstance(value, tuple):
            chrom, start, end, ref, alt = value
        elif isinstance(value, dict):
            chrom = value["chrom"]
            start = value["start"]
            end = value["end"]
            ref = value["ref"]
            alt = value["alt"]
        elif isinstance(value, str):
            return Variant.from_str(value).as_tuple()
        elif pd.isna(value):
            chrom = None
            start = -1
            end = -1
            ref = None
            alt = None
        else:
            raise TypeError(
                "can only insert Variant-like objects into a VariantArray"
            )

        # validate dtypes
        start = int(start)
        end = int(end)

        return chrom, start, end, ref, alt

    @classmethod
    def _parse_fill_value(cls, value) -> Tuple[str, int, int, str, str]:
        return cls._parse_scalar(value)

    @classmethod
    def _parse_setitem_value(cls, value):
        if cls._is_valid_scalar(value):
            return cls._parse_scalar(value)
        else:
            return cls._parse_listlike(value)

    @staticmethod
    def _is_valid_scalar(value):
        return (
                isinstance(value, tuple) or
                isinstance(value, dict) or
                isinstance(value, Variant)
        )

    @classmethod
    def from_arrays(
            cls: type[VariantArrayT],
            chrom,
            start,
            end,
            ref,
            alt,
            verify=True,
            copy: bool = False,
    ) -> VariantArrayT:
        val = (chrom, start, end, ref, alt)
        return cls(value=val, verify=verify, copy=copy)

    @classmethod
    def from_tuples(
            cls: type[VariantArrayT],
            data,
            verify=True,
            copy: bool = False,
    ) -> VariantArrayT:
        if len(data) == 0:
            chrom = np.asarray([], dtype=object)
            start = np.asarray([], dtype=np.int32)
            end = np.asarray([], dtype=np.int32)
            ref = np.asarray([], dtype=object)
            alt = np.asarray([], dtype=object)
        elif cls._is_valid_scalar(data):
            chrom, start, end, ref, alt = cls._parse_scalar(data)
        else:
            df = pd.DataFrame.from_records(data, columns=["chrom", "start", "end", "ref", "alt"])
            chrom = np.asarray(df["chrom"], dtype=object)
            start = np.asarray(df["start"], dtype=np.int32)
            end = np.asarray(df["end"], dtype=np.int32)
            ref = np.asarray(df["ref"], dtype=object)
            alt = np.asarray(df["alt"], dtype=object)

        return cls.from_arrays(chrom, start, end, ref, alt, verify=verify, copy=copy)

    @classmethod
    def from_vcf_df(cls, df: pd.DataFrame) -> VariantArrayT:
        """
        Creates a new VariantArray from VCF format.

        :param df: Pandas dataframe with the following columns:
            - chrom: chromosome name
            - pos: variant position (1-based, inclusive)
            - ref: reference sequence
            - alt: alternative sequence
        :return: VariantArray
        """
        expected_cols = [
            "chrom",
            "pos",
            "ref",
            "alt",
        ]
        if not pd.Series(expected_cols).isin(df.columns).all():
            raise ValueError("Missing columns in passed dataframe")

        return cls(
            {
                "chrom": df["chrom"],
                "start": df["pos"] - 1,
                "end": df["pos"] + df["ref"].str.len().fillna(0).astype("int32") - 1,
                "ref": df["ref"],
                "alt": df["alt"],
            },
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> VariantArrayT:
        """
        Creates a new VariantArray from a DataFrame.

        :param df: Pandas dataframe with the following columns:
            - chrom: chromosome name
            - start: variant start (0-based, inclusive)
            - end: variant end (1-based, exclusive)
            - ref: reference sequence
            - alt: alternative sequence
        :return: VariantArray
        """
        cols = df.columns

        expected_cols = ["chrom", "start", "end", "ref", "alt"]
        if np.all(np.isin(expected_cols, cols)):
            return cls.from_arrays(
                chrom=df["chrom"],
                start=df["start"],
                end=df["end"],
                ref=df["ref"],
                alt=df["alt"],
            )

        # check if in VCF format
        expected_cols = ["chrom", "pos", "ref", "alt"]
        if np.all(np.isin(expected_cols, cols)):
            return cls.from_vcf_df(df)

        raise ValueError("Could not create VariantArray from dataframe")

    def as_frame(self):
        return pd.DataFrame({
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "ref": self.ref,
            "alt": self.alt,
        })

    @property
    def length(self):
        """
        Return an Index with entries denoting the length of each Variant in the VariantArray.
        """
        return self.end - self.start

    # ---------------------------------------------------------------------
    # ExtensionArray interface

    @classmethod
    def _concat_same_type(
            cls: Type[VariantArrayT], to_concat: Sequence[VariantArrayT]
    ) -> VariantArrayT:
        """
        Concatenate multiple arrays of this dtype.

        Parameters
        ----------
        to_concat: sequence of this type

        Returns
        -------
        ExtensionArray
        """
        chrom = np.concatenate([v._chrom for v in to_concat])
        start = np.concatenate([v._start for v in to_concat])
        end = np.concatenate([v._end for v in to_concat])
        ref = np.concatenate([v._ref for v in to_concat])
        alt = np.concatenate([v._alt for v in to_concat])

        return cls.from_arrays(chrom, start, end, ref, alt)

    @classmethod
    def _from_sequence(
            cls: type[VariantArrayT],
            scalars,
            *,
            dtype=None,
            copy: bool = False,
    ) -> VariantArrayT:
        """
        Construct a newExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars: Sequence

            Each element will be an instance of the scalar type for this array, ``cls.dtype.type``.
        dtype: dtype, optional

            Construct for this particular dtype.This should be a Dtype compatible with the ExtensionArray.
        copy: boolean, default False

            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        val = cls._parse_listlike(scalars)
        return cls.from_arrays(*val)

    def _values_for_factorize(self):
        # type: () -> Tuple[np.ndarray, Any]
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray

            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinel` and not included in `uniques`. By default,
            ``np.nan`` is used.

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`.
        """
        return np.asarray(self), self.na_value

    @classmethod
    def _from_factorized(
            cls: type[VariantArrayT], values: np.ndarray, original: VariantArrayT
    ) -> VariantArrayT:
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.
        ExtensionArray.factorize : Encode the extension array as an enumerated type.
        """
        return cls(values)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        return cls.from_str(strings)

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Return the VariantArray's data as a numpy array of Variant objects (with dtype='object')
        """
        mask = self.isna()

        result = np.empty(len(self._chrom), dtype=object)
        for i in range(len(self._chrom)):
            if mask[i]:
                result[i] = pd.NA
            else:
                result[i] = Variant(
                    self._chrom[i],
                    self._start[i],
                    self._end[i],
                    self._ref[i],
                    self._alt[i],
                )

        if dtype != None:
            result = result.astype(dtype)
        return result

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        variant_type = ArrowVariantType()
        storage_array = pa.StructArray.from_arrays(
            arrays=[
                pa.array(self._chrom, type=pa.string(), from_pandas=True),
                pa.array(self._start, type=pa.int32(), from_pandas=True),
                pa.array(self._end, type=pa.int32(), from_pandas=True),
                pa.array(self._ref, type=pa.string(), from_pandas=True),
                pa.array(self._alt, type=pa.string(), from_pandas=True),
            ],
            names=["chrom", "start", "end", "ref", "alt"],
        )
        mask = self.isna()
        if mask.any():
            # if there are missing values, set validity bitmap also on the array level
            null_bitmap = pa.array(~mask).buffers()[1]
            storage_array = pa.StructArray.from_buffers(
                storage_array.type,
                len(storage_array),
                [null_bitmap],
                children=[storage_array.field(i) for i in range(5)],
            )

        if type is not None:
            if type.equals(variant_type.storage_type):
                return storage_array
            else:
                raise TypeError(
                    f"Not supported to convert VariantArray to '{type}' type"
                )

        return pa.ExtensionArray.from_storage(variant_type, storage_array)

    # ---------------------------------------------------------------------
    # Descriptive
    def copy(self: VariantArrayT) -> VariantArrayT:
        """
        Return a copy of the array.
        """
        return VariantArray(self, copy=True)

    @property
    def dtype(self) -> VariantDtype:
        return VariantDtype()

    @property
    def nbytes(self) -> int:
        return (
                self.chrom.nbytes +
                self.start.nbytes +
                self.end.nbytes +
                self.ref.nbytes +
                self.alt.nbytes
        )

    @property
    def size(self) -> int:
        # Avoid materializing self.values
        return self._chrom.size

    def __iter__(self):
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return len(self._chrom)

    def __getitem__(self, key):
        key = pd.api.indexers.check_array_indexer(self, key)
        return self._filter(key)

    def __setitem__(self, key, value):
        chrom, start, end, ref, alt = self._parse_setitem_value(value)

        key = pd.api.indexers.check_array_indexer(self, key)

        self._chrom[key] = chrom
        self._start[key] = start
        self._end[key] = end
        self._ref[key] = ref
        self._alt[key] = alt

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif self._is_valid_scalar(fill_value):
                fill_value = self._parse_scalar(fill_value)
            else:
                raise TypeError("provide Variant-like as fill value")

        chrom = take(self.chrom, indices, allow_fill=allow_fill, fill_value=fill_value)
        start = take(self.start, indices, allow_fill=allow_fill, fill_value=fill_value)
        end = take(self.end, indices, allow_fill=allow_fill, fill_value=fill_value)
        ref = take(self.ref, indices, allow_fill=allow_fill, fill_value=fill_value)
        alt = take(self.alt, indices, allow_fill=allow_fill, fill_value=fill_value)
        result = VariantArray.from_arrays(chrom, start, end, ref, alt)

        if allow_fill and fill_value is None:
            result[pd.isna(result)] = None

        return result

    def isna(self):
        """
        Returns boolean NumPy array indicating if eachvalue is missing
        """
        retval = pd.isna(self._chrom)
        return retval

    def unique(self):
        """
        Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques: ExtensionArray
        """
        uniques = VariantArray(self.as_frame().drop_duplicates())

        # TODO: test alternatives, e.g.:
        # factors, uniques = pd.factorize(self)
        # if np.any(factors < 0):
        #     uniques = self._concat_same_type([uniques, VariantArray._from_sequence([self.na_value])])

        return uniques

    def astype(self, dtype, copy=True):
        dtype = pd.api.types.pandas_dtype(dtype)

        if pd.api.types.is_string_dtype(dtype):
            return pd.array(self.to_str(), dtype=dtype)
        elif dtype == VariantDtype or dtype == Variant:
            if copy:
                return self.copy()
            else:
                return self
        else:
            raise TypeError(f"Cannot cast VariantDtype to {dtype}")
            # return super().astype(dtype, copy)

    def min(self, axis=None, skipna: bool = True, **kwargs) -> Variant:
        raise NotImplementedError()

    def max(self, axis=None, skipna: bool = True, **kwargs) -> Variant:
        raise NotImplementedError()

    def value_counts(self, dropna=False):
        return pd.value_counts(np.asarray(self), dropna=dropna).astype("Int64")

    def _cmp_method(self, other, op, fail_on_missing=False):
        # ensure pandas array for list-like and eliminate non-variant scalars
        if pd.api.types.is_list_like(other):
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")
            other = pd.array(other)
        else:
            if not isinstance(other, Variant):
                # non-variant scalar -> no matches
                return TypeError(f"Invalid comparison: {other} using {op}")

        # determine the dtype of the elements we want to compare
        if isinstance(other, Variant):
            other_dtype = pd.api.types.pandas_dtype("Variant")
        elif not pd.api.types.is_categorical_dtype(other.dtype):
            other_dtype = other.dtype
        else:
            # for categorical defer to categories for dtype
            other_dtype = other.categories.dtype

            other = other.categories.take(
                other.codes, allow_fill=True, fill_value=other.categories._na_value
            )

        # variant-like -> need same closed and matching endpoints
        if isinstance(other_dtype, VariantDtype):
            if op is operator.eq:
                return (
                        (self._chrom == other.chrom) &
                        (self._start == other.start) &
                        (self._end == other.end) &
                        (self._ref == other.ref) &
                        (self._alt == other.alt) |
                        (pd.isna(self) == pd.isna(other))  # return true if both values are NA
                )
            elif op is operator.ne:
                return (
                        (self._chrom != other.chrom) |
                        (self._start != other.start) |
                        (self._end != other.end) |
                        (self._ref != other.ref) |
                        (self._alt != other.alt) &
                        ~ (pd.isna(self) == pd.isna(other))  # return false if both values are NA
                )
            elif op in {
                operator.gt,
                operator.ge,
                operator.lt,
                operator.le,
            }:
                equal = np.stack([
                    (self._chrom != other.chrom),
                    (self._start != other.start),
                    (self._end != other.end),
                    (self._ref != other.ref),
                    (self._alt != other.alt),
                    # ~ (pd.isna(self) == pd.isna(other)), # return false if both values are NA
                ])
                first_false_value = np.argmin(equal, axis=0)
                axis_idx = np.arange(len(first_false_value))

                comp = np.stack([
                    op(self._chrom, other.chrom),
                    op(self._start, other.start),
                    op(self._end, other.end),
                    op(self._ref, other.ref),
                    op(self._alt, other.alt),
                    # ~ (pd.isna(self) == pd.isna(other)), # return false if both values are NA
                ])
                retval = comp[first_false_value, axis_idx]
                # all_equal = equal[first_false_value, axis_idx]

                any_of_both_missing = (pd.isna(self) | pd.isna(other))
                if fail_on_missing:
                    if np.any(any_of_both_missing):
                        raise TypeError("boolean value of NA is ambiguous")
                return pd.arrays.BooleanArray(retval, mask=any_of_both_missing)
            else:
                raise ValueError(f"Unknown op {op}")
        else:
            raise ValueError(f"Unknown type of other: {other_dtype}")

    _arith_method = _cmp_method

    # @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    # @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    # @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    # @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    # @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    # @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    # ---------------------------------------------------------------------
    # Rendering Methods

    def _format_data(self) -> str:
        n = len(self)

        max_seq_items = min((pd.get_option("display.max_seq_items") or n) // 10, 10)

        formatter = str

        if n == 0:
            summary = "[]"
        elif n == 1:
            first = formatter(self[0])
            summary = f"[{first}]"
        elif n == 2:
            first = formatter(self[0])
            last = formatter(self[-1])
            summary = f"[{first}, {last}]"
        else:

            if n > max_seq_items:
                n = min(max_seq_items // 2, 10)
                head = [formatter(x) for x in self[:n]]
                tail = [formatter(x) for x in self[-n:]]
                head_str = ", ".join(head)
                tail_str = ", ".join(tail)
                summary = f"[{head_str} ... {tail_str}]"
            else:
                tail = [formatter(x) for x in self]
                tail_str = ", ".join(tail)
                summary = f"[{tail_str}]"

        return summary

    def __repr__(self) -> str:
        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        data = self._format_data()
        class_name = f"<{type(self).__name__}>\n"

        template = f"{class_name}{data}\nLength: {len(self)}, dtype: {self.dtype}"
        return template

    def _format_space(self) -> str:
        space = " " * (len(type(self).__name__) + 1)

        return f"\n{space}"

    def __str__(self):
        return self.__repr__()

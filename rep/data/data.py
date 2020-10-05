from __future__ import annotations

from typing import Dict, Set, List
import abc

import os
import sys

import numpy as np
import pandas as pd

import desmi
import kipoiseq
import pyranges as pr


class BaseGenotype(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sel(
            self,
            variant: desmi.objects.Variant = None,
            genomic_range: pr.PyRanges = None,
            sample=None,
            variables=None
    ) -> BaseGenotype:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def samples(self) -> pd.Index:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def variants(self) -> desmi.objects.Variant:
        raise NotImplementedError


class DesmiGenotype(BaseGenotype):
    gt: desmi.genotype.Genotype
    selectors: Dict[str, object]

    def __init__(self, gt, selectors=None):
        self.gt = gt
        if selectors is None:
            self.selectors = {}
        else:
            self.selectors = selectors

    @property
    def variant(self):
        return self.selectors.get("variant", None)

    @property
    def genomic_range(self):
        return self.selectors.get("genomic_range", None)

    @property
    def samples(self):
        return self.selectors.get("samples", self.gt.samples())

    @property
    def variables(self):
        return self.selectors.get("variables", pd.Index(["GT", "GQ", "DP"]))

    def sel(
            self,
            variant: desmi.objects.Variant = None,
            genomic_range: pr.PyRanges = None,
            sample=None,
            variables=None
    ) -> BaseGenotype:
        if self.selectors.get("variant", None) is not None and variant is not None:
            raise NotImplementedError("Joining Variant objects not implemented yet")

        variant = self.selectors.get("variant", variant)
        variant: pr.PyRanges = variant.to_pyranges()

        if self.selectors.get("genomic_range") is not None:
            genomic_range = genomic_range.intersect(self.selectors.get("genomic_range"))


class AbstractXBag(meta=abc.ABCMeta):
    _dims: Set[str]

    def __init__(self, data, dims):
        self._data = data
        self._dims = dims

    @abc.abstractmethod
    def sel(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def isel(self, **kwargs):
        raise NotImplementedError()

    def _check_dims(self, kwargs_dict: Dict[str, object]):
        for k in kwargs_dict.keys():
            if k not in self._dims:
                raise ValueError("Unknown dimension: %s" % k)

    def fetch(self):
        raise NotImplementedError()

    @property
    def dims(self):
        return self._dims

    @property
    def data(self):
        return self._data

    @property
    def index(self):


class PandasXBag(AbstractXBag):

    def __init__(self, data: pd.DataFrame, index_cols: List[str]):
        for c in index_cols:
            if c not in data.columns:
                raise ValueError("Column could not be found in dataframe: %s" % c)

        self._index_cols = index_cols
        super().__init__(data, ["columns", *index_cols])

    def sel(self, **kwargs):
        self._check_dims(kwargs)


if __name__ == "__main__":
    df = pd.DataFrame({
        "x": [2, 3, 1, 4, 5],
        "y": ["asdf", "d", "ö", "l", "k"],
    })
    PandasXBag(df)

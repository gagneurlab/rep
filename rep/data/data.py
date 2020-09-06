from typing import Dict, Set, List
import abc

import os
import sys

import pandas as pd


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

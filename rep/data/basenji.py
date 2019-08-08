import os
import sys

import collections
import random
import math
import numpy.random as nr
import numpy as np
import pandas as pd

import xarray as xr
import dask.array as da
import zarr

# TODO: VERY!! bad practice
BASENJI_BASE_DIR = "/s/project/rep/predictions/basenji/"
BASENJI_DIMS = ("observations", "genes", "alleles", "genePositions", "basenjiFeatures")


def _zarr_to_xarray(zarr_da, basenji_dims=BASENJI_DIMS):
    xrda = xr.DataArray(da.from_zarr(zarr_da), dims=basenji_dims)

    dim = np.empty(xrda.sizes["observations"], dtype=object)
    for k, v in zarr_da.attrs["samples"].items():
        dim[v] = k
    xrda.coords["observations"] = dim

    dim = np.empty(xrda.sizes["genes"], dtype=object)
    for k, v in zarr_da.attrs["genes"].items():
        dim[v] = k
    xrda.coords["genes"] = dim

    dim = np.empty(xrda.sizes["genePositions"], dtype=int)
    for k, v in zarr_da.attrs["genePositions"].items():
        dim[v] = int(k)
    xrda.coords["genePositions"] = dim

    return xrda


def basenji(data_name, dims=BASENJI_DIMS):
    path = os.path.join(BASENJI_BASE_DIR, data_name)

    xrds = list()
    for d in sorted([f.path for f in os.scandir(path) if f.is_dir()]):
        print(d)
        xrds.append(_zarr_to_xarray(zarr.open(d), basenji_dims=dims))

    xrds = xr.concat(xrds, dim="genes")
    return xrds



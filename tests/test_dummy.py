from typing import Union, List, Tuple, Dict

import pytest

import xarray as xr
import numpy as np
from xarray import DataArray


def concat_axis(
        darrs: Union[List[xr.DataArray], Tuple[xr.DataArray]],
        dims: Union[List[str], Tuple[str]],
        axis: int = None,
        **kwargs
):
    """
    Concat arrays along some axis similar to `np.concatenate`. Automatically renames the dimensions to `dims`.
    Please note that this renaming happens by the axis position, therefore make sure to transpose all arrays
     to the correct dimension order.

    :param darrs: List or tuple of xr.DataArrays
    :param dims: The dimension names of the resulting array. Renames axes where necessary.
    :param axis: The axis which should be concatenated along
    :param kwargs: Additional arguments which will be passed to `xr.concat()`
    :return: Concatenated xr.DataArray with dimensions `dim`.
    """

    # get depth of nested lists
    if axis is None:
        axis = 0
        l = darrs
        # while l is a list or tuple and contains elements:
        while isinstance(l, List) or isinstance(l, Tuple) and l:
            # increase depth by one
            axis -= 1
            l = l[0]
        if axis == 0:
            raise ValueError("`darrs` has to be a (possibly nested) list or tuple of xr.DataArrays!")

    # align arrays by force-renaming dimensions
    to_concat = list()
    for i, da in enumerate(darrs):
        # recursive call for nested arrays;
        # most inner call should have axis = -1,
        # most outer call should have axis = - depth_of_darrs
        if isinstance(da, list) or isinstance(da, tuple):
            da = concat_axis(da, dims=dims, axis=axis + 1, **kwargs)

        if not isinstance(da, xr.DataArray):
            raise ValueError("Input %d must be a xr.DataArray" % i)
        if len(da.dims) != len(dims):
            raise ValueError("Input %d must have the same number of dimensions as specified in the `dims` argument!")
        da = da.rename(dict(zip(da.dims, dims)))
        to_concat.append(da)

    return xr.concat(to_concat, dim=dims[axis], **kwargs)

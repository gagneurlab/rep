from typing import List, Dict, Tuple

import xarray as xr
import numpy as np
import dask.array


# import pandas as pd
# def core_dim_locs_from_multiindex(multi_index, coords: Dict[str, pd.Index], new_dim_name, core_dims=None) -> List[
#     Tuple[str, xr.DataArray]]:
#     if core_dims is None:
#         core_dims = np.asarray(multi_index.names)
#
#     core_dim_locs = []
#     for dim in core_dims:
#         core_dim_locs.append(
#             pd.Index(coords[dim]).get_indexer(multi_index.get_level_values(dim))
#         )
#
#     core_dim_locs_xr = []
#     for i, dim in enumerate(core_dims):
#         labels = multi_index.get_level_values(dim)
#         locs = pd.Index(coords[dim]).get_indexer(labels)
#         core_dim_locs_xr.append((
#             dim,
#             xr.DataArray(
#                 locs,
#                 coords={
#                     dim: ((new_dim_name,), labels)
#                 },
#                 dims=(new_dim_name,),
#                 name=dim
#             )
#         ))
#     return core_dim_locs_xr


def core_dim_locs_from_cond(cond, new_dim_name, core_dims=None) -> List[Tuple[str, xr.DataArray]]:
    if core_dims is None:
        core_dims = cond.dims

    core_dim_locs = np.argwhere(cond.data)
    if isinstance(core_dim_locs, dask.array.core.Array):
        core_dim_locs = core_dim_locs.persist().compute_chunk_sizes()

    core_dim_locs_xr = []
    for i, dim in enumerate(core_dims):
        locs = core_dim_locs[:, i]
        labels = np.asanyarray(cond[dim])
        core_dim_locs_xr.append((
            dim,
            xr.DataArray(
                locs,
                coords={
                    dim: ((new_dim_name,), dask.array.asanyarray(labels)[locs])
                },
                dims=(new_dim_name,),
                name=dim
            )
        ))
    return core_dim_locs_xr


def subset_variable(variable: xr.DataArray, core_dim_locs, new_dim_name, mask):
    core_dims = np.array([dim for dim, locs in core_dim_locs])

    variable, mask_bcast = xr.broadcast(variable, mask)
    variable = variable.transpose(*core_dims, ...)

    other_dims = np.asarray(variable.dims)
    other_dims = other_dims[~np.isin(variable.dims, core_dims)]

    flattened_mask = mask.data.flatten()
    flattened_variable_data = dask.array.reshape(
        variable.data,
        shape=[*flattened_mask.shape, *[variable.sizes[d] for d in other_dims]]
    )

    subset = flattened_variable_data[flattened_mask]

    # force-set chunk size from known chunks
    chunk_sizes = core_dim_locs[0][1].chunks[0]
    subset._chunks = (chunk_sizes, *subset._chunks[1:])

    subset_xr = xr.DataArray(subset, dims=(new_dim_name, *other_dims), coords={
        **{dim: idx.coords[dim] for dim, idx in core_dim_locs},
        **{dim: variable[dim] for dim in other_dims},
    })

    return subset_xr


def dataset_masked_indexing(ds: xr.Dataset, mask: xr.DataArray, new_dim_name: str):
    mask.data = dask.array.asanyarray(mask.data)
    core_dim_locs = core_dim_locs_from_cond(mask, new_dim_name=new_dim_name)
    core_dims = np.array([dim for dim, locs in core_dim_locs])

    new_variables = {}
    for name, variable in ds.items():
        if np.any(np.isin(variable.dims, core_dims)):
            variable = subset_variable(variable, core_dim_locs, new_dim_name=new_dim_name, mask=mask)
        new_variables[name] = variable

    return xr.Dataset(new_variables)


def test_array_indexing():
    test_ds = xr.Dataset({
        "x": xr.DataArray(dask.array.random.randint(0, 1000, size=[100, 100, 100])),
        "y": xr.DataArray(dask.array.random.randint(0, 1000, size=[100, 100])),
        "missing": xr.DataArray(dask.array.random.randint(0, 2, size=[100, 100, 100], dtype=bool)),
    })

    # test single array indexing
    data = test_ds["missing"]
    mask = test_ds["missing"]
    new_dim_name = "newdim"

    core_dim_locs = core_dim_locs_from_cond(mask, new_dim_name=new_dim_name)
    indexed_test_da = subset_variable(data, core_dim_locs, new_dim_name=new_dim_name, mask=mask)

    assert indexed_test_da.all().compute().item()
    assert indexed_test_da.sum().compute().item() == test_ds["missing"].sum().compute().item()
    assert indexed_test_da.dims == ("newdim",)


def test_ds_indexing():
    test_ds = xr.Dataset({
        "a": xr.DataArray(dask.array.random.randint(0, 1000, size=[100, 100, 100, 2])),
        "x": xr.DataArray(dask.array.random.randint(0, 1000, size=[100, 100, 100])),
        "y": xr.DataArray(dask.array.random.randint(0, 1000, size=[100, 100])),
        "z": xr.DataArray(dask.array.random.randint(0, 1000, size=[100, 100]), dims=("nodim_a", "nodim_b")),
        "missing": xr.DataArray(dask.array.random.randint(0, 2, size=[100, 100, 100], dtype=bool)),
    })

    indexed_test_ds = dataset_masked_indexing(test_ds, test_ds["missing"], "newdim")

    assert indexed_test_ds["missing"].all().compute().item()
    assert indexed_test_ds["missing"].sum().compute().item() == test_ds["missing"].sum().compute().item()
    assert indexed_test_ds["a"].dims == ("newdim", "dim_3")
    assert indexed_test_ds["x"].dims == ("newdim",)
    assert indexed_test_ds["y"].dims == ("newdim",)
    assert indexed_test_ds["z"].dims == ("nodim_a", "nodim_b")
    assert indexed_test_ds["missing"].dims == ("newdim",)

    ref_indexed_test_ds = test_ds.stack(newdim=["dim_0", "dim_1", "dim_2"])
    ref_indexed_test_ds = ref_indexed_test_ds.isel(newdim=ref_indexed_test_ds["missing"])
    ref_indexed_test_ds = ref_indexed_test_ds.compute()

    all_is_equal = (indexed_test_ds == ref_indexed_test_ds).all().compute()

    assert all_is_equal["a"].item()
    assert all_is_equal["x"].item()
    assert all_is_equal["y"].item()
    assert all_is_equal["missing"].item()

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python [conda env:anaconda-florian3]
#     language: python
#     name: conda-env-anaconda-florian3-py
# ---

# +
# code autoreload
# %load_ext autoreload
# %autoreload 2
import os
import sys

import collections
import random
import math
import numpy.random as nr
import numpy as np
import pandas as pd

import joblib

import xarray as xr
import dask
import dask.dataframe as ddf
import dask.array as da
import zarr

# set default scheduler to threaded
dask.config.set(scheduler='threads')

import tqdm

import scanpy.api as sc
import anndata as ad

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

import seaborn as sns

import plotnine as pn

## init plotly
# from plotly.offline import iplot, init_notebook_mode
# init_notebook_mode(connected=True)
import plotly.io as pio
pio.renderers.default = 'iframe_connected'
import plotly.graph_objs as go
import plotly.express as px

import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hd
hd.shade.cmap=["lightblue", "darkblue"]
hv.extension("bokeh", "matplotlib")
import hvplot

import sklearn

from scipy import (
    stats as scistats,
    special as scispecial,
)

import tensorflow as tf
import keras as k


# +
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# -

# !echo "$(nvidia-smi)"

from dask.cache import Cache
cache = Cache(8e9)  # Leverage eight gigabytes of memory
cache.register()

sys.path.append(os.path.expanduser("~/Projects/REP/rep"))
import rep.random as rnd

# + {"tags": ["parameters"]}
# absolute paths:
CACHE_DIR="/s/project/rep/cache/"
RAW_DATA_DIR="/s/project/rep/raw/"
PROCESSED_DATA_DIR="/s/project/rep/processed/"
# per default, relative to PROCESSED_DATA_DIR:
MODEL_DIR="training_results"
# per default, relative to MODEL_DIR:
CURRENT_MODEL_DIR="MMSplice+VEP+expr_multitask"

# parameters for model training
l2=0.00001
epochs=100

# +
MODEL_DIR=os.path.join(PROCESSED_DATA_DIR, "training_results")
CURRENT_MODEL_DIR=os.path.join(MODEL_DIR, "VEP+MMSplice+expr_multitask")

if not os.path.exists(CURRENT_MODEL_DIR):
    os.mkdir(CURRENT_MODEL_DIR)
# -

# # Load input data

xrds = xr.open_zarr(os.path.join(PROCESSED_DATA_DIR, "gtex/OUTRIDER/xarray_unstacked.zarr"))
xrds

stacked_xrds = xrds.stack(observations=["individual", "genes"])

# ## Select blood gene expression as input feature

blood = stacked_xrds.sel(subtissue="Whole_Blood")
blood

# ## Load MMSplice features and stack them (`observations = ["individual", "gene"]`)

mmsplice = xr.open_zarr(os.path.join(PROCESSED_DATA_DIR, "MMSplice", "agg_gene.zarr"))

mmsplice

mmsplice_stacked = mmsplice.stack(observations=["individual", "genes"])
mmsplice_stacked

# ## Load VEP features and stack them (`observations = ["individual", "gene"]`)

vep = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "VEP", "gtex_vep.csv.gz"))
vep.set_index(["individual", "genes"], inplace=True)
vep.head()

vep_xr = xr.DataArray(vep.values.astype("float32"), dims=("observations", "vep_features"), coords={"observations": vep.index, "vep_features": vep.columns})
vep_xr

stacked_xrds, vep_xr, mmsplice_stacked = xr.align(stacked_xrds, vep_xr, mmsplice_stacked, copy=False, join="left", fill_value=0.)
mmsplice_stacked

# ## Construct feature set

# +
from typing import Union, Tuple, List
import numpy as np
import xarray as xr

def concat_by_axis(
    darrs: Union[List[xr.DataArray], Tuple[xr.DataArray]],
    dims: Union[List[str], Tuple[str]],
    axis: int = None,
    drop_coords=True,
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

    # Get depth of nested lists. Assumes `darrs` is correctly formatted as list of lists.
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
            raise ValueError("Input %d must have the same number of dimensions as specified in the `dims` argument!" % i)

        # force-rename dimensions
        da = da.rename(dict(zip(da.dims, dims)))
        
        # remove coordinates
        if drop_coords:
            da = da.reset_coords(drop=True)

        to_concat.append(da)

    return xr.concat(to_concat, dim=dims[axis], **kwargs)
# -

features = concat_by_axis([
    blood.cdf.expand_dims({"features": ["cdf"]}, axis=-1), 
    blood.padj.expand_dims({"features": ["padj"]}, axis=-1), 
    blood.hilo_padj.expand_dims({"features": ["hilo_padj"]}, axis=-1), 
    blood.missing.expand_dims({"features": ["missing"]}, axis=-1),
    vep_xr,
    mmsplice_stacked["gene_sum"].expand_dims({"features": ["mmsplice_gene_sum"]}, axis=-1),
    mmsplice_stacked["gene_max"].expand_dims({"features": ["mmsplice_gene_max"]}, axis=-1),
], dims=("observations", "features"), fill_value=0, coords="minimal")
features = features.fillna(0)
features

stacked_xrds["c_features"] = features
stacked_xrds["cdf"] = stacked_xrds["cdf"].fillna(0)
stacked_xrds["normppf"] = stacked_xrds["normppf"].fillna(0)
stacked_xrds["padj"] = stacked_xrds["padj"].fillna(0)

stacked_xrds

# ## Prepare class weights
# There are about 10^4 more non-outliers than outliers. Therefore, outliers should be weighted 10^4 times higher than non-outliers. <br>
# Also, some tissues were not measured for certain individuals. Therefore, missing values should be weighted '0'.

# calculate number of outliers per tissue
n_outlier = ((~ stacked_xrds.missing) & (stacked_xrds.hilo != 0)).sum(dim="observations")
n_corrected_outlier = ((~ stacked_xrds.missing) & (stacked_xrds.hilo_padj != 0)).sum(dim="observations")
n_non_outlier = ((~ stacked_xrds.missing) & (stacked_xrds.hilo_padj == 0)).sum(dim="observations")

n_non_outlier = n_non_outlier.compute()
n_non_outlier

n_outlier = n_outlier.compute()
n_outlier

n_corrected_outlier = n_corrected_outlier.compute()
n_corrected_outlier

outliers = xr.Dataset({
    "n_non_outlier": n_non_outlier,
    "n_outlier": n_outlier,
    "n_corrected_outlier": n_corrected_outlier,
})
outliers

df = outliers.to_dataframe().reset_index().melt(id_vars="subtissue", var_name="outlier_type")
(
    pn.ggplot(df, pn.aes(x="subtissue", y="value", fill="outlier_type")) + 
    pn.geom_bar(stat="identity", position="dodge") + 
    pn.scale_y_log10() + 
    pn.theme(axis_text_x=pn.element_text(rotation=90, hjust=1)) + 
    pn.ggtitle("Number of non-outliers, outliers and multiple testing - corrected outliers")
)

class_weight = stacked_xrds.missing.transpose("observations", "subtissue", transpose_coords=False)


outlier_weight = (n_corrected_outlier + n_non_outlier) / n_corrected_outlier
non_outlier_weight = (n_corrected_outlier + n_non_outlier) / n_non_outlier

assert np.all(non_outlier_weight < outlier_weight)

outlier_weight

class_weight = class_weight * (
    (stacked_xrds.hilo_padj != 0) * outlier_weight + # matrix of outliers, non-outliers are '0'
    (stacked_xrds.hilo_padj == 0) * non_outlier_weight # matrix of non-outliers, outliers are '0'
)
class_weight[:10, :5].reset_coords(drop=True).compute()

stacked_xrds["class_weight"] = class_weight

pd.Series(blood.hilo_padj.values[~ blood.missing.values]).value_counts()

pd.Series(stacked_xrds.hilo_padj.values[~ stacked_xrds.missing.values]).value_counts()

# +
# split data into test and train sets
testDelim = int(blood.dims["observations"] * 0.8)

train = stacked_xrds.isel(observations=slice(None, testDelim))
test  = stacked_xrds.isel(observations=slice(testDelim, None))

print("train:")
print(train)
print("test:")
print(test)


# -

n_input_features = train.dims["features"]
n_input_features

n_targets = train.dims["subtissue"]
n_targets

# +
model = k.Sequential([
#     k.layers.Dense(units=20, activation="relu", input_shape=(n_input_features,)),
    k.layers.Dense(units=n_targets, activation="linear", kernel_regularizer=k.regularizers.l2(l2), input_shape=(n_input_features,)),
])

model.summary()
# -

model.compile(
    optimizer='adam',
    loss="mse",
    metrics=["mse", "mae", "mape", 'cosine'],
)

train.c_features.transpose("observations", "features", transpose_coords=False).data

train.normppf.transpose("observations", "subtissue", transpose_coords=False).data

# +
# batch_size = 2**30 // n_input_features # 1GB of memory

model.fit(
    x=train.c_features.transpose("observations", "features", transpose_coords=False).data, 
    y=train.normppf.transpose("observations", "subtissue", transpose_coords=False).data, 
    batch_size=315230, 
    shuffle=False,
    validation_split=0.1,
    epochs=epochs,
    class_weight=train.class_weight.transpose("observations", "subtissue", transpose_coords=False).data,
    callbacks=[
        k.callbacks.EarlyStopping(patience=4),
    ],
    workers=4,
    use_multiprocessing=True
)

# -

CURRENT_MODEL_DIR

model.save(os.path.join(CURRENT_MODEL_DIR, "model.h5"))


def predict_lazy(features, model, feature_dim="features", output_dim="subtissue", output_size=None):
    """
    Predicts using a (Keras-) model with a two-dimensional input and two-dimensional output,
    keeps xarray metadata and dask chunks
    """
    if output_size==None:
        output_size = model.output.shape[-1].value
    
    model_predict_lazy = da.gufunc(
        model.predict, 
        signature="(features)->(classes)", 
        output_dtypes="float32", 
        output_sizes={"classes": output_size}, 
        allow_rechunk=True, 
        vectorize=False
    )
    if isinstance(features, xr.DataArray):
        return xr.apply_ufunc(
            predict_lazy, features, 
            kwargs={"model": model}, 
            input_core_dims=[[feature_dim]], 
            output_core_dims=[[output_dim]], 
            dask="allowed",
        )
    else:
        return model_predict_lazy(features)


predicted = predict_lazy(test["c_features"], model)

to_save = predicted.reset_coords(drop=True).reset_index("observations").rename("predicted")
to_save

print("size of data to save: %.2f MB" % (to_save.nbytes/2**20))

to_save.sizes

with dask.config.set(scheduler='single-threaded'):
    to_save.chunk({"observations": 2**22 // to_save.sizes["subtissue"]}).to_dataset(name="predicted").to_zarr(os.path.join(CURRENT_MODEL_DIR, "predicted.zarr"), mode="w")

np.any(np.isnan(to_save)).compute()



import re
import numpy as np
import pandas as pd
import math
from collections import OrderedDict
from sklearn.metrics import r2_score
from scipy.stats.mstats import gmean

import comet_ml
import gin
from gin_train.metrics import RegressionMetrics

from rep.constants import METADATA_CST as mcst
from rep.constants import GTEX_CST as gcst


def filter_entries(tissue_name, column_name, metadata, y_true, y_pred):
    """Assumes the index in nparray matches the metadata index. 
    
    Args:
        tissue_name (str): tissue for which to filter the data
        column_name (str): column in the metadata where tissue_name its found
        metadata (:obj:DataFrame): metadata from the cross tissue matrix - this is a matrix
        y_true: -> matrix (samples_tissue_cross x genes)
        y_pred: -> matrix (samples_tissue_cross x genes)
    """
    if tissue_name and len(tissue_name) > 0:
        tmp = np.where(metadata[column_name] == tissue_name)[0]
        row_indexes_slice = tmp.tolist()
        return y_true[row_indexes_slice,:], y_true[row_indexes_slice,:]

    # return whole dataset
    return y_true, y_pred


def compute_metrics_per_tissue(gm, y_true, y_pred, label, metric_collection_obj):
    """Apply all metrics for the predictions
    """
    genes_count = y_true.shape[1]
    for gene_i in range(genes_count):

#       # compute mse var_explained, preasonr, mad, 
        m_temp = metric_collection_obj(y_true[:, gene_i], y_pred[:, gene_i])
    
        # compute other metrics
        m_temp = metrics_extension(m_temp, y_true[:, gene_i], y_pred[:, gene_i])

        # rename to tissue specific
        m_temp = rename(m_temp, label)

        gm.append(m_temp)
        
    return gm


def distribution_summary(x):
    """
    Args:
        x: list of floats
    """
    return {"mean": np.mean(x),
            "min": np.min(x, 0),
            "p10": np.quantile(x, 0.1),
            "p25": np.quantile(x, 0.25),
            "median": np.quantile(x, 0.5),
            "p75": np.quantile(x, 0.75),
            "p90": np.quantile(x, 0.9),
            "max": np.max(x)
           }


def rquare(y_true, y_pred):
    """R square
    """
    return r2_score(y_true,y_pred)


def geo_arithm_mean_ratio(y_true, y_pred):
    """Geometric-arithmetic mean ratio. In house defined metric
    
       mean(geo_mean(i,j)/arthm(i,j)), 
       where i - true, j - predicted values across samples for a single feature (gene)
    """    
    np_matrix = np.array([y_true + 0.0001, y_pred + 0.0001])
    
    return np.mean(np.divide(gmean(np_matrix, axis=0),
                             np.mean(np_matrix, axis=0))
                  )


def metrics_extension(m_temp, y_true, y_pred):
    
    # metrics = {"rsquare":rquare,
    #           "geo_arithm_mean_ratio":geo_arithm_mean_ratio}
    metrics = {"rsquare": rquare}
    for m in metrics:
        m_temp[m] = metrics[m](y_true, y_pred)
    
    return m_temp


def rename(m_temp, label):
    """Rename labels. Ex. instead of mse -> blood/mse    
    """
    if label:
        new_m_temp = OrderedDict()
        for key, value in m_temp.items():
            new_m_temp[label + "/" + key] =  value
        return new_m_temp
    
    # if label None return unchanged Collection
    return m_temp

    
def collapse_name(name):
    """Replace spaces with _ and make every word lower case
    """
    if name:
        return re.sub("\s+", "_", str(name).lower()).strip()
    
    # for None values
    return None


@gin.configurable
def rep_metric(y_true, y_pred, tissue_specific_metadata = None):
    
    # replace Nan with 0 / avoid 0 entries
    y_true = np.nan_to_num(np.array(y_true)) + 0.001
    y_pred = np.nan_to_num(np.array(y_pred)) + 0.001
    
    # create an object with following matrics metrics
    metric_collection = RegressionMetrics() 
    
    tissues = [None] # None signalize to run metrics over all tissues
    parent_tissue = None

    # tissue specific setup
    if tissue_specific_metadata is not None:
        # append all other tissues
        tissues += list(set(tissue_specific_metadata[gcst.TO_TISSUE].tolist()))
        parent_tissue = list(set(tissue_specific_metadata[gcst.TO_PARENT_TISSUE].tolist()))
        tissues = [None, 'Bladder', 'Spleen']

    # metrics dictionary
    gm = []    
    out = {}
    # per tissue
    for t in tissues:
        gm = []
        label = collapse_name(t)
        (y_true_slice, y_pred_slice) = filter_entries(t, gcst.TO_TISSUE, tissue_specific_metadata, y_true, y_pred)
        gm = compute_metrics_per_tissue(gm, y_true_slice, y_pred_slice, label, metric_collection)
        metric_names = list(gm[0])
        for metric_name in metric_names:
            out[metric_name] = distribution_summary([x[metric_name] if math.isnan(x[metric_name]) == False else 0 for x in gm ])
        
#     # per parent tissue
#     if parent_tissue:
#         for p in parent_tissue:
#             label = collapse_name(t)
#             (y_true_slice, y_pred_slice) = filter_entries(t, gcst.TO_PARENT_TISSUE, tissue_specific_metadata[mcst.INDIV_TISSUE_METADATA], y_true, y_pred)
#             gm = compute_metrics_per_tissue(gm, y_true_slice, y_pred_slice, label, metric_collection)
    
    # compute distribution summary for all the metrics
#     metric_names = list(gm[0])
#     print(metric_names)
#     print(gm)
#     out = {}
#     for metric_name in metric_names:
#         # set to 0 values which are not a number
# #         xlist = [x[metric_name] if math.isnan(x[metric_name]) == False else 0 for x in gm]
# #         out[metric_name] = distribution_summary(xlist)
#         out[metric_name] = distribution_summary([x[metric_name] if math.isnan(x[metric_name]) == False else 0 for x in gm ])
     
    return out
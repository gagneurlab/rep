import numpy as np
import pandas as pd
import math
from collections import OrderedDict
from sklearn.metrics import r2_score
from scipy.stats.mstats import gmean

import gin
from gin_train.metrics import RegressionMetrics

from rep.constants import METADATA_CST as mcst
from rep.constants import GTEX_CST as gcst


def filter_entries(tissue_name, column_name, metadata, y_true, y_pred):
    """Assumes the index in nparray matches the metadata index. 
    """
    pass

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
    np_matrix = np.array([y_true + 0.0001, y_pred + 0.0001])
    
    return np.mean(np.divide(gmean(np_matrix, axis=0),
                             np.mean(np_matrix, axis=0))
                  )

def metrics_extension(m_temp, y_true, y_pred):
    
    metrics = {"rquare":rquare,
              "geo_arithm_mean_ratio":geo_arithm_mean_ratio}
    
    return m_temp


def rename()

@gin.configurable
def rep_metric(y_true, y_pred, tissue_specific_metadata = None):
        
    # create an object with following matrics metrics
    m = RegressionMetrics() 
    
    # tissue specific setup
    tissues = [None] # None signalize to run metrics over all tissues
    parent_tissue = None
    if tissue_specific_metadata:
        # append all other tissues
        tissues += list(set(tissue_specific_metadata[mcst.INDIV_TISSUE_METADATA][gcst.TO_TISSUE].to_list()))
        parent_tissue = list(set(tissue_specific_metadata[mcst.INDIV_TISSUE_METADATA][gcst.TO_PARENT_TISSUE].to_list()))

    # metrics dictionary
    gm = []    
    
    
    genes_count = y_true.shape[1]
    for gene_i in range(genes_count):
        
        # compute mse var_explained, preasonr, mad, 
        m_temp = m(y_true[:, gene_i], y_pred[:, gene_i])
        
        # compute other metrics
        m_temp = metrics_extension(m_temp, y_true[:, gene_i], y_pred[:, gene_i])
        
        # rename to tissue specific
        m_temp = rename(m_temp, label)
        
        gm.append(m_temp)        

    
    metric_names = list(gm[0])
        
    out = {}
    for metric_name in metric_names:
        # set to 0 values which are not a number
        xlist = [x[metric_name] if math.isnan(x[metric_name]) == False else 0 for x in gm]
        out[metric_name] = distribution_summary(xlist)
     
    return out
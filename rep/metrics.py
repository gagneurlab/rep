import gin
from gin_train.metrics import RegressionMetrics

import numpy as np
import pandas as pd

import math

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

@gin.configurable
def rep_metric(y_true, y_pred):
        
    # create an object with following matrics metrics
    m = RegressionMetrics() 

    gm = []
    for gene_i in range(y_true.shape[1]):
        m_temp = m(y_true[:, gene_i], y_pred[:, gene_i])
        gm.append(m_temp)        

    
    metric_names = list(gm[0])
        
    out = {}
    for metric_name in metric_names:
        xlist = [x[metric_name] if math.isnan(x[metric_name]) == False else 0 for x in gm]
        out[metric_name] = distribution_summary(xlist)
     
    return out
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr, spearmanr


def evaluate(y_pred,y_valid,title):
    # evaluate model
    print("R2: ",r2_score(y_valid, y_pred))
    
    plt.figure(figsize=(15,15))
    performances = pd.Series([spearmanr(y_valid[:,i], y_pred[:, i])[0]  for i in range(y_pred.shape[1])])
    plt.hist(performances.dropna())
    plt.title(title)

def regression_eval_multioutput(y_true, y_pred, alpha=0.5, markersize=2, task="", ax=None, same_lim=False, loglog=False):
    
    for i in range(y_true.shape[1]):
        regression_eval(y_true[:,i], y_pred[:,i], alpha=alpha, markersize=markersize, task=task, ax=ax, same_lim=same_lim, loglog=loglog)
    
def regression_eval(y_true, y_pred, alpha=0.5, markersize=2, task="", ax=None, same_lim=False, loglog=False):
    
    if ax is None:
        fig, ax = plt.subplots(1)
    from scipy.stats import pearsonr, spearmanr
    xmax = max([y_true.max(), y_pred.max()])
    xmin = min([y_true.min(), y_pred.min()])
    
    if loglog:
        pearson, pearson_pval = pearsonr(np.log10(y_true), np.log10(y_pred))
        spearman, spearman_pval = spearmanr(np.log10(y_true), np.log(y_pred))
    else:
        pearson, pearson_pval = pearsonr(y_true, y_pred)
        spearman, spearman_pval = spearmanr(y_true, y_pred)
    if loglog:
        plt_fn = ax.loglog
    else:
        plt_fn = ax.plot
        
    plt_fn(y_pred, y_true, ".", 
           markersize=markersize, 
           alpha=alpha)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    
    if same_lim:
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((xmin, xmax))
    rp = r"$R_{p}$"
    rs = r"$R_{s}$"
    ax.set_title(task)
    ax.text(.95, .2, f"{rp}={pearson:.2f}",
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes)
    ax.text(.95, .05, f"{rs}={spearman:.2f}", 
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes)
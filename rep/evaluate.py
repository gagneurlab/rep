import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr, spearmanr

import gin

class Evaluate(object):

    __quantile__ = {'q0':0,
               'q10':10,
               'q25':25,
               'q50':50,
               'q75':75,
               'q90':90,
               'q100':100}


#     __methods__ = {'spearman':Evaluate.spearman,
#                    'pearson':Evaluate.pearson,
#                    'r2_score':sklearn.metrics.r2_score}

    __ALL__ = 'all'

    __ALL_METRICS__ = {'spearman':__quantile__,
                       'pearson':__quantile__,
                       'r2_score':None}

    def __init__(self,metrics_set):
        '''
        Takes a list of metrics that should be computed

        Args:
            metrics_set (list): [all | spearman | pearson | r2_score]
        '''

        self.metrics_set = metrics_set


    def __call__(self, y_true, y_pred):

        dict_metrics = {}

        # all metrics computed
        if Evaluate.__ALL__ in self.metrics_set:
            self.metrics_set = [Evaluate.__ALL_METRICS__]

        for m in self.metrics_set:

            dict_metrics[m] = {}

            # single number
            if self.metrics_set[m]:
                dict_metrics[m] = Evaluate.__methods__[m](y_true, y_pred)

            # distribution
            else:

                distr = Evaluate.__methods__[m](y_true, y_pred)

                # compute all quatiles
                for q in self.metrics_set[m]:
                    dict_metrics[m][q] = np.percentile(distr, Evaluate.__quantile__[q])


    @staticmethod
    def spearmanr(y_true, y_pred):

        return pd.Series([scipy.stats.spearmanr(y_true[:,i], y_pred[:, i])[0] for i in range(y_pred.shape[1])])

    @staticmethod
    def pearsonr(y_true, y_pred):

        return pd.Series([scipy.stats.pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_pred.shape[1])])




    # def rquare_eval(y_pred,y_valid,title):
    #     # evaluate model
    #     print("R2: ",r2_score(y_valid, y_pred))
    #
    #     plt.figure(figsize=(5,5))
    #     performances = pd.Series([spearmanr(y_valid[:,i], y_pred[:, i])[0]  for i in range(y_pred.shape[1])])
    #     plt.hist(performances.dropna())
    #     plt.title(title)
    #
    # def regression_eval_multioutput(y_true, y_pred, alpha=0.5, markersize=2, task="", ax=None, same_lim=False, loglog=False):
    #
    #     for i in range(y_true.shape[1]):
    #         regression_eval(y_true[:,i], y_pred[:,i], alpha=alpha, markersize=markersize, task=task, ax=ax, same_lim=same_lim, loglog=loglog)
    #
    # def correlation_plot(y_pred,y_valid,n=10):
    #
    #     plt.figure(figsize=(5,5))
    # #     ensembl = gene_id_train
    #     for i in range(n): # number of genes
    #         plt.subplot(n/2,n/2,(i+1))
    #         plt.scatter(y_pred[:,i],y_valid[:,i], alpha=0.1)
    #         plt.xlabel("Predicted")
    #         plt.ylabel("Observed")
    # #         plt.title(gtex[ensembl[i]].obs['symbol'].tolist()[0])
    #         plt.grid(True)
    #
    # def regression_eval(y_true, y_pred, alpha=0.5, markersize=2, task="", ax=None, same_lim=False, loglog=False):
    #
    #     if ax is None:
    #         fig, ax = plt.subplots(1)
    #     from scipy.stats import pearsonr, spearmanr
    #     xmax = max([y_true.max(), y_pred.max()])
    #     xmin = min([y_true.min(), y_pred.min()])
    #
    #     if loglog:
    #         pearson, pearson_pval = pearsonr(np.log10(y_true), np.log10(y_pred))
    #         spearman, spearman_pval = spearmanr(np.log10(y_true), np.log(y_pred))
    #     else:
    #         pearson, pearson_pval = pearsonr(y_true, y_pred)
    #         spearman, spearman_pval = spearmanr(y_true, y_pred)
    #     if loglog:
    #         plt_fn = ax.loglog
    #     else:
    #         plt_fn = ax.plot
    #
    #     plt_fn(y_pred, y_true, ".",
    #            markersize=markersize,
    #            alpha=alpha)
    #     ax.set_xlabel("Predicted")
    #     ax.set_ylabel("Observed")
    #
    #     if same_lim:
    #         ax.set_xlim((xmin, xmax))
    #         ax.set_ylim((xmin, xmax))
    #     rp = r"$R_{p}$"
    #     rs = r"$R_{s}$"
    #     ax.set_title(task)
    #     ax.text(.95, .2, f"{rp}={pearson:.2f}",
    #             verticalalignment='bottom',
    #             horizontalalignment='right',
    #             transform=ax.transAxes)
    #     ax.text(.95, .05, f"{rs}={spearman:.2f}",
    #             verticalalignment='bottom',
    #             horizontalalignment='right',
    #             transform=ax.transAxes)
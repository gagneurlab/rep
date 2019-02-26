"""
Sklearn model
PyTorch nn.Module classes or other models
"""
import sklearn as sk
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import gin
from gin import config

####################### Linear Regression / Sklearn ######################


# configurable params
config.external_configurable(sk.decomposition.PCA,module='sk.decomposition')
config.external_configurable(sk.linear_model.LassoLarsCV,module='sk.linear_model')

# dim_reduction = PCA(10)
# lasso_lars = MultiOutputRegressor(LassoLarsCV(cv=5, max_iter=20, normalize=False, n_jobs = 5))

# @gin.configurable
# def lasso_model(dim_reducer, lasso_lars):
#     p = Pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
#                     ('DimensionalityReductionPCA',dim_reducer),
#                     ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(lasso_lars))])
#     return p

@gin.configurable
def lasso_model(n_components, cv, max_iter, normalize, n_jobs):
    p = Pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('DimensionalityReductionPCA',PCA(n_components = n_components)),
                    ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(LassoLarsCV(cv = cv, max_iter = max_iter, normalize = normalize, n_jobs = n_jobs), n_jobs = 1))])
    return p

####################### Linear Regression / PyTorch ######################


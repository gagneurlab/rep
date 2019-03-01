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

import torch
import torch.nn as nn

####################### Linear Regression / Sklearn ######################

# configurable params
gin.external_configurable(sk.decomposition.PCA,module='sk.decomposition')
gin.external_configurable(sk.linear_model.LassoLarsCV,module='sk.linear_model')

dim_reduction = PCA(10)
lasso_lars = MultiOutputRegressor(LassoLarsCV(cv=5, max_iter=20, normalize=False, n_jobs = 5))

@gin.configurable
def lasso_model(n_components, cv, max_iter, normalize, n_jobs):
    """Linear Regression Pipeline using:
        (i) StandardScaler as preprocessing step
        (ii) Dimensionality reduction using PCA - and its variation i.e. KernelPCA / IncrementalPCA
        (iii) Multioutput Regression using LassoLars
    """
    p = Pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('DimensionalityReductionPCA',PCA(n_components = n_components)),
                    ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(LassoLarsCV(cv = cv, max_iter = max_iter, normalize = normalize, n_jobs = n_jobs), n_jobs = 1))])
    return p

# dim_reduction = PCA(10)
# lasso_lars = MultiOutputRegressor(LassoLarsCV(cv=5, max_iter=20, normalize=False, n_jobs = 5))

# @gin.configurable
# def lasso_model(dim_reducer, lasso_lars):
#     p = Pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
#                     ('DimensionalityReductionPCA',dim_reducer),
#                     ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(lasso_lars))])
#     return p

# @gin.configurable
# def lasso_model(dim_reducer, cv, max_iter, normalize, n_jobs):
#     """Linear Regression Pipeline using:
#         (i) StandardScaler as preprocessing step
#         (ii) Dimensionality reduction using PCA - and its variation i.e. KernelPCA / IncrementalPCA
#         (iii) Multioutput Regression using LassoLars
#     """
#     p = Pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
#                     ('DimensionalityReductionPCA',dim_reducer),
#                     ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(LassoLarsCV(cv = cv, max_iter = max_iter, normalize = normalize, n_jobs = n_jobs), n_jobs = 1))])
#     return p

# @gin.configurable
# def lasso_model(dim_reducer, lasso_lars):
#     """Linear Regression Pipeline using:
#         (i) StandardScaler as preprocessing step
#         (ii) Dimensionality reduction using PCA - and its variation i.e. KernelPCA / IncrementalPCA
#         (iii) Multioutput Regression using LassoLars
#     """
#     p = Pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
#                     ('DimensionalityReductionPCA',dim_reducer),
#                     ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(lasso_lars))])
#     return p

# @gin.configurable
# def lasso_model(dim_reducer, lasso_lars):
#     """Linear Regression Pipeline using:
#         (i) StandardScaler as preprocessing step
#         (ii) Dimensionality reduction using PCA - and its variation i.e. KernelPCA / IncrementalPCA
#         (iii) Multioutput Regression using LassoLars
#     """
#     p = make_pipeline([('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
#                     ('DimensionalityReductionPCA',dim_reducer),
#                     ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(lasso_lars, n_jobs=10))])
#     return p



####################### Linear Regression / PyTorch ######################
@gin.configurable
class LinearRegressionCustom(nn.Module):
    """
        
    Attributes:
        input_dim:
        output_dim:
        criterion (:obj:nn): loss function to optimize, i.e. nn.MSELoss()
        optimiser (:class:torch.optim): reference to optimize class i.e. torch.optim.SGD
    """
    def __init__(self, input_dim, output_dim, criterion, optimiser, l_rate):
        
        super(LinearRegressionCustom, self).__init__() 
        
        # linear model
        self.linear = nn.Linear(input_dim, output_dim)
        
        # metric to optimize
        self.criterion = criterion
        
        # optimization method
        params_dict = dict(self.linear.named_parameters())
        params = []
        for key, value in params_dict.items():
            params += [{'params':[value],'lr':l_rate}]
        self.optimiser = optimiser(params)
    
    
    @property
    def get_criterion(self):
        return self.criterion
    
    
    @property
    def get_optimiser(self):
        return self.optimiser
    
    
    @property
    def get_model(self):
        return self.linear
    
    
    def set_criterion(self, crt):
        self.criterion = crt
        
    
    def set_optimiser(self, opt):
        self.optimiser = opt
        
    
    def set_model(self, m):
        self.linear = m
        
        
if __name__ == "__main__":
    
    # how to create a Linear regression using pytorch
    m = LinearRegressionCustom(3,4,nn.MSELoss(),torch.optim.SGD,0.001)

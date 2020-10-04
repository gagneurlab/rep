"""
Sklearn model
PyTorch nn.Module classes or other models
"""
import numpy as np
import pandas as pd
from typing import Union,List

import sklearn as sk
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoLarsCV, Lasso, LinearRegression, RidgeCV, SGDRegressor, LassoLars, Ridge, HuberRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

import keras
from keras.layers import Input, Dense, Add, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, Adamax, Nadam
from keras.losses import *
from keras import regularizers
from keras.callbacks import EarlyStopping, ProgbarLogger, ReduceLROnPlateau, Callback

import matplotlib.pyplot as plt

import gin
from gin import config

import torch
import torch.nn as nn

torch.manual_seed(7)
np.random.seed(1)

####################### Linear Regression / Sklearn ######################

# configurable params
gin.external_configurable(sk.decomposition.PCA, module='sk.decomposition')
gin.external_configurable(sk.decomposition.KernelPCA, module='sk.decomposition')
gin.external_configurable(sk.decomposition.IncrementalPCA, module='sk.decomposition')
gin.external_configurable(sk.linear_model.LassoLarsCV, module='sk.linear_model')

dim_reduction = PCA(10)
lasso_lars = MultiOutputRegressor(LassoLarsCV(cv=5, max_iter=20, normalize=False, n_jobs=5))


@gin.configurable
def pca_train(pca_type=PCA, n_components=10, features_file=None, train_individuals_file=None, tissue=None):
    """Dimensionality reduction of the inputs. Assumes a very well defined structure.
            (i) StandardScaler as preprocessing step
            (ii) Dimensionality reduction using PCA - and its variation i.e. KernelPCA / IncrementalPCA
        Args:
            pca_type (:class) : reference to class in sklearn to compute PCA [PCA, KernelPCA,IncrementalPCA]
            n_components (int):
            features_file (str): filename - stores an RepAnnData object
            train_individuals_file (str): filename - list of individuals
            tissue (str): perform PCA only over a specific tissue
    """

    # read inputs
    train_individuals = p.read_csv_one_column(train_individuals_file)
    features = p.RepAnnData.read_h5ad(features_file)

    # filter RepAnnData by Blood and train individuals
    gtex_filtered = features[features.samples['Individual'].isin(train_individuals)]
    if tissue:
        gtex_blood = gtex_filtered[gtex_filtered.obs['Tissue'] == tissue]
    else:
        gtex_blood = gtex_filtered

    print(f'Start {pca_type} with {n_components} for {gtex_blood.X.shape[0]} individuals and {gtex_blood.X.shape[1]} genes')

    # fit to normal distribuation
    X_std = StandardScaler().fit_transform(gtex_blood.X)

    #     if pca_type not in [PCA, KernelPCA, IncrementalPCA]:
    #         raise ValueError("pca_type value not known. Please use [PCA, KernelPCA, IncrementalPCA]")

    # Instantiate
    pca = pca_type(n_components=n_components)
    # Fit and Apply dimensionality reduction on X
    pca.fit_transform(X_std)

    return pca


@gin.configurable
def lasso_model(dim_reducer: Union[PCA, IncrementalPCA, KernelPCA], cv: int, max_iter: int, normalize: bool,
                n_jobs: int) -> Pipeline:
    """Linear Regression Pipeline using:
        (i) StandardScaler as preprocessing step
        (ii) Feature reduction
        (iii) Multioutput Regression using LassoLars

    Args:
        features_annotation (list): regression features (genes + tissue_category)
        dim_reducer:
        cv:
        max_iter:
        normalize:
        n_jobs:
    """


    p = Pipeline(steps=[('StandardScaler',StandardScaler()),
                        ('DimReduction', dim_reducer),
                        ('LassoLarsMultiOutputRegressor',
                   MultiOutputRegressor(LassoLarsCV(cv=cv, max_iter=max_iter, normalize=normalize, n_jobs=n_jobs),
                                        n_jobs=5))])
    return p

@gin.configurable
def linear_regression():
    """Linear Regression Pipeline using:
        (i) StandardScaler as preprocessing step
        (ii) Multioutput Regression using LinearRegression

    Returns:
        Pipeline object
    """

    p = Pipeline(steps=[('StandardScaler',StandardScaler()),
                        ('LassoLarsMultiOutputRegressor', MultiOutputRegressor(LinearRegression(n_jobs=10)))])
    return p


@gin.configurable
def lasso_model_onehot(features_annotation: List[str], dim_reducer: Union[PCA, IncrementalPCA, KernelPCA], cv: int, max_iter: int, normalize: bool,
                n_jobs: int) -> Pipeline:
    """Linear Regression Pipeline using:
        (i) StandardScaler as preprocessing step
        (ii) Feature reduction
        (iii) Multioutput Regression using LassoLars

    Args:
        features_annotation (list): regression features (genes + tissue_category)
        dim_reducer:
        cv:
        max_iter:
        normalize:
        n_jobs:
    """

#    p = Pipeline([  ('StandardScaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
#                    ('DimensionalityReductionPCA',PCA(n_components = n_components)),
#                    ('LassoLarsMultiOutputRegressor',MultiOutputRegressor(LassoLarsCV(cv = cv, max_iter = max_iter, normalize = normalize, n_jobs = n_jobs), n_jobs = 1))])


    # all genes
    numeric_features = features_annotation[:-1]
    numeric_transformer = Pipeline(steps=[
        ('StandardScaler', StandardScaler()),
        ('DimensionalityReductionPCA', dim_reducer)])

    # tissue category
    categorical_features = features_annotation[-1]
    categorical_transformer = Pipeline(steps=[
        ('OneHot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    p = Pipeline(steps=[('preprocessor', preprocessor),
                        ('LassoLarsMultiOutputRegressor',
                   MultiOutputRegressor(LassoLarsCV(cv=cv, max_iter=max_iter, normalize=normalize, n_jobs=n_jobs),
                                        n_jobs=5))])
    return p


############################################# Autoencoders ################################################
@gin.configurable
def pca_autoencoder(n_comp=128):
    """Reconstruct input gene expression using PCA
    """
    pca = PCA(n_components=n_comp)
   
    return pca
    
# @gin.configurable
# def linear_ae(code_size = 128):
#     """Autoencoder with one hidden space
#     """
#     # autoencoder layers size
#     input_size = y_blood_train.shape[1]
#     code_size = code_size
#
#     # optimizer
#     adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.000001,decay=0.01)
#
#     # stop condition
#     callbacks = [EarlyStopping(monitor='val_loss',min_delta=0.0001, patience=3)]
#
#     # model
#     linear_ae = Sequential()
#     linear_ae.add(Dense(code_size, input_shape = (input_size,)))
#     linear_ae.add(Dense(input_size))
#     linear_ae.compile(optimizer=adam, loss=mean_squared_error)
#
#     return linear_ae
    


####################### Linear Regression / PyTorch ######################

# configurable params
gin.external_configurable(torch.nn.MSELoss, module='torch.nn')
gin.external_configurable(torch.optim.Adam, module='torch.optim')


@gin.configurable
def torch_linear_model(input_dim, output_dim, criterion, optimiser, l_rate):
    """Linear Regression Pipeline in Pytorch
    """
    # how to create a Linear regression using pytorch
    m = LinearRegressionCustom(input_dim,
                               output_dim,
                               criterion,
                               optimiser,
                               l_rate)

    return m


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
            params += [{'params': [value], 'lr': l_rate}]
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


####################### Baseline Model ######################

# def compute_baseline(annobj):
#     """Compute average for each tissue across samples

#     Args:
#         annobj (:obj:Anndata): summarized experiment genes x samples

#     Returns:
#         DataFrame containing x_ij - mean expression of gene j in tissue i.
#     """
#     tissues = sorted(annobj.var['Tissue'].drop_duplicates().tolist())
#     Y_mean_tissue = np.zeros((len(annobj.obs_names),len(tissues)))

#     for t in tissues:
#         slice_bytissue = annobj[:,annobj.var['Tissue'] == t]    
#         mean_value = np.mean(slice_bytissue.X,axis=1)
#         Y_mean_tissue[:,tissues.index(t)] = mean_value

#     out = pd.DataFrame(data = Y_mean_tissue, index = annobj.obs_names, columns = tissues)
#     return out.transpose() 


if __name__ == "__main__":
    # how to create a Linear regression using pytorch
    # m = LinearRegressionCustom(3,4,nn.MSELoss(),torch.optim.SGD,0.001)

    # test PCA

    # pca model
    import os

    features_file = os.path.join("/", "s", "project", "rep", "processed", "gtex", "recount",
                                 "recount_gtex_logratios.h5ad")
    train_individuals_file = os.path.join("/", "s", "project", "rep", "processed", "gtex", "recount",
                                          "train_individuals.txt")
    pca_model = pca_train(n_components=10,
                          features_file=features_file,
                          train_individuals_file=train_individuals_file,
                          tissue='Whole Blood')

    # fit model over crosstissue matrix
    inputs = os.path.join("/", "s", "project", "rep", "processed", "gtex", "input_data", "X_inputs_pc_onlyblood.h5")

    X_inputs = p.RepAnnData.read_h5ad(inputs)
    X_train = X_inputs[X_inputs.samples['Type'] == 'train']
    X_train_pca = pca_model.fit(X_train.X)
    p.writeh5(X_train_pca, "test_train_pca.h5ad")

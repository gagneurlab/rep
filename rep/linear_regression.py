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
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr, spearmanr

from rep import evaluate as e

class Linear_Regression():
    
    def __init__(self, Xs_train, Ys_train, Xs_valid, Ys_valid):
        
        self.Xs = Xs_train
        self.Ys = Ys_train
        self.Ys_valid = Ys_valid
        self.Xs_valid = Xs_valid
        
        # dictionary with all models that we would like to test
        self.dict_models = {'train_lassolars_model':self.train_lassolars_model,
                            'train_lassolars_model_multioutput':self.train_lassolars_model_multioutput}
        
        # avoid same value for the entire column error for lasso
        self.Xs[0,:] = self.Xs[0,:] + 0.001
        self.Xs_valid[0,:] = self.Xs_valid[0,:] + 0.001
    
    
    def run(self,model='train_lassolars_model'):   
       
        # train    
        reg = self.dict_models[model](self.Xs,self.Ys)
               
        # predict
        predict_y = self.predict_lasso(reg, self.Xs_valid)
        
        # evaluate
#         e.evaluate(predict_y,self.Ys_valid,"LassoLars Linear Regression")   
#         e.regression_eval_multioutput(self.Ys_valid[:,:3],predict_y[:,:3]) # plot for the first 3 features the correlation
        
        return predict_y
    
    
    def run_batches(self,model='train_lassolars_model',n=300):
        '''Linear regression over batches. The final model is given by the avg()
           This works only with lasso lars 
        '''
        
        # train
        reg = []
        for i in range(0,self.Xs.shape[0],n):
            if i + n < self.Xs.shape[0]:
                reg.append(self.dict_models[model](self.Xs[i:i+n,:], self.Ys[i:i+n,:]))
            else:
                reg.append(self.dict_models[model](self.Xs[i:,:], self.Ys[i:,:]))
        
        # average over the parameters
        for i in range(len(reg[0].estimators_)):
            e = np.zeros(self.Xs.shape[1])
            intercept = 0
            for r in reg:
                e += r.estimators_[i].coef_
                intercept +=  r.estimators_[i].intercept_

            reg[0].estimators_[i].coef_ = e/len(reg[0].estimators_)
            reg[0].estimators_[i].intercept_ = intercept/len(reg[0].estimators_)
        
        # predict
        predict_y = self.predict_lasso(reg[0],self.Xs_valid)
        
        # evaluate
        
        return predict_y
    

    def train_lassolars_model(self,train_x, train_y):
        
        train_x[0,:] = train_x[0,:] + 0.001
#         train_y[0,:] = train_y[0,:] + 0.001

        reg = LassoLarsCV(cv=5, n_jobs=1, max_iter=20, normalize=False)
        reg.fit(train_x,train_y) 
        return reg
    
    def train_lassolars_model_multioutput(self,train_x, train_y):
        
        train_x[0,:] = train_x[0,:] + 0.001
        train_y[0,:] = train_y[0,:] + 0.001

        reg = MultiOutputRegressor(LassoLarsCV(cv=5, max_iter=50, normalize=False), n_jobs = 10)
        reg.fit(train_x,train_y) 
        return reg
    

    def predict_lasso(self,reg, valid_x):
        
        predict_y = reg.predict(valid_x)
        return predict_y
    

class Transform():
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def fit_transform(self):
        
        # Center data to N(0,1) distribution
        x_preproc = StandardScaler()
        y_preproc = StandardScaler()

        Xs = x_preproc.fit_transform(self.x)
        Ys = y_preproc.fit_transform(self.y)

        return (Xs, Ys, x_preproc, y_preproc)

    def transform(self, x_preproc, y_preproc):
        
        Xs = x_preproc.transform(self.x)
        Ys = y_preproc.transform(self.y)

        return (Xs, Ys)


class FeatureReduction():
    
    def __init__(self,x):
        self.x = x
    
    def pca_svd(self,components = 2,fit_transform = True, scaler = None):
        
        if fit_transform == True:
            pca = PCA(n_components = components, svd_solver='randomized')
        else:
            pca = scaler
        
        if fit_transform:
            Xs_pca = pca.fit_transform(self.x)
        else:
            Xs_pca = pca.transform(self.x)
        
        return (Xs_pca,pca)
    
    def sparse_pca(self,components = 2,fit_transform = True):
        
        pca = SparsePCA(n_components = components, max_iter = 50, random_state = 123)
        
        if fit_transform:
            Xs_pca = pca.fit_transform(self.x)
        else:
            Xs_pca = pca.transform(self.x)
        
        return (Xs_pca,pca)
    

    
# # log-normalize the count data
# Y = np.log10(Y + 1)  # these are the `train` individuals
# X = np.log10(Y + 1)
# # Make sure you have made the train,valid,test split
# Y_valid = np.log10(Y_valid + 1)  # `valid` individuals
# X_valid = np.log10(Y_valid + 1)

# # standardize the data
# from sklearn.preprocessing import StandardScaler
# x_preproc = StandardScaler()
# y_preproc = StandardScaler()
# Xs = x_preproc.fit_transform(X)
# Ys = y_preproc.fit_transform(Y)
# Xs_valid = x_preproc.transform(X_valid)
# Ys_valid = y_preproc.transform(Y_valid)

# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import LassoLarsCV

# m = MultiOutputRegressor(LassoLarsCV(), n_jobs=10)

# # Train the model using the training sets
# m.fit(Xs, Ys)
# Y_pred = m.predict(Xs_valid)
# i = 10 # some random gene
# plt.scatter(Y_pred[:, i], Ys_valid[:,i], alpha=0.1)

# # Evaluate the performance
# from scipy.stats import pearsonr, spearmanr

# # Get the performance for all the genes
# performances = pd.Series([spearmanr(Ys_valid[:,i], Y_pred[:, i])[0]
# 				         for i in range(Y_pred.shape[1])],
# 						 index=gene_idx)
# performances.plot.hist(30)
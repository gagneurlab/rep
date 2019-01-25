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

from rep import evaluate as e

class Linear_Regression():
    
    def __init__(self,Xs_train,Ys_train,Xs_valid,Ys_valid):
        
        self.Xs = Xs_train
        self.Ys = Ys_train
        self.Ys_valid = Ys_valid
        self.Xs_valid = Xs_valid
        
        # avoid same value for the entire column error for lasso
        self.Xs[1,:] = self.Xs[1,:] + 0.001
        self.Ys[1,:] = self.Ys[1,:] + 0.001
        self.Xs_valid[1,:] = self.Xs_valid[1,:] + 0.001
        self.Ys_valid[1,:] = self.Ys_valid[1,:] + 0.001
    
    
    def run(self):   
        
        
        # train        
        reg = MultiOutputRegressor(LassoLarsCV(cv=5, n_jobs=10, max_iter=50, normalize=False), n_jobs = 10)
        reg.fit(self.Xs,self.Ys)
        
        # predict
        predict_y = reg.predict(self.Xs_valid)
        
        # evaluate
#         e.evaluate(predict_y,self.Ys_valid,"LassoLars Linear Regression")   
#         e.regression_eval_multioutput(self.Ys_valid[:,:3],predict_y[:,:3]) # plot for the first 3 features the correlation
        
        return predict_y
    
    
    def run_batches(self,n=300):
        # train
        
        # predict
        
        # evaluate
        pass




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
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
import matplotlib.pyplot as plt


############################### Linear regression using coordinate descent ################################
class SimpleRegression(object):
    '''

    Args:
        fit_intercept (bool):
        type_estimator (str): allowed values ['ridge','huber']
        loss_function (str)L allowed values ['mse','lad']
    '''

    ESTIMATOR = ['ridge', 'huber']
    LOSS_FUNCTION = ['mse', 'lad']
    FIT_INTERCEPT = 'fit_intercept'
    EPSILON = 'epsilon'
    MAX_ITER = 'max_iter'
    ALPHA = 'alpha'

    def __init__(self,
                 fit_intercept=True,
                 type_estimator='ridge',
                 loss_function='mse',
                 fit_parallel=False,
                 params_estimator={}):

        self.fit_intercept = fit_intercept
        self.fit_parallel = fit_parallel
        self.type_estimator = type_estimator
        self.loss_function = loss_function

        # estimator parameters
        self.params_estimator = params_estimator

        if self.__check__(self.type_estimator, SimpleRegression.ESTIMATOR) == False:
            raise ValueError(f'Estimator does not exist! Please choose from the following list: {SimpleRegression.ESTIMATOR}')

        if self.__check__(self.loss_function, SimpleRegression.LOSS_FUNCTION) == False:
            raise ValueError(f'Estimator does not exist! Please choose from the following list: {SimpleRegression.LOSS_FUNCTION}')


        self.model = self.__init_model__()

        # train multiple models of same type at the same time
        self.model_assembly = None



    def __init_model__(self):

        m = None
        if self.type_estimator == 'huber':
            m = HuberRegressor(fit_intercept=self.params_estimator[SimpleRegression.FIT_INTERCEPT],
                               epsilon=self.params_estimator[SimpleRegression.EPSILON],
                               tol=self.params_estimator[SimpleRegression.TOL],
                               max_iter=self.params_estimator[SimpleRegression.MAX_ITER],
                               alpha=self.params_estimator[SimpleRegression.ALPHA])
        if self.type_estimator == 'ridge':
            m = Ridge(fit_intercept=self.params_estimator[SimpleRegression.FIT_INTERCEPT],
                      alpha=self.params_estimator[SimpleRegression.ALPHA])

        return m


    def __check__(self, value, allowed_values):
        '''Check validity of the input

        Args:
            value:
            allowed_values:

        Returns:
        '''
        if value in allowed_values: return True
        return False


    def train(self, X, y):

        if self.fit_parallel:
            lm_array = Parallel(n_jobs=4)(
                delayed(self.__train_single__(X[:, gene].reshape(-1, 1), y[:, gene].reshape(-1, 1)) for gene in range(X.shape[1])))
            self.model_assembly = lm_array
        else:
            self.model = self.__train_single__(X,y)


    def __train_single__(self, X, y):
        return self.model.fit(X,y)


    def predict(self,X):
        if self.fit_parallel:
            # X is a multi input
            y_first = self.model_assembly[0].predict(X[:, 0].reshape(-1, 1))
            y_pred = np.zeros((y_first.shape[0], len(self.model_assembly)))
            y_pred[:, 0] = y_first.squeeze()
            for i in range(1, len(self.model_assembly)):
                y_pred[:, i] = (self.model_assembly[i].predict(X[:, i].reshape(-1, 1))).squeeze()
            return y_pred
        else:
            # X is a single input array (n-dim)
            return self.model.predict(X)

    @property
    def coef(self):
        coef_list = []
        if self.model_assembly:
            for m in self.model_assembly:
                coef_list.append(m.coef_)
            return coef_list
        return self.model.coef_

    @property
    def intercept(self):
        intercept_list = []
        if self.model_assembly:
            for m in self.model_assembly:
                intercept_list.append(m.intercept_)
            return intercept_list
        return self.model.intercept_

    @property
    def print_model(self):
        print("Coef:",self.coef)
        print("Intercept:", self.intercept)




class CoordinateDescent(object):
    '''

    '''

    def __init__(self,
                 components_dict={},
                 loss_function='mse',
                 epsilon_stop=0.0001, # early stopping
                 enable_plot=True):
        self.components_dict = components_dict
        self.loss_function = loss_function
        self.enable_plot = enable_plot
        self.epsilon_stop = epsilon_stop

    def train(self, components_X, y):

        prediction_per_component = {}

        # initialize predictions
        for key in self.components_dict: prediction_per_component[key] = self.__init_prediction__(y.shape)

        # run training
        i = 0
        loss = 0
        while (True):
            i += 1
            pass


    def predict(self):
        pass


    def __init_prediction(self, shape):
        return np.zeros(shape)


############################################### Declarative fashion


def train_model(X_, y_, fit_intercept=True, type_estimator='ridge'):
    m = Ridge(fit_intercept=fit_intercept, alpha=1)
    if type_estimator == 'huber':
        m = HuberRegressor(fit_intercept=fit_intercept,epsilon=1.5,tol=0.001,max_iter = 50, alpha=0)
    return m.fit(X_, y_)


def model2(X,y,size_genes, fit_intercept, type_estimator='ridge'):
    lm_array = []
    lm_array = Parallel(n_jobs=4)(delayed(train_model)(X[:,gene].reshape(-1,1), y[:,gene].reshape(-1,1), fit_intercept, type_estimator=type_estimator) for gene in range(size_genes))
    return lm_array


def predict_model2(model, X):
    y_first = model[0].predict(X[:,0].reshape(-1,1))
    y_pred = np.zeros((y_first.shape[0], len(model)))
    y_pred[:,0] = y_first.squeeze()
    for i in range(1,len(model)):
        y_pred[:,i] = (model[i].predict(X[:,i].reshape(-1,1))).squeeze()
    return y_pred


def coef_model2(model):
    print("\nFitted Param:")
    print("Tetas: ")
    for lm in model: print(f'coef={lm.coef_};intercept={lm.intercept_}')
    print("")

    
def model_Q(X,y,fit_intercept, type_estimator='ridge'):
    if type_estimator == 'huber':
        return HuberRegressor(fit_intercept=fit_intercept,epsilon=1.35,tol=0.0001).fit(X, y)
    return Ridge(fit_intercept=fit_intercept, alpha=10).fit(X, y)


def predict_model_Q(model, X):
    return model.predict(X)


def coef_model_Q(model):
    print("\nFitted Param:")
    print("Betas: ")
    print(f'coef={model.coef_};intercept={model.intercept_}')

    
def model_P(X,y,fit_intercept, type_estimator='ridge'):
    if type_estimator == 'huber':
        return HuberRegressor(fit_intercept=fit_intercept,epsilon=1.35,tol=0.0001).fit(X, y)
    return Ridge(fit_intercept=fit_intercept, alpha=10).fit(X, y)


def predict_model_P(model,X):
    return model.predict(X)


def coef_model_P(model):
    print("\nFitted Param:")
    print("Aphas: ")
    print(f'coef={model.coef_};intercept={model.intercept_}')


def compute_loss(y_true, y_pred, type_loss='mse'):

    if type_loss == 'mse':
        return mean_squared_error(y_true, y_pred)

    if type_loss == 'lad': # L1-norm loss function (least absolute deviations)
        return np.sum(np.abs(y_true-y_pred)) # convert from tensor to numpy

    if type_loss == 'lms':
        return None

    if type_loss == 'mape': # Mean absolute percentage error
        loss = np.median(np.sum(np.divide(np.abs(y_true - y_pred), (np.abs(y_true)+0.000001)), axis=0) / y_true.shape[0])
#         print(loss)
        return loss
        
    return None


def model4(X, X_P, X_Q, y, size_genes, size_indiv, title_loss = 'Loss', loss_function='mse', estimator='ridge', enable_plot_loss=True):

    print(estimator)
    q_pred = np.zeros(y.shape)
    p_pred = np.zeros(y.shape)
    model2_pred = np.zeros(y.shape)

    loss = 0
    epsilon = 0.001
    i = 0

    # loss - total, partial loss for P (indiv effect), Q (gene effect) and model2 (gene-tissue variance)
    x_plot = []
    y_plot = []
    q_plot = []
    p_plot = []
    model2_plot = []

    y_flat = y.flatten()
    X_P_block = np.repeat(X_P, repeats=size_genes, axis=0)
    X_Q_block = np.tile(X_Q, (size_indiv, 1))

    while (True):
        i += 1

        # fit partial models
        m_p = model_P(X_P_block, (y - q_pred - model2_pred).flatten(), fit_intercept=False, type_estimator=estimator)
        m_q = model_Q(X_Q_block, (y - p_pred - model2_pred).flatten(), fit_intercept=True, type_estimator=estimator)
        m_model2 = model2(X, y - p_pred - q_pred, size_genes, fit_intercept=False, type_estimator=estimator)

        # predict
        q_pred = predict_model_Q(m_q, X_Q_block).reshape((size_indiv, size_genes))
        p_pred = predict_model_P(m_p, X_P_block).reshape((size_indiv, size_genes))
        model2_pred = predict_model2(m_model2, X)

        #         loss = compute_loss(y,gene_effect_pred.reshape(size_indiv, size_genes) + indiv_effect_pred)
        # loss MSE
        loss = compute_loss(y, q_pred + p_pred + model2_pred, type_loss=loss_function)
        x_plot.append(i)
        y_plot.append(loss)

        # partial loss
        # q_plot.append(compute_loss(y, q_pred, type=loss_function))
        # p_plot.append(compute_loss(y, p_pred, type=loss_function))
        # model2_plot.append(compute_loss(y, model2_pred, type=loss_function))
        if len(y_plot) >= 2 and abs(y_plot[-1] - y_plot[-2]) <= epsilon:
            if loss_function == 'lad':
                print("lad_mean=",loss/(size_indiv*size_genes))
            elif estimator == 'huber':
                print(f'loss={loss}')
            break

    # save model
    model_dict = {
        'P': m_p,
        'Q': m_q,
        'model2': m_model2
    }

    # plot loss
    if enable_plot_loss:

        plt.ioff()  ## Note this correction
        fig = plt.figure()
        #    plt.axis([0,i,0,np.max(y_plot)+0.2])
        plt.plot(x_plot, y_plot)
        #     plt.plot(x_plot,q_plot)
        #     plt.plot(x_plot,p_plot)
        #     plt.plot(x_plot,model2_plot)
        #     plt.semilogy()
        plt.legend(['loss=' + loss_function], loc='upper left')
        #     plt.legend(['loss', 'Q_loss', 'P_loss', 'X_loss'], loc='upper left')
        plt.title(title_loss)
        plt.show()

    return model_dict


def predict_model4(model, X, X_P, X_Q, size_genes, size_indiv):
    X_P_block = np.repeat(X_P, repeats=size_genes, axis=0)
    X_Q_block = np.tile(X_Q, (size_indiv, 1))

    predq = predict_model_Q(model['Q'], X_Q_block).reshape((size_indiv, size_genes))
    predp = predict_model_P(model['P'], X_P_block).reshape((size_indiv, size_genes))
    predx = predict_model2(model['model2'], X)
    return predq + predp + predx


def coef_model4(model):
    coef_model_Q(model["Q"])
    coef_model_P(model["P"])

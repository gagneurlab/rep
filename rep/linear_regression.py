import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from anndata import AnnData

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
import rep.preprocessing_new as prep
import rep.datasets as d
import rep.models as m
from anndata import AnnData

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



################################ auxiliary methods

def prepare_linear_regression_input(tissue, n_comp_indiv_effect=25, n_comp_gene_effect=25, gene_list=None):

    # Step 1. load data

    path = "/s/project/rep/"
    # path = "/home/mada/Uni/Masterthesis/online_repo/rep/data/"
    y_targets_h5 = path + "processed/gtex/input_data/Y_targets_pc_onlyblood_log_tpm.h5"
    x_inputs_h5 = path + "processed/gtex/input_data/X_inputs_pc_onlyblood_log_tpm.h5"
    train_dataset, valid_dataset = d.rep_blood_expression(x_inputs_h5, y_targets_h5, gene_list=gene_list, to_tissue=[tissue], from_tissue=['Whole Blood'])


    metadata_samples_train, metadata_samples_valid = train_dataset.metadata, valid_dataset.metadata
    x_train, y_train = train_dataset.inputs, train_dataset.targets
    x_valid, y_valid = valid_dataset.inputs, valid_dataset.targets

    # samples train and valid - use the blood samples to compute PCA
    samples_blood_train_all = metadata_samples_train['From_sample'].tolist()
    samples_blood_valid_all = metadata_samples_valid['From_sample'].tolist()


    # load gtex data - individual effect (all blood samples)
    features_file = path + "processed/gtex/recount/recount_gtex_norm_tmp.h5ad"
    gtex = prep.RepAnnData.read_h5ad(features_file)
    gtex_filtered = gtex[gtex.samples.index.isin(samples_blood_valid_all + samples_blood_train_all)]
    
        
    print(np.isnan(np.array(gtex_filtered.X)))


    # load gtex data - gene effect (all samples all tissues - only individuals in the training set)
    features_file = path + "processed/gtex/recount/recount_gtex_norm_tmp.h5ad"
    train_individuals_file = path + "processed/gtex/recount/train_individuals.txt"
    train_individuals = prep.read_csv_one_column(train_individuals_file)

    gtex_raw = prep.RepAnnData.read_h5ad(features_file)
    gtex_raw_train = gtex_raw[gtex_raw.samples['Individual'].isin(train_individuals)]
 

    # Step 2. compute gene effect (Q) and individual effect (P)

    Q = pca_gene_effect(gtex_raw_train, n_comp=n_comp_gene_effect, gene_list=gene_list)

    P_blood_all = pca_individual_effect(gtex_filtered, n_comp=n_comp_indiv_effect)
    P_train_unsorted = P_blood_all[P_blood_all.obs.index.isin(samples_blood_train_all)]
    P_valid_unsorted = P_blood_all[P_blood_all.obs.index.isin(samples_blood_valid_all)]



    # 3. Filter data by tissue

    (y_blood_train, y_tissue_train, meta_train), (y_blood_valid, y_tissue_valid, meta_valid) = get_linear_regression_data(tissue,
                                    metadata_samples_train,
                                    metadata_samples_valid,
                                    x_train,
                                    y_train,
                                    x_valid,
                                    y_valid)

    # sort PCA to match the input data
    P_train = sort_by_index(P_train_unsorted, samples_blood_train_all)
    P_valid = sort_by_index(P_valid_unsorted, samples_blood_valid_all)


    n_indiv_train, n_genes_train = y_blood_train.shape
    n_indiv_valid, n_genes_valid = y_blood_valid.shape

    # replace inf values if exists

    return (y_blood_train, y_tissue_train, meta_train, P_train,
            Q.values.astype(np.float32), n_genes_train, n_indiv_train), \
           (y_blood_valid, y_tissue_valid, meta_valid, P_valid,
            Q.values.astype(np.float32), n_genes_valid, n_indiv_valid)


def sort_by_individual(anndata, sorter_list, column):
    df = pd.DataFrame(anndata.X, index=anndata.obs.index.tolist())
    df = pd.concat([df, anndata.obs], axis=1, join_axes=[df.index])

    # keep only one entry
    already_visited = []
    to_keep = []

    for i in range(df.shape[0]):
        indiv = df[column].iloc[i]
        if indiv in sorter_list and indiv not in already_visited:
            to_keep.append(i)
            already_visited.append(indiv)

    df = df.iloc[to_keep, :]

    df.sort_values([column], ascending=[sorter_list], inplace=True)

    anndata_new = AnnData(X=df.iloc[:, :anndata.X.shape[1]], obs=df.iloc[:, anndata.X.shape[1]:], var=anndata.var)
    anndata_new.obs = df.iloc[:, anndata.X.shape[1]:]
    anndata_new.var = anndata.var

    # sort matrix according to index
    return anndata_new


def sort_by_index(anndata, sorter_list):

    # sort matrix according to index (insert duplicates if necessary - some individuals have twice sequenced some tissues)
    df = pd.DataFrame(columns=anndata.var.index.tolist())
    for i, sample_id in enumerate(sorter_list):
        df.loc[i] = anndata[anndata.obs.index == sample_id].X.flatten()

    return df.values



def pca_individual_effect(gtex, n_comp=25, tissue='Whole Blood', pca_type=PCA):
    '''Compute PCA - individual effect

    Args:
        gtex:
        n_comp:
        tissue:
        pca_type:

    Returns:

    '''
    slice_blood = gtex[gtex.obs['Tissue'] == tissue]
    metadata = slice_blood.obs
    features = slice_blood.X
    if features.shape[0] >= n_comp:
        features_centered = StandardScaler().fit_transform(features)
        pca = pca_type(n_components=n_comp)
        features_pca = pca.fit_transform(features_centered)

    # wrap PCA into AnndataObject
    return AnnData(X=features_pca, obs=metadata)


# compute PCA - gene effect
def pca_gene_effect(gtex, n_comp = 25, pca_type = PCA, gene_list=None):
    metadata = gtex.var
    features = gtex.X.transpose()

    if features.shape[0] >= n_comp:
        features_centered  = StandardScaler().fit_transform(features)
        pca = pca_type(n_components=n_comp)
        features_pca = pca.fit_transform(features_centered)
    gene_effect = pd.DataFrame(features_pca, index=metadata['gene_id'])

    if gene_list is not None:
        return gene_effect[gene_effect.index.isin(gene_list)]
    return gene_effect


def get_linear_regression_data(tissue, mt_train, mt_valid, x_train, y_train, x_valid, y_valid):

    # Filter by tissue
    index_train = np.where(mt_train['To_tissue'] == tissue)[0]
    index_valid = np.where(mt_valid['To_tissue'] == tissue)[0]

    xs_train = x_train[index_train, :]
    xs_valid = x_valid[index_valid, :]

    ys_train = y_train[index_train, :]
    ys_valid = y_valid[index_valid, :]

    mt_train_filtered = mt_train.iloc[index_train, :]
    mt_valid_filtered = mt_valid.iloc[index_valid, :]

    #     print(xs_train.shape, mt_train_filtered.shape)

    # unique_indiv_train = mt_train_filtered['Individual'].drop_duplicates().tolist()
    # visited_train = []
    # to_keep_train = []
    #
    # for i in range(xs_train.shape[0]):
    #     indiv = mt_train_filtered['Individual'].iloc[i]
    #     if indiv not in visited_train:
    #         to_keep_train.append(i)
    #         visited_train.append(indiv)
    # index_train = to_keep_train
    #
    # unique_indiv_valid = mt_valid_filtered['Individual'].drop_duplicates().tolist()
    # visited_valid = []
    # to_keep_valid = []
    # for i in range(xs_valid.shape[0]):
    #     indiv = mt_valid_filtered['Individual'].iloc[i]
    #     if indiv not in visited_valid:
    #         to_keep_valid.append(i)
    #         visited_valid.append(indiv)
    # index_valid = to_keep_valid
    #
    # xs_train = xs_train[index_train, :]
    # xs_valid = xs_valid[index_valid, :]
    # ys_train = ys_train[index_train, :]
    # ys_valid = ys_valid[index_valid, :]

    xs_train = xs_train.astype(np.float32) + 0.00001
    ys_train = ys_train.astype(np.float32) + 0.00001

    xs_valid = xs_valid.astype(np.float32) + 0.00001
    ys_valid = ys_valid.astype(np.float32) + 0.00001

    # mt_train_filtered = mt_train_filtered.iloc[index_train, :]
    # mt_valid_filtered = mt_valid_filtered.iloc[index_valid, :]

    return (xs_train, ys_train, mt_train_filtered), (xs_valid, ys_valid, mt_valid_filtered)



    
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
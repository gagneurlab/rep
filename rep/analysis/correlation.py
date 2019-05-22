import os
import sys
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from rep import preprocessing_new as p
import pickle
import warnings;
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.simplefilter('ignore')


# def compute_correlation_pairwise(var1, var2):
#     '''Compute correlation for two samples

#     Args:
#         var1 (:AnnData):
#         var2 (:AnnData):

#     Returns:
#     '''
#     return {'Corr': np.corrcoef(np.array(var1.X).flatten(), np.array(var2.X).flatten())[0][1],
#             'Sample1': var1.obs.index.tolist()[0],
#             'Sample2': var2.obs.index.tolist()[0],
#             'From_Tissue': var1.obs['Tissue'].tolist()[0],
#             'To_Tissue': var2.obs['Tissue'].tolist()[0],
#             'From_Individual': var1.obs['Individual'].tolist()[0],
#             'To_Individual': var2.obs['Individual'].tolist()[0]}


# def compute_correlation_one_against_all(dataobj, s1, list_s2):
#     array_corr = []

#     for s2 in list_s2:
#         array_corr.append(compute_correlation_pairwise(dataobj[dataobj.obs.index == s1],
#                                                        dataobj[dataobj.obs.index == s2]))

#     df = pd.DataFrame(data=array_corr)
#     df.to_pickle(f'samples_correlation_{s1}.pkl')


def compute_correlation_pairwise(dataobj, s1, s2):
    '''Compute correlation for two samples

    Args:
        dataobj (:AnnData):
        s1 (str):
        s2 (str):

    Returns:
        correlation between 2 samples
    '''
    print(s1, s2)
    return np.corrcoef(np.array(dataobj[dataobj.obs.index == s1].X).flatten(), 
                       np.array(dataobj[dataobj.obs.index == s2].X).flatten())[0][1]
    
    
def compute_parallel(dataobj):

    samples = dataobj.obs.index.tolist()
    print("Total number of samples:",len(samples))
    print("Expect total number of pairs:",len(samples)**2)
#     pairs = []
#     for i, s1 in enumerate(samples):
#         pairs.append((s1, samples[i:]))

    corr_array = Parallel(n_jobs=20)(
        delayed(compute_correlation_pairwise)(dataobj, s1, s2) for s1 in samples for s2 in samples)
    
    return np.array(corr_array).reshape(-1, len(samples))

if __name__ == "__main__":

    gtex = p.load(os.path.join(os.readlink(os.path.join("..", "..", "data")), "processed", "gtex", "recount",
                               "recount_gtex_norm_tmp.h5ad"))
    gtex_filtered = gtex[gtex.samples['Tissue'].isin(["Whole Blood", "Muscle - Skeletal"])]

    data_array = compute_parallel(gtex_filtered)
    np.save("sample_correlation.txt", data_array)



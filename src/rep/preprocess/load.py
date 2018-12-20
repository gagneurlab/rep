import sys
import time
from itertools import permutations

import argh
import os


# import scanpy
import anndata
import pandas as pd

from rep.src.rep.preprocess.constants import ANNDATA_CST as a

def print_anndata(toprintanndata):
    print("anndata.X")
    print(toprintanndata.X)
    print("anndata.var")
    print(toprintanndata.var_names)
    print(toprintanndata.var)
    print("anndata.obs")
    print(toprintanndata.obs_names)
    print(toprintanndata.obs)
    print()


def load_vars(obj, csvfile, sep=","):
    """
    Load var (columns) description for the summarized experiment
    :param annobj:
    :param csvfile:
    :param sep:
    """

    if csvfile:
        varaux = pd.DataFrame(pd.read_csv(os.path.abspath(csvfile), header=None, delimiter=sep, index_col=0))
        obj.var = varaux
        obj.var_names = list(varaux.index)


def load_obs(obj, csvfile, sep=","):
    """
    Load obs (rows) description for the summarized experiment
    :param annobj:
    :param csvfile:
    :param sep:
    """
    if csvfile:
        obsaux = pd.DataFrame(pd.read_csv(os.path.abspath(csvfile), header=None, delimiter=sep, index_col=0))
        obj.obs = obsaux
        obj.obs_names = list(obsaux.index)


def load_count_matrix(filename, sep=",", varanno=None, obsanno=None):
    """
    Load count matrix and put this into a summarized experiment.
    Add anndata.var (col description) and anndata.obs (row description) annotation

        ---- var ---
    |   T1_s1,T2_s2,T3_s,T4_1
    |   G1,10,20,30,40
    obs G2,5,10,20,30
    |   G3,6,7,8,9

    varanno input example:
    T1_s1  F  Tissue1  Sample1  rnaseq
    T2_s2  M  Tissue2  Sample2  rnaseq
    T3_s1  F  Tissue3  Sample1  rnaseq
    T4_s2  M  Tissue4  Sample2  rnaseq

    obsanno input example:
    G1,hg19,t1,chr1,1111,-
    G2,hg19,t2,chr2,2222,-
    G3,hg19,t3,chr3,3333,-

    :param filename: .csv file containing a n_obs x n_vars count matrix
    :param sep: separator, this should be the same for count_marix, varanno and obsanno
    :param varanno: additional annotation for cols (e.g. sample_tissue description)
    :param obsanno: additional annotation for rows (e.g. gene id description)
    :return: anndata object
    """
    abs_path = os.path.abspath(filename)

    # read count matrix
    annobj = anndata.read_csv(abs_path, delimiter=sep)

    # read var data (samples description)
    load_vars(annobj, varanno, sep)

    # read obs data (index description e.g gene annotation)
    load_obs(annobj, obsanno, sep)

    return annobj


def load_anndata_from_file(filename, backed=False, varanno=None, obsanno=None, sep=","):
    """
    Load anndata format specific data into an anndata object.
    :param filename: .h5ad file containing n_obs x n_vars count matrix and further annotations
    :param backed: default False - see anndata.read_h5ad documentation https://media.readthedocs.org/pdf/anndata/latest/anndata.pdf
                   if varanno and obsanno are provided please set backed = r+
    :param varanno: additional annotation for cols (e.g. sample_tissue description)
    :param obsanno: additional annotation for rows (e.g. gene id description)
    :param sep: separator for varanno and obsanno files
    :return: anndata object
    """
    abspath = os.path.abspath(filename)
    annobj = anndata.read_h5ad(abspath, backed=backed)
    try:

        if varanno or obsanno:
            if backed != 'r+':
                raise BackedError

    except BackedError:
        print("Exception [varanno] or [obsanno] provided! Please set [backed='r+']")
        exit(1)

    # read var data
    load_vars(annobj, varanno, sep)
    # read obs data
    load_obs(annobj, obsanno, sep)

    return annobj


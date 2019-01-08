"""
Prepare the training data for gene expression imputation (all pairs of tissues per samples)

Features:

Convert the raw count matrix (e.g. genes x samples) into a summarized expriment using
anndata.py https://media.readthedocs.org/pdf/anndata/latest/anndata.pdf

Allow flexible filtering over the count matrix...
"""

import argh
import os
import sys
import time
from itertools import chain
from itertools import permutations
from itertools import combinations
import traceback

import numpy
# import scanpy
import math
import anndata
import pandas as pd
import numpy as np
from scipy import sparse

from rep.constants import ANNDATA_CST as a


def print_anndata(toprintanndata):
    print("anndata.X ----")
    print(toprintanndata.X)
    print("anndata.var ----")
    print(toprintanndata.var[:5,:5])
    print("anndata.obs ----")
    print(toprintanndata.obs[:5,:5])
    print()


def load_df(csvfile, header=None, delimiter=",", index_col=0):
    return pd.read_csv(os.path.abspath(csvfile), header=header, delimiter=delimiter, index_col=index_col)


def load_samples(obj, csvfile, sep=","):
    """Load samples (columns) description for the summarized experiment

    Args:
        annobj (:obj:`AnnData`):
        csvfile (str):
        sep (str):
    """

    if csvfile:
        varaux = load_df(csvfile, header=0, delimiter=sep, index_col=0)
        obj.var = varaux
        obj.var.columns = list(varaux.columns.values)
        obj.var_names = list(varaux.index)


def load_genes(obj, csvfile, sep=","):
    """Load genes (rows) description for the summarized experiment

     Args:
        annobj (:obj:`AnnData`):
        csvfile (str):
        sep (str):
    """
    if csvfile:
        obsaux = load_df(csvfile, header=0, delimiter=sep, index_col=0)
        obj.obs = obsaux
        obj.obs.columns = list(obsaux.columns.values)
        obj.obs_names = list(obsaux.index)


def load_count_matrix(obj, filename, sep=","):
    """Load count matrix and put this into a summarized experiment.
       Add anndata.samples (col description)and  anndata.genes (row description) annotation

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

    Args:
        filename (str): .csv file containing a n_obs x n_vars count matrix
        sep (str): separator, this should be the same for count_marix, varanno and obsanno
        samples_anno (str,optional): additional annotation file for cols (e.g. sample_tissue description). Default None
        genes_anno (str,optional): additional gene annotation file (e.g. gene id description). Default None

    Returns:
        Anndata object
    """
    abs_path = os.path.abspath(filename)

    # read count matrix
    annobj = anndata.read_csv(abs_path, delimiter=sep)

    # read samples data (samples description)
    load_samples(annobj, samples_anno, sep)

    # read genes data (index description e.g gene annotation)
    load_genes(annobj, genes_anno, sep)

    return annobj


def create_anndata(counts_file, samples_anno=None, genes_anno=None, sep=","):
    """Creates an AnnData Object

    Args:
        counts_file (str): file containing the counts
        samples_anno (str):
        genes_anno:
        sep:

    Returns:
        AnnData object
    """
    x = load_df(counts_file, header=0, delimiter=sep)
    annobj = anndata.AnnData(X=x)
    load_genes(annobj, genes_anno, sep)
    load_samples(annobj, samples_anno, sep)

    return annobj


def load(filename, backed=False, samples_anno=None, genes_anno=None, sep=","):
    """Load anndata object

    Args:
        filename (str): .h5ad file containing n_obs x n_vars count matrix and further annotations
        backed (bool): default False - see anndata.read_h5ad documentation https://media.readthedocs.org/pdf/anndata/latest/anndata.pdf
                       if varanno and obsanno are provided please set backed = r+
        samples_anno (str,optional) : sample_tissue description file
        genes_anno (str,optional): gene id description file
        sep (str): separator for varanno and obsanno files

    Returns:
        Anndata object
    """
    abspath = os.path.abspath(filename)
    annobj = anndata.read_h5ad(abspath, backed=backed)
    try:

        if samples_anno or genes_anno:
            if backed != 'r+':
                raise BackedError

    except BackedError:
        print("Exception [varanno] or [obsanno] provided! Please set [backed='r+']")
        exit(1)

    # read samples description
    load_samples(annobj, samples_anno, sep)
    # read genes description
    load_genes(annobj, genes_anno, sep)

    return annobj


def filter_df_by_value(df, jsonFilters):
    """Find rows matching the filtering criteria.

    Args:
        df (:obj:Data.frame): count matrix
        jsonFilters (dict): json file, where key is the column, and value are the admited (filtering) values

    Returns:
        row.names for which the filtering applies
    """
    names = list(df.index)
    for key in jsonFilters:
        if key == 0:  # index of data.frame
            names = list(set(names) & set(jsonFilters[key]))
            continue

        # regular columns
        name_per_key = []
        for val in jsonFilters[key]:
            # get index of rows which column matches certain value
            aux_names = df.index[df[str(key)] == val].tolist()

            # intersect 2 lists
            name_per_key = list(set(name_per_key) | set(aux_names))

        names = list(set(names) & set(name_per_key))


    # rows which mach all filtering criteria
    return names


def filter_anndata_by_value(annobj, filters):
    """Apply filtering on anndata.samples and anndata.genes dataframes using value filtering

    Args:
        filters (dict): Please follow the structure bellow (if no 'obs' filtering, then the key does not have to be in the json)
                    {'var':{
                            1:['F'],
                            4:['rnaseq']
                            #col1:['value1','value2',...],
                            #col2:['value3',...]
                        },
                    'obs':{
                            3:['chr3']
                        }
                    }

    Returns:
        tuple (filtered var_names, filtered obs_names) - use this output as input to filter the count matrix (anndata.X)
    """
    for key in filters:
        if key == a.SAMPLES:  # filtering by var (col)
            filtered_var_names = filter_df_by_value(annobj.var, filters[key])
        elif key == a.GENES:  # filtering by var (col)
            filtered_obs_names = filter_df_by_value(annobj.obs, filters[key])

    if a.SAMPLES not in filters:
        filtered_var_names = list(annobj.var_names)  # no filtering
    if a.GENES not in filters:
        filtered_obs_names = list(annobj.obs_names)

    return (filtered_var_names, filtered_obs_names)


def filter_anndata_by_region(annobj, filterJson, regions, format):
    """Filtering by region over the anndata.obs (the return its a tuple just for completeness, the anndata.var stay the same)

    Args:
        annobj (:obj:AnnData):
        filterJson (dict): {
                        'chromosome': col1,
                        'start' : col2,
                        'stop' : col3,
                        'strand' : col4
                        } - mark columns which indicate the regions in the anndata obj
        regions (str): file in standard format, e.g. bed or gtf
        format (str): bed or gtf

    Returns:
        tuple (filtered var_names, filtered obs_names)
    """
    pass


def save(annobj, outname=None):
    """Write .h5ad-formatted hdf5 file and close a potential backing file. Default gzip file

    Args:
        annobj (:obj:AnnData):
        outname (str): name of the output file (needs to end with .h5ad)

    Returns:
        output filename (if none specified then this will be random generated)
    """

    if outname:
        abspath = os.path.abspath(outname)
        name = abspath
    else:
        name = os.path.abspath("tmp" + str(int(time.time())) + ".h5ad")

    # convert header to string (avoid bug)
    annobj.var_names = [str(v) for v in annobj.var_names]
    annobj.obs_names = [str(o) for o in annobj.obs_names]
    annobj.var.rename(index={r: str(r) for r in list(annobj.var.index)},
                      columns={c: str(c) for c in list(annobj.var.columns)},
                      inplace=True)
    annobj.obs.rename(index={r: str(r) for r in list(annobj.obs.index)},
                      columns={c: str(c) for c in list(annobj.obs.columns)},
                      inplace=True)

    annobj.write(name)

    return name


def group_by(df, column, index_subset):
    """Group for instance e.g. tissues per sample.

    Args:
        df (:obj:Data.frame): data.frame with the values on which we would like to group
        column (str,int): column over we will group
        index_subset (list(str)): filtered list of indexes (using e.g filter_anndata_by_value)

    Returns:
        {'value':[list of index_subset matching the value]}
    """
    dict = {}
    df_new = df.groupby(str(column))
    for group in df_new.groups:
        # rearrage the index_subset by grouping over e.g. tissue
        dict[group] = list(set(list(df_new.groups[group])) & set(index_subset))

    return dict


def arrangements(list_of_samples, n=None):
    """This generates all arrangements of length n (the order is relevant)

    Args:
        list_of_samples (list(str)):
        n (int): number of elements in sublist

    Returns:
        list of tuples of samples
    """
    if not n:  # assume permutations = arrangements(k,n) where k == n
        n = len(list_of_samples)
    return [p for p in permutations(list_of_samples, n)]


def build_x_y(annobj, cross_list):
    """Build the two matrices X (train) and Y (labels)

    Args:
        annobj (:obj:AnnData):
        cross_list (list((str,str))): pairs of indexes
        obs_names (str): filtering over the features (e.g. genes)

    Returns:
        (df_X,def_Y) where X  and Y of size len(cross_list) x len(obs_names)
    """
    # create indexes
    length = len(annobj.obs_names)
    index_elem = [str(str(x) + "_" + str(y)) for i, (x,y) in enumerate(cross_list)]

    df_X = pd.DataFrame(index=annobj.obs_names, columns=index_elem)
    df_Y = pd.DataFrame(index=annobj.obs_names, columns=index_elem)
    
    print("Total pairs: " + str(len(cross_list)))
    
    # build accessing dictionary
    access = {x:i for i,x in enumerate(annobj.var_names)}
    
    # convert to sparse matrix
    m = sparse.csr_matrix(np.matrix(annobj.X))
    
    for i, (x, y) in enumerate(cross_list):
        custom_index = str(str(x) + "_" + str(y))
        
        # add X
        vector = m[:,access[x]]
        df_X.loc[:,custom_index] = vector
        
        # add Y
        vector = m[:,access[y]]
        df_Y.loc[:,custom_index] = vector

    return (df_X.T, df_Y.T)

def rnaseq_cross_tissue(anndata_obj, individuals, gene_ids, target_transform=None,
                        input_transform=None, shuffle=False):
    """Prepare the traning data by:
        1. Filtering the anndata matrix, by col (gene_ids) and rows (sample_ids)
        2. Normalize data if necessary
        3. Stratify samples
        4. Create all pair between all tissues available for one individuum
        this uses var_names and obs_names. These can be either a list of keys or a dictionary (grouping the samples
        by condition - this means the pairs will be generated only within a group)

    Args:
        anndata_obj (:obj:AnnData): h5ad format
        individuals (list(str)): (individuals) in the summarized experiment (assume there is a row called Individuals)
        gene_ids (list(str)): (gene_ids/rows) in the summarized experiment
        traget_transform (str): reference to a function  function
        input_transform (str): reference to a function
        shuffle (bool):

    Returns:
        (df.X,df.Y) # index=sample_tissue

    """

    # get samples
    samples_df = anndata_obj.var
    sample_ids = filter_df_by_value(samples_df, {'Individual': individuals})
    sample_ids = group_by(samples_df, 'Individual', sample_ids)

    n = 2  # pairs of tissues
    cross_list = []
    samples = []
    if isinstance(sample_ids, (list,)):
        cross_list = arrangements(sample_ids, n)

    if isinstance(sample_ids, dict):
        sample_aux = []
        for key in sample_ids:
            cross_list += arrangements(sample_ids[key], n)
            
            # flatten var_names dict
            sample_aux += sample_ids[key]

        samples = list(set(sample_aux))

    anndata_filtered_var = anndata_obj[:, samples]
    anndata_sliced = anndata_filtered_var[gene_ids,:]
    
    (X, Y) = build_x_y(anndata_sliced, cross_list)
    print("4.4 Build the two matrices X and Y")
    
    return (X, Y)

def split_by_individuals(annobj, fraction=[3. / 5, 1. / 5, 1. / 5], groupby=['Gender','Seq'], stratified=True, shuffle=False):
    """Split dataset using stratified individuals by Gender ..
    
    Args:
        annobj (:obj:AnnData): Assumes a column its named Individual
        percent (list(float,float,float)): split fraction for train valid and test
        stratified (bool):
        shuffle (bool):

    Returns:
        (train_individuals,valid_individuals, test_individuals)
    """

    # Stratify
    df = annobj.var.reset_index()[['Individual'] + groupby]
    df.drop_duplicates(inplace = True)
    print(df.shape)
    print(df[:10])
    df_grouped = df.groupby(groupby, as_index=False)
    print(df_grouped)
    
    train_individuals = []
    valid_individuals = []
    test_individuals = []
    for name, group in df_grouped:  # get same fraction from each group
        if group.shape[0] > 3:
            index_aux = list(map(lambda x: math.ceil(group.shape[0] * x), fraction))
        elif group.shape[0] == 3:
            index_aux = [1,1,1]
        elif group.shape[0] == 2:
            index_aux = [1,0,1]
        else:
            index_aux = [1,0,0]

        train_individuals += group.iloc[:index_aux[0],:]['Individual'].tolist()
        valid_individuals += group.iloc[index_aux[0]: (index_aux[0] + index_aux[1]),:]['Individual'].tolist()
        test_individuals += group.iloc[(index_aux[0] + index_aux[1]):,:]['Individual'].tolist()

    return (train_individuals,valid_individuals,test_individuals)

if __name__ == '__main__':

    # assembling:
    parser = argh.ArghParser()
    parser.add_commands([load, create_anndata])

    if sys.argv[1] == 'explore':
        # dispatching:
        parser.dispatch()

    elif sys.argv[1] == 'example':
        # run example

        # create a summarized experiment
        # print("1. Read csv + anno matrices:")
        # print()
        # annobj = create_anndata("../data.csv", sep=",", samples_anno="../anno.csv", genes_anno="../anno_obs.csv")
        # save(annobj,"test.h5ad")

        print("1*.Reload from file HDF5")
        print()
        annobj = load("test.h5ad")
        print_anndata(annobj)

        print("2. Stratify individuals")
        (train,valid,test) = split_by_individuals(annobj)
        print(train,valid,test)

        (X_train, Y_train) = rnaseq_cross_tissue(annobj, individuals=train, gene_ids=annobj.obs_names)
        (X_valid, Y_valid) = rnaseq_cross_tissue(annobj, individuals=valid, gene_ids=annobj.obs_names)
        (X_test, Y_test) = rnaseq_cross_tissue(annobj, individuals=test, gene_ids=annobj.obs_names)

        print(X_train)
        print(Y_train)

        # # filter
        # print("4. Filter by value anndata.var and anndata.obs")
        # print()
        # (var, obs) = filter_anndata_by_value(annobj, {a.SAMPLES : {"Gender": ['M']},
        #                                            a.GENES: {0: ['G1', 'G2']}})

        # stratify and compute the cross tissue
        # (X, Y) = rnaseq_cross_tissue(annobj, individuals=['Indiv1'], gene_ids=annobj.obs_names)
        # print(X)
        # print()
        # print(Y)

        # not working - https://github.com/theislab/anndata/issues/84
        # # save in the h5ad format
        # print("2. Annobj saved here: ")
        # name = save(annobj)
        # print(name)
        #
        # # # load from h5ad file
        # print("3. Load obj from h2ad file and change annotation")
        # annobj = load_anndata_from_file(name, backed='r+', varanno="../../anno_2.csv")


    else:
        print("python preprocessing.py [explore|example]")

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
import json

from itertools import chain
from itertools import permutations
from itertools import combinations
import traceback
import h5py


import numpy
# import scanpy
import math
import anndata
import pandas as pd
import numpy as np
from scipy import sparse

from rep.constants import ANNDATA_CST as a

########################################## I/O #########################################################
########################################################################################################

def readh5(name):
    filename = name
    f = h5py.File(filename, 'r')
    X = np.array(f[list(f.keys())[0]])
#     f.close()
    
    return X

def writeh5(obj,obj_name,filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(obj_name, data=obj)
    h5f.close()
    

def readJSON(name):
    with open(name,'r') as json_file:  
        data = json.load(json_file)
    return data

def writeJSON(obj,filename):
    with open(filename,'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)

    
def print_anndata(toprintanndata):
    print("anndata.X ----")
    print(toprintanndata.X)
    print("anndata.var ----")
    print(toprintanndata.var.iloc[:5,:5])
    print("anndata.obs ----")
    print(toprintanndata.obs.iloc[:5,:5])
    print()

def load_df(csvfile, header=None, delimiter=",", index_col=0):
    return pd.read_csv(os.path.abspath(csvfile), header=header, delimiter=delimiter, index_col=index_col)


########################################## Transform function ##########################################
########################################################################################################

def mylog(df):
    """log2(df) using pseudocounts
    """
    return df.apply(lambda x: math.log(x+1))

def mylog10(df):
    """log10(df) using pseudocounts
    """
    return np.log10(df+1)



# variable which stores reference to function
function_mappings = {'log':mylog, 'log10':mylog10 }

########################################## Anndata Summ Exp.  ##########################################
########################################################################################################

def create_anndata(counts_file, samples_anno=None, genes_anno=None, sep=","):
    """Creates an AnnData Object

    Args:
        counts_file (str): file containing the counts
        samples_anno (str): file containing sampple annotation
        genes_anno (str): file containing genes annotation
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

    annobj.write_h5ad(name)
    
    
    return name

########################################## Filter anndata Summ Exp.  ###################################
########################################################################################################

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


        name_per_key = []
        # regular columns
        for val in jsonFilters[key]:
            # get index of rows which column matches certain value
            aux_names = df.index[df[str(key)] == val].tolist()
            name_per_key += aux_names
            
        # remove duplicates
        name_per_key = list(set(name_per_key))
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


########################################## Split and compute pairs   ###################################
########################################################################################################

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


def build_x_y(annobj, cross_list, input_transform=None):
    """Build the two matrices X (train) and Y (labels)

    Args:
        annobj (:obj:AnnData):
        cross_list (list((str,str))): pairs of indexes
        obs_names (str): filtering over the features (e.g. genes)
        input_transform (str): reference to a function

    Returns:
        (df_X,def_Y) where X  and Y of size len(cross_list) x len(obs_names)
    """
    # create indexes T1_T2
    index_elem = [str(str(x) + "_" + str(y)) for i, (x,y) in enumerate(cross_list)]
    
    # build accessing dictionary
    access = {x:i for i,x in enumerate(annobj.var_names)}
    
    print("Total pairs: " + str(len(cross_list)))    
    
    # apply transformation
    if input_transform:
        try:
            m = function_mappings[input_transform](annobj.X) # call the function
        except:
            return "Invalid function - please check the processing.rnaseq_cross_tissue documentation"
    else:
        m = annobj.X

    slice_x = [access[x] for (x,_) in cross_list]
    slice_y = [access[y] for (_,y) in cross_list]
    
#     mydata = sparse.csc_matrix(m)
    mydata = np.array(m)
    
    return (mydata[:,slice_x].T, mydata[:,slice_y].T, index_elem, annobj.obs_names)


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
        traget_transform (str): reference to a function  function - still not implemented
        input_transform (str): reference to a function; choose [log]
        shuffle (bool):

    Returns:
        (df.X,df.Y) # index=sample_tissue

    """

    # get samples
    samples_df = anndata_obj.var
    print("samples_df ", samples_df.shape)
    
    # slice data.frame only for the subset of individuals
    _ids = filter_df_by_value(samples_df, {'Individual': individuals})
    
    samples_df_sliced = samples_df[samples_df.index.isin(_ids)]
    print("samples_df_sliced ", samples_df_sliced.shape)
    
    # group samples per individual
    sample_ids = group_by(samples_df_sliced, 'Individual', _ids)
    
    n = 2  # pairs of tissues
    cross_list = []
    samples = []
    if isinstance(sample_ids, (list,)):
        cross_list = arrangements(sample_ids, n)

    if isinstance(sample_ids, dict):
        print("compute all arrangements")
        sample_aux = []
        for key in sample_ids:
            if len(sample_ids[key]) >= n:
                cross_list += arrangements(sample_ids[key], n)
            
            sample_aux += sample_ids[key]

        samples = list(set(sample_aux))

    anndata_filtered_var = anndata_obj[:, samples]
    anndata_sliced = anndata_filtered_var[gene_ids,:]
    
    (X, Y, rownames, columns) = build_x_y(anndata_sliced, cross_list, input_transform=input_transform)
    
    return (X, Y, rownames, columns)


def sum_tissues_per_individual(info, individuals):
    return sum(info[x] for x  in info if x in individuals)


def remove_best(real,expected,subset,tissue_info,epsilon,n_samples):
    
    print("remove best")
    min_error = 1
    sample_id = None
    
    for i in range(len(subset)): 
        f = float(sum_tissues_per_individual(tissue_info, subset[:i] + subset[i+1:])/n_samples)
                
        if -epsilon <= (expected - f) <= epsilon: # found optimun
            
            sample_id = subset[i]
            subset.pop(i)
            
            print("Min error: ", expected - f)
            print(sample_id)
            
            return sample_id
        
        diff = abs(expected - f)
        
        if diff<=min_error:
            sample_id = subset[i]
            min_error = diff
    
    subset.remove(sample_id)
    
    print("Min error: ",min_error)
    print(sample_id)
    
    return sample_id
        
    

def rebalance(train, valid, test, tissues_info, n_samples, fraction=[3. / 5, 1. / 5, 1. / 5]):
    """Rebalance train valid test to be proportional also in terms of number of tissues.
       The initial split its using the Invidivuals and not the #samples, so it might be that some individuals
       have more samples as other
    
    Args:
        train (list): list of individuals included for the training set
        valid (list):
        test (list):
        n_samples (int): number of samples
        tissue_info (dict): {'Individual':#tissues} - dict holding the tissue counts per individual 
        
        fraction (bool): (list(float,float,float)): split fraction for train valid and test

    Returns:
        (train_individuals,valid_individuals, test_individuals)
    """
    
    sets = [train, valid, test]
    balanced = False
    i = 1
    iterations = 100
    epsilon = 0.005
    
    while(not balanced and i < iterations):
        
        print("Iteration: ", i)
        i += 1
        
        
        # check if the train valid test are balances in terms of samples
        (c_train, c_valid, c_test) = (sum_tissues_per_individual(tissues_info, sets[0]),
                                      sum_tissues_per_individual(tissues_info, sets[1]),
                                      sum_tissues_per_individual(tissues_info, sets[2]))

        # print state
        print("\tExpc counts: ", [math.floor(x*n_samples) for x in fraction])
        print("\tReal counts: ", c_train, c_valid, c_test)


        (f_train, f_valid, f_test) = (float(c_train/n_samples),
                                      float(c_valid/n_samples),
                                      float(c_test/n_samples))

        print("\tExpc fraction: ", fraction)
        print("\tReal fraction: ", f_train, f_valid, f_test)
        
        
        # compute difference
        measured_fractions = [f_train, f_valid, f_test]        
        diff =  list(np.array(measured_fractions) - np.array(fraction))
        print("\tDiff: ",diff)
        # swap elements
        count_balanced = 0
        for k, f in enumerate(diff):

            # if the subset already balanced
            if -epsilon <= f <= epsilon:
                count_balanced += 1
                continue
            
            if f > 0: # to many samples 
                sample_id = remove_best(measured_fractions[k],fraction[k],sets[k],tissues_info,epsilon,n_samples)
                
                # add this to the next group
                sets[(k+1)%3].append(sample_id)
        
        
        if count_balanced == len(sets):
            balanced = True
        
    
    return sets


def split_by_individuals(annobj, fraction=[3. / 5, 1. / 5, 1. / 5], groupby=['Gender','Seq'], stratified=True, shuffle=False):
    """Split dataset using stratified individuals by Gender ..
    
    Args:
        annobj (:obj:AnnData): Assumes a column its named Individual
        fraction (list(float,float,float)): split fraction for train valid and test
        stratified (bool):
        shuffle (bool):

    Returns:
        (train_individuals,valid_individuals, test_individuals)
    """

    # Stratify
    df = annobj.var.reset_index(drop=True)[['Individual'] + groupby]
    df.drop_duplicates(inplace = True)
    df_grouped = df.groupby(groupby, as_index=False)
    
    train_individuals = []
    valid_individuals = []
    test_individuals = []
    
    for name, group in df_grouped:  # get same fraction from each group
        if group.shape[0] > 3:
            index_aux = list(map(lambda x: math.floor(group.shape[0] * x), fraction))
            # correct error
            index_aux[2] = group.shape[0] - index_aux[0] - index_aux[1]
        elif group.shape[0] == 3:
            index_aux = [1,1,1]
        elif group.shape[0] == 2:
            index_aux = [1,0,1]
        else:
            index_aux = [1,0,0]
        
        train_individuals += group.iloc[:index_aux[0],:]['Individual'].tolist()
        valid_individuals += group.iloc[index_aux[0]: (index_aux[0] + index_aux[1]),:]['Individual'].tolist()
        test_individuals += group.iloc[(index_aux[0] + index_aux[1]):,:]['Individual'].tolist()
       
        
    # count tissues per individual
    info_tissues = {indiv: annobj.var[annobj.var['Individual'] == indiv].shape[0] for indiv in df['Individual'].tolist()}
    print("Total individuals: " + str(len(df['Individual'].tolist())))
   
    print("Individual split before balancing: ", len(train_individuals),len(valid_individuals),len(test_individuals))
    
    rebalance(train_individuals,valid_individuals,test_individuals,info_tissues,annobj.var.shape[0])

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

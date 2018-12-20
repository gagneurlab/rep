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
import anndata
import pandas as pd

from rep.constants import ANNDATA_CST as a


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
        varaux = pd.DataFrame(pd.read_csv(os.path.abspath(csvfile), header=0, delimiter=sep, index_col=0))
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


def filter_df_by_value(df, jsonFilters):
    """
    Find rows matching the filtering criteria.
    :param df: dataframe
    :param jsonFilters: json file, where key is the column, and value are the admited (filtering) values
    :return: row.names for which the filtering applies
    """
    names = list(df.index)

    for key in jsonFilters:
        if key == 0: # index of data.frame
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


def filter_anndata_by_value(annobj, jsonFilters):
    """
    Apply filtering on anndata.var and anndata.obs dataframes using value filtering
    :jsonFilters: Please follow the structure bellow (if no 'obs' filtering, then the key does not have to be in the json)
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
    :return: tuple (filtered var_names, filtered obs_names) - use this output as input to filter the count matrix (anndata.X)
    """
    print(annobj.var)
    for key in jsonFilters:
        if key == a.VAR:  # filtering by var (col)
            filtered_var_names = filter_df_by_value(annobj.var, jsonFilters[key])
        elif key == a.OBS:  # filtering by var (col)
            filtered_obs_names = filter_df_by_value(annobj.obs, jsonFilters[key])

    if a.VAR not in jsonFilters:
        filtered_var_names = list(annobj.var_names)  # no filtering
    if a.OBS not in jsonFilters:
        filtered_obs_names = list(annobj.obs_names)

    return (filtered_var_names, filtered_obs_names)


def filter_anndata_by_region(annobj, filterJson, regions, format):
    """
    Filtering by region over the anndata.obs (the return its a tuple just for completeness, the anndata.var stay the same)
    :param annobj:
    :param filterJson: {
                        'chromosome': col1,
                        'start' : col2,
                        'stop' : col3,
                        'strand' : col4
                        } - mark columns which indicate the regions in the anndata obj
    :param regions: file in standard format, e.g. bed or gtf
    :param format: bed or gtf
    :return: tuple (filtered var_names, filtered obs_names)
    """
    pass


def save(annobj, outname=None):
    """
    Write .h5ad-formatted hdf5 file and close a potential backing file. Default gzip file
    :param annobj:
    :param outname: name of the output file (needs to end with .h5ad)
    :return output filename (if none specified then this will be random generated)
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
    """
    Group for instance e.g. tissues per sample.

    :param df: data.frame with the values on which we would like to group
    :param column: column over we will group
    :param index_subset: filtered list of indexes (using e.g filter_anndata_by_value)
    :return: {'value':[list of index_subset matching the value]}
    """
    dict = {}
    df_new = df.groupby(str(column))
    for group in df_new.groups:
        # rearrage the index_subset by grouping over e.g. tissue
        dict[group] = list(set(list(df_new.groups[group])) & set(index_subset))

    return dict


def arrangements(list_of_samples, n=None):
    """
    This generates all arrangements of length n (the order is relevant)

    :param list_of_samples:
    :param n: number of elements in sublist
    :return:
    """
    if not n:  # assume permutations = arrangements(k,n) where k == n
        n = len(list_of_samples)
    return [p for p in permutations(list_of_samples, n)]


def build_x_y(annobj, cross_list):
    """
    Build the two matrices X (train) and Y (labels)
    :param annobj:
    :param cross_list: pairs of indexes
    :param obs_names: filtering over the features (e.g. genes)
    :return: (df_X,def_Y) where X  and Y of size len(cross_list) x len(obs_names)
    """
    # create indexes
    length = len(annobj.obs_names)
    index = []
    for i, (x, y) in enumerate(cross_list):
        index.append(str(str(x) + "_" + str(y)))

    df_X = pd.DataFrame(index=index, columns=annobj.obs_names)
    df_Y = pd.DataFrame(index=index, columns=annobj.obs_names)

    for i, (x, y) in enumerate(cross_list):
        custom_index = str(str(x) + "_" + str(y))

        vector = annobj[:, [x]].X
        df_X.loc[custom_index] = vector.reshape(1, length)

        vector = annobj[:, [y]].X
        df_Y.loc[custom_index] = vector.reshape(1, length)

    return (df_X, df_Y)


def rnaseq_cross_tissue(anndata_obj, individuals, gene_ids, target_transform=None, target_transform_param=None,
                        input_transform=None, input_transform_param=None, shuffle=False):
    """
    Prepare the traning data by:
        1. Filtering the anndata matrix, by col (gene_ids) and rows (sample_ids)
        2. Normalize data if necessary
        3. Stratify samples
        4. Create all pair between all tissues available for one individuum
        this uses var_names and obs_names. These can be either a list of keys or a dictionary (grouping the samples
        by condition - this means the pairs will be generated only within a group)

    Args:
        anndata_obj: h5ad format
        individuals: (individuals) in the summarized experiment (assume there is a row called Individuals)
        obs_names: (gene_ids/rows) in the summarized experiment
        traget_transform: reference to a function  function
        input_transform: reference to a function
        shuffle (bool):
    Return:
        (df.X,df.Y) # index=sample_tissue

    """

    # get samples
    sample_ids = filter_df_by_value(anndata_obj.var,{'Individual':individuals})
    sample_ids = group_by(anndata_obj.var, 'Individual', sample_ids)

    n = 2  # pairs of tissues
    cross_list = []

    if isinstance(sample_ids, (list,)):
        cross_list = arrangements(sample_ids, n)

    if isinstance(sample_ids, dict):
        individuals_aux = []
        for key in sample_ids:
            cross_list += arrangements(sample_ids[key], n)

            # flatten var_names dict
            individuals_aux += sample_ids[key]

        individuals = list(set(individuals_aux))

    print("4.3 Slice anndata")
    print()
    anndata_filtered_var = anndata_obj[:, individuals]
    anndata_sliced = anndata_filtered_var[gene_ids, :]
    print_anndata(anndata_sliced)

    print("4.4 Build the two matrices X and Y")
    print()
    (X, Y) = build_x_y(anndata_sliced, cross_list)

    return (X, Y)


if __name__ == '__main__':

    # assembling:
    parser = argh.ArghParser()
    parser.add_commands([load_count_matrix, load_anndata_from_file])

    if sys.argv[1] == 'explore':
        # dispatching:
        parser.dispatch()

    elif sys.argv[1] == 'example':
        # run example

        # create a summarized experiment
        print("1. Read csv + anno matrices:")
        print()
        annobj = load_count_matrix("../../data.csv", sep=",", varanno="../../anno.csv", obsanno="../../anno_obs.csv")
        print_anndata(annobj)

        # filter
        print("4. Filter by value anndata.var and anndata.obs")
        print()
        (var, obs) = filter_anndata_by_value(annobj, {a.VAR: {1: ['M']},
                                                      a.OBS: {0: ['G1', 'G2']}})
        print(var, obs)
        # group tissues by sample
        dict_samples_var = group_by(annobj.var, 3, var)
        (X, Y) = rnaseq_cross_tissue(annobj, var_names=dict_samples_var, obs_names=obs)
        print(X)
        print()
        print(Y)

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

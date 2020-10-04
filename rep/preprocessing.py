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
import re
import logging

from itertools import permutations
import h5py

import numpy
# import scanpy
import math
import anndata
import pandas as pd
import numpy as np

from rep.constants import ANNDATA_CST as a
from rep.constants import GTEX_CST as gcst
from rep.constants import METADATA_CST as mcst

import gin
from gin import config

# set your log level
# logging.basicConfig(level=logging.CRITICAL)
# logger = logging.get#logger('preprocessing')

# set print statement as logging statement
# print = __builtins__.print if debug else logging.debug

ANNO_TRANSCRIPTS = 'transcripts_list'
ANNO_EXONS = 'exons_list'
ANNO_CODING_EXONS = 'exons_coding_list'
ANNO_EXONIC_LENGHT = 'len_exonic'
ANNO_START = 'start'
ANNO_STOP = 'stop'
ANNO_STRAND = 'strand'
ANNO_LENGHT = 'len'
ANNO_CHR = 'chr'


class RepAnnData(anndata.AnnData):
    """Thin wrapper over anndata.Anndata object
       Specify concrete obs and var, namely, obs will be sample, var will be gene for this project.

    """

    def __init__(self, X=None, samples_obs=None, genes_var=None, uns=None):
        anndata.AnnData.__init__(self,
                                 X=X,
                                 obs=samples_obs,
                                 var=genes_var,
                                 uns=uns)
        if samples_obs is not None:
            self.set_samples(samples_obs)
        if genes_var is not None:
            self.set_genes(genes_var)

    @property
    def genes(self):
        return self.var

    @property
    def samples(self):
        return self.obs

    @property
    def genes_names(self):
        return self.var_names

    @property
    def samples_names(self):
        return self.obs_names

    def set_genes(self, df_genes_description):
        self.var = df_genes_description
        self.var.columns = list(df_genes_description.columns.values)
        self.var_names = list(df_genes_description.index)

    def set_samples(self, df_samples_description):
        self.obs = df_samples_description
        self.obs.columns = list(df_samples_description.columns.values)
        self.obs_names = list(df_samples_description.index)

    def set_genes_names(self, list_names):
        self.var_names = list_names

    def set_samples_names(self, list_names):
        self.obs_names = list_names

    def save(self, outname=None):
        """Write .h5ad-formatted hdf5 file and close a potential backing file. Default compression type = gzip

        Args:
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
        self.set_genes_names([
            str(v) for v in self.genes_names
        ])
        self.set_samples_names([
            str(o) for o in self.samples_names
        ])
        self.genes.rename(
            index={
                r: str(r) for r in list(self.genes.index)
            },
            columns={
                c: str(c) for c in list(self.genes.columns)
            },
            inplace=True)
        self.samples.rename(
            index={
                r: str(r) for r in list(self.samples.index)
            },
            columns={
                c: str(c) for c in list(self.samples.columns)
            },
            inplace=True
        )

        self.write(name)

        return name

    @staticmethod
    def filter_genes(repobj, key='gene_id', values=[]):
        obj = repobj[:, repobj.genes[key].isin(values)]
        return RepAnnData(X=np.array(obj.X), genes_var=obj.var, samples_obs=obj.obs)

    @staticmethod
    def filter_samples(repobj, key='To_tissue', values=[]):
        obj = repobj[repobj.samples[key].isin(values)]
        return RepAnnData(X=np.array(obj.X), genes_var=obj.var, samples_obs=obj.obs)

    @staticmethod
    def read_h5ad(filename, backed=None):
        obj = anndata.read_h5ad(filename, backed=backed)
        r = RepAnnData(X=obj.X, genes_var=obj.var, samples_obs=obj.obs)
        return r

    @staticmethod
    def read_csv(abs_path, delimiter=","):
        obj = anndata.read_csv(abs_path, delimiter=delimiter)
        r = RepAnnData(X=obj.X, genes_var=obj.var, samples_obs=obj.obs)
        return r


########################################## I/O #########################################################
########################################################################################################


def readh5(name):
    filename = name
    f = h5py.File(filename, 'r')
    X = np.array(f[list(f.keys())[0]])
    #     f.close()

    return X


def writeh5(obj, obj_name, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(obj_name, data=obj)
    h5f.close()


def readJSON(name):
    with open(name, 'r') as json_file:
        data = json.load(json_file)
    return data


def writeJSON(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)


def read_csv_one_column(filename):
    """Reads a list of elements from the file

    Args:
        filename:

    Returns:
         list of elements
    """
    l = []
    with open(filename, 'r') as f:
        for line in f: l.append(line.replace("\n", ""))
    return l


# def print_anndata(toprintanndata):
#     print("anndata.X ----")
#     print(toprintanndata.X)
#     print("anndata.var ----")
#     print(toprintanndata.var.iloc[:5, :5])
#     print("anndata.obs ----")
#     print(toprintanndata.obs.iloc[:5, :5])
#     print()


def load_df(csvfile, header=None, delimiter=",", index_col=0):
    return pd.read_csv(os.path.abspath(csvfile), header=header, delimiter=delimiter, index_col=index_col)


def save_list(filename, data):
    '''Save array of strings/numeric using a \n delimiter.
    '''
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


@gin.configurable
def load_list(filename):
    '''Load array of strings using a \n delimiter.
    '''
    arr = []
    with open(filename, 'r') as f:
        for line in f:
            arr.append(line.split("\n")[0].strip())

    return arr


########################################## Genomic annotation ##########################################
########################################################################################################
def transcript_to_keep(gene, l_candidates, compare_value):
    value = 1
    nr = 2

    # print(l_candidates)
    # find all transcripts with max coding exons
    max_keys = list(filter(lambda k: k[value] == compare_value, l_candidates))

    # found one transcript
    if len(max_keys) == 1:
        return max_keys[0][0]
    if len(max_keys) == 0:
        raise ValueError("could not found transcript for gene: ", gene)

    # min ENST number
    min_number = min([nr for _, _, nr in l_candidates])

    # return key
    return list(filter(lambda k: k[nr] == min_number, l_candidates))[0][0]


def raw_counts2fpkm(annobj, annotation):
    """Convert raw counts to FPKM

    Args:
        annobj (:obj:RepAnndata):
                           sample1 sample2
                    gene1
                    gene2

        annotation (:obj:json): contains the gene annotation, gene length, exonic length
    """

    norm_X = np.zeros(annobj.X.shape)
    recount_genes = annobj.genes.index.tolist()
    # per sample count fragments
    mapped_fragments = np.sum(annobj.X, axis=1)

    # compute per gene
    for i in range(0, annobj.X.shape[1]):
        col = annobj.X[:, i]
        gene = recount_genes[i]
        # len of transcript in kb
        transcript_len = annotation[gene][ANNO_EXONIC_LENGHT] / 1000.0

        # frag per kilobase million
        norm_X[:, i] = (col * 1000 * 1000) / (transcript_len * mapped_fragments)

    return norm_X


def raw_counts2tpm(annobj, annotation):
    """Convert raw counts to TPM

    Args:
        annobj (:obj:RepAnndata):
                           sample1 sample2
                    gene1
                    gene2

        annotation (:obj:json): contains the gene annotation, gene length, exonic length
    """

    norm_X = np.zeros(annobj.X.shape)
    recount_genes = annobj.genes.index.tolist()

    # array with gene_len
    transcript_len = np.zeros(len(recount_genes))
    for i, gene in enumerate(recount_genes):
        transcript_len[i] = annotation[gene][ANNO_EXONIC_LENGHT] / 1000.0

    # compute per sample
    for i in range(0, annobj.X.shape[0]):
        counts_per_sample = annobj.X[i, :]
        norm_factor_per_sample = counts_per_sample / transcript_len

        # frag per kilobase million
        norm_X[i, :] = (1000 * 1000 * norm_factor_per_sample) / np.sum(norm_factor_per_sample)

    return norm_X


########################################## Transform function ##########################################
########################################################################################################

def mylog(df):
    """log2(df) using pseudocounts
    """
    return df.apply(lambda x: math.log(x + 1))


def mylog10(df):
    """log10(df) using pseudocounts
    """
    return np.log10(df + 1)


# variable which stores reference to function
function_mappings = {'log': mylog, 'log10': mylog10}


########################################## Anndata Summ Exp.  ##########################################
########################################################################################################

def create_anndata(counts_file, samples_anno=None, genes_anno=None, sep=","):
    """Creates an AnnData Object

    Args:
        counts_file (str): file containing the counts
        samples_anno (str): file containing sampple annotation
        genes_anno (str): file containing genes annotation
        sep (str): separator

    Returns:
        AnnData object
    """
    x = load_df(counts_file, header=0, delimiter=sep)
    annobj = anndata.RepAnnData(X=x)
    load_genes(annobj, genes_anno, sep)
    load_samples(annobj, samples_anno, sep)

    return annobj


def load(filename, backed=False, samples_anno=None, genes_anno=None, sep=","):
    """Load anndata object

    Args:
        filename (str): .h5ad file containing n_obs x n_vars count matrix and further annotations
        backed (bool):  default False - see anndata.read_h5ad documentation
                        https://media.readthedocs.org/pdf/anndata/latest/anndata.pdf
                        if varanno and obsanno are provided please set backed = r+
        samples_anno (str,optional) : sample_tissue description file
        genes_anno (str,optional): gene id description file
        sep (str): separator for varanno and obsanno files

    Returns:
        Anndata object
    """
    abspath = os.path.abspath(filename)
    annobj = RepAnnData.read_h5ad(abspath, backed=backed)
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
        annobj (:obj:`RepAnnData`):
        csvfile (str):
        sep (str):
    """

    if csvfile:
        varaux = load_df(csvfile, header=0, delimiter=sep, index_col=0)
        obj.set_samples(varaux)


def load_genes(obj, csvfile, sep=","):
    """Load genes (rows) description for the summarized experiment

     Args:
        annobj (:obj:`RepAnnData`):
        csvfile (str):
        sep (str):
    """
    if csvfile:
        obsaux = load_df(csvfile, header=0, delimiter=sep, index_col=0)
        obj.set_genes(obsaux)


def load_count_matrix(filename, sep=",", samples_anno=None, genes_anno=None):
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
    annobj = RepAnnData.read_csv(abs_path, delimiter=sep)

    # read samples data (samples description)
    load_samples(annobj, samples_anno, sep)

    # read genes data (index description e.g gene annotation)
    load_genes(annobj, genes_anno, sep)

    return annobj


# def save(annobj, outname=None):
#     """Write .h5ad-formatted hdf5 file and close a potential backing file. Default compression type = gzip
#
#     Args:
#         annobj (:obj:AnnData):
#         outname (str): name of the output file (needs to end with .h5ad)
#
#     Returns:
#         output filename (if none specified then this will be random generated)
#     """
#
#     if outname:
#         abspath = os.path.abspath(outname)
#         name = abspath
#     else:
#         name = os.path.abspath("tmp" + str(int(time.time())) + ".h5ad")
#
#     # convert header to string (avoid bug)
#     annobj.var_names = [str(v) for v in annobj.var_names]
#     annobj.obs_names = [str(o) for o in annobj.obs_names]
#     annobj.var.rename(index={r: str(r) for r in list(annobj.var.index)},
#                       columns={c: str(c) for c in list(annobj.var.columns)},
#                       inplace=True)
#     annobj.obs.rename(index={r: str(r) for r in list(annobj.obs.index)},
#                       columns={c: str(c) for c in list(annobj.obs.columns)},
#                       inplace=True)
#
#     annobj.write_h5ad(name)
#
#     return name


########################################## Filter anndata Summ Exp.  ###################################
########################################################################################################

def filter_df_by_value(df, jsonFilters):
    """Find rows matching the filtering criteria (allows multiple filters).

    Args:
        df (:obj:Data.frame): count matrix
        jsonFilters (dict): json file, where key is the column, and value are the admited (filtering) values

    Returns:
        row.names for which the filtering applies
    """
    names = list(df.index)
    for key in jsonFilters:

        # filtering by index column
        if key == 0:
            # make sure the values to filter are part of the index column
            names = list(set(names) & set(jsonFilters[key]))
            continue

        # filter by regular columns
        name_per_key = []
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
        annobj (:obj:`RepAnnData`):
        filters (dict): Please follow the structure bellow (if no 'obs' filtering, then the key does not
                        have to be in the json)
                    {'samples':{
                            1:['F'],
                            4:['rnaseq']
                            #col1:['value1','value2',...],
                            #col2:['value3',...]
                        },
                    'genes':{
                            3:['chr3']
                        }
                    }

    Returns:
        tuple (filtered var_names, filtered obs_names) - use this output as input to filter the count matrix (anndata.X)
    """
    for key in filters:
        if key == a.SAMPLES:  # filtering by row
            filtered_samples_names = filter_df_by_value(annobj.samples, filters[key])
        elif key == a.GENES:  # filtering by col
            filtered_genes_names = filter_df_by_value(annobj.genes, filters[key])

    if a.SAMPLES not in filters:
        filtered_samples_names = list(annobj.samples_names)  # no filtering
    if a.GENES not in filters:
        filtered_genes_names = list(annobj.genes_names)

    return (filtered_samples_names, filtered_genes_names)


def filter_anndata_by_region(annobj, filterJson, regions, format):
    """Filtering by region over the anndata.obs (the return its a tuple just for completeness,
    the anndata.var stay the same)

    Args:
        annobj (:obj:RepAnnData):
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

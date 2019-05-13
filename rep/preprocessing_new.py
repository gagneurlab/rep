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
        self.set_genes_names([str(v) for v in self.genes_names])
        self.set_samples_names([str(o) for o in self.samples_names])
        self.genes.rename(index={r: str(r) for r in list(self.genes.index)},
                          columns={c: str(c) for c in list(self.genes.columns)},
                          inplace=True)
        self.samples.rename(index={r: str(r) for r in list(self.samples.index)},
                            columns={c: str(c) for c in list(self.samples.columns)},
                            inplace=True)

        self.write(name)

        return name

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
        for line in f: l.append(line.replace("\n",""))
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


def get_annotation(file):
    """Parse GTF annotation file. Choose major transcript per gene: (approach is deterministic)
       Rules:
        - most coding exons, if not then:
        - if two transcripts same #coding_exons:
            - choose transcript with less non-coding exons
            - if equal: choose lowest ENS number
        - if no coding exons, then choose most #exons:
            - if equal: choose lowest ENS number
        GTF annotation file is "1-based", meaning an interval lenght = stop - start + 1

    Args:
        file (str): gtf annotation file
        
    Returns:
        dictionary with genes and its longest transcript
        {'ENSG00000223972.5':
            { 'start': 11869,
              'stop': 14409,
              'chr': 'chr1',
              'strand': '+',
              'len': 2540,
              'transcript_list':
                    {'ENST00000450305.2':
                        {   'start': 12010,
                            'stop': 13670,
                            'chr': 'chr1',
                            'strand': '+',
                            'len': 1660,
                           'exon_list': [(12010, 12057),
                 (12179, 12227),
                 (12613, 12697),
                 (12975, 13052),
                 (13221, 13374),
                 (13453, 13670)]}},
              'len_exonic': 626},
    """

    dict_genes = {}

    with open(file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            arr = line.replace("\n", "").split("\t")
            gene_id = re.search("gene_id \"(.*?)\";", arr[8]).group(1)

            entry = {}
            entry[ANNO_START] = int(arr[3])
            entry[ANNO_STOP] = int(arr[4])
            entry[ANNO_CHR] = arr[0]
            entry[ANNO_STRAND] = arr[6]
            entry[ANNO_LENGHT] = abs(entry[ANNO_STOP] - entry[ANNO_START] + 1)

            # parse gene entry -> add gene annotation the structure
            if arr[2] == 'gene':

                if gene_id not in dict_genes:
                    entry[ANNO_TRANSCRIPTS] = {}
                else:
                    entry[ANNO_TRANSCRIPTS] = dict_genes[gene_id][ANNO_TRANSCRIPTS]
                dict_genes[gene_id] = entry

            # parse transcript entry 
            if arr[2] == 'transcript':

                transcript_id = re.search("transcript_id \"(.*?)\";", arr[8]).group(1)

                # add gene as parent for the transcript
                if gene_id not in dict_genes:
                    dict_genes[gene_id] = {}
                    dict_genes[gene_id][ANNO_TRANSCRIPTS] = {}

                # add transcript
                transcript_dict = dict_genes[gene_id][ANNO_TRANSCRIPTS]
                if transcript_id in transcript_dict:
                    entry[ANNO_EXONS] = transcript_dict[transcript_id][ANNO_EXONS]
                else:
                    entry[ANNO_EXONS] = []
                transcript_dict[transcript_id] = entry

            # parse exon entry
            if arr[2] == 'exon':

                transcript_id = re.search("transcript_id \"(.*?)\";", arr[8]).group(1)

                # add parent gene and transcript if not exist
                if gene_id not in dict_genes:
                    dict_genes[gene_id] = {}
                    dict_genes[gene_id][ANNO_TRANSCRIPTS] = {}

                transcript_dict = dict_genes[gene_id][ANNO_TRANSCRIPTS]
                if transcript_id not in transcript_dict: transcript_dict[transcript_id] = {}
                if ANNO_EXONS not in transcript_dict[transcript_id]: transcript_dict[transcript_id][ANNO_EXONS] = []

                transcript_dict[transcript_id][ANNO_EXONS].append((entry[ANNO_START], entry[ANNO_STOP]))

            # parse CDS
            if arr[2] == 'CDS':

                transcript_id = re.search("transcript_id \"(.*?)\";", arr[8]).group(1)

                # add parent gene and transcript if not exist
                if gene_id not in dict_genes:
                    dict_genes[gene_id] = {}
                    dict_genes[gene_id][ANNO_TRANSCRIPTS] = {}

                transcript_dict = dict_genes[gene_id][ANNO_TRANSCRIPTS]
                if transcript_id not in transcript_dict: transcript_dict[transcript_id] = {}
                if ANNO_CODING_EXONS not in transcript_dict[transcript_id]: transcript_dict[transcript_id][
                    ANNO_CODING_EXONS] = []

                transcript_dict[transcript_id][ANNO_CODING_EXONS].append((entry[ANNO_START], entry[ANNO_STOP]))

    # find major transcript
    for gene in dict_genes:

        transcript_candidates = dict_genes[gene][ANNO_TRANSCRIPTS]

        # generate list [(ENS, #coding_exon_count, #ENS_number),...] (#coding_exon_count =0 for the transcripts with no coding exons)
        count_coding_exons = list(
            map(lambda k: (k, len(transcript_candidates[k][ANNO_CODING_EXONS]), int(k[4:].split(".")[0])) \
                if ANNO_CODING_EXONS in transcript_candidates else (k, 0, int(k[4:].split(".")[0])),
                transcript_candidates))
        max_coding_exons = max([value for _, value, _ in count_coding_exons])

        # at least one coding exon
        if max_coding_exons > 0:
            tr_tokeep = transcript_to_keep(gene, count_coding_exons, max_coding_exons)

        else:  # no coding exons

            # generate list [(ENS, #exon_count, #ENS_number),...]
            count_exons = list(map(lambda k: (k, len(transcript_candidates[k][ANNO_EXONS]), int(k[4:].split(".")[0])),
                                   transcript_candidates))
            max_exons = max([value for _, value, _ in count_exons])
            tr_tokeep = transcript_to_keep(gene, count_exons, max_exons)

        # remove all other transcripts
        aux = dict_genes[gene][ANNO_TRANSCRIPTS][tr_tokeep]
        dict_genes[gene][ANNO_TRANSCRIPTS] = {}
        dict_genes[gene][ANNO_TRANSCRIPTS][tr_tokeep] = aux

        # compute exonic length
        dict_genes[gene][ANNO_EXONIC_LENGHT] = sum(
            [y - x + 1 for (x, y) in dict_genes[gene][ANNO_TRANSCRIPTS][tr_tokeep][ANNO_EXONS]])

    return dict_genes


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
    for i, gene in enumerate(recount_genes): transcript_len[i] = annotation[gene][ANNO_EXONIC_LENGHT] / 1000.0

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
        # rearrange the index_subset by grouping over e.g. tissue
        gr = list(set(list(df_new.groups[group])) & set(index_subset))
        if len(gr) > 0:
            dict[group] = gr

    return dict


def arrangements(list_of_samples, n=None):
    """This generates all arrangements of length n (the order is relevant)

    Args:
        list_of_samples (list(str)):
        n (int): number of elements in sublist

    Returns:
        list of tuples of samples
    """
    if len(list_of_samples) < 2:
        return []
    if not n:  # assume permutations = arrangements(k,n) where k == n
        n = len(list_of_samples)

    return [p for p in permutations(list_of_samples, n)]


def get_metadata(annobj, cross_list, index_pat_tissue):
    """Get additional metadata for the cross tissue matrix pairs
    
    Args:
        annobj (:obj:RepAnnData):
        cross_list (:array(tuple)): tissue pairs
        index_pat_tissue: index for the cross tissue matrix pairs
    
    Return:
        json object containing information about the rows and columns of cross tissue matrix
        row: gene, gene annotation
        column: individual, from_tissue, to_tissue, available WGS/WES
    """

    # Get Individual x Tissue (from, to) info
    patient_tissue_info = pd.DataFrame( index = index_pat_tissue,
                                        columns=[gcst.INDEX,
                                                gcst.INDIVIDUAL,
                                                gcst.INDIV_SEQ_ASSAY,
                                                gcst.FROM_SAMPLE,
                                                gcst.GENDER,
                                                gcst.FROM_TISSUE,
                                                gcst.FROM_PARENT_TISSUE,
                                                gcst.TO_TISSUE,
                                                gcst.TO_PARENT_TISSUE,
                                                gcst.TO_SAMPLE])

    for i, (sample1, sample2) in enumerate(cross_list):

        info1 = annobj.samples.loc[sample1, :]
        info2 = annobj.samples.loc[sample2, :]

        if info1[gcst.INDIVIDUAL] != info2[gcst.INDIVIDUAL]:
            raise ValueError('Samples where mixed up. Found cross tissue tuples coming from different individuals.')

        patient_tissue_info.iloc[i] = [index_pat_tissue[i]] + \
                                      info1[[gcst.INDIVIDUAL,
                                             gcst.INDIV_SEQ_ASSAY]].tolist() + \
                                             [sample1] + \
                                      info1[[gcst.GENDER,
                                             gcst.TISSUE,
                                             gcst.PARENT_TISSUE]].tolist() + \
                                      info2[[gcst.TISSUE,
                                             gcst.PARENT_TISSUE]].tolist() + \
                                             [sample2]

    # # Gene information
    # gene_info = pd.DataFrame(index=annobj.gene_names)info1[[
    return patient_tissue_info


def build_x_y(annobj, cross_list, input_transform=None, onlyBlood=False):
    """Build the cross tissue matrix pairs X (train) and Y (labels)
       Cross tissue matrix pairs:
            - x_ij correspond to expression of tissue i in gene j , x_ij in X
            - y_ij correspond to expression of tissue i in gene j, y_ij in Y
            - (x_ij, y_ij) -> all arrangements of 2 elements for the expression in invidividual K acroos all available tissues
            - where x_ij = y_ij are removed (same tissue)

    Args:
        annobj (:obj:RepAnnData):
        cross_list (list((str,str))): pairs of indexes
        obs_names (str): filtering over the features (e.g. genes)
        input_transform (str): reference to a function

    Returns:
        (df_X,def_Y) where X  and Y of size len(cross_list) x len(obs_names)
    """

    # cross_list - filter only blood
    filtered_list = []
    if onlyBlood == True:
        for (x, y) in cross_list:
            if annobj.samples.loc[annobj.samples.index == x, 'Tissue'].tolist()[0] == 'Whole Blood':
                filtered_list.append((x, y))
        cross_list = filtered_list

    # create indexes T1_T2
    index_elem = [str(str(x) + "_" + str(y)) for i, (x, y) in enumerate(cross_list)]

    # build accessing dictionary
    access = {x: i for i, x in enumerate(annobj.samples_names)}

    print("Total pairs: " + str(len(cross_list)))

    # apply transformation
    if input_transform:
        try:
            m = function_mappings[input_transform](annobj.X)  # call the function
        except:
            return "Invalid function - please check the processing.rnaseq_cross_tissue documentation"
    else:
        m = annobj.X

    slice_x = [access[x] for (x, _) in cross_list]
    slice_y = [access[y] for (_, y) in cross_list]

    mydata = np.array(m)

    # compute the metadata for the cross tissue matrix
    metadata = get_metadata(annobj, cross_list, index_elem)

    return (mydata[slice_x, :], mydata[slice_y, :], metadata)


def compute_tissue_pairs(sample_ids):
    """Computer pairs of tissues per individual
    
    Args:
        samples_ids (dict): key = individual
        
    Returns:
        pair if sample ids,
        list of unique samples
    """

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

    return cross_list, samples


def rnaseq_cross_tissue(anndata_obj, individuals, gene_ids, target_transform=None,
                        input_transform=None, shuffle=False, onlyBlood=False):
    """Prepare the traning data by:
        1. Filtering the anndata matrix, by col (gene_ids) and rows (sample_ids)
        2. Normalize data if necessary
        3. Stratify samples
        4. Create all pair between all tissues available for one individuum
        this uses var_names and obs_names. These can be either a list of keys or a dictionary (grouping the samples
        by condition - this means the pairs will be generated only within a group)

    Args:
        anndata_obj (:obj:RepAnnData): h5ad format
        individuals (list(str)): (individuals) in the summarized experiment (assume there is a row called Individuals)
        gene_ids (list(str)): (gene_ids/rows) in the summarized experiment
        traget_transform (str): reference to a function  function - still not implemented
        input_transform (str): reference to a function; choose [log]
        shuffle (bool):

    Returns:
        (df.X,df.Y) # index=sample_tissue

    """
    if len(individuals) == 0:
        return (None, None, None)

    # get samples
    samples_df = anndata_obj.samples
    # logger.debug("samples_df ", samples_df.shape)

    # slice data.frame only for the subset of individuals
    _ids = filter_df_by_value(samples_df, {'Individual': individuals})

    samples_df_sliced = samples_df[samples_df.index.isin(_ids)]
    # logger.debug("samples_df_sliced ", samples_df_sliced.shape)

    # group samples per individual
    sample_ids = group_by(samples_df_sliced, 'Individual', _ids)

    # compute list of tissues pairs 
    cross_list, samples = compute_tissue_pairs(sample_ids)

    # slice anndata by samples and genes
    anndata_filtered_var = anndata_obj[samples, :]
    anndata_sliced = anndata_filtered_var[:, gene_ids]

    # ensure X_array to be 2D and not 1D
    X_array = anndata_sliced.X.reshape(len(samples),len(gene_ids))
    # cast again to RepAnnData
    repandata = RepAnnData(X=X_array,samples_obs=anndata_sliced.obs,genes_var=anndata_sliced.var)

    (X, Y, metadata) = build_x_y(repandata,
                                 cross_list,
                                 input_transform=input_transform,
                                 onlyBlood=onlyBlood)

    return (X, Y, metadata)


def rnaseq_train_valid_test(anndata_obj, individuals, gene_ids, target_transform=None,
                            input_transform=None, shuffle=False, onlyBlood=False):
    """Generate train, valid, test datasets and store them within a large anndata object

        Args:
            anndata_obj (:obj:RepAnnData): h5ad format
            individuals (dict): (individuals) in the summarized experiment (assume there is a row called Individuals)
                                {train: list(individuals),
                                test: list(individuals),
                                valid: list(individuals)}
            gene_ids (list(str)): (gene_ids/rows) in the summarized experiment
            traget_transform (str): reference to a function  function - still not implemented
            input_transform (str): reference to a function; choose [log]
            shuffle (bool):

        Returns:
            inputs, targets

    """
    df_genes = anndata_obj.genes[anndata_obj.genes.index.isin(gene_ids)]
    df_samples = pd.DataFrame(columns=[gcst.INDEX,
                                       gcst.INDIVIDUAL,
                                       gcst.INDIV_SEQ_ASSAY,
                                       gcst.FROM_SAMPLE,
                                       gcst.GENDER,
                                       gcst.FROM_TISSUE,
                                       gcst.FROM_PARENT_TISSUE,
                                       gcst.TO_TISSUE,
                                       gcst.TO_PARENT_TISSUE,
                                       gcst.TO_SAMPLE,
                                       gcst.TYPE])
    X_large = pd.DataFrame(columns=gene_ids)
    Y_large = pd.DataFrame(columns=gene_ids)
    for key in ['train', 'valid', 'test']:
        (X, Y, metadata) = rnaseq_cross_tissue(anndata_obj, individuals[key], gene_ids,
                                               target_transform=target_transform, input_transform=input_transform,
                                               shuffle=shuffle, onlyBlood=onlyBlood)

        # skip samples if one of the X, Y, metadata are none
        if X is None or Y is None or metadata is None: continue

        # add type (train, valid, test)
        metadata[gcst.TYPE] = pd.Series(key, index=metadata.index)

        # extend df_samples
        df_samples = df_samples.append(metadata)

        x_aux = pd.DataFrame(columns=gene_ids, index=metadata[gcst.INDEX].tolist(), data=X)
        X_large = X_large.append(x_aux)

        y_aux = pd.DataFrame(columns=gene_ids, index=metadata[gcst.INDEX].tolist(), data=Y)
        Y_large = Y_large.append(y_aux)

    # set index of df_samples
    df_samples.set_index(gcst.INDEX)

    rx = RepAnnData(X=X_large, samples_obs=df_samples, genes_var=df_genes)
    ry = RepAnnData(X=Y_large, samples_obs=df_samples, genes_var=df_genes)

    # return inputs, targets
    return rx, ry


def sum_tissues_per_individual(info, individuals):
    return sum(info[x] for x in info if x in individuals)


def remove_best(real, expected, subset, tissue_info, epsilon, n_samples):
    # logger.debug("remove best")
    min_error = 1
    sample_id = None

    for i in range(len(subset)):
        f = float(sum_tissues_per_individual(tissue_info, subset[:i] + subset[i + 1:]) / n_samples)

        # reach convergence
        if -epsilon <= (expected - f) <= epsilon:  # found optimum
            sample_id = subset[i]
            subset.pop(i)

            # logger.debug("Min error: ", expected - f)
            # logger.debug(sample_id)

            return sample_id

        diff = abs(expected - f)

        if diff <= min_error:
            sample_id = subset[i]
            min_error = diff

    subset.remove(sample_id)

    # logger.debug("Min error: ", min_error)
    # logger.debug(sample_id)

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

    while (not balanced and i < iterations):

        # print("Iteration: ", i)
        i += 1

        # check if the train valid test are balances in terms of samples
        (c_train, c_valid, c_test) = (sum_tissues_per_individual(tissues_info, sets[0]),
                                      sum_tissues_per_individual(tissues_info, sets[1]),
                                      sum_tissues_per_individual(tissues_info, sets[2]))

        # print state
        # logger.debug("\tExpc counts: ", [math.floor(x * n_samples) for x in fraction])
        # logger.debug("\tReal counts: ", c_train, c_valid, c_test)

        (f_train, f_valid, f_test) = (float(c_train / n_samples),
                                      float(c_valid / n_samples),
                                      float(c_test / n_samples))

        # logger.debug("\tExpc fraction: ", fraction)
        # logger.debug("\tReal fraction: ", f_train, f_valid, f_test)

        # compute difference
        measured_fractions = [f_train, f_valid, f_test]
        diff = list(np.array(measured_fractions) - np.array(fraction))
        # logger.debug("\tDiff: ", diff)

        # swap elements
        count_balanced = 0
        for k, f in enumerate(diff):

            # if the subset already balanced
            if -epsilon <= f <= epsilon:
                count_balanced += 1
                continue

            if f > 0:  # to many samples
                sample_id = remove_best(measured_fractions[k], fraction[k], sets[k], tissues_info, epsilon, n_samples)

                # add this to the next group
                sets[(k + 1) % 3].append(sample_id)

        if count_balanced == len(sets):
            balanced = True

    return sets


def split_by_individuals(annobj, fraction=[3. / 5, 1. / 5, 1. / 5], groupby=['Gender', 'Seq'], stratified=True,
                         shuffle=False):
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

    # subset dataframe by Individual and Gender/Seq
    df = annobj.samples.reset_index(drop=True)[['Individual'] + groupby]
    df.drop_duplicates(inplace=True)

    # group individuals by gender and seq - basically max 4 subgroups
    # namely F-WGS, F-WES, M-WGS, M-WES
    df_grouped = df.groupby(groupby, as_index=False)

    train_individuals = []
    valid_individuals = []
    test_individuals = []

    for name, group in df_grouped:  # get same fraction from each  of the 4 groups

        # when group has more than 3 individuals, then compute fractions for train, valid, test
        if group.shape[0] > 3:

            # array with fraction of individuals which are chosen to be in [train, valid, test]
            index_aux = list(map(lambda x: math.floor(group.shape[0] * x), fraction))

            # correct error (number of individuals per train, valid, test is an integer)
            index_aux[2] = group.shape[0] - index_aux[0] - index_aux[1]

        elif group.shape[0] == 3:
            index_aux = [1, 1, 1]
        elif group.shape[0] == 2:
            # choose one invid for train and test
            index_aux = [1, 0, 1]
        else:
            index_aux = [1, 0, 0]

        train_individuals += group.iloc[:index_aux[0], :]['Individual'].tolist()
        valid_individuals += group.iloc[index_aux[0]: (index_aux[0] + index_aux[1]), :]['Individual'].tolist()
        test_individuals += group.iloc[(index_aux[0] + index_aux[1]):, :]['Individual'].tolist()

    # count tissues per individual
    info_tissues = {indiv: annobj.samples[annobj.samples['Individual'] == indiv].shape[0] for indiv in
                    df['Individual'].tolist()}
    print("Total individuals: " + str(len(df['Individual'].tolist())))

    print("Individual split before balancing: ", len(train_individuals), len(valid_individuals), len(test_individuals))

    # rebalance if at least 3xindividuals
    if (len(train_individuals) + len(valid_individuals) + len(test_individuals)) > 3:
        rebalance(train_individuals, valid_individuals, test_individuals, info_tissues, annobj.samples.shape[0])
        print(
        "Individual split after balancing: ", len(train_individuals), len(valid_individuals), len(test_individuals))

    return (train_individuals, valid_individuals, test_individuals)


if __name__ == '__main__':
    X = np.arange(18).reshape(6, -1)
    df_genes = pd.DataFrame(index=['g1', 'g2', 'g3'], data={'Anno': ['b1', 'b2', 'b3']})
    df_samples = pd.DataFrame(data={'Tissue': ['t1', 't2', 't3', 't2', 't1','t2'],
                                 'Individual': ['i1', 'i1', 'i2', 'i2', 'i3','i3'],
                                 'Gender': ['f', 'f', 'm', 'm', 'm','m'],
                                 'Seq': ['WES', 'WES', 'WGS', 'WGS', 'WGS','WGS']},
                           index=['s1', 's2', 's3', 's4', 's5','s6'])

    annobj = RepAnnData(X=pd.DataFrame(data=X, index=df_samples.index, columns=df_genes.index),
                        genes_var=df_genes,samples_obs=df_samples)
    # output = annobj.save(outname="test.h5")
    # print("File saved here", output)
    # print(type(annobj))
    #
    # test_annobj = RepAnnData.read_h5ad(output)
    # print(test_annobj.X)
    # print(test_annobj.genes)
    # print(test_annobj.samples)

    (train, valid, test) = split_by_individuals(annobj)
    # print(train, valid, test)
    (X,Y) = rnaseq_train_valid_test(annobj,{'train':train,'valid':valid,'test':test},annobj.genes_names)
    print("Finish")
    print(X.X)
    print(Y.X)
    print("samples")
    print(X.samples)
    print(Y.samples)
    print(X.genes)

    # # assembling:
    # parser = argh.ArghParser()
    # parser.add_commands([load, create_anndata])
    #
    # if sys.argv[1] == 'explore':
    #     # dispatching:
    #     parser.dispatch()
    #
    # elif sys.argv[1] == 'example':
    #     # run example
    #
    #     # create a summarized experiment
    #     # print("1. Read csv + anno matrices:")
    #     # print()
    #     # annobj = create_anndata("../data.csv", sep=",", samples_anno="../anno.csv", genes_anno="../anno_obs.csv")
    #     # save(annobj,"test.h5ad")
    #
    #     print("1*.Reload from file HDF5")
    #     print()
    #     annobj = load("test.h5ad")
    #     print_anndata(annobj)
    #
    #     print("2. Stratify individuals")
    #     (train, valid, test) = split_by_individuals(annobj)
    #     print(train, valid, test)
    #
    #     (X_train, Y_train) = rnaseq_cross_tissue(annobj, individuals=train, gene_ids=annobj.obs_names)
    #     (X_valid, Y_valid) = rnaseq_cross_tissue(annobj, individuals=valid, gene_ids=annobj.obs_names)
    #     (X_test, Y_test) = rnaseq_cross_tissue(annobj, individuals=test, gene_ids=annobj.obs_names)
    #
    #     print(X_train)
    #     print(Y_train)
    #
    #     # # filter
    #     # print("4. Filter by value anndata.var and anndata.obs")
    #     # print()
    #     # (var, obs) = filter_anndata_by_value(annobj, {a.SAMPLES : {"Gender": ['M']},
    #     #                                            a.GENES: {0: ['G1', 'G2']}})
    #
    #     # stratify and compute the cross tissue
    #     # (X, Y) = rnaseq_cross_tissue(annobj, individuals=['Indiv1'], gene_ids=annobj.obs_names)
    #     # print(X)
    #     # print()
    #     # print(Y)
    #
    #     # not working - https://github.com/theislab/anndata/issues/84
    #     # # save in the h5ad format
    #     # print("2. Annobj saved here: ")
    #     # name = save(annobj)
    #     # print(name)
    #     #
    #     # # # load from h5ad file
    #     # print("3. Load obj from h2ad file and change annotation")
    #     # annobj = load_anndata_from_file(name, backed='r+', varanno="../../anno_2.csv")
    #
    #
    # else:
    #     print("python preprocessing.py [explore|example]")

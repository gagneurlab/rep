__author__ = 'Mada'

import argh
import anndata
import pandas as pd
# import scanpy
import os


def load_count_matrix(filename,sep=",",varanno=None):
    """
    Load count matrix and put this into a summarized experiment

        ---- var ---
    |   0,T1,T2,T3,T4
    |   G1,10,20,30,40
    obs G2,5,10,20,30
    |   G3,6,7,8,9

    :return:
    Anndata object
    """
    abs_path = os.path.abspath(filename)
    print("My file: " + abs_path)

    # read count matrix
    annobj = anndata.read_csv(abs_path, delimiter=sep, first_column_names=1)

    # read var data (samples description)
    varaux = None
    if varanno:
        varaux = pd.read_csv(os.path.abspath(varanno),header=None,delimiter=sep)
    annobj.var = pd.DataFrame(varaux)



    # useful commands
    # print(annobj.X) # print count matrix
    # print(annobj.obs_names) # print rownames (observations)
    # print(annobj.var_names)

    return annobj


def filter_count_matrix():
    """
    Filter count matrix by columns (e.g. tissues), by row (e.g. genes), by region (e.g. gtf, bed)
    :return:
    """
    pass


# assembling:

parser = argh.ArghParser()
parser.add_commands([load_count_matrix, filter_count_matrix])

# dispatching:

if __name__ == '__main__':
    parser.dispatch()

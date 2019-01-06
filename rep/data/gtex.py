import sys
import os

import pandas as pd

from rep import preprocessing as p

def load_gtex_to_anndata():
    data_path = os.readlink(os.path.join("..","..","data"))
    path = os.path.join(data_path,"raw","gtex","recount","version2")
    counts_file = "recount_gene_counts.csv"
    rowdata_file = "recount_rowdata.csv"
    coldata_file = "recount_coldata.csv"

    # might take longer to create a large object
    annobj = p.create_anndata(os.path.join(path,counts_file), sep="\t", samples_anno=os.path.join(path,coldata_file), genes_anno=os.path.join(path,rowdata_file))
    
    # add Gender, Individual, Tissue columns
    sample_description = annobj.var
    sample_description.loc[:,'Gender'] = pd.Series(sample_description['bigwig_file']).str.split("_",expand=True)[3]
    
    aux = pd.Series(sample_description['sampid']).str.split("-",expand=True)
    sample_description.loc[:,'Individual'] = aux[0].map(str) + "-" + aux[1]
    
    sample_description = sample_description.rename(index=str, columns={"smts": "Parent_Tissue", "smtsd": "Tissue"})
    
    # redefine sample description
    annobj.var = sample_description
    
    # save AnnData object to HDF5 format
    output_file = os.path.join(data_path,"processed","gtex","recount","recount_gtex.h5ad")
    p.save(annobj,output_file)

if __init__ == "__main__":
    load_gtex_to_anndata()




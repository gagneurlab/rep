"""
Datasets 
"""
import json
import pandas as pd

import gin
import rep.preprocessing as p



class RepDataset(object):
    """
    Attributes:
        inputs (nparray):
        targets (nparray):
        metadata (json): use to compute tissue specific metrics
        label: dataset_name
    """
    def __init__(self, inp, tar, met=None, ds_name=None):
        self.inp = inp
        self.tar = tar
        self.met = met
        self.ds_name= ds_name
    
    @property
    def targets(self):
        return self.tar
    
    @property
    def inputs(self):
        return self.inp
    
    @property
    def metadata(self):
        return self.met
    
    @property
    def dataset_name(self):
        return self.ds_name
    

def read_decompress(file):
    """Read and decompress metadata
        
    Args:
        file (str): file name which contain a valid serialized json
                    dict_keys(['gene_metadata', 'patient_tissue_metadata'])
                     value of the key should be dataframes
    Returns:
        uncompress json format of the metadata
    """
    if file is None:
        return None
      
    with open(file,'r') as json_file:  
        data = json.load(json_file)
            
        # decompress dataframes:
        for key in data:
            data[key] = pd.read_json(data[key])
        
        return data
    
    return None


@gin.configurable
def rep_blood_expression(x_train_h5, y_train_h5, 
                         x_valid_h5, y_valid_h5, 
                         metadata_train_file=None, 
                         metadata_valid_file=None,
                         label=None):
    
    x_train = p.readh5(x_train_h5)
    y_train = p.readh5(y_train_h5)
    x_valid = p.readh5(x_valid_h5)
    y_valid = p.readh5(y_valid_h5)
    
    # avoid zero entries
    x_train[0,:] = x_train[0,:] + 0.001
    y_train[0,:] = y_train[0,:] + 0.001
    x_valid[0,:] = x_valid[0,:] + 0.001
    y_valid[0,:] = y_valid[0,:] + 0.001
    
    if label:
        dataset_name_valid = label + "_valid"
        dataset_name_train = label + "_train"
    else:
        dataset_name_valid = "valid"
        dataset_name_train = "train"
        
    metadata_train = read_decompress(metadata_train_file) if metadata_train_file else None
    metadata_valid = read_decompress(metadata_valid_file) if metadata_valid_file else None
    
    train_dataset = RepDataset(x_train, y_train, metadata_train, dataset_name_train)
    valid_dataset = RepDataset(x_valid, y_valid, metadata_valid, dataset_name_valid)
    
    return train_dataset, valid_dataset 
    
    
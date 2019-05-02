"""
Datasets 
"""
import json
import pandas as pd
import numpy as np

import gin
import rep.preprocessing_new as p



class RepDataset(object):
    """
    Attributes:
        inp (nparray): inputs
        tar (nparray): targets
        metadata (json): use to compute tissue specific metrics
        ds_name: dataset_name
    """
    def __init__(self, inp, tar, met=None, ds_name=None, features=None):
        self.inp = inp
        self.tar = tar
        self.met = met
        self.ds_name= ds_name
        self.feat = features

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
    def features(self):
        return self.feat

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
def rep_blood_expression(x_inputs_h5, y_targets_h5, label=None):

    x_inputs = p.RepAnnData.read_h5ad(x_inputs_h5)
    y_targets = p.RepAnnData.read_h5ad(y_targets_h5)

    # keep X and samples description
    x_train_all = x_inputs[x_inputs.samples['Type'] == 'train']
    x_valid_all = x_inputs[x_inputs.samples['Type'] == 'valid']

    x_train = np.array(x_train_all.X)
    y_train = np.array(y_targets[y_targets.samples['Type'] == 'train'].X)
    x_valid = np.array(x_valid_all.X)
    y_valid = np.array(y_targets[y_targets.samples['Type'] == 'valid'].X)

    # avoid zero entries
    x_train[0,:] = x_train[0,:] + 0.00000001
    y_train[0,:] = y_train[0,:] + 0.00000001
    x_valid[0,:] = x_valid[0,:] + 0.00000001
    y_valid[0,:] = y_valid[0,:] + 0.00000001

    if label:
        dataset_name_valid = label + "_valid"
        dataset_name_train = label + "_train"
    else:
        dataset_name_valid = "valid"
        dataset_name_train = "train"

    metadata_train = x_train_all.obs
    metadata_valid = x_valid_all.obs

    features_train = x_train_all.var
    features_valid = x_valid_all.var

    train_dataset = RepDataset(x_train, y_train, metadata_train, dataset_name_train, features_train)
    valid_dataset = RepDataset(x_valid, y_valid, metadata_valid, dataset_name_valid,features_valid)

    return train_dataset, valid_dataset


@gin.configurable
def rep_blood2blood_expression(x_inputs_h5, label=None):
    
    x_inputs = p.RepAnnData.read_h5ad(x_inputs_h5)

    # keep X and samples description
    x_train_all = x_inputs[x_inputs.samples['Type'] == 'train']
    x_valid_all = x_inputs[x_inputs.samples['Type'] == 'valid']

    x_train = np.array(x_train_all.X)
    x_valid = np.array(x_valid_all.X)
  
    # avoid zero entries
    x_train[0,:] = x_train[0,:] + 0.00000001
    x_valid[0,:] = x_valid[0,:] + 0.00000001
  
    if label:
        dataset_name_valid = label + "_valid"
        dataset_name_train = label + "_train"
    else:
        dataset_name_valid = "valid"
        dataset_name_train = "train"

    metadata_train = x_train_all.obs
    metadata_valid = x_valid_all.obs

    features_train = x_train_all.var
    features_valid = x_valid_all.var

    train_dataset = RepDataset(x_train, x_train, metadata_train, dataset_name_train, features_train)
    valid_dataset = RepDataset(x_valid, x_valid, metadata_valid, dataset_name_valid, features_valid)

    return train_dataset, valid_dataset
import os

from tqdm import tqdm
import logging
from copy import deepcopy
from collections import OrderedDict

from kipoi.data_utils import numpy_collate_concat
from kipoi.external.flatten_json import flatten

from gin_train.utils import write_json
from gin_train.utils import prefix_dict
from gin_train.metrics import RegressionMetrics
import gin

import sklearn
import torch
from torch.utils.data import TensorDataset

import joblib
import pandas as pd
from abc import ABCMeta, abstractmethod

import rep.models

torch.manual_seed(7)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Trainer(object):
    """Generic trainer object. This class has twio major components:
            (i) train model
            (ii) evalute model (this implies also predicting using the model)
    
        
    Attributes:
        model: compiled sklearn.pipeline.Pipeline
        train (:obj:RepDataset): training dataset
        valid (:obj:RepDataset): validation dataset 
        output_dir: output directory where to log the training
        cometml_experiment: if not None, append logs to commetml
        wandb_run: send evaluation scores to the dashbord
    """
    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):
        
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run

        # setup the output directory
        self.set_output_dir(output_dir)


    def set_output_dir(self, output_dir):
        """Set output folder structure for the model.

        Args:
            output_dir (str): output directory name
        """

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"
    
    

    ##########################################################################
    ###########################    Train   ###################################
    ##########################################################################
    
    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_samples_per_epoch=None,
              validation_samples=None,
              train_batch_sampler=None,
              **kwargs):

        # **kwargs won't be used, they are just included for compatibility with gin_train.
        """Train the model
        
        Args:
            num_workers: how many workers to use in parallel
        """
        
        # define dataset
        X_train, y_train = (self.train_dataset.inputs, self.train_dataset.targets)
        if self.valid_dataset is None:
            raise ValueError("len(self.valid_dataset) == 0")

        # check model type
        self.check_model()

        # fit model
        self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, num_workers=num_workers)
        
        # save model
        self.save()
        
    
    @abstractmethod
    def check_model(self):
        """Check if the model has the specified data type."""
        pass
    
    
    @abstractmethod
    def fit(self, inputs, targets):
        """Generic method for fitting using a given model.
           This method has to be implemented in subclass
        """
        pass
    
    
    @abstractmethod
    def save(self):
        """Generic method for saving the model.
           This method has to be implemented in subclass.
        """
        pass
    
    
    ##########################################################################
    ########################### Evaluation ###################################
    ##########################################################################
    

    def evaluate(self, 
                 metric, 
                 batch_size = None,
                 num_workers = 8,
                 eval_train=False,
                 eval_skip=False,
                 save=True):
        """Evaluate the model on the validation set

        Args:
            metric: a list or a dictionary of metrics
            batch_size: None - means full dataset
            num_workers: number of threads
            eval_train: if True, also compute the evaluation metrics on the training set
            save: save the json file to the output directory
        """
        
        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [self.train_dataset, self.valid_dataset]
        else:
            eval_datasets = [self.valid_dataset]
        
        metric_res = OrderedDict()
        eval_metric = metric
        
        for i, data in enumerate(eval_datasets):

            lpreds = []
            llabels = []
            
            inputs = data.inputs
            targets = data.targets
            
            lpreds.append(self.predict(inputs))
            llabels.append(deepcopy(targets))

            preds = numpy_collate_concat(lpreds)
            labels = numpy_collate_concat(llabels)
            del lpreds
            del llabels
            
            metric_res[data.dataset_name] = eval_metric(labels, preds, tissue_specific_metadata = data.metadata)
                
        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(prefix_dict(metric_res, prefix="eval/"), separator='/'))

        return metric_res
    
    
    @abstractmethod
    def predict(self, inputs):
        """Generic method for predicting using a given model.
           This method has to be implemented in subclass.
        """
        pass


@gin.configurable
class SklearnPipelineTrainer(Trainer):

    """Simple Scikit model trainer
    """

    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):

        Trainer.__init__(self,model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment,
                 wandb_run)

    
    def check_model(self):
        if not isinstance(self.model, sklearn.pipeline.Pipeline):
            raise ValueError("model is not a sklearn.pipeline.Pipeline")
    
    
    def fit(self, inputs, targets, epochs=10, batch_size=256, num_workers=8):        
        self.model.fit(inputs, targets)
    
    
    def save(self):        
        import pickle
        with open(self.ckp_file, 'wb') as file:
            pickle.dump(self.model, file)


    def predict(self, inputs):
        return self.model.predict(inputs)
    

@gin.configurable
class PyTorchTrainer(Trainer):

    """Simple PyTorch model trainer
    
    Attributes:
            model: object of class LinearRegressionCustom extending nn.Module
            train_dataset:
            valid_dataset:
            output_dir:
            cometml_experiment:
            wandb_run:
    """

    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):
        """
        
        Args:
            model: object of class LinearRegressionCustom extending nn.Module
            train_dataset:
            valid_dataset:
            output_dir:
            cometml_experiment:
            wandb_run:
        """
        
        super(PyTorchTrainer,self).__init__(model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment,
                 wandb_run)
        
        # initilize the output data
        self.loss = None
        self.outputs = None
        
   
    def check_model(self):
        if not isinstance(self.model, rep.models.LinearRegressionCustom):
            raise ValueError("model is not a rep.models.LinearRegressionCustom")
    
    
    def fit(self, inputs, targets, epochs=10, batch_size=256, num_workers=8):  
        
        for epoch in range(epochs):
            epoch +=1
            
            # define dataset
            trainset = torch.utils.data.TensorDataset(torch.Tensor(inputs),torch.Tensor(targets))
            
            # reset gradients
            self.model.get_optimiser.zero_grad()
            
            # use mini-batches to train the model
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                      shuffle=False, num_workers=num_workers)
            
            for i, data in enumerate(trainloader, 0):
                inp, labels = data    
                inp, labels = torch.autograd.Variable(inp), torch.autograd.Variable(labels)
#                 print('Batch idx {}, data shape {}, target shape {}'.format(i, inp.shape, 
#                                                                             lab.shape))
                self.fit_minibatch(inp, labels)
                
            print('epoch {}, loss {}'.format(epoch,self.loss))
    
    
    def fit_minibatch(self, inputs, labels):
            
        # forward to get predicted values
        self.outputs = self.model.get_model.forward(inputs)
        
        # compute loss
        self.loss = self.model.get_criterion(self.outputs, labels)
        
        # back propagation - compute gradients
        self.loss.backward()
        
        # update the parameters
        self.model.get_optimiser.step()
    
    
    def predict(self, inputs):
        return self.model.get_model(inputs)
    
    
    def save(self):
        torch.save(self.model.get_model, self.ckp_file)

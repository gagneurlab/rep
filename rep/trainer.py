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
import joblib


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Trainer(object):

    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):
        """
        Args:
            model: compiled sklearn.pipeline.Pipeline
            train: training Dataset (object inheriting from kipoi.data.Dataset)
            valid: validation Dataset (object inheriting from kipoi.data.Dataset)
            output_dir: output directory where to log the training
            cometml_experiment: if not None, append logs to commetml
            wandb_run:
        """
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
    ########################### Evaluation ###################################
    ##########################################################################
    

    def evaluate(self, 
                 metric, 
                 batch_size = None,
                 num_workers = 8,
                 eval_train=False,
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
            eval_datasets = self.valid_dataset
        
        metric_res = OrderedDict()
        eval_metric = metric
        
        for i, (inputs, targets) in enumerate(eval_datasets):

            lpreds = []
            llabels = []

            lpreds.append(self.model.predict(inputs))
            llabels.append(deepcopy(targets))

            preds = numpy_collate_concat(lpreds)
            labels = numpy_collate_concat(llabels)
            del lpreds
            del llabels
            
            if eval_train and i == 0:
                metric_res["dataset_train"] = eval_metric(labels, preds)
            else:
                metric_res["dataset_" + str(i)] = eval_metric(labels, preds)
                
        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(prefix_dict(metric_res, prefix="eval/"), separator='/'))

        return metric_res
    

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
        super(SklearnPipelineTrainer,self).__init__(model, train_dataset, valid_dataset, output_dir, cometml_experiment, wandb_run)

        
    def train(self,
              num_workers=8,
              **kwargs):

        # **kwargs won't be used, they are just included for compatibility with gin_train.
        """Train the model
        Args:
            num_workers: how many workers to use in parallel
        """

        X_train, y_train = (self.train_dataset[0], self.train_dataset[1])

        if self.valid_dataset is None:
            raise ValueError("len(self.valid_dataset) == 0")

        if not isinstance(self.model, sklearn.pipeline.Pipeline):
            raise ValueError("model is not a sklearn.pipeline.Pipeline")

        # fit model
        self.model.fit(X_train, y_train)
        
        # save model
        self.save()
#         self.model.save(self.ckp_file)
    
    def save(self):
        
        import pickle
        with open(self.ckp_file, 'wb') as file:
            pickle.dump(self.model, file)

#         import joblib
#         joblib.dump(self.model, self.ckp_file, compress = 1)


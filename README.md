# SNA Final Project

## How to use the code
1. make facebook directory
```
mkdir facebook
```
2. download the certain file `combined-adj-sparsefeat.pkl` which is the combined version of the facebook files
3. install dependencies
```
pip install -r requirements.txt
```

## Sample Code of training model
* train the GCN model with `hidden-size=16`, `lr=0.001`
* the AUC performance converges to **0.967920**
  
```python
import itertools
import configparser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv, GraphConv
import dgl.function as fn

from utils import compute_loss, compute_auc
from dataset import LinkPredictionDataset


# testing for the class
if __name__ == "__main__":

    # read from setting file
    configs = configparser.ConfigParser()
    configs.read('setting.ini')

    # load the training setting parameters
    model_name = configs["LP Parameter"]["model-name"]
    hidden_size = int(configs['LP Parameter']['hidden-size'])
    lr = float(configs['LP Parameter']['learning-rate'])

    # load the dataset settings
    filepath = os.path.join(configs["Data Process"]["entry"], "combined-adj-sparsefeat.pkl")
    test_ratio = float(configs["Data Process"]["test-ratio"])
    seed = int(configs["Reproduce"]["seed"])

    # create dataset
    link_prediction_ds = LinkPredictionDataset(filepath, seed)
    # build up pipeline
    torch.manual_seed( seed )
    pipeline = Pipeline(model_name, hidden_size, link_prediction_ds.get_feature_size())
    # split the dataset before training
    ds = link_prediction_ds.split(test_ratio)
    # train the model
    pipeline.train(ds, lr)
```

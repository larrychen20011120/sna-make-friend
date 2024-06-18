# SNA Final Project

## How to use the code
1. clone down the project and enter the project
```shell
git clone https://github.com/larrychen20011120/sna-make-friend.git
cd sna-make-friend
```
2. make facebook directory in order to keep the installed dataset
```shell
mkdir facebook
```
3. download the certain file `combined-adj-sparsefeat.pkl` which is the combined version of the facebook files
```shell
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WzV_rQ9oTxsw3s7yyXzdqEcrQgW591fj' -O ./facebook/combined-adj-sparsefeat.pkl
```
4. install dependencies
* simple installation for **none gpu-relavant** packages
```
pip install -r requirements.txt
```
* install your own **pytorch** version wtih the correct cuda version (you can type `nvidia-smi` to see cuda version)
* install dgl
   * first run the following code to get the correct torch version and cuda version
     
     ```
     !python -c "import torch; print(torch.__version__)"
     ```
   * install the dgl lib
     ```
     pip install  dgl -f https://data.dgl.ai/wheels/torch-{torch_version_here}/cu{cuda_version_here}/repo.html
     ```

## Sample Code of training model
* GCN
    * train the GCN model with `hidden-size=16`, `lr=0.001`
    * the AUC performance converges to **0.967920**
* SAGE
    * train the GCN model with `hidden-size=16`, `lr=0.0002`
    * the AUC performance converges to **0.922333**
* Conclusion -> learning rate has a great impact on training graph neural network

```python
import itertools
import configparser
import os
import torch
import matplotlib.pyplot as plt
from model import Pipeline
from dataset import LinkPredictionDataset


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

## Example of running testing
```python

import configparser
import os
import torch
from dataset import LinkPredictionDataset, sample_friend_pairs
from model import Pipeline
from utils import compute_rank_error
from tqdm import tqdm
from main import RandomWalkWithRestart
import numpy as np
import dgl
import math
import random

# read from setting file
configs = configparser.ConfigParser()
configs.read('setting.ini')

# load the training setting parameters
model_name = configs["LP Parameter"]["model-name"]
hidden_size = int(configs['LP Parameter']['hidden-size'])
lr = float(configs['LP Parameter']['learning-rate'])
epochs = int(configs["LP Parameter"]['epoch'])

# load the dataset settings
filepath = os.path.join(configs["Task Setting"]["entry"], "combined-adj-sparsefeat.pkl")
test_ratio = float(configs["Task Setting"]["test-ratio"])
seed = int(configs["Reproduce"]["seed"])
count = int(configs["Task Setting"]["friend-sample-count"])
max_alter_count = int(configs["Task Setting"]["max-alter-count"])

# load algorithm parameter
restart_ratio = float(configs['Random Walk']['restart-ratio'])
epsilon = float(configs["Random Walk"]["epsilon"])
walk_graph = configs["Random Walk"]["walk-graph"]

# create dataset
link_prediction_ds = LinkPredictionDataset(filepath, seed)
# build up pipeline
torch.manual_seed( seed )
pipeline = Pipeline(model_name, hidden_size, link_prediction_ds.get_feature_size())
# split the dataset before training
ds = link_prediction_ds.split(test_ratio)
pipeline.train(ds, lr=lr, epochs=epochs)

future_friend_pairs = sample_friend_pairs(ds, count=count, seed=seed)

method = RandomWalkWithRestart(restart_ratio, max_alter_count, walk_graph, epsilon)
origin_ranks, new_ranks = [], []

with tqdm(total=count, desc="Total Progress", position=0) as outer_progress:
    for pair in future_friend_pairs:
        src, tgt = pair
        g = method.fit(ds["train_g"], pair)
        
        # change new graph to dataset
        link_prediction_ds.set_new_graph(g)
        # split the dataset before training
        ds = link_prediction_ds.split(test_ratio, pairs=future_friend_pairs)

        # build up pipeline
        torch.manual_seed( seed )
        new_pipeline = Pipeline(model_name, hidden_size, link_prediction_ds.get_feature_size())
        new_pipeline.train(ds, lr, epochs, outer_progress)

        # get new rank
        origin_ranks.append( pipeline.predict_rank(ds["train_g"], src, tgt) )
        new_ranks.append( new_pipeline.predict_rank(ds["train_g"], src, tgt) )
    
# output the score
print( compute_rank_error(origin_ranks, new_ranks) )
```

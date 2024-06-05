# SNA Final Project

## How to use the code
1. clone down the project and enter the project
```shell
git clone https://github.com/larrychen20011120/sna-make-friend.git
cd sna-make-friend
```
2. make facebook directory
```shell
mkdir facebook
```
3. download the certain file `combined-adj-sparsefeat.pkl` which is the combined version of the facebook files
```shell
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WzV_rQ9oTxsw3s7yyXzdqEcrQgW591fj' -O ./facebook/combined-adj-sparsefeat.pkl
```
4. install dependencies
* simple installation
```
pip install -r requirements.txt
```
* if there is a version problem of DGL library, you should run the following command instead of installing the DGL lib from requirements.txt
```
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
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

# SNA Final Project

## How to use the code
1. make facebook directory
```
mkdir facebook
```
2. download the certain file `combined-adj-sparsefeat.pkl` which is the combined version of the facebook files

## Sample Code of training model
```python=
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

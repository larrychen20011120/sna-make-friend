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

## training model parameters
* GCN
    * train the GCN model with `hidden-size=16`, `lr=0.001`
    * the AUC performance converges to **0.967920**
* SAGE
    * train the GCN model with `hidden-size=16`, `lr=0.0002`
    * the AUC performance converges to **0.922333**
* Conclusion -> learning rate has a great impact on training graph neural network
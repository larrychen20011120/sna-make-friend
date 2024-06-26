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
     python -c "import torch; print(torch.__version__)"
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

## the project structures
* model.py: GCN, SAGE, and recommendation pipeline
* utils.py: the way to compute loss value and mean rank error
* our experiments is running on three python notebooks
	* Experiment 1: change_feature_only.ipynb
	* Experiment 2: rwr.ipynb
	* Experiment 3: sa.ipynb  (it's recommended to run it with CUDA)

## our proposed methods
* Rule Based

  * based on target's friends

    ![](assets/exp1.png)

  * based on target's ranking list

    ![](assets/exp2.png)

* Random Walk with Restart

  ![](assets/rwr.png)

* Simulated Annealing

  ![](assets/sa.png)

## experiments result
The whole comparison the like the following table. To see more details, please open the `report.pdf` to see the concrete experiment settings and alogrithm definition.

![](assets/table.png)

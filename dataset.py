import pickle
import os
import configparser
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import dgl

warnings.filterwarnings("ignore", category=DeprecationWarning)

class LinkPredictionDataset:

    def __init__(self, filepath, seed=42):

        self.friendship_matrix, self.feature_matrix = self.__load_data(filepath)
        self.num_nodes = self.friendship_matrix.shape[0]
        self.graph = self.__construct_graph()
        self.seed = seed
        

    def split(self, test_ratio=0.3):
        u, v = self.graph.edges()
        train_pos_g, train_neg_g, test_pos_g, test_neg_g = self.__sampling(u, v, test_ratio)
        dataset = {
            "train_pos_g": train_pos_g,
            "train_neg_g": train_neg_g,
            "test_pos_g":  test_pos_g,
            "test_neg_g":  test_neg_g,
            "train_g":     self.graph
        }
        return dataset

    def get_num_nodes(self):
        return self.num_nodes
    def get_feature_size(self):
        return self.feature_matrix.shape[1]

    # private methods
    def __load_data(self, filepath):
        with open(filepath, "rb") as f:
            friendship_matrix, feature_matrix = pickle.load(f, encoding='latin1')
        return friendship_matrix, feature_matrix

    def __construct_graph(self):
        src, dst = np.nonzero(self.friendship_matrix)
        g = dgl.graph((src, dst))
        g.ndata['feat'] = torch.tensor(self.feature_matrix.todense(), dtype=torch.float32)
        return g

    def __sampling(self, u, v, test_ratio):
        
        eids = np.arange(self.graph.number_of_edges())
        # setting random seed to get the same shuffle
        np.random.seed( self.seed )
        eids = np.random.permutation(eids)
        # calculate train and test size
        test_size = int(len(eids) * test_ratio)
        # get positive edges for test and train
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

        # Find all negative edges
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(self.graph.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        # split the negative edges for training and testing
        np.random.seed( self.seed )
        test_size = int(len(eids) * test_ratio)
        neg_eids = np.random.choice(len(neg_u), self.graph.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

        # construct positive and negative graphs for training and testing
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.graph.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.graph.number_of_nodes())
        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.graph.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.graph.number_of_nodes())

        # remove from origin graph g
        self.graph = dgl.remove_edges(self.graph, eids[:test_size])
        self.graph = dgl.add_self_loop(self.graph)

        return train_pos_g, train_neg_g, test_pos_g, test_neg_g

    ### Magic functions
    def __str__(self):
        informations = [
            f"Friendship matrix shape : { self.friendship_matrix.shape }",
            f"Feature matrix shape    : { self.feature_matrix.shape }"
        ]
        return "\n".join(informations)
    

# testing for the class
if __name__ == "__main__":
    # read from setting file
    configs = configparser.ConfigParser()
    configs.read('setting.ini')
    # load the setting parameters
    filepath = os.path.join(configs["Data Process"]["entry"], "combined-adj-sparsefeat.pkl")
    test_ratio = float(configs["Data Process"]["test-ratio"])
    seed = int(configs["Reproduce"]["seed"])
    # create dataset
    link_prediction_ds = LinkPredictionDataset(filepath, seed)
    print(link_prediction_ds)
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
        self.train_neg_g = None

    def split(self, test_ratio=0.3, pairs=[]):
        u, v = self.graph.edges()
        train_pos_g, train_neg_g, test_pos_g, test_neg_g = self.__sampling(u, v, test_ratio, pairs)

        dataset = {
            "train_pos_g": train_pos_g,
            "train_neg_g": train_neg_g,
            "test_pos_g":  test_pos_g,
            "test_neg_g":  test_neg_g,
            "train_g":     self.graph
        }
        return dataset

    def set_new_graph(self, g):
        del self.graph
        self.graph = g.clone()

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

    def __sampling(self, u, v, test_ratio, pairs=[]):
        
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
        # avoid the testing data be the negative samples
        if len(pairs) != 0:
            pair_u, pair_v = [], []
            for a, b in pairs:
                pair_u.append(a)
                pair_v.append(b)
                pair_v.append(a)
                pair_u.append(b)
            adj_pair = sp.coo_matrix((np.ones(len(pair_u)), (np.array(pair_u), np.array(pair_v))), shape=adj.shape)
            adj_neg = 1 - adj.todense() - np.eye(self.graph.number_of_nodes()) - adj_pair.toarray()

        else:
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

    ### Magic functions -> for printing out the dataset
    def __str__(self):
        informations = [
            f"Friendship matrix shape : { self.friendship_matrix.shape }",
            f"Feature matrix shape    : { self.feature_matrix.shape }"
        ]
        return "\n".join(informations)
    

def sample_friend_pairs(ds, count=500, seed=42):
    u, v = ds["train_g"].edges()
    pos_u_list, pos_v_list = list(u), list(v)
    neg_u_list, neg_v_list = [], []
    for k, g in ds.items():
        if k == "train_g":
            continue
        if "pos" in k:
            u, v = g.edges()
            pos_u_list.extend( list(u) ) 
            pos_v_list.extend( list(v) )
        elif "neg" in k:
            u, v = g.edges()
            neg_u_list.extend( list(u) )
            neg_v_list.extend( list(v) )
    
    # Find all negative edges
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    pos = sp.coo_matrix((np.ones(len(pos_u_list)), (np.array(pos_u_list), np.array(pos_v_list))), shape=adj.shape)
    adj_neg = 1 - adj.todense() - pos - np.eye(ds["train_g"].number_of_nodes())
    # set negative samples as not friend in the future
    adj_neg[neg_u_list, neg_v_list] = 0
    neg_u, neg_v = np.where(adj_neg != 0)

    np.random.seed( seed )
    neg_eids = np.random.choice(len(neg_u), count)

    return [ (u, v) for u, v in zip(neg_u[neg_eids], neg_v[neg_eids]) ]




# testing for the class
if __name__ == "__main__":
    # read from setting file
    configs = configparser.ConfigParser()
    configs.read('setting.ini')
    # load the setting parameters
    filepath = os.path.join(configs["Task Setting"]["entry"], "combined-adj-sparsefeat.pkl")
    test_ratio = float(configs["Task Setting"]["test-ratio"])
    seed = int(configs["Run Setting"]["seed"])
    count = int(configs["Task Setting"]["friend-sample-count"])
    # create dataset
    link_prediction_ds = LinkPredictionDataset(filepath, seed)
    print(link_prediction_ds)

    # split the dataset before training
    ds = link_prediction_ds.split(test_ratio)
    future_friend_pairs = sample_friend_pairs(ds, count=count, seed=seed)
    print(future_friend_pairs[:10])
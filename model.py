import itertools
import configparser
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv, GraphConv
import dgl.function as fn

from utils import compute_loss, compute_auc


# build two-layer GraphConv model
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class Pipeline:

    def __init__(self, model_name="GCN", hidden_size=16, in_feats=1283):

        self.model_name = model_name
        self.model = None
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.pred = DotPredictor()


    def train(self, ds, lr=0.001, epochs=100):

        if self.model_name == 'GCN':
            self.model = GCN(self.in_feats, self.hidden_size)
            lr=0.001
        elif self.model_name == 'SAGE':
            self.model = GraphSAGE(self.in_feats, self.hidden_size)
            lr=0.0002

        training_losses = []
        # ----------- set up loss and optimizer -------------- #
        # in this case, loss will in training loop
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.pred.parameters()), lr=lr)

        # ----------- training -------------------------------- #
        for _ in tqdm(range(epochs)):
            # forward
            h = self.model(ds["train_g"], ds["train_g"].ndata['feat'])  # get node embeddings
            pos_score = self.pred(ds["train_pos_g"], h)
            neg_score = self.pred(ds["train_neg_g"], h)
            loss = compute_loss(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())


        with torch.no_grad():
            pos_score = self.pred(ds["test_pos_g"], h)
            neg_score = self.pred(ds["test_neg_g"], h)
            print()
            print(f"AUC: {compute_auc(pos_score, neg_score):.6f}")

        return training_losses
        
    def recommend(self, input_graph, user_id=0):
        
        # `h` represents the node embeddings, with shape [num_nodes, hidden_size]
        h = self.model(input_graph, input_graph.ndata['feat'])
        # generate a graph with (num_nodes - num_friends_of_user) edges
        # one end of the edge is user_id
        # the other end is a user that's NOT friends with user_id
        u, v = input_graph.edges()
        user_friends = set([user_id])
        user_neg_u, user_neg_v = [], []
        for n1, n2 in zip(u, v):   # get all friends of user_id
            if int(n1) == user_id:
                user_friends.add(int(n2))
            if int(n2) == user_id:
                user_friends.add(int(n1))

        for i in range(input_graph.num_nodes()):  # generate "negative edges" for user_id
            if i != user_id and i not in user_friends:
                user_neg_u.append(user_id)
                user_neg_v.append(i)

        user_g = dgl.graph((user_neg_u, user_neg_v), num_nodes=input_graph.number_of_nodes())

        pred = DotPredictor()

        # calculate the score of each user
        with torch.no_grad():
            scores = [(i, score.item()) for i, score in enumerate(pred(user_g, h))]

        # produce final ranked list
        scores.sort(key=lambda x: -x[1])

        return scores
    
    def predict_rank(self, input_graph, src=0, tgt=100):

        result = self.recommend(input_graph, tgt)
        rank = -1
        for i, r in enumerate(result):
            if r[0] == src:
                rank = i
        return rank


# testing for the class
if __name__ == "__main__":

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
    model_path = os.path.join(configs['LP Parameter']["model-entry"], f"{model_name}.pth")
    test_ratio = float(configs["Task Setting"]["test-ratio"])
    seed = int(configs["Run Setting"]["seed"])

    # create dataset
    link_prediction_ds = LinkPredictionDataset(filepath, seed)
    # build up pipeline
    torch.manual_seed( seed )
    pipeline = Pipeline(model_name, hidden_size, link_prediction_ds.get_feature_size())

    # split the dataset before training
    ds = link_prediction_ds.split(test_ratio)

    pipeline.train(ds, lr, epochs)
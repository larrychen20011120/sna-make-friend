import numpy as np
import cupy as cp
import torch
import dgl
import math
import random

from utils import to_cpu, to_gpu


# ### with GPU speeding up
# def random_walk_with_restart(graph, start_point, restart_ratio, device="cpu"):
#     init_vector = np.zeros((graph.shape[0], 1))
#     init_vector[start_point, 0] = 1

#     # calculate closed-form of RWR
#     if device == "cpu":
#         proximity = (1 - restart_ratio) * np.linalg.inv( (np.eye(graph.shape[0]) - restart_ratio*graph) ) @ init_vector
#     elif device == "gpu":
#         init_vector = to_gpu(init_vector)
#         proximity = (1 - restart_ratio) * cp.linalg.inv( (cp.eye(graph.shape[0]) - restart_ratio*graph) ) @ init_vector
#     else:
#         print(f"No Such Device: {device}!!")
#         exit(-1)
#     return proximity

def random_walk_with_restart(graph, start_point, restart_ratio, max_iter=100, tol=1e-6, device="cpu"):
    if device == "cpu":
        graph = np.array(graph)
        init_vector = np.zeros((graph.shape[0], 1))
        init_vector[start_point, 0] = 1
        p = init_vector.copy()
        
        for _ in range(max_iter):
            prev_p = p.copy()
            p = (1 - restart_ratio) * np.dot(graph, p) + restart_ratio * init_vector
            if np.linalg.norm(p - prev_p, 1) < tol:
                break
                
    elif device == "gpu":
        graph = cp.array(graph)
        init_vector = cp.zeros((graph.shape[0], 1))
        init_vector[start_point, 0] = 1
        p = init_vector.copy()
        
        for _ in range(max_iter):
            prev_p = p.copy()
            p = (1 - restart_ratio) * cp.dot(graph, p) + restart_ratio * init_vector
            if cp.linalg.norm(p - prev_p, 1) < tol:
                break
        
    else:
        raise ValueError(f"No Such Device: {device}!!")
    
    return p

def construct_normalized_graph(friends, features, epsilon=1e-4, walk_graph="weighted", device="cpu"):
    if walk_graph == "adj":
        # normalize friends
        if device == "cpu":
            friends += (epsilon * np.eye(friends.shape[0], dtype=float))
            normalized_friends = friends / np.sum(friends, axis=1)
        else:
            friends += (epsilon * cp.eye(friends.shape[0]))
            normalized_friends = friends / cp.sum(friends, axis=1)

    elif walk_graph == "weighted":

        if device == "cpu":
            # Compute feature inner product matrix
            feature_inner_product = np.dot(features, features.T)
            # Replace the original adjacency matrix values with feature inner product values
            neighbors = (friends == 1)
            friends[neighbors] = feature_inner_product[neighbors]
            # Normalize the friends matrix
            friends += (epsilon * np.eye(friends.shape[0], dtype=float))
            normalized_friends = friends / np.sum(friends, axis=1)
        else:
            # Compute feature inner product matrix
            feature_inner_product = cp.dot(features, features.T)
            # Replace the original adjacency matrix values with feature inner product values
            neighbors = (friends == 1)
            friends[neighbors] = feature_inner_product[neighbors]
            # Normalize the friends matrix
            friends += (epsilon * cp.eye(friends.shape[0]))
            normalized_friends = friends / cp.sum(friends, axis=1)

    else:
        print("Error in reading parameter walk graph!!")
        exit(1)

    return normalized_friends

## 利用 target 鄰居共同特徵來修改特徵
###### 不使用 GPU 加速
class RuleBase:
    def __init__(self, k_feat=10):
        self.k_feat = k_feat

    def fit(self, input_graph, pair):
        user_id, tgt = pair
        users = []
        friends = input_graph.adj().to_dense().numpy()
        features = input_graph.ndata['feat'].numpy()

        for u, is_friend in enumerate(friends[tgt]):
            if is_friend:
                users.append(u)
        users = np.array(users)
        all_users_list = users.tolist()
        all_users_features = features[all_users_list, :]

        combined_features = np.sum(all_users_features, axis=0)

        indices = np.flip(np.argsort(combined_features))
        top_indices, cnt = [], 0

        for index in indices:
            if features[user_id, index] == 0 and combined_features[index] != 0:
                top_indices.append(index)
                cnt += 1
            elif features[user_id, index] != 0 and combined_features[index] == 0:
                top_indices.append(index)
                cnt += 1
            if cnt == self.k_feat:
                break
        
        src, dst = np.nonzero(friends)
        g = dgl.graph((src, dst))

        modified_features = features.copy()
        modified_features[user_id, top_indices] = 1
        g.ndata['feat'] = torch.tensor(modified_features, dtype=torch.float32)

        return g
    
## 利用 RWR 找出最重要的鄰居，與他們建立朋友關係
class RandomWalkWithRestart:

    def __init__(self, **param):
        self.restart_ratio = param["restart_ratio"]
        self.k_new_friend = param["max_alter_count"]
        self.walk_graph = param["walk_graph"]
        self.epsilon = param["epsilon"]
        self.device = param["device"]

    def fit(self, input_graph, pair):
        user_id, tgt = pair
        input_graph = dgl.remove_self_loop(input_graph.clone())
        
        # get friends and feature metrix
        friends = input_graph.adj().to_dense().numpy()
        features = input_graph.ndata['feat'].numpy()
        # turn to float
        friends = friends.astype(float)

        #### put it on the device
        if self.device == "gpu":
            friends = to_gpu(friends)
            features = to_gpu(features)

        normalized_friends = construct_normalized_graph(friends, features, self.epsilon, self.walk_graph, self.device)
        proximity = random_walk_with_restart(normalized_friends, tgt, self.restart_ratio, device=self.device)

        #### convert to numpy
        if self.device == "gpu":
            proximity = to_cpu(proximity)

        proximity = proximity.reshape(-1)
        top_friends = np.flip(np.argsort(proximity))[1:self.k_new_friend+1]

        friends[top_friends, user_id] = 1
        friends[user_id, top_friends] = 1

        src, dst = np.nonzero(friends)
        g = dgl.graph((src, dst))
        g.ndata['feat'] = torch.tensor(features, dtype=torch.float32)

        return g

## 利用最佳化方法結合 RWR 來尋找最佳解
class SimulatedAnnealing:
    
    def __init__(self, **param):
        self.start_temp = param["start_temp"]
        self.end_temp = param["end_temp"]
        self.cooling_rate = param["cooling_rate"]
        self.max_iter = param["max_iter"]
        self.answer_size = param["answer_size"]
        self.seed = param["seed"]
        self.device = param["device"]

    def fit(self, input_graph, pair, restart_ratio=0.1):

        user_id, tgt = pair
        input_graph = input_graph.clone()
        node_size = input_graph.ndata['feat'].shape[0]
        feature_size = input_graph.ndata['feat'].shape[1]


        temperature = self.start_temp

        # 隨機挑選初始值並產生初始解
        curr_answer = self.__init_answer(node_size, feature_size)
        alter_graph = self.__alter_graph(input_graph, curr_answer, user_id)

        # get friends and feature metrix
        alter_graph = dgl.remove_self_loop(alter_graph)
        friends = alter_graph.adj().to_dense().numpy()
        features = alter_graph.ndata['feat'].numpy()
        # turn to float
        friends = friends.astype(float)

        #### put it on the device
        if self.device == "gpu":
            friends = to_gpu(friends)
            features = to_gpu(features)

        normalized_friends = construct_normalized_graph(friends, features)
        proximity = random_walk_with_restart(normalized_friends, tgt, restart_ratio=restart_ratio, device=self.device)

        #### convert to numpy
        if self.device == "gpu":
            proximity = to_cpu(proximity)
        proximity = proximity.reshape(-1)
        curr_fitness = proximity[user_id]

        for _ in tqdm(range(self.max_iter)):
            if temperature == self.end_temp:
                return curr_answer, curr_fitness
        
            # generate next answer -> random pick & alter it
            alter_pos = np.random.choice(len(curr_answer), 1)[0]
            r = np.random.randint(0, node_size+feature_size)

            while r in curr_answer:
                r = np.random.randint(0, node_size+feature_size)

            next_answer = curr_answer[:]
            next_answer[alter_pos] = r
            alter_graph = self.__alter_graph(input_graph, next_answer, user_id)

           # get friends and feature metrix
            alter_graph = dgl.remove_self_loop(alter_graph)
            friends = alter_graph.adj().to_dense().numpy()
            features = alter_graph.ndata['feat'].numpy()
            # turn to float
            friends = friends.astype(float)

            #### put it on the device
            if self.device == "gpu":
                friends = to_gpu(friends)
                features = to_gpu(features)

            normalized_friends = construct_normalized_graph(friends, features)
            proximity = random_walk_with_restart(normalized_friends, tgt, restart_ratio=restart_ratio, device=self.device)

            #### convert to numpy
            if self.device == "gpu":
                proximity = to_cpu(proximity)
            proximity = proximity.reshape(-1)

            next_fitness = proximity[user_id]
            improved_fitness = next_fitness - curr_fitness

            if improved_fitness > 0:
                curr_fitness = next_fitness
                curr_answer = next_answer
            else:
                try: # 避免指數計算 overflow
                    prob = math.exp(-improved_fitness/temperature)
                except OverflowError:
                    prob = 0 # 值過小設為零

                if random.random() < prob:
                    # 允許錯誤=>更新
                    curr_fitness = next_fitness
                    curr_answer = next_answer

            temperature *= self.cooling_rate

        return self.__alter_graph(input_graph, curr_answer, src)


    def __init_answer(self, node_size, feature_size):

        feature_change_size = np.random.choice(self.answer_size, 1)[0]
        node_change_size = self.answer_size - feature_change_size
        feature_list = np.arange(feature_size)
        node_list = np.arange(feature_size, feature_size + node_size)
        np.random.shuffle(feature_list)
        np.random.shuffle(node_list)

        return list(feature_list[:feature_change_size]) + list(node_list[:node_change_size])
    
    def __alter_graph(self, input_graph, answer, src):
        input_graph = input_graph.clone()
        feature_size = input_graph.ndata['feat'].shape[1]

        for a in answer:
            if a < feature_size:
                replaced_value = 1 - input_graph.ndata['feat'][src, a]
                input_graph.ndata['feat'][src, a] = 0 if replaced_value < 0 else replaced_value
            else:
                a -= feature_size
                if not input_graph.has_edges_between([src], [a])[0]:
                    input_graph.add_edges([src], [a])
                    input_graph.add_edges([a], [src])
                else:
                    eid = input_graph.edge_ids(src, a)
                    input_graph.remove_edges(eid)

        return input_graph


import configparser
import os
import torch
from dataset import LinkPredictionDataset, sample_friend_pairs
from model import Pipeline
from utils import compute_rank_error
from tqdm import tqdm

# testing for the class
if __name__ == "__main__":

    # read from setting file
    configs = configparser.ConfigParser()
    configs.read('setting.ini')

    # load running setting
    seed = int(configs["Run Setting"]["seed"])
    device = configs["Run Setting"]["device"]

    # load the training setting parameters
    model_name = configs["LP Parameter"]["model-name"]
    hidden_size = int(configs['LP Parameter']['hidden-size'])
    lr = float(configs['LP Parameter']['learning-rate'])
    epochs = int(configs["LP Parameter"]['epoch'])

    # load the dataset settings
    filepath = os.path.join(configs["Task Setting"]["entry"], "combined-adj-sparsefeat.pkl")
    test_ratio = float(configs["Task Setting"]["test-ratio"])
    count = int(configs["Task Setting"]["friend-sample-count"])
    max_alter_count = int(configs["Task Setting"]["max-alter-count"])

    # load algorithm parameter
    restart_ratio = float(configs['Random Walk']['restart-ratio'])
    epsilon = float(configs["Random Walk"]["epsilon"])
    walk_graph = configs["Random Walk"]["walk-graph"]
    start_temp = int(configs["Simulated Annealing"]["start-temp"])
    end_temp = int(configs["Simulated Annealing"]["end-temp"])
    max_iter = int(configs["Simulated Annealing"]["max-iter"])
    cooling_rate = float(configs["Simulated Annealing"]["cooling-rate"])

    # create dataset
    link_prediction_ds = LinkPredictionDataset(filepath, seed)
    # build up pipeline
    torch.manual_seed( seed )
    pipeline = Pipeline(model_name, hidden_size, link_prediction_ds.get_feature_size())
    # split the dataset before training
    ds = link_prediction_ds.split(test_ratio)
    pipeline.train(ds, lr=lr, epochs=epochs)

    future_friend_pairs = sample_friend_pairs(ds, count=count, seed=seed)

    # method = RuleBase(max_alter_count) #
    method = RandomWalkWithRestart(
        restart_ratio=restart_ratio, max_alter_count=max_alter_count, 
        walk_graph=walk_graph, epsilon=epsilon, device=device
    )
    # method = SimulatedAnnealing(
    #     start_temp=start_temp, end_temp=end_temp, max_iter=max_iter,
    #     cooling_rate=cooling_rate, answer_size=max_alter_count,
    #     seed=seed, device=device
    # )

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
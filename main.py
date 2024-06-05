import numpy as np
import math
import random
from tqdm.auto import tqdm

class GeneticAlgorithm:
    def __init__(self):
        pass


class SimulatedAnnealing:
    
    def __init__(self, **param):
        self.start_temp = param["start_temp"]
        self.end_temp = param["end_temp"]
        self.cooling_rate = param["cooling_rate"]
        self.max_iter = param["max_iter"]
        self.answer_size = param["max_change_count"]
        self.seed = param["seed"]

    def fit(self, pair, input_graph, pipeline):

        src, tgt = pair
        input_graph = input_graph.clone()
        node_size = input_graph.ndata['feat'].shape[0]
        feature_size = input_graph.ndata['feat'].shape[1]

        temperature = self.start_temp
        
        init_fitness = pipeline.predict(input_graph, src, tgt)
        curr_answer = self.__init_answer(node_size, feature_size)
        alter_graph = self.__alter_graph(input_graph, curr_answer, src)
        next_fitness = pipeline.predict(alter_graph, src, tgt)

        curr_fitness = next_fitness

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
            alter_graph = self.__alter_graph(input_graph, next_answer, src)
            next_fitness = pipeline.predict(alter_graph, src, tgt)  
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

        init_rank, curr_rank = None, None
        for rank, (user, _) in enumerate(pipeline.recommend(input_graph, tgt)):
            if user == src:
                init_rank = rank
        alter_graph = self.__alter_graph(input_graph, curr_answer, src)
        for rank, (user, _) in enumerate(pipeline.recommend(alter_graph, tgt)):
            if user == src:
                curr_rank = rank
        
        return {
            "adjustment":    curr_answer,
            "improved-fitness": curr_fitness - init_fitness,
            "improved-rank": curr_rank - init_rank
        }


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
    filepath = os.path.join(configs["Task Setting"]["entry"], "combined-adj-sparsefeat.pkl")
    test_ratio = float(configs["Task Setting"]["test-ratio"])
    seed = int(configs["Reproduce"]["seed"])
    count = int(configs["Task Setting"]["friend-sample-count"])

    # create dataset
    link_prediction_ds = LinkPredictionDataset(filepath, seed)
    # build up pipeline
    torch.manual_seed( seed )
    pipeline = Pipeline(model_name, hidden_size, link_prediction_ds.get_feature_size())
    # split the dataset before training
    ds = link_prediction_ds.split(test_ratio)
    # train the model
    pipeline.train(ds, lr)
    future_friend_pairs = sample_friend_pairs(ds, count=count, seed=seed)

    # SA
    start_temp = int(configs["SA Parameter"]["init-temp"])
    end_temp = int(configs["SA Parameter"]["end-temp"])
    max_iter = int(configs["SA Parameter"]["max-iter"])
    cooling_rate = float(configs["SA Parameter"]["cooling-rate"])
    max_change_count = int(configs["Task Setting"]["max-change-count"])

    sa = SimulatedAnnealing(
        start_temp=start_temp, end_temp=end_temp, max_iter=max_iter,
        cooling_rate=cooling_rate, max_change_count=max_change_count, seed=seed
    )

    for pair in future_friend_pairs:
        print(sa.fit(pair, ds["train_g"], pipeline))
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import cupy as cp

def compute_loss(pos_score, neg_score):  # computes the loss based on binary cross entropy
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):  # computes AUC (Area-Under-Curve) score
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_rank_error(prev_rank, curr_rank):
    improved_rank = [ c - p for c, p in zip(curr_rank, prev_rank) ]
    return {
         "average": np.sum(improved_rank) / len(curr_rank),
         "max": np.max(improved_rank),
         "min": np.min(improved_rank)
    }
     

def compute_hit_ratio(prev_rank, curr_rank, top_n=5):
    hit = 0
    for p, c in zip(prev_rank, curr_rank):
        if p > top_n and c <= top_n:
             hit += 1
    return hit / len(curr_rank)

def to_gpu(nparray):
    return cp.asarray(nparray)

def to_cpu(cparray):
    return cp.asnumpy(cparray)
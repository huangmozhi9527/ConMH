import numpy as np
import scipy.io as sio 
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
        yescnt_all[qid] = x
        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all  


def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()
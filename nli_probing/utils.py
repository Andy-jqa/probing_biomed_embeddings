__author__ = 'Qiao Jin'

import json
import model
import numpy as np
import random
import sys
import time
import torch
import os

def Batching(ids, batch_size, shuffle=True):
    if shuffle:
       random.shuffle(ids)

    if len(ids) % batch_size == 0:
        batches = int(len(ids) / batch_size)
    else:
        batches = int(len(ids) / batch_size) + 1

    for batch in range(batches):
        yield ids[batch * batch_size: (batch + 1) * batch_size]

def readCache(caches, instances, sentid2idx, sent_num, embed_type):
    embedding_list = [caches[sentid2idx[instance + '_' + sent_num]] for instance in instances] # List[np.ndarray(3 x len x 1024)]
    if embed_type == 'biomed_w2v':
       lengths = [embedding.shape[0] for embedding in embedding_list] 
    else:
        lengths = [embedding.shape[1] for embedding in embedding_list]
    max_len = max(lengths)
    batch_size = len(lengths)

    if embed_type == 'biomed_w2v':
        embeddings, token_mask = np.zeros((batch_size, max_len, 200)), np.zeros((batch_size, max_len))
    else:
        embeddings, token_mask = np.zeros((batch_size, 3, max_len, 1024)), np.zeros((batch_size, max_len))
        # B x 3 x L x 1024, B x L

    for i in range(batch_size):
        if embed_type == 'biomed_w2v':
            embeddings[i, :lengths[i], :] = embedding_list[i] 
        else:
            embeddings[i, :, :lengths[i], :] = embedding_list[i]
        token_mask[i, :lengths[i]] = 1

    embeddings = torch.Tensor(embeddings).cuda()
    token_mask = torch.Tensor(token_mask).cuda()

    return embeddings, token_mask

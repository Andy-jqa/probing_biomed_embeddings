__author__ = 'Qiao Jin'

import math
import numpy as np
import os
import torch
from torch import nn

from utils import *

class Model(nn.Module):
    def __init__(self, num_tags, h1=512, h2=128, h3=32):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(3, 1))
        nn.init.xavier_normal_(self.weight)

        self.linears = nn.Sequential(
        nn.Linear(1024, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, h3),
        nn.ReLU())

        self.output = nn.Sequential(
        nn.Linear(h3, num_tags),
        nn.ReLU())

    def forward(self, word, mask):
        B = word.size(0)
        L = word.size(2) 

        word = word.view(B, 3, L*1024)
        word = word.permute(0, 2, 1) # B x L*1024 x 3

        weights = nn.functional.softmax(self.weight, dim=0) # 3 x 1

        word = torch.matmul(word, weights) # B x L*1024 x 1
        word = word.squeeze(dim=2).view(B, L, 1024) # B x L x 1024

        word = self.linears(word) # B x L x 32
        word = self.output(word) # B x L x 5

        return word
    
    def get_repr(self, word):
        '''
        word: B x 3 x 1024
        '''
        B = word.size(0)

        word = word.permute(0, 2, 1) # B x 1024 x 3

        weights = nn.functional.softmax(self.weight, dim=0) # 3 x 1

        word = torch.matmul(word, weights).squeeze(dim=2) # B x 1024

        word = self.linears(word) # B x 32
        word = self.output(word)

        return word # B x 1024


class w2v_Model(nn.Module):
    def __init__(self, num_tags, h1=128, h2=64, h3=32):
        super(w2v_Model, self).__init__()

        self.linears = nn.Sequential(
        nn.Linear(200, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, h3),
        nn.ReLU())
        
        self.output = nn.Sequential(
        nn.Linear(h3, num_tags),
        nn.ReLU())

    def forward(self, word, mask):
        B = word.size(0)
        L = word.size(2) 

        word = self.linears(word) # B x L x 32
        word = self.output(word) #

        return word

def Inference(path, caches, sentid2idx, Model, crf, embed_type):

    test_set = json.load(open('data/test'))
    test_data_loader = Batching(list(test_set), batch_size=64, shuffle=False)

    os.system('rm -rf %s' % path)

    for instances in test_data_loader:
        embedding, token_mask = readCache(caches, instances, sentid2idx, embed_type) 
        outputs = Model(embedding, token_mask)  # B x L x 5
        outputs = crf.viterbi_tags(logits=outputs, mask=token_mask)

        outputs = list(zip(*outputs))[0]

        for instance, output in zip(instances, outputs):
            instance = test_set[instance]
            id = instance['id']
            text = instance['sent']
            text_len = [len(token) for token in text]
            text_accum_len = [0] + [sum(text_len[:i+1]) for i in range(len(text_len))]
            
            output = output[:len(text_len)]

            s_idx = []
            e_idx = []

            for idx, label in enumerate(output):
                if label == 4:
                    if len(s_idx) > len(e_idx):
                        e_idx.append(idx-1)
                    s_idx.append(idx)
                    e_idx.append(idx)
                elif label == 2:
                    if len(s_idx) == len(e_idx):
                        s_idx.append(idx)
                elif label == 3:
                    if len(s_idx) > len(e_idx):
                        e_idx.append(idx)
                    else:
                        s_idx.append(idx)
                elif label == 0:
                    if len(s_idx) > len(e_idx):
                        e_idx.append(idx-1)
            
            for s, e in zip(s_idx, e_idx):
                with open(path, 'a') as f:
                    f.write('%s|%d %d|%s\n' % (id, text_accum_len[s], text_accum_len[e+1]-1, ' '.join([text[idx] for idx in range(s, e+1)])))

__author__ = 'Qiao Jin'

from allennlp.modules import conditional_random_field as CRF

import h5py
import json
import model
import numpy as np
import random
import sys
import os
import time
import torch

from utils import *

def main(params):

    num_epoch = params['num_epoch']
    batch_size = params['batch_size']
    lr = params['lr']
    embed_type = params['embed_type']
    seed = params['seed']
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    crf = CRF.ConditionalRandomField(num_tags=5)
    crf.cuda()
    
    if embed_type == 'biomed_w2v':
        Model = model.w2v_Model(num_tags=5)
    else:
        Model = model.Model(num_tags=5)
    Model.cuda()

    optimizer = torch.optim.Adam([param for param in crf.parameters() if param.requires_grad==True] + list(Model.parameters()),lr=lr)
    iter_num = 1
    t_start = time.time()

    whole_train_set = json.load(open('data/training'))
    whole_train_ids = list(whole_train_set)
    train_set =  {_id:whole_train_set[_id] for _id in whole_train_ids[:12500]}
    dev_set = {_id:whole_train_set[_id] for _id in whole_train_ids[12500:]}

    print('Loading caches...')
    if embed_type == 'biomed_w2v':
        caches = np.load('caches_%s/embedding.npy' % embed_type).item()
    else:
        caches = h5py.File('caches_%s/elmo_embeddings.hdf5' % embed_type, 'r')
    print('Loaded')

    idx2sentid = json.load(open('dict/idx2sentid_%s' % embed_type))
    if embed_type == 'biomed_w2v':
        sentid2idx = {v:int(k) for k,v in idx2sentid.items()}
    else:
        sentid2idx = {v:k for k,v in idx2sentid.items()}

    log_dir = '%s_seed%d_log' % (embed_type, seed)
    os.system('rm -rf %s' % log_dir)
    os.system('mkdir %s' % log_dir)

    min_loss = float('inf')
    min_loss_iter = 0

    for epoch in range(num_epoch):
        train_data_loader = Batching(list(train_set), batch_size, shuffle=True)

        for instances in train_data_loader:
            embedding, token_mask = readCache(caches, instances, sentid2idx, embed_type) # B x 3 x L x 1024
            embedding = Model(embedding, token_mask)

            labels = iobesLabel([train_set[instance] for instance in instances], embedding)

            loss = -crf(inputs=embedding, tags=labels, mask=token_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num % 25 == 0:
                t_elapsed = time.time() - t_start

                message = ('Epoch %d Iter %d TRAIN loss=%.4e elapsed=%.1f'  % (epoch, iter_num, loss.detach(), t_elapsed))
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')
            
            if iter_num % 200 == 0:
                dev_data_loader = Batching(list(dev_set), batch_size, shuffle=False)
                
                total_loss = 0
                step = 0

                for instances in dev_data_loader:
                    embedding, token_mask = readCache(caches, instances, sentid2idx, embed_type) # B x 3 x L x 1024
                    embedding = Model(embedding, token_mask)
                    
                    labels = iobesLabel([dev_set[instance] for instance in instances], embedding)
                    loss = -crf(inputs=embedding, tags=labels, mask=token_mask).detach()
                    
                    total_loss += loss
                    step += 1

                average_loss = total_loss / step

                message = ('Epoch %d EVAL loss=%.4e' % (epoch, average_loss))
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')
                
                if embed_type != 'biomed_w2v':
                    weights = Model.weight.data.squeeze().detach().cpu().numpy()
                    weights = [np.exp(weight)/np.sum(np.exp(weights)) for weight in weights]

                    message = 'weights: %.4f %.4f %.4f' % (weights[0], weights[1], weights[2])
                    print(message)
                    with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')
                
                if average_loss <= min_loss:
                    min_loss_iter = iter_num
                    min_loss = average_loss
                    torch.save(crf.state_dict(), '%s/best_crf' % log_dir)
                    torch.save(Model.state_dict(), '%s/best_model' % log_dir)
                    model.Inference('%s/predictions' % (log_dir), caches, sentid2idx, Model, crf, embed_type)

                message = 'best iter %s loss %.4e' % (min_loss_iter, min_loss)
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')

            iter_num += 1

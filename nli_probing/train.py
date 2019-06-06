__author__ = 'Qiao Jin'

import h5py
import json
import model
import numpy as np
import random
import sys
import time

import torch
from torch import nn

from utils import *

def main(params):
    num_epoch = params['num_epoch']
    batch_size = params['batch_size']
    lr = params['lr']
    embed_type = params['embed_type']
    seed = params['seed']
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if embed_type == 'biomed_w2v':
        Model = model.w2v_Model()
    else:
        Model = model.Model()
    Model.cuda()

    optimizer = torch.optim.Adam(Model.parameters(),lr=lr)
    iter_num = 1
    t_start = time.time()

    train_set = json.load(open('data/std_training'))
    dev_set = json.load(open('data/std_dev'))
    test_set = json.load(open('data/std_test'))

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

    toClass = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

    criterion = nn.CrossEntropyLoss()

    log_dir = '%s_seed%d_log' % (embed_type, seed)
    os.system('rm -rf %s' % log_dir)
    os.system('mkdir %s' % log_dir)

    max_dev_acc = float('-inf')

    for epoch in range(num_epoch):
        train_data_loader = Batching(list(train_set), batch_size, shuffle=True)

        for instances in train_data_loader:
            embedding_1, token_mask_1 = readCache(caches, instances, sentid2idx, '1', embed_type) # B x 3 x L x 1024
            embedding_2, token_mask_2 = readCache(caches, instances, sentid2idx, '2', embed_type) # B x 3 x L x 1024

            labels = [toClass[train_set[instance]['label']] for instance in instances]
            labels = torch.Tensor(labels).type(torch.LongTensor).cuda()

            logits = Model(embedding_1, token_mask_1, embedding_2, token_mask_2)        

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num % 10 == 0:
                t_elapsed = time.time() - t_start
                message = ('Epoch %d Iter %d TRAIN loss=%.4e elapsed=%.1f'  % (epoch, iter_num, loss.detach(), t_elapsed))
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')
            
            if iter_num % 50 == 0:
                dev_data_loader = Batching(list(dev_set), batch_size, shuffle=False)
                
                total_loss = 0
                step = 0
                total_instances = 0
                right_predictions = 0

                for instances in dev_data_loader:
                    embedding_1, token_mask_1 = readCache(caches, instances, sentid2idx, '1', embed_type) # B x 3 x L x 1024
                    embedding_2, token_mask_2 = readCache(caches, instances, sentid2idx, '2', embed_type)
                    
                    labels = [toClass[dev_set[instance]['label']] for instance in instances]
                    labels = torch.Tensor(labels).type(torch.LongTensor).cuda()

                    logits = Model(embedding_1, token_mask_1, embedding_2, token_mask_2).detach()
                    pred = torch.argmax(logits, dim=1)
                    
                    total_instances += logits.size(0)
                    right_predictions += torch.sum(pred == labels).detach().cpu().numpy()

                    loss = criterion(logits, labels).detach()

                    total_loss += loss
                    step += 1

                dev_accuracy = right_predictions/total_instances
                max_dev_acc = max(max_dev_acc, dev_accuracy)

                message = ('Epoch %d EVAL loss=%.4e acc=%.4f' % (epoch, total_loss/step, right_predictions/total_instances))
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')

                test_data_loader = Batching(list(test_set), batch_size, shuffle=False)
                
                total_loss = 0
                step = 0
                total_instances = 0
                right_predictions = 0

                for instances in test_data_loader:
                    embedding_1, token_mask_1 = readCache(caches, instances, sentid2idx, '1', embed_type) # B x 3 x L x 1024
                    embedding_2, token_mask_2 = readCache(caches, instances, sentid2idx, '2', embed_type)
                    
                    labels = [toClass[test_set[instance]['label']] for instance in instances]
                    labels = torch.Tensor(labels).type(torch.LongTensor).cuda()

                    logits = Model(embedding_1, token_mask_1, embedding_2, token_mask_2).detach()
                    pred = torch.argmax(logits, dim=1)
                    
                    total_instances += logits.size(0)
                    right_predictions += torch.sum(pred == labels).detach().cpu().numpy()

                    loss = criterion(logits, labels).detach()

                    total_loss += loss
                    step += 1

                test_accuracy = right_predictions/total_instances

                if dev_accuracy == max_dev_acc:
                    test_at_max = test_accuracy
                    torch.save(Model.state_dict(), '%s/best_model' % log_dir)

                message = ('Epoch %d TEST loss=%.4e acc=%.4f' % (epoch, total_loss/step, right_predictions/total_instances))
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')

                message = ('TEST acc=%.4f at best dev acc' % test_at_max)
                print(message)
                with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')

                if embed_type != 'biomed_w2v':
                    weights = Model.weight.data.squeeze().detach().cpu().numpy()
                    weights = [np.exp(weight)/np.sum(np.exp(weights)) for weight in weights]

                    message = 'weights: %.4f %.4f %.4f' % (weights[0], weights[1], weights[2])
                    print(message)
                    with open('%s/training_log' % log_dir, 'a') as f: f.write(message + '\n')

            iter_num += 1


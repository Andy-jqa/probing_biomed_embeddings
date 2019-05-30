# modified from https://github.com/allenai/bilm-tf/blob/master/usage_cached.py

import argparse
import os

import h5py
from bilm import dump_bilm_embeddings
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

import json

def w2v(tokens):
    _output = np.zeros((len(tokens), 200))

    for idx, token in enumerate(tokens):
        if token in model.vocab:
            _output[idx] = model[token]
        else:
            _output[idx] = average

    return _output

parser = argparse.ArgumentParser(description='Cache dataset')
parser.add_argument('--embed_type', dest='embed_type', type=str, help='Cache the specified embedding type of the dataset. Possible types: "biomed_elmo", "biomed_w2v", "general_elmo"')

args = parser.parse_args()
params = vars(args)

train = json.load(open('data/training'))
test = json.load(open('data/test'))

idx2sentid = {}

if not os.path.isdir('caches_%s' % params['embed_type']): os.mkdir('caches_%s' % params['embed_type'])

if params['embed_type'] == 'biomed_w2v':
    output = {}

    model = KeyedVectors.load_word2vec_format('../weights/biomed_w2v.bin', binary=True)
    average = np.load('../weights/w2v_average.npy')

    for sentid in list(train):
        output[len(output)]= w2v(train[sentid]['sent'])
        idx2sentid[len(idx2sentid)] = sentid

    for sentid in list(test):
        output[len(output)]= w2v(test[sentid]['sent'])
        idx2sentid[len(idx2sentid)] = sentid

    np.save('caches_%s/embedding' % params['embed_type'], output)

else:
    dataset_file = 'caches_%s/dataset_file.txt' % params['embed_type']

    with open(dataset_file, 'w') as fout:
        for sent_id in list(train):
            sentence = train[sent_id]['sent'] # List[Tokens]
            fout.write(' '.join(sentence) + '\n')
            idx2sentid[len(idx2sentid)] = sent_id
        for sent_id in list(test):
            sentence = test[sent_id]['sent'] # List[Tokens]
            fout.write(' '.join(sentence) + '\n')
            idx2sentid[len(idx2sentid)] = sent_id

    if params['embed_type'] in ['general_elmo', 'biomed_elmo']:
        vocab_file = '../dict/vocabulary.txt'
        options_file = '../weights/%s_options.json' % params['embed_type']
        weight_file = '../weights/%s_weights.hdf5' % params['embed_type']

    embedding_file = 'caches_%s/elmo_embeddings.hdf5' % params['embed_type']
    dump_bilm_embeddings(
        vocab_file, dataset_file, options_file, weight_file, embedding_file
    )

if not os.path.isdir('dict'): os.mkdir('dict')

with open('dict/idx2sentid_%s' % params['embed_type'], 'w') as f:
    json.dump(idx2sentid, f)

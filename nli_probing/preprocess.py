__author__ = 'Qiao Jin'

'''
Preprocess the jsonlines data format to previously used json format
'''

import json
import jsonlines
import nltk

def transform(part):
    '''
    part: Str, train or dev or test
    '''
    output = {}

    with jsonlines.open('data/mli_%s_v1.jsonl' % part) as reader:
        for obj in reader:
            pairID = obj['pairID']
            label = obj['gold_label']

            output[pairID] = {}
            output[pairID]['label'] = label
            output[pairID]['sentence1'] = nltk.word_tokenize(obj['sentence1'])
            output[pairID]['sentence2'] = nltk.word_tokenize(obj['sentence2'])

    return output

with open('data/std_training', 'w') as f:
    json.dump(transform('train'), f)
with open('data/std_dev', 'w') as f:
    json.dump(transform('dev'), f)
with open('data/std_test', 'w') as f:
    json.dump(transform('test'), f)

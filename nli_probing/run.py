__author__ = 'Qiao Jin'

import argparse
import numpy as np
import random
import train

parser = argparse.ArgumentParser(description='Run probing experiments on mednli')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size (Default: 32)')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed (Default: 0)')
parser.add_argument('--lr', dest='lr', type=float, default=0.002, help='Adam learning rate (Default: 0.002)')
parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=10, help='number of epochs to train (Default: 10)')
parser.add_argument('--embed_type', dest='embed_type', type=str, help='Cache the specified embedding type of the dataset. Possible types: "biomed_elmo", "biomed_w2v", "general_elmo"')

args = parser.parse_args()
params = vars(args)

np.random.seed(params['seed'])
random.seed(params['seed'])

train.main(params)

__author__ = 'Qiao Jin'

import math
import numpy as np
import os
import torch
from torch import nn

from utils import *

class Model(nn.Module):
    def __init__(self, R=16, h1=512, h2=1024):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(3, 1))
        nn.init.xavier_normal_(self.weight)
        '''
        self.linears1 = nn.Sequential(
        nn.Linear(1024, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU())
        
        self.linears2 = nn.Sequential(
        nn.Linear(1024, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU())
        '''

        self.R = R
        self.rel = nn.Parameter(torch.Tensor(self.R, 1024, 1024))
        nn.init.xavier_normal_(self.rel)

        self.linear = nn.Linear(self.R, 3, bias=False)

    def forward(self, word_1, mask_1, word_2, mask_2):
        B = word_1.size(0)
        L_1 = word_1.size(2)
        L_2 = word_2.size(2)

        word_1 = word_1.view(B, 3, L_1*1024)
        word_1 = word_1.permute(0, 2, 1) # B x L*1024 x 3
        word_2 = word_2.view(B, 3, L_2*1024)
        word_2 = word_2.permute(0, 2, 1) # B x L*1024 x 3

        weights = nn.functional.softmax(self.weight, dim=0) # 3 x 1

        word_1 = torch.matmul(word_1, weights) # B x L*1024 x 1
        word_1 = word_1.squeeze(dim=2).view(B, L_1, 1024) # B x L x 1024
        word_2 = torch.matmul(word_2, weights) # B x L*1024 x 1
        word_2 = word_2.squeeze(dim=2).view(B, L_2, 1024) # B x L x 1024

        word_1 = word_1.unsqueeze(dim=0).expand(self.R, B, L_1, 1024).contiguous().view(self.R*B, L_1, 1024) # BR x L1 x 1024
        word_2 = word_2.unsqueeze(dim=0).expand(self.R, B, L_2, 1024).contiguous().view(self.R*B, L_2, 1024) # BR x L2 x 1024
        rel = self.rel.unsqueeze(dim=1).expand(self.R, B, 1024, 1024).contiguous().view(self.R*B, 1024, 1024)# BR x 1024 x 1024

        mask = torch.bmm(mask_1.unsqueeze(dim=2), mask_2.unsqueeze(dim=1)) # B x L1 x L2
        mask = mask.unsqueeze(dim=0).expand(self.R, B, L_1, L_2).contiguous().view(self.R*B, L_1, L_2) # BR x L1 x L2

        bilinear = torch.bmm(torch.bmm(word_1, rel), word_2.permute(0, 2, 1)) # BR x L1 x L2
        bilinear[mask == 0] = float('-inf')
        bilinear = bilinear.view(self.R, B, L_1, L_2)
        bilinear = bilinear.view(self.R, B, L_1*L_2) # R x B x L1L2
        output = torch.max(bilinear, dim=2)[0].permute(1, 0) # B x R
        logits = self.linear(output)
        
        return logits

    def get_relations(self, word_1, mask_1, word_2, mask_2):
        B = word_1.size(0)
        L_1 = word_1.size(2)
        L_2 = word_2.size(2)

        word_1 = word_1.view(B, 3, L_1*1024)
        word_1 = word_1.permute(0, 2, 1) # B x L*1024 x 3
        word_2 = word_2.view(B, 3, L_2*1024)
        word_2 = word_2.permute(0, 2, 1) # B x L*1024 x 3

        weights = nn.functional.softmax(self.weight, dim=0) # 3 x 1

        word_1 = torch.matmul(word_1, weights) # B x L*1024 x 1
        word_1 = word_1.squeeze(dim=2).view(B, L_1, 1024) # B x L x 1024
        word_2 = torch.matmul(word_2, weights) # B x L*1024 x 1
        word_2 = word_2.squeeze(dim=2).view(B, L_2, 1024) # B x L x 1024

        word_1 = word_1.unsqueeze(dim=0).expand(self.R, B, L_1, 1024).contiguous().view(self.R*B, L_1, 1024) # BR x L1 x 1024
        word_2 = word_2.unsqueeze(dim=0).expand(self.R, B, L_2, 1024).contiguous().view(self.R*B, L_2, 1024) # BR x L2 x 1024
        rel = self.rel.unsqueeze(dim=1).expand(self.R, B, 1024, 1024).contiguous().view(self.R*B, 1024, 1024)# BR x 1024 x 1024

        mask = torch.bmm(mask_1.unsqueeze(dim=2), mask_2.unsqueeze(dim=1)) # B x L1 x L2
        mask = mask.unsqueeze(dim=0).expand(self.R, B, L_1, L_2).contiguous().view(self.R*B, L_1, L_2) # BR x L1 x L2

        bilinear = torch.bmm(torch.bmm(word_1, rel), word_2.permute(0, 2, 1)) # BR x L1 x L2
        bilinear[mask == 0] = float('-inf')
        bilinear = bilinear.view(self.R, B, L_1, L_2)

        return bilinear.permute(1, 0, 2, 3) # B x R x L1 x L2
        '''
        bilinear = bilinear.view(self.R, B, L_1*L_2) # R x B x L1L2

        pair_max = torch.argmax(bilinear, dim=2).permute(1, 0) # B x R

        output = torch.max(bilinear, dim=2)[0].permute(1, 0) # B x R
        output = output.unsqueeze(dim=1) # B x 1 x R
        weight = self.linear.weight.unsqueeze(dim=0) # 1 x 3 x R
        
        output = output * weight # B x 3 x R
        rel_max = torch.argmax(output, dim=2) # B x 3

        return pair_max, rel_max
        '''


class w2v_Model(nn.Module):
    def __init__(self, R=16):
        super(w2v_Model, self).__init__()

        self.R = R
        self.rel = nn.Parameter(torch.Tensor(self.R, 200, 200))
        nn.init.xavier_normal_(self.rel)

        self.linear = nn.Linear(self.R, 3, bias=False)

    def forward(self, word_1, mask_1, word_2, mask_2):
        B = word_1.size(0)
        L_1 = word_1.size(1)
        L_2 = word_2.size(1)

        word_1 = word_1.unsqueeze(dim=0).expand(self.R, B, L_1, 200).contiguous().view(self.R*B, L_1, 200) # BR x L1 x 200
        word_2 = word_2.unsqueeze(dim=0).expand(self.R, B, L_2, 200).contiguous().view(self.R*B, L_2, 200) # BR x L2 x 200
        rel = self.rel.unsqueeze(dim=1).expand(self.R, B, 200, 200).contiguous().view(self.R*B, 200, 200)# BR x 200 x 200

        mask = torch.bmm(mask_1.unsqueeze(dim=2), mask_2.unsqueeze(dim=1)) # B x L1 x L2
        mask = mask.unsqueeze(dim=0).expand(self.R, B, L_1, L_2).contiguous().view(self.R*B, L_1, L_2) # BR x L1 x L2

        bilinear = torch.bmm(torch.bmm(word_1, rel), word_2.permute(0, 2, 1)) # BR x L1 x L2
        bilinear[mask == 0] = float('-inf')
        bilinear = bilinear.view(self.R, B, L_1, L_2)
        bilinear = bilinear.view(self.R, B, L_1*L_2) # B x R x L1L2
        output = torch.max(bilinear, dim=2)[0].permute(1, 0) # B x R
        logits = self.linear(output)
        
        return logits

    def get_relations(self, word_1, mask_1, word_2, mask_2):
        B = word_1.size(0)
        L_1 = word_1.size(1)
        L_2 = word_2.size(1)

        word_1 = word_1.unsqueeze(dim=0).expand(self.R, B, L_1, 200).contiguous().view(self.R*B, L_1, 200) # BR x L1 x 200
        word_2 = word_2.unsqueeze(dim=0).expand(self.R, B, L_2, 200).contiguous().view(self.R*B, L_2, 200) # BR x L2 x 200
        rel = self.rel.unsqueeze(dim=1).expand(self.R, B, 200, 200).contiguous().view(self.R*B, 200, 200)# BR x 200 x 200

        mask = torch.bmm(mask_1.unsqueeze(dim=2), mask_2.unsqueeze(dim=1)) # B x L1 x L2
        mask = mask.unsqueeze(dim=0).expand(self.R, B, L_1, L_2).contiguous().view(self.R*B, L_1, L_2) # BR x L1 x L2

        bilinear = torch.bmm(torch.bmm(word_1, rel), word_2.permute(0, 2, 1)) # BR x L1 x L2
        bilinear[mask == 0] = float('-inf')
        bilinear = bilinear.view(self.R, B, L_1, L_2)

        return bilinear.permute(1, 0, 2, 3)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np
from models.compact_bilinear_pooling import CountSketch, CompactBilinearPooling


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class TOAD_fc_mtl_concat(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(TOAD_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]


        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 2)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_tasks = 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)


        
        fc_ = [nn.Linear(6, 6), nn.ReLU()]
        self.other_feature = nn.Sequential(*fc_)
        '''
        # MCB
        self.mcb = CompactBilinearPooling(size[1], 6, size[1])
        self.relu = nn.ReLU()'''
        # concat
        self.mcb = nn.Linear(size[1] + 6, size[1])
        self.relu = nn.ReLU()


        self.classifier = nn.Linear(size[1], n_classes)
        self.site_classifier = nn.Linear(size[1], 2)

        initialize_weights(self)
                
    def relocate(self):
        device=self.device
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)
            self.mcb = self.mcb.to(device)
            self.relu = self.relu.to(device)

        self.other_feature = self.other_feature.to(device)

        self.classifier = self.classifier.to(device)
        self.site_classifier = self.site_classifier.to(device)
        
    def forward(self, h, gender,lymph,stage_T,stage_N,stage_M,age,return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)

        # 结合临床
        omis_feature = torch.cat((gender, lymph, stage_T, stage_N, stage_M, age))
        feature = self.other_feature(omis_feature)
        # M = self.mcb(M, feature.repeat(M.size(0), 1))  #mcb
        M = torch.cat([M, feature.repeat(M.size(0), 1)], dim=1) #concat
        M = self.mcb(M)#concat
        M = self.relu(M)


        logits  = self.classifier(M[0].unsqueeze(0)) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        print(Y_prob)
        if Y_prob[:, 1] > 0.3:
            Y_hat = torch.tensor([[1]], device=self.device)
        else:
            Y_hat = torch.tensor([[0]], device=self.device)

        site_logits  = self.site_classifier(M[1].unsqueeze(0)) 
        site_hat = torch.topk(site_logits, 1, dim = 1)[1]
        site_prob = F.softmax(site_logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,
                'site_logits': site_logits, 'site_prob': site_prob, 'site_hat': site_hat, 'A': A_raw})

        return results_dict


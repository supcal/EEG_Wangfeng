import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear, Chebynet
from utils.utils import normalize_A
import copy

class DGCNN(nn.Module):
    def __init__(self, args, device='cpu'):
        super(DGCNN, self).__init__()
        self.layer1 = Chebynet(args.feature_len*5, args.kadj, args.graph_out)
        self.BN1 = nn.BatchNorm1d(args.feature_len*5)
        self.fc = Linear(args.channels_num*args.graph_out, args.nclass)
        self.A = nn.Parameter(torch.FloatTensor(
            args.channels_num, args.channels_num).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)

    def forward(self, x):
        tsne1 = copy.deepcopy(x.detach())
        tsne1 = tsne1.reshape(tsne1.shape[0], -1)
        x=x.reshape(x.shape[0],x.shape[1],-1)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        tsne2 = copy.deepcopy(result.detach())
        return result,tsne1,tsne2
    
    def loss(self, model, pred, label):
        focal = nn.CrossEntropyLoss()(pred[0], label)
        return focal

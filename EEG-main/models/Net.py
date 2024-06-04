import torch
import torch.nn as nn
from models.DGCNN import DGCNN


import math
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args

        self.graph_layer = DGCNN(
            args.feature_len*5, args.channels_num, 3, args.graph_out)

        fc_in_dim = args.graph_out*args.channels_num
        self.classify_layer1 = nn.Sequential(
            nn.Linear(fc_in_dim, 4096),
            # nn.BatchNorm1d(4096),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(inplace=True)
        )
        self.classify_layer2 = nn.Linear(4096, args.nclass)

    def forward(self, x):
        result = x.reshape(x.shape[0], x.shape[1], -1)
        result = self.graph_layer(result)

        result = self.classify_layer1(result)
        result = self.classify_layer2(result)

        return result

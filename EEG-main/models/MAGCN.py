import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear, Chebynet, SELayer
import numpy as np


class MAGCN(nn.Module):
    def __init__(self, args):
        super(MAGCN, self).__init__()

        self.args = args

        self.hierarchical_attention = HierarchicalAttention(args)

        gfk_adj = torch.load(
            "/home/wf/EEG_GTN/data/dataset/SEED/adjacency_matrix/adjacency_matrix_based_on_spatial_distance.pt").unsqueeze(0).to(args.device)
        self.gfk_adj = torch.cat(
            [gfk_adj for i in range(args.batch_size)])

        self.graph_layer_pcc = DGCNN(
            args.feature_len*args.bands, args.channels_num, args.kadj, args.graph_out)
        self.graph_layer_mi = DGCNN(
            args.feature_len*args.bands, args.channels_num, args.kadj, args.graph_out)
        self.graph_layer_gfk = DGCNN(
            args.feature_len*args.bands, args.channels_num, args.kadj, args.graph_out)

        fc_in_dim = args.graph_out * args.channels_num * 3
        self.classify_layer1 = nn.Sequential(
            nn.Linear(fc_in_dim, 4096),
            # nn.BatchNorm1d(4096),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(inplace=True)
        )
        self.classify_layer2 = nn.Linear(4096, args.nclass)

    def forward(self, x):
        x = self.hierarchical_attention(x)

        pcc_adj = batch_pcc(x)
        mi_adj = batch_cosine_similarity(x)
        gfk_adj = self.gfk_adj

        pcc_feature = self.graph_layer_pcc(x, pcc_adj)
        mi_feature = self.graph_layer_mi(x, mi_adj)
        gfk_feature = self.graph_layer_gfk(x, gfk_adj)

        x = torch.cat((pcc_feature, mi_feature, gfk_feature), 1)
        x = self.classify_layer1(x)
        x = self.classify_layer2(x)

        return x


class DGCNN(nn.Module):
    def __init__(self, feature_len, channels_num, k_adj, dims_out):
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(feature_len, k_adj, dims_out)
        self.BN1 = nn.BatchNorm1d(feature_len)
        self.W = nn.Parameter(torch.FloatTensor(
            channels_num, channels_num).cuda())
        nn.init.uniform_(self.W, 0.01, 0.5)

    def forward(self, x, A):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        A_weighted = torch.matmul(A, self.W)
        L = normalize_A(A_weighted)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        return result


class HierarchicalAttention(nn.Module):
    def __init__(self, args):
        '''
        使用残差的图读出层
        '''
        super(HierarchicalAttention, self).__init__()

        self.args = args
        # spatil & spectral attenion
        self.band_att = SELayer(args.bands, 2)
        self.channel_att = SELayer(
            args.channels_num, args.channels_num//args.se_squeeze_ratio)

        # conv merge band
        self.band_conv = nn.Conv2d(5, 1, 1, 1)

        # # BN
        # self.BN = nn.BatchNorm1d(
        #     (config['gh']*2)*args.channels_num//args.rsr+args.channels_num*in_dim)

    def forward(self, x):
        '''
        x:[batch,channel,feature,band]
        '''
        x = self.band_att(x.transpose(1, 3)).transpose(1, 3)
        x = self.channel_att(x)

        x = x.flatten(start_dim=2)

        # x = self.band_conv(x.transpose(1, 3)).squeeze(1).transpose(1, 2)

        return x

    def PCA_svd(self, X, k, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n, 1])
        h = ((1/n) * torch.mm(ones, ones.t())
             ) if center else torch.zeros(n*n).view([n, n])
        H = torch.eye(n) - h
        H = H.cuda()
        X_center = torch.mm(H.double(), X.double())
        u, s, v = torch.svd(X_center)
        components = v[:k].t()
        # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
        return components


def normalize_A(A, lmax=2):
    # Apply ReLU to remove any negative weights in the adjacency matrix
    A = F.relu(A)

    # Get the number of nodes (assuming the adjacency matrices are square)
    batch_size, num_nodes, _ = A.shape

    # Remove self-loops by subtracting identity matrices from A
    A = A * (torch.ones(batch_size, num_nodes, num_nodes, device=A.device) -
             torch.eye(num_nodes, device=A.device).unsqueeze(0))

    # Make the matrix symmetric by adding its transpose
    A = A + A.transpose(1, 2)

    # Compute the degree matrix D
    d = torch.sum(A, dim=2)
    d = 1 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)

    # Compute the normalized Laplacian L
    I = torch.eye(num_nodes, device=A.device).unsqueeze(
        0).repeat(batch_size, 1, 1)
    L = I - torch.matmul(torch.matmul(D, A), D)

    # Scale the Laplacian matrix
    Lnorm = (2 * L / lmax) - I

    return Lnorm


def batch_pcc(x):
    # x shape: (batch, channels_num, features_len)
    mean_x = torch.mean(x, dim=2, keepdim=True)
    xm = x - mean_x
    c1 = torch.matmul(xm, xm.transpose(1, 2))
    c2 = torch.sqrt(torch.sum(xm ** 2, dim=2, keepdim=True))
    correlation_matrix = c1 / torch.matmul(c2, c2.transpose(1, 2))
    return correlation_matrix


def batch_cosine_similarity(features, eps=1e-8):
    """
    计算批次中特征的余弦相似度矩阵。
    :param features: 特征矩阵，形状为 (batch, num_nodes, features_len)
    :param eps: 用于数值稳定性的小数
    :return: 余弦相似度矩阵，形状为 (batch, num_nodes, num_nodes)
    """
    # 归一化特征以获得单位向量
    norm_features = F.normalize(features, p=2, dim=2, eps=eps)

    # 计算余弦相似度
    cosine_sim = torch.matmul(norm_features, norm_features.transpose(1, 2))

    return cosine_sim

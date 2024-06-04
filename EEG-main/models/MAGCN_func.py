import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear, Chebynet, SELayer
from utils.utils import loss_fuction
import numpy as np


class MAGCN_func(nn.Module):
    def __init__(self, args):
        super(MAGCN_func, self).__init__()
        self.args = args
        self.feature_extractor = FeatureExtractor(args)
        self.label_classifier = LabelClassifier(args)
        self.domain_classifier = DomainClassifier(args)
        self.grl = GRL()

    def forward(self, x):
        tsne1 = x[0].reshape(x[0].shape[0], -1)
        features = self.feature_extractor(x)
        reversed_features = self.grl(features)
        class_outputs, tsne_class = self.label_classifier(features)
        domain_outputs, tsne_domain = self.domain_classifier(reversed_features)
        return [class_outputs, domain_outputs, tsne1, tsne_class, tsne_domain]

    def loss(self, model, pred, label):
        focal = nn.CrossEntropyLoss()(pred[0], label[0])
        w = torch.cat([x.view(-1) for x in model.parameters()])
        l2_loss = model.args.loss2 * torch.sum(torch.abs(w))
        l1_loss = model.args.loss1 * torch.sum(w.pow(2))
        class_loss = focal + l1_loss + l2_loss

        domain_loss = nn.CrossEntropyLoss()(pred[1], label[1])
        total_loss = class_loss + self.args.loss_beta * domain_loss
        return total_loss


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()

        self.args = args

        self.hierarchical_attention = HierarchicalAttention(args)

        gfk_adj = torch.load(
            "/home/wf/EEG_GTN/data/dataset/SEED/adjacency_matrix/adjacency_matrix_based_on_spatial_distance.pt").unsqueeze(0).to(args.device)
        self.gfk_adj = torch.cat(
            [gfk_adj for i in range(args.batch_size)])

        self.DGCNNs = nn.ModuleList()

        if args.bands != 5:
            garph_in_dim = args.feature_len
        else:
            garph_in_dim = args.feature_len*args.bands

        for i in range(args.adj_num):
            self.DGCNNs.append(DGCNN(garph_in_dim,
                               args.channels_num, args.kadj, args.graph_out))

    def forward(self, x):
        (data, pcc_adj, nmi_adj, coh_adj, plv_adj) = x
        data = self.hierarchical_attention(data)

        adj_set = [pcc_adj, nmi_adj, coh_adj, plv_adj, self.gfk_adj]  # 0.9042
        # adj_set = [pcc_adj, nmi_adj, coh_adj, plv_adj]  # 0.9042
        # adj_set = [pcc_adj, nmi_adj, coh_adj]  # 0.9229
        result_set = []

        for i in range(self.args.adj_num):
            result_set.append(self.DGCNNs[i](data, adj_set[i]))

        data = torch.cat(result_set, 1)
        return data


class LabelClassifier(nn.Module):
    def __init__(self, args):
        super(LabelClassifier, self).__init__()
        fc_in_dim = args.graph_out * args.channels_num * args.adj_num
        self.classify_layer1 = nn.Sequential(
            nn.Linear(fc_in_dim, 4096),
            # nn.BatchNorm1d(4096),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(inplace=True)
        )
        self.classify_layer2 = nn.Linear(4096, args.nclass)

    def forward(self, x):
        tsne = self.classify_layer1(x)
        x = self.classify_layer2(tsne)
        return x, tsne


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
        if args.bands == 5:
            self.band_att = SELayer(args.bands, 2)
        self.channel_att = SELayer(
            args.channels_num, args.channels_num//args.se_squeeze_ratio)

    def forward(self, x):
        '''
        x:[batch,channel,feature,band]
        '''
        if self.args.bands == 5:
            x = self.band_att(x.transpose(1, 3)).transpose(1, 3)
        else:
            x = x[:, :, :, self.args.bands].unsqueeze(3)
        x = self.channel_att(x)

        x = x.flatten(start_dim=2)
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


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class GradientReverseFunction(torch.autograd.Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff=1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class DomainClassifier(nn.Module):
    def __init__(self, args):
        """
        初始化领域分类器。
        :param feature_dim: 特征提取器输出的特征维数。
        :param num_domains: 需要区分的领域数量。
        """
        super(DomainClassifier, self).__init__()
        fc_in_dim = args.graph_out * args.channels_num * args.adj_num
        self.layers = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.layers2 = nn.Linear(128, args.domain_class)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入特征。
        :return: 领域的预测。
        """
        tsne = self.layers(x)
        return self.layers2(tsne), tsne

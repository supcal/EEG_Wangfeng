import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from models.layers import TemporalConvNet, SELayer, ReadoutLayer
import copy
from utils.utils import laplacian_torch, custom_init, truncated_normal_, calculate_adj_matrix

# 自适应的邻接矩阵


class ATGRNet(nn.Module):
    def __init__(self, args, device='cpu'):
        '''
        通道注意力机制 频段注意力机制 
        图卷积神经网络 TCN 残差
        '''
        super(ATGRNet, self).__init__()

        self.args = args
        self.device = args.device
        self.windows_num = args.windows_num

        self.SSTG_array = nn.ModuleList()

        self.A = custom_init(torch.FloatTensor(
            args.channels_num, args.channels_num))
        self.A = nn.Parameter(self.A)

        self.laplacian = laplacian_torch(
            self.A.detach()).to_sparse().to(self.device)

        self.BN = nn.BatchNorm2d(args.feature_len)

        for i in range(self.windows_num):
            self.SSTG_array.append(SSTGBlockResidual(
                args, math.ceil(args.feature_len/self.windows_num), device))
            # self.SSTG_array.append(SSTGBlock(args, device))

        tcn_hidden_channels = [args.tcn_hidden]*(args.tcn_layers)
        # tcn_hidden_channels = [128, 64, 32, 16, 8]
        tcn_hidden_channels = tcn_hidden_channels[:args.tcn_layers]
        self.tcn = TemporalConvNet(
            (args.graph_out*2)*args.channels_num//args.rsr+math.ceil(args.feature_len/self.windows_num)*args.channels_num, tcn_hidden_channels)
        fc_in_dim = (self.windows_num)*tcn_hidden_channels[-1]  # *2

        self.classify_layer1 = nn.Sequential(
            nn.Linear(fc_in_dim, 4096),
            # nn.BatchNorm1d(4096),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(inplace=True)
        )
        self.classify_layer2 = nn.Linear(4096, args.nclass)

    def forward(self, x):
        # x:(batch,32,232,5)
        tsne1 = copy.deepcopy(x.detach())
        tsne1 = tsne1.reshape(tsne1.shape[0], -1)
        x = self.BN(x.transpose(1, 2)).transpose(1, 2)
        x = list(torch.chunk(x, self.windows_num, dim=2))
        # x = x[:-1]

        x = [self.SSTG_array[i](x[i], self.laplacian).unsqueeze(1)
             for i in range(self.windows_num)]
        x = torch.cat(x, dim=1)
        x = F.leaky_relu(self.tcn(x.transpose(1, 2)))
        # x = F.leaky_relu(self.lstm(x)[0])
        x = x.reshape(x.shape[0], -1)
        x = self.classify_layer1(x)

        x = self.classify_layer2(x)
        tsne2 = copy.deepcopy(x.detach())
        # x = F.leaky_relu(self.classify_layer(x))
        # x = F.softmax(x, dim=1)
        self.A = self.update_A(self.A, int(
            self.args.channels_num*self.args.channels_num/self.args.k_ratio))
        self.laplacian = laplacian_torch(
            self.A.detach()).to_sparse().to(self.device)

        return x, tsne1, tsne2

    def keep_k_largest_edges(self, A, k):
        # A is the adjacency matrix
        # k is the number of edges to keep
        # return the adjacency matrix with only the k largest edges
        A = A.abs()
        A = A.triu()
        A = A.flatten()
        A = A.sort(descending=True).values
        threshold = A[k].to(self.device)
        A = torch.where(A.abs().to(self.device) >= threshold, A.abs().to(
            self.device), torch.tensor([1e-10]).to(self.device))
        A = A.reshape((self.args.channels_num, self.args.channels_num))
        return A

    def update_A(self, A, k):
        # A is the adjacency matrix
        # k is the number of edges to keep
        # return the updated adjacency matrix with only the k largest edges
        A = self.keep_k_largest_edges(A, k)
        A = nn.Parameter(A)
        return A

    def loss(self, model, pred, label):
        focal = nn.CrossEntropyLoss()(pred[0], label)
        w = torch.cat([x.view(-1) for x in model.parameters()])
        l2_loss = model.args.loss2 * torch.sum(torch.abs(w))
        l1_loss = model.args.loss1 * torch.sum(w.pow(2))
        class_loss = focal + self.args.loss1*l1_loss + self.args.loss2*l2_loss
        return class_loss


class SSTGBlockResidual(nn.Module):
    def __init__(self, args, in_channels, device='cpu'):
        '''
        使用残差的图读出层
        '''
        super(SSTGBlockResidual, self).__init__()

        self.args = args
        self.device = device
        # spatil & spectral attenion
        self.band_att = SELayer(args.bands, 2)
        self.channel_att = SELayer(
            args.channels_num, args.channels_num//args.se_squeeze_ratio)

        # conv merge band
        self.band_conv = nn.Conv2d(5, 1, 1, 1)

        # graph convolution
        in_dim = math.ceil(args.feature_len/args.windows_num)
        self.graph_layer_g = PolyGCLayer(
            in_channels, args.graph_out, 25, args.pooling_size, args.feature_len, device)
        self.graph_layer_s = PolyGCLayer(
            in_channels, args.graph_out, 25, args.pooling_size, args.feature_len, device)

        # readout
        # self.readout = ReadoutLayer(
        #     args, args.graph_out*2+in_dim, (args.graph_out*2+in_dim)*args.channels_num//args.rsr)
        self.readout_g = ReadoutLayer(
            args, args.graph_out, (args.graph_out)*args.channels_num//args.rsr)
        self.readout_s = ReadoutLayer(
            args, args.graph_out, (args.graph_out)*args.channels_num//args.rsr)

        # BN
        self.BN = nn.BatchNorm1d(
            (args.graph_out*2)*args.channels_num//args.rsr+args.channels_num*in_dim)

    def forward(self, x, laplacian):
        '''
        x:[batch,channel,feature,band]
        '''

        x = self.band_att(x.transpose(1, 3)).transpose(1, 3)
        x = self.channel_att(x)

        x = self.band_conv(x.transpose(1, 3)).squeeze(1).transpose(1, 2)

        residual = x
        x_g = self.readout_g(self.graph_layer_g(x, laplacian))
        x_s = self.readout_s(self.graph_layer_s(x))

        x = torch.cat(
            [residual.reshape(residual.shape[0], -1), x_g, x_s], dim=1)
        del x_g, x_s
        if self.args.batch_size !=1:
            x = self.BN(x)
        return x


class PolyGCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, poly_degree, pooling_size, feature_len, device):
        """
        Args:
            in_channels (int): 输入特征的通道数
            out_channels (int): 输出特征的通道数
            poly_degree (int): Chebyshev多项式的阶数
            pooling_size (int): 最大池化的大小，应为2的幂次方
        """
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(
            poly_degree*in_channels, out_channels)).to(device)
        self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels)).to(device)
        self.pooling_size = pooling_size
        self.poly_degree = poly_degree

        self.W_t = nn.Parameter(torch.Tensor(
            in_channels, in_channels)).to(device)
        nn.init.kaiming_normal_(self.W_t)

        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        """
        初始化权重参数和偏置参数
        """
        truncated_normal_(self.weight, mean=0.0, std=0.1)
        truncated_normal_(self.bias, mean=0.0, std=0.1)

    def chebyshev(self, x, laplacian):
        """
        Chebyshev多项式的计算，包括多项式的递归计算以及乘上权重矩阵。
        Args:
            x (tensor): 输入特征张量，shape为(batch_size, node_num, in_features)
        Returns:
            tensor: 输出特征张量，shape为(batch_size, node_num, out_features)
        """
        batch_size, node_num, in_features = x.size()
        out_features = self.weight.size(1)
        x0 = x.permute(1, 2, 0)  # node_num x in_features x batch_size
        x0 = torch.reshape(x0, [node_num, in_features * batch_size])
        x_list = [x0]
        if self.poly_degree > 1:
            x1 = torch.sparse.mm(laplacian, x0)
            x_list.append(x1)
        for k in range(2, self.poly_degree):
            # node_num x in_features*batch_size
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            x_list.append(x2)
            x0, x1 = x1, x2
        # poly_degree x node_num x in_features*batch_size
        x = torch.stack(x_list, dim=0)
        x = torch.reshape(
            x, [self.poly_degree, node_num, in_features, batch_size])
        # batch_size x node_num x in_features x poly_degree
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(
            x, [batch_size*node_num, in_features*self.poly_degree])
        x = torch.matmul(x, self.weight)  # batch_size*node_num x out_features
        # batch_size x node_num x out_features
        x = torch.reshape(x, [batch_size, node_num, out_features])
        return x

    def brelu(self, x):
        """Bias and ReLU. One bias per filter."""
        return F.relu(x + self.bias)

    def pool(self, x):
        """Max pooling of size p. Should be a power of 2."""
        if self.pooling_size > 1:
            x = x.permute(0, 2, 1)  # batch_size x out_features x node_num
            x = F.max_pool1d(x, kernel_size=self.pooling_size,
                             stride=self.pooling_size)
            x = x.permute(0, 2, 1)  # batch_size x node_num x out_features
        return x

    def forward(self, x, laplacian=None):
        if laplacian is None:
            laplacian = calculate_adj_matrix(self, x)
        x = self.chebyshev(x, laplacian)
        x = self.brelu(x)
        x = self.pool(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, in_dim, out_dim, channels_num, k_adj=3, dropout=0.0, device='cpu'):
        '''
        in_dim:输入的特征长度
        out_dim:期望的图编码后的维度
        channels_num:图的节点数
        k_adj:聚合邻居节点的深度
        '''
        super(DGCNN, self).__init__()
        self.device = device
        self.K = k_adj
        self.layer1 = Chebynet(in_dim, out_dim, k_adj, dropout, device)
        self.BN1 = nn.BatchNorm1d(in_dim)
        self.A = nn.Parameter(torch.FloatTensor(channels_num, channels_num))
        nn.init.xavier_normal_(self.A)
        # self.A=t.nn.Parameter(getAdj(self.A))

    def forward(self, x):  # , adj
        '''
        x: [batch_size,channels_num,features_num]
        '''
        # self.A.data = adj
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = self.normalize_A(self.A)
        result = self.layer1(x, L)
        return result

    def normalize_A(self, A, symmetry=False):
        A = F.relu(A)
        if symmetry:
            A = A + torch.transpose(A, 0, 1)
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        return L


class Chebynet(nn.Module):
    def __init__(self, in_dim, out_dim, hid_layers, dropout, device):
        super(Chebynet, self).__init__()
        self.device = device
        self.hid_layers = hid_layers
        self.gc1 = nn.ModuleList()
        for i in range(hid_layers):
            self.gc1.append(GraphConvolution(in_dim, out_dim))

    def forward(self, x, L):
        adj = self.generate_cheby_adj(L, self.hid_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

    def generate_cheby_adj(self, A, hid_layers):
        support = []
        for i in range(hid_layers):
            if i == 0:
                support.append(torch.eye(A.shape[1]).to(self.device))
            elif i == 1:
                support.append(A)
            else:
                temp = torch.matmul(support[-1], A)
                support.append(temp)
        return support


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from utils.utils import generate_cheby_adj


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(
            in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x:[batch,band_size,H,W]
        '''
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).type_as(x)


class ReadoutLayer(nn.Module):
    """Graph Readout Layer."""

    def __init__(self, args,
                 input_dim,
                 output_dim,
                 act=nn.ReLU(),
                 dropout_p=0.):
        super(ReadoutLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.att_weight = self.glorot([self.input_dim, 1])
        self.emb_weight = self.glorot([self.input_dim, self.input_dim])
        self.mlp_weight = self.glorot([self.input_dim, self.output_dim])
        self.att_bias = nn.Parameter(torch.zeros(1))
        self.emb_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.mlp_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, x):  # _ not used
        # soft attention
        att = torch.sigmoid(torch.matmul(x, self.att_weight)+self.att_bias)
        emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        # graph summation
        g = att * emb
        g = torch.cat([torch.sum(g, dim=1) + torch.max(g, dim=1)[0]], dim=1)
        # g = self.dropout(g)
        # classification
        output = torch.matmul(g, self.mlp_weight)+self.mlp_bias
        return output

    def glorot(self, shape):
        init_range = np.sqrt(6.0/(shape[0]+shape[1]))
        initial = nn.Parameter(torch.nn.init.uniform_(
            tensor=torch.empty(shape), a=-init_range, b=init_range))
        return initial


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            # 确定每一层的输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution(in_channels,  out_channels))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result

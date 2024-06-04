import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class TemporalAttention(nn.Module):
    """
    计算时序注意力分数
    --------
    输入：(batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    输出：(batch_size, num_of_timesteps, num_of_timesteps)
    """
    def __init__(self, num_of_timesteps=265, num_of_vertices=62, num_of_features=5):
        super(TemporalAttention, self).__init__()
        self.U_1 = nn.Parameter(torch.randn(num_of_vertices, 1))
        self.U_2 = nn.Parameter(torch.randn(num_of_features, num_of_vertices))
        self.U_3 = nn.Parameter(torch.randn(num_of_features))
        self.b_e = nn.Parameter(torch.randn(1, num_of_timesteps, num_of_timesteps))
        self.V_e = nn.Parameter(torch.randn(num_of_timesteps, num_of_timesteps))

    def forward(self, x):
        # x shape: (batch_size, T, V, F)
        batch_size, T, V, F = x.size()

        # 计算左侧
        lhs = torch.matmul(x.transpose(1, 3), self.U_1).view(batch_size, T, F)
        lhs = torch.matmul(lhs, self.U_2)

        # 计算右侧
        rhs = torch.matmul(self.U_3, x.permute(2, 0, 3, 1))
        rhs = rhs.permute(1, 0, 2)

        # 计算乘积
        product = torch.bmm(lhs, rhs)

        # 应用 V_e 和 b_e
        S = torch.sigmoid(torch.matmul(self.V_e, product.permute(1, 2, 0)) + self.b_e).permute(2, 0, 1)

        # 标准化
        S = S - torch.max(S, dim=1, keepdim=True)[0]
        exp_S = torch.exp(S)
        S_normalized = exp_S / torch.sum(exp_S, dim=1, keepdim=True)

        return S_normalized



class SpatialAttention(nn.Module):
    """
    计算空间注意力分数
    --------
    输入：(batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    输出：(batch_size, num_of_vertices, num_of_vertices)
    """
    def __init__(self, num_of_timesteps=265, num_of_vertices=62, num_of_features=5):
        super(SpatialAttention, self).__init__()
        self.W_1 = nn.Parameter(torch.randn(num_of_timesteps, 1))
        self.W_2 = nn.Parameter(torch.randn(num_of_features, num_of_timesteps))
        self.W_3 = nn.Parameter(torch.randn(num_of_features))
        self.b_s = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices))
        self.V_s = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices))

    def forward(self, x):
        # x shape: (batch_size, T, V, F)
        batch_size, T, V, F = x.size()

        # 计算左侧
        lhs = torch.matmul(x.transpose(1, 3), self.W_1).view(batch_size, V, F)
        lhs = torch.matmul(lhs, self.W_2)

        # 计算右侧
        rhs = torch.matmul(self.W_3, x.permute(1, 0, 3, 2))
        rhs = rhs.permute(1, 0, 2)

        # 计算乘积
        product = torch.bmm(lhs, rhs)

        # 应用 V_s 和 b_s
        S = torch.sigmoid(torch.matmul(self.V_s, product.permute(1, 2, 0)) + self.b_s).permute(2, 0, 1)

        # 标准化
        S = S - torch.max(S, dim=1, keepdim=True)[0]
        exp_S = torch.exp(S)
        S_normalized = exp_S / torch.sum(exp_S, dim=1, keepdim=True)

        return S_normalized
    
class Graph_Learn(nn.Module):
    """
    基于中间时间片的图结构学习
    --------
    输入：(batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    输出：(batch_size, num_of_vertices, num_of_vertices)
    """
    def __init__(self, alpha):
        super(Graph_Learn, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.randn(1, 1))  # 类似于 Keras 中的可训练权重

    def forward(self, x):
        # x shape: (batch_size, T, V, F)
        batch_size, T, V, F = x.size()
        x_transposed = x.transpose(1, 2)  # 改变维度以便于运算

        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            xt = x[:, time_step, :, :]  # 当前时间步的切片
            diff = xt.unsqueeze(2) - xt.unsqueeze(1)  # 计算差异

            S = torch.exp(torch.matmul(diff.abs().view(batch_size * V, V * F), self.a).view(batch_size, V, V))
            S_normalized = F.normalize(S, p=1, dim=2)  # 归一化

            diff_tmp += diff.abs()
            outputs.append(S_normalized)

        outputs = torch.stack(outputs, dim=1)
        self.S = outputs.mean(dim=0)  # 计算所有时间步的平均值
        self.diff = diff_tmp.mean(dim=0) / T  # 计算所有时间步的平均差异

        return outputs

    def graph_learning_loss(self):
        # 计算图学习损失
        diff_loss = (self.diff ** 2 * self.S).mean()
        F_norm_loss = self.alpha * (self.S ** 2).mean()

        return diff_loss + F_norm_loss
    

class ChebConvWithAttGL(nn.Module):
    """
    带注意力机制的 K 阶切比雪夫图卷积，结合图学习（Graph Learn）
    --------
    输入：[x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
           Att (batch_size, num_of_vertices, num_of_vertices),
           S   (batch_size, num_of_vertices, num_of_vertices)]
    输出：(batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    """
    def __init__(self, num_of_filters, k):
        super(ChebConvWithAttGL, self).__init__()
        self.k = k
        self.num_of_filters = num_of_filters
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(F, num_of_filters)) for _ in range(k)])

    def forward(self, x, Att, S):
        _, T, V, F = x.size()
        S = torch.min(S, S.transpose(1, 2))  # 确保对称性

        outputs = []
        for time_step in range(T):
            graph_signal = x[:, time_step, :, :]
            output = torch.zeros((x.size(0), V, self.num_of_filters), device=x.device)

            A = S[:, time_step, :, :]
            # 计算切比雪夫多项式 (设 lambda_max=2)
            D = torch.diag_embed(torch.sum(A, dim=1))
            L = D - A
            L_t = L - torch.eye(V, device=x.device)
            cheb_polynomials = [torch.eye(V, device=x.device), L_t]
            for i in range(2, self.k):
                cheb_polynomials.append(2 * L_t @ cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

            for kk in range(self.k):
                T_k = cheb_polynomials[kk]
                T_k_with_at = T_k * Att[:, None, :, :]  # 扩展维度以适应批量大小
                theta_k = self.Theta[kk]

                rhs = torch.bmm(T_k_with_at.transpose(1, 2), graph_signal)
                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1).permute(0, 3, 1, 2))


class ChebConvWithAttStatic(nn.Module):
    """
    带静态图结构的 K 阶切比雪夫图卷积
    --------
    输入：[x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
           Att (batch_size, num_of_vertices, num_of_vertices)]
    输出：(batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    """
    def __init__(self, num_of_filters, k, cheb_polynomials):
        super(ChebConvWithAttStatic, self).__init__()
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = [torch.tensor(p, dtype=torch.float32) for p in cheb_polynomials]
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(F, num_of_filters)) for _ in range(k)])

    def forward(self, x, Att):
        _, T, V, F = x.size()

        outputs = []
        for time_step in range(T):
            graph_signal = x[:, time_step, :, :]
            output = torch.zeros((x.size(0), V, self.num_of_filters), device=x.device)

            for kk in range(self.k):
                T_k = self.cheb_polynomials[kk].to(x.device)
                T_k_with_at = T_k * Att[:, None, :, :]  # 扩展维度以适应批量大小
                theta_k = self.Theta[kk]

                rhs = torch.bmm(T_k_with_at.transpose(1, 2), graph_signal)
                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1).permute(0, 3, 1, 2))


def reshape_dot(x, TAtt):
    """
    重新整形和点乘操作
    --------
    输入：[x (Tensor), TAtt (Tensor)]
    输出：经过点乘操作后的 Tensor
    """
    batch_size, T, V, F = x.size()
    x_transposed = x.permute(0, 2, 3, 1).contiguous()  # 调整 x 的维度
    x_reshaped = x_transposed.view(batch_size, -1, T)  # 重新整形 x

    result = torch.bmm(x_reshaped, TAtt)
    return result.view(-1, T, V, F)  # 将结果重塑回原始维度

def layer_norm(x):
    """
    层标准化
    --------
    输入：x (Tensor)
    输出：标准化后的 Tensor
    """
    relu_x = F.relu(x)
    return nn.LayerNorm(relu_x.size()[1:], elementwise_affine=True)(relu_x)


class ReverseGradient(Function):
    """
    反转梯度的自定义函数
    """
    @staticmethod
    def forward(ctx, x, hp_lambda):
        # 存储 hp_lambda 以供 backward 使用
        ctx.hp_lambda = hp_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反转梯度
        hp_lambda = ctx.hp_lambda
        return grad_output.neg() * hp_lambda, None

class GradientReversalLayer(nn.Module):
    """
    反转梯度的层，在训练期间改变梯度的符号
    """
    def __init__(self, hp_lambda):
        super(GradientReversalLayer, self).__init__()
        self.hp_lambda = hp_lambda

    def forward(self, x):
        return ReverseGradient.apply(x, self.hp_lambda)


class MSTGCN_Block(nn.Module):
    """
    封装的空间-时间卷积块
    -------
    输入参数：
    x: 输入数据
    k: k 阶 Chebyshev GCN
    i: 块编号
    """
    def __init__(self, k, num_of_chev_filters, num_of_time_filters, time_conv_strides, 
                 cheb_polynomials, time_conv_kernel, GLalpha, i=0):
        super(MSTGCN_Block, self).__init__()
        self.temporal_attention = TemporalAttention()
        self.spatial_attention = SpatialAttention()
        self.graph_learn = Graph_Learn(GLalpha)
        self.cheb_conv_GL = ChebConvWithAttGL(num_of_chev_filters, k)
        self.cheb_conv_SD = ChebConvWithAttStatic(num_of_chev_filters, k, cheb_polynomials)
        self.time_conv_GL = nn.Conv2d(num_of_chev_filters, num_of_time_filters, 
                                      (time_conv_kernel, 1), stride=(1, time_conv_strides), padding='same')
        self.time_conv_SD = nn.Conv2d(num_of_chev_filters, num_of_time_filters, 
                                      (time_conv_kernel, 1), stride=(1, time_conv_strides), padding='same')
        self.layer_norm_GL = nn.LayerNorm([num_of_time_filters, ...])  # Fill in appropriate dimensions
        self.layer_norm_SD = nn.LayerNorm([num_of_time_filters, ...])  # Fill in appropriate dimensions

    def forward(self, x):
        temporal_Att = self.temporal_attention(x)
        x_TAtt = reshape_dot(x, temporal_Att)  # Ensure reshape_dot is implemented correctly in PyTorch

        spatial_Att = self.spatial_attention(x_TAtt)
        S = self.graph_learn(x)
        S = F.dropout(S, 0.3)

        spatial_gcn_GL = self.cheb_conv_GL([x, spatial_Att, S])
        spatial_gcn_SD = self.cheb_conv_SD([x, spatial_Att])

        time_conv_output_GL = self.time_conv_GL(spatial_gcn_GL)
        time_conv_output_SD = self.time_conv_SD(spatial_gcn_SD)

        end_output_GL = self.layer_norm_GL(time_conv_output_GL)
        end_output_SD = self.layer_norm_SD(time_conv_output_SD)

        return end_output_GL, end_output_SD


class MSTGCN(nn.Module):
    """
    MSTGCN 模型实现
    """
    def __init__(self, k, num_of_chev_filters, num_of_time_filters, time_conv_strides, 
                 cheb_polynomials, time_conv_kernel, sample_shape, num_block, dense_size, 
                 GLalpha, lambda_reversal, num_classes=5, num_domain=9):
        super(MSTGCN, self).__init__()

        # Building MSTGCN blocks
        self.blocks = nn.ModuleList()
        for i in range(num_block):
            self.blocks.append(MSTGCN_Block(k, num_of_chev_filters, num_of_time_filters,
                                            time_conv_strides, cheb_polynomials, time_conv_kernel, 
                                            GLalpha, i))

        self.flatten = nn.Flatten()
        self.dense_layers = nn.ModuleList([nn.Linear(in_features=dense_size[0], out_features=dense_size[1])])
        self.softmax = nn.Linear(dense_size[-1], num_classes)
        self.flip_layer = GradientReversalLayer(lambda_reversal)
        self.domain_layers = nn.ModuleList([nn.Linear(dense_size[0], dense_size[1]), 
                                            nn.Linear(dense_size[-1], num_domain)])

    def forward(self, x):
        for block in self.blocks:
            x_GL, x_SD = block(x)
            x = torch.cat((x_GL, x_SD), dim=1)
        
        x = self.flatten(x)
        for dense in self.dense_layers:
            x = F.relu(dense(x))

        label_output = self.softmax(x)

        domain_input = self.flip_layer(x)
        for domain_layer in self.domain_layers:
            domain_input = F.relu(domain_layer(domain_input))
        domain_output = F.softmax(domain_input, dim=1)

        return label_output, domain_output
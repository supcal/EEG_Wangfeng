import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from configparser import ConfigParser


def arg_parse():
    r"""初始化参数 超参"""

    parser = argparse.ArgumentParser(description='GTN')

    # basic parser
    parser.add_argument('--epochs', dest='epochs',
                        type=int, help='num of epoch')
    parser.add_argument('--batch_size', dest="batch_size",
                        type=int, help="batch_size")
    parser.add_argument('--nclass', dest='nclass', type=int,
                        help='number of classification')
    parser.add_argument('--gpu', dest='gpu', type=int, help='number of gpu')
    parser.add_argument('--device', dest='device', help='number of gpu')
    parser.add_argument('--dropout', type=float,  help='Dropout.')
    parser.add_argument("--lr", type=float,  help='learning rate.')
    parser.add_argument("--k", type=int,  help='k fold.')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--train_mode',  help='sub nums',
                        choices=['si', 'sd', 'debug'])
    parser.add_argument('--model', help='model selection')
    parser.add_argument('--save_tsne_cm', type=int,
                        help='save tsne confusion matrix data')
    parser.add_argument('--save_model', type=int, help='save model')
    parser.add_argument('--save_path', help='dataset')
    parser.add_argument('--have_domain', help='have_domain')

    # about sample
    parser.add_argument('--bands', type=int, help='num bands',
                        choices=[0, 1, 2, 3, 4, 5])  # 0 means all bands selected
    parser.add_argument('--dataset', dest='dataset', help='dataset')
    parser.add_argument("--feature_type", help='the feature type',
                        choices=['de', 'psd', 'dasm', 'rasm', 'dcau'])
    parser.add_argument('--label_type', help='label type',
                        choices=['label', 'valence', 'arousal', 'dominance'])
    parser.add_argument('--channels_num', type=int, help='num channels')
    parser.add_argument('--feature_len', type=int, help='length of features')
    parser.add_argument('--raw_len', type=int, help='length of raw data')
    parser.add_argument('--session', type=int, help='test session')

    # loss
    parser.add_argument('--loss2', type=float, help='L2 Loss parameter.')
    parser.add_argument('--loss1', type=float, help='L1 Loss parameter.')

    # data split
    parser.add_argument("--split_method",  help='data split method')
    parser.add_argument('--cur_sub_index', type=int,
                        help='current subject index')
    parser.add_argument('--cur_session_index', type=int,
                        help='current session index')
    parser.add_argument('--cur_exp_index', type=int,
                        help='current experiment index')
    parser.add_argument('--clip_length', type=int,
                        help='clip length')
    parser.add_argument('--k_fold_nums', type=int,
                        help='k_fold_nums')

    # model parameter
    parser.add_argument('--graph_out', type=int, help='graph layer output dim')
    parser.add_argument('--attention_out', type=int,
                        help='attention layer output dim')
    parser.add_argument('--spp', type=bool, help='is spp method')
    parser.add_argument('--num_levels', type=bool, help='spp num_levels')
    parser.add_argument('--kadj', type=int, help='DGCNN k adj')
    parser.add_argument('--se_squeeze_ratio', type=int,
                        help='se_squeeze_ratio')
    parser.add_argument('--graph_readout_dim', type=int,
                        help='graph readout dim')
    parser.add_argument('--domain_class', type=int,
                        help='domain_class')
    parser.add_argument('--adj_num', type=int,
                        help='adj_num')
    parser.add_argument('--windows_num', type=int,
                        help='windows_num')
    parser.add_argument('--tcn_hidden', type=int,
                        help='tcn_hidden')
    parser.add_argument('--tcn_layers', type=int,
                        help='tcn_layers')
    parser.add_argument('--pooling_size', type=int,
                        help='pooling_size')
    parser.add_argument('--grl_alpha', type=float,
                        help='grl_alpha')
    parser.add_argument('--rsr', type=float,
                        help='rsr')
    parser.add_argument('--k_ratio', type=float,
                        help='k_ratio')
    parser.add_argument('--loss_beta', type=float,
                        help='loss_beta')

    # config path
    parser.add_argument('--config_path', help='config file path')

    parser.set_defaults(
        # basic
        epochs=200,
        batch_size=1,
        nclass=3,
        gpu=0,
        device='cuda:0',
        dropout=.5,
        lr=1e-3,
        k=10,
        seed=3407,
        train_mode='sd',
        model='ATGRNet',  # MAGCN
        save_tsne_cm=0,
        save_model=0,
        save_path='/home/wf/EEG_GTN/data/parser_save',
        have_domain=False,

        # sample
        bands=5,
        # 'seed', 'deap', 'dreamer', 'seed_origin', 'amigos', 'seed_iv','seed_iv_adj'
        dataset='deap',
        feature_type='de',  # 'de', 'psd', 'dasm', 'rasm', 'dcau'
        label_type='valence',  # 'label','valence', 'arousal', 'dominance'
        channels_num=62,
        feature_len=265,
        raw_len=53001,
        session=0,

        # loss
        loss2=1e-4,
        loss1=5e-6,

        # split
        split_method='by_exp',  # by_exp, by_sess, loso, k_fold
        cur_sub_index=0,
        cur_session_index=0,
        cur_exp_index=0,
        clip_length=1,
        k_fold_nums=10,

        # model parameter
        graph_out=30,
        attention_out=30,
        spp=True,
        num_levels=3,
        kadj=3,
        se_squeeze_ratio=2,
        graph_readout_dim=30,
        domain_class=15,
        adj_num=5,
        windows_num=3,
        tcn_hidden=30,
        grl_alpha=1.0,
        tcn_layers=3,
        pooling_size=1,
        loss_beta=1.0,
        rsr=1,  # readout layer squeeze ratio
        k_ratio=6,
        
        # config
        config_path='/home/wf/EEG_GTN/global.config',
    )

    return parser.parse_args()


def setup_seed(seed=3364):
    r"""设置随机种子"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_device(args):
    r"""设置GPU"""

    if torch.cuda.is_available():
        args.device = 'cuda:{}'.format(0)
    else:
        args.device = 'cpu'


def get_time(t1, t2):
    r"""计算时分秒；

    Args:
        t1: 程序开始时的系统时间.
        t2: 程序结束后的系统时间.

    Returns:
        运行时分秒
    """

    run_time = round(t2-t1)
    # 计算时分秒
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    # 输出
    return f'该程序运行时间：{hour}小时{minute}分钟{second}秒'


def setup_save_path(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    args.save_path = os.path.join(args.save_path, args.dataset, current_time)
    if args.train_mode != 'debug':
        os.makedirs(args.save_path)
        logging.basicConfig(filename=os.path.join(args.save_path, 'log.log'),
                            level=logging.DEBUG, format='%(asctime)s - %(message)s')


def custom_init(tensor, positive=True, distribution='uniform', range=(0, 1)):
    # tensor: a torch.Tensor or a nn.Module
    # positive: a boolean indicating whether to initialize with positive values only
    # distribution: a string indicating the type of distribution to sample from, either 'uniform' or 'normal'
    # range: a tuple indicating the lower and upper bound of the distribution

    # apply the initialization function recursively to every submodule if tensor is a nn.Module
    if isinstance(tensor, nn.Module):
        tensor.apply(lambda t: custom_init(t, positive, distribution, range))
        return tensor

    # check the validity of the arguments
    assert distribution in [
        'uniform', 'normal'], "distribution must be either 'uniform' or 'normal'"
    assert len(
        range) == 2 and range[0] < range[1], "range must be a tuple of two numbers with the first one smaller than the second one"

    # sample from the specified distribution
    if distribution == 'uniform':
        init.uniform_(tensor, range[0], range[1])
    else:
        init.normal_(tensor, range[0], range[1])

    # make sure the values are positive if positive is True
    if positive:
        tensor = torch.abs(tensor)

    return tensor


def laplacian_torch(W, normalized=True, symmetry=True):
    A = W
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


def truncated_normal_(tensor, mean=0, std=0.1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def calculate_adj_matrix(model, X):
    # 计算分子部分
    diff = X.unsqueeze(2) - X.unsqueeze(1)  # (batchsize, N, N, D)
    abs_diff = torch.abs(diff)  # (batchsize, N, N, D)
    relu_diff = torch.relu(torch.matmul(
        abs_diff, model.W_t))  # (batchsize, N, N, D)
    exp_diff = torch.exp(torch.sum(relu_diff, dim=3))  # (batchsize, N, N)
    # 计算分母部分
    softmax_denominator = torch.sum(
        exp_diff, dim=2, keepdim=True)  # (batchsize, N, 1)

    # 计算邻接矩阵
    A = torch.mean(exp_diff / softmax_denominator, 0)  # (N, N)

    return A


def normalize_A(A, lmax=2):
    A = F.relu(A)
    N = A.shape[0]
    A = A*(torch.ones(N, N).cuda()-torch.eye(N, N).cuda())
    A = A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N, N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm = (2*L/lmax)-torch.eye(N, N).cuda()
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L, support[-1],)-support[-2]
            support.append(temp)
    return support


def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
    K = args.K
    # if not args.knn:
    # pdb.set_trace()
    x = F.relu(feat_trans(H))
    # D_ = torch.sigmoid(x@x.t())
    D_ = x@x.t()
    _, D_topk_indices = D_.t().sort(dim=1, descending=True)
    D_topk_indices = D_topk_indices[:, :K]
    D_topk_value = D_.t()[torch.arange(
        D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
    edge_j = D_topk_indices.reshape(-1)
    edge_i = torch.arange(
        D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    return [edge_index, edge_value]


def set_default_config(args):
    config = ConfigParser()
    config.read(args.config_path, encoding='UTF-8')


def loss_fuction(model, y_pred, y_true):
    """
    计算损失函数

    参数:
    - y_pred: 预测结果
    - y_true: 真实结果

    返回: 
    损失值
    """
    if model.args.model == 'svm':
        output = torch.softmax(y_pred, dim=1)  # 将输出转换为概率分布的形式
        # 将目标值从二元分类格式转换为多类别格式
        y_true = 2 * \
            torch.nn.functional.one_hot(
                y_true, num_classes=output.shape[1]).float() - 1
        return torch.mean(torch.clamp(1 - y_pred * y_true, min=0))
    else:
        return loss_with_l1_l2(model, y_pred, y_true)


def loss_with_l1_l2(model, y_pred, y_true):
    focal = nn.CrossEntropyLoss()(y_pred, y_true)
    # w = torch.cat([x.view(-1) for x in model.parameters()])
    # l2_loss = model.args.loss2 * torch.sum(torch.abs(w))
    # l1_loss = model.args.loss1 * torch.sum(w.pow(2))
    total_loss = focal  # + l1_loss + l2_loss
    return total_loss

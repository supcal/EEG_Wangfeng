import numpy as np
from scipy import signal

def cal_pcc_matrix(features):
    """
    使用皮尔森相关系数计算邻接矩阵

    参数:
    - features: 二维数组，表示图的节点特征矩阵，每行是一个节点的特征向量

    返回:
    - adjacency_matrix: 二维数组，表示计算得到的邻接矩阵
    """
    num_nodes = features.shape[0]

    return np.corrcoef(features,features)[:num_nodes,:num_nodes]


def cal_coherence_matrix(features,fs=200, threshold = 0.3):
    """
    计算特征矩阵的相干矩阵
    """
    # 获取特征矩阵的行数y
    num_nodes = features.shape[0]
    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 遍历特征矩阵的每一行
    for i in range(num_nodes):
        # 遍历特征矩阵的每一列
        for j in range(i, num_nodes):
            # 计算特征矩阵两行之间的相干性
            _,similarity= signal.coherence(features[i], features[j],fs, window='hann')
            # 计算相干性的均值
            similarity= np.mean(similarity)
            # 将相干性赋值给邻接矩阵
            adjacency_matrix[i, j] = similarity
            adjacency_matrix[j, i] = similarity  # 邻接矩阵是对称的


    # 将邻接矩阵中小于0.3的值赋值为1e-10
    np.where(adjacency_matrix < threshold, 1e-10, adjacency_matrix)
    
    # 返回邻接矩阵
    return adjacency_matrix


def cal_nmi_matrix(features, threshold=0.3):
    num_nodes = features.shape[0]
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            similarity= calculate_nmi(features[i], features[j])
            adjacency_matrix[i, j] = similarity
            adjacency_matrix[j, i] = similarity  # 邻接矩阵是对称的
    
    np.where(adjacency_matrix < threshold, 1e-10, adjacency_matrix)

    return adjacency_matrix

def calculate_nmi(x, y):
    """
    计算两个信号x和y之间的互信息。
    :param x: 第一个信号
    :param y: 第二个信号
    :return: 互信息值
    """
    bins = 3
    c_xy = np.histogram2d(x, y, bins)[0]
    p_xy = c_xy / float(np.sum(c_xy))  # 联合概率分布
    p_xy = np.where(p_xy < 1e-10, 1e-10, p_xy)
    p_x = np.sum(p_xy, axis=1)  # x的边缘概率分布
    p_y = np.sum(p_xy, axis=0)  # y的边缘概率分布
    
    # 计算互信息
    mi = np.nansum(p_xy * np.log(p_xy / (p_x[:, None] * p_y[None, :])))
    # 熵
    h_x = -np.sum(p_x * np.log(p_x))
    h_y = -np.sum(p_y * np.log(p_y))

    # 归一化互信息
    nmi = 2 * mi / (h_x + h_y)
    return nmi

import numpy as np
from scipy.signal import hilbert

def cal_plv_matrix(eeg_data):
    """
    计算相位锁定值（PLV）矩阵
    :param eeg_data: 形状为 (channels_num, time_step) 的脑电数据
    :return: PLV矩阵
    """
    n_channels = eeg_data.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    phase_data = np.angle(hilbert(eeg_data, axis=1))

    for i in range(n_channels):
        for j in range(i, n_channels):
            plv = np.abs(np.mean(np.exp(1j * (phase_data[i] - phase_data[j]))))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    return plv_matrix

import numpy as np


# 这个方法是将原始数据与特征合并起来, 共MAGCN使用, 原始数据用来计算邻接矩阵, 特征用于图卷积
import scipy.io as sio
import os
import torch
import numpy as np
import re

all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]


def get_data_by_feature_type(subject_num, session_num, experiment_num, feature_type='de', smooth_type='LDS', root_path="/home/wf/EEG_GTN/data/dataset/SEED/ExtractedFeatures"):
    '''
    feature_type:de,psd,dasm,rasm,asm,dacu 
    smooth_type:movingAve,LDS
    subject_num:1-15
    experiment_num:1-15
    session_num:1-3
    '''
    file_name = str(subject_num)+'_'+str(session_num)+'.mat'
    key = feature_type+'_'+smooth_type+str(experiment_num)
    data = torch.from_numpy(sio.loadmat(
        os.path.join(root_path, file_name))[key])
    label = torch.from_numpy(np.array(all_label[experiment_num-1]))
    return data, label


def get_row_data(subject_num, session_num, experiment_num, root_path="/home/wf/EEG_GTN/data/dataset/SEED/Preprocessed_EEG"):
    '''
    feature_type:de,psd,dasm,rasm,asm,dacu 
    smooth_type:movingAve,LDS
    subject_num:1-15
    experiment_num:1-15
    session_num:1-3
    '''
    file_name = str(subject_num)+'_'+str(session_num)+'.mat'
    data = sio.loadmat(os.path.join(root_path, file_name))
    kyes = get_trail_keys(data.keys())
    data = torch.from_numpy(data[kyes[experiment_num-1]])
    label = torch.from_numpy(np.array(all_label[experiment_num-1]))

    return data, label


def get_trail_keys(data_keys):
    s = str(data_keys)

    keys = re.findall(r"'(.*?)'", s)[3:]
    return keys

import torch
import os
import logging
import random

from configparser import ConfigParser
from utils.dataset_class import *
from torch.utils.data import DataLoader

class DataSplit():
    r"""获得划分好的数据集"""

    def __init__(self, args):
        """
        初始化DataSplit类的实例。

        参数:
        - args: 包含配置信息的对象

        返回值:
        无
        """
        self.feature_type = args.feature_type
        self.label_type = args.label_type
        self.args = args
        self.dataset = self.get_dataset_class()

        config = ConfigParser()
        config.read(args.config_path, encoding='UTF-8')

        self.X, self.y = self.load_data(config['path'])
        args.nclass = int(config[args.dataset]['nclass'])
        args.domain_class = int(config[args.dataset]['sub_num'])
        # 划分 self.X_train, self.y_train, self.X_test, self.y_test
        self.data_split(args)

        # 获得dataloader
        self.train_loader = self.get_dataloader(is_train=True)
        self.test_loader = self.get_dataloader(is_train=False)

    def load_data(self, path_set):
        r"""从数据集的文件夹中读取出数据和标签

        参数:
        - path_set: 包含数据集路径的字典

        返回值:
        - X: 数据
        - y: 标签
        """

        X = torch.load(os.path.join(
            path_set[self.args.dataset], f'{self.feature_type}', 'data.pt'))
        y = torch.load(os.path.join(
            path_set[self.args.dataset], f'{self.feature_type}', 'label.pt'))
        
        if self.args.dataset == 'deap':
            y = y[:, :, :, self.get_label_dict(
                self.args.dataset, self.args.label_type)]
        
        return X, y

    def data_split(self, args):
        """
        对数据集进行切分:

        参数:

        args: 参数对象,包含切分方法等参数信息

        功能:

        根据切分方法如'by_exp'、'by_sess'等,按受试者、实验或者session等特征划分数据集为训练和测试子集
        将数据集形状改变为(index, channel, feature, band)的形式
        打印日志,记录特征长度和通道数量信息

        返回:

        无返回,但会将训练和测试子集结果赋值给类成员属性:

        self.X_train: 训练子集输入特征
        self.y_train: 训练子集目标变量
        self.X_test: 测试子集输入特征
        self.y_test: 测试子集目标变量
        """

        split_methods = {

            # SD
            'by_exp': lambda: self.split_by_exp(args),
            'by_sess': lambda: self.split_by_sess(args),

            # SI
            'loso': lambda: self.split_loso(args),
            'k_fold': lambda: self.split_k_fold(args),
        }
        self.X_train, self.y_train, self.X_test, self.y_test = split_methods[args.split_method](
        )
        self.X_train, self.y_train = self.reshape_sample(
            self.X_train, self.y_train)
        self.X_test, self.y_test = self.reshape_sample(
            self.X_test, self.y_test)
        if not args.spp:
            args.feature_len = self.X_train.shape[2]
            args.channels_num = self.X_train.shape[1]
        if args.dataset == 'seed_adj' or args.dataset == 'seed_iv_adj':
            args.channels_num = self.X_train[0]['data'].shape[0]
            args.feature_len = self.X_train[0]['data'].shape[1]
        else:
            args.feature_len = self.X_train.shape[2]
            args.channels_num = self.X_train.shape[1]

        print(
            'features length:', args.feature_len, '   ',
            'channels nums:', args.channels_num)
        logging.info(
            f'features length: {args.feature_len}, channels nums: {args.channels_num}')

    def split_by_exp(self, args):
        r"""受试者依赖的按试验留一交叉验证划分

        根据试验进行划分:

        参数:

        无

        功能:

        按当前受试者和实验索引,利用滑动窗口方式划分实验数据
        将对应受试者和实验的输入输出数据分为训练和测试子集

        返回:

        X_train: 输入训练子集
        y_train: 输出训练子集
        X_test: 输入测试子集
        y_test: 输出测试子集
        """

        sub_index = args.cur_sub_index
        exp_index = args.cur_exp_index
        clip_length = args.clip_length

        exp_nums = [i for i in range(self.X.shape[2])]
        # if args.dataset == 'faced':
        #     exp_test = []
        #     exp_test += random.sample(exp_nums[:12], 3)
        #     exp_test += random.sample(exp_nums[12:16], 1)
        #     exp_test += random.sample(exp_nums[16:], 3)
        #     random.shuffle(exp_test)
        # else:
        exp_test = exp_nums[exp_index *
                            clip_length:exp_index*clip_length+clip_length]
        exp_train = [x for x in exp_nums if x not in exp_test]

        # X_train = self.X[sub_index, :, exp_train]
        # X_test = self.X[sub_index, :, exp_test]
        # y_train = self.y[sub_index, :, exp_train]
        # y_test = self.y[sub_index, :, exp_test]

        X_train = self.X[:, :, exp_train]
        X_test = self.X[:, :, exp_test]
        y_train = self.y[:, :, exp_train]
        y_test = self.y[:, :, exp_test]

        return X_train, y_train, X_test, y_test

    def split_by_sess(self, args):
        r"""受试者依赖的按session留一交叉验证划分

        根据session进行划分:

        参数:

        无

        功能:

        按当前受试者和session索引,将数据集分为训练和测试子集
        训练子集包含session索引以外的所有session数据
        测试子集包含指定session索引的数据

        返回:

        X_train: 输入训练子集
        y_train: 输出训练子集
        X_test: 输入测试子集
        y_test: 输出测试子集

        """

        sub_index = args.cur_sub_index
        session_index = args.cur_session_index

        sess_nums = [0, 1, 2]
        sess_nums.remove(session_index)
        sess_train = sess_nums
        sess_test = [session_index]
        if isinstance(self.X, torch.Tensor):
            X_train = self.X[sub_index, sess_train]
            X_test = self.X[sub_index, sess_test]
            y_train = self.y[sub_index, sess_train]
            y_test = self.y[sub_index, sess_test]
        else:
            X_train = [[self.X[sub_index][x] for x in sess_train]]
            X_test = [[self.X[sub_index][x] for x in sess_test]]
            y_train = [[self.y[sub_index][x] for x in sess_train]]
            y_test = [[self.y[sub_index][x] for x in sess_test]]

        return X_train, y_train, X_test, y_test

    def split_loso(self, args):
        """
        根据受试者进行留一交叉验证划分:

        参数:

        args: 参数对象

        功能:

        根据当前受试者索引,将数据集分为训练和测试子集
        训练子集包含除指定受试者外的所有受试者数据
        测试子集包含指定受试者数据

        返回:

        X_train: 输入训练子集
        y_train: 输出训练子集
        X_test: 输入测试子集
        y_test: 输出测试子集
        """
        sub_train = [i for i in range(len(self.X))]
        sub_train.remove(args.cur_sub_index)
        sub_test = [args.cur_sub_index]

        if isinstance(self.X, torch.Tensor):
            X_train = self.X[sub_train]
            X_test = self.X[sub_test]
            y_train = self.y[sub_train]
            y_test = self.y[sub_test]
        else:
            X_train = [self.X[x] for x in sub_train]
            y_train = [self.y[x] for x in sub_train]
            X_test = [self.X[x] for x in sub_test]
            y_test = [self.y[x] for x in sub_test]

        return X_train, y_train, X_test, y_test

    def split_k_fold(self, args):
        """
        根据受试者进行k折交叉验证划分:

        参数:

        args: 参数对象

        功能:

        根据当前受试者索引,将数据集分为训练和测试子集
        训练子集包含除指定受试者外的所有受试者数据
        测试子集包含指定受试者数据

        返回:

        X_train: 输入训练子集
        y_train: 输出训练子集
        X_test: 输入测试子集
        y_test: 输出测试子集
        """
        sub_num = len(self.X)
        one_fold_nums = int(sub_num / args.k_fold_nums)

        sub_train = [i for i in range(len(self.X))]
        sub_test = list(range(args.cur_sub_index,args.cur_sub_index+one_fold_nums))
        sub_train = [i for i in sub_train if i not in sub_test]
        
        if isinstance(self.X, torch.Tensor):
            X_train = self.X[sub_train]
            X_test = self.X[sub_test]
            y_train = self.y[sub_train]
            y_test = self.y[sub_test]
        else:
            X_train = [self.X[x] for x in sub_train]
            y_train = [self.y[x] for x in sub_train]
            X_test = [self.X[x] for x in sub_test]
            y_test = [self.y[x] for x in sub_test]

        return X_train, y_train, X_test, y_test
    def reshape_sample(self, X, y):
        r"""
        重塑样本形状:

        参数:

        X: 输入特征
        y: 输出目标

        功能:

        如果数据是张量,直接reshape成(index, channel, feature, band)形式
        如果数据是列表,需要对每个样本作滑动窗口操作使其张量化
        返回数据形状为(index, channel, feature, band)的张量

        返回:

        out_X: 重塑后输入特征
        out_y: 重塑后输出目标

        """

        # 如果输入的是tensor 那么数据是提前对齐的 直接reshape
        if isinstance(self.X, torch.Tensor):
            out_X = X.reshape(-1, X.shape[-3],
                              X.shape[-2], X.shape[-1])
            out_y = y.reshape(-1)

        # 如果不是对齐的那么输入的是列表，需要滑动窗口分割使其对齐
        elif self.args.spp:
            out_X, out_y = [], []
            for i in range(len(X)):
                for j in range(len(X[0])):
                    for k in range(len(X[0][0])):
                        out_X.append(X[i][j][k])
                        out_y.append(y[i][j][k].unsqueeze(0))

            out_y = torch.cat(out_y, 0)
        else:
            out_X, out_y = [], []
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    for k in range(len(self.X[0][0])):
                        temp_X = X[i][j][k].unsqueeze(0)
                        temp_y = y[i][j][k].unsqueeze(0)
                        temp_X, temp_y = self.window_slide(
                            self.args, temp_X, temp_y)
                        out_X.append(temp_X)
                        out_y.append(temp_y)
            out_X = torch.cat(out_X, 0)
            out_y = torch.cat(out_y, 0)
        return out_X, out_y

    def window_slide(self, args, X, y):
        r"""滑动窗口分割, 根据args里面的window_len和step参数进行切割

        对样本进行滑动窗口分割:

        参数:

        args: 参数对象,包含窗口长度和步长参数
        X: 输入特征
        y: 输出目标

        功能:

        根据窗口长度和步长参数对每个样本进行滑动窗口分割
        返回窗口化后的输入和输出组成的张量

        返回:

        X_slide: 滑动窗口分割后的输入特征张量
        y_slide: 滑动窗口分割后的输出目标张量

        """

        X_slide = []
        y_slide = []
        for i in range(len(y)):
            start = 0
            end = args.window_len
            X_temp = []
            y_temp = []
            while end < X.shape[-2]:
                X_temp.append(X[i, :, start:end, :].unsqueeze(0))
                y_temp.append(y[i].unsqueeze(0))
                start += args.step
                end += args.step
            X_temp.append(
                X[i, :, X.shape[-2]-args.window_len:, :].unsqueeze(0))
            y_temp.append(y[i].unsqueeze(0))
            X_slide.append(torch.cat(X_temp, dim=0))
            y_slide.append(torch.cat(y_temp, dim=0))
        X_slide = torch.cat(X_slide, dim=0)
        y_slide = torch.cat(y_slide, dim=0)
        return X_slide, y_slide

    def get_dataset_class(self):
        """
        获取数据集类:

        参数:

        无

        功能:

        根据配置中的数据集名称,从工厂字典中获取对应的数据集类

        返回:

        dataset工厂字典中对应的Dataset类,用于实例化后加载数据集
        """
        data_factory = {
            'seed': SEEDDataset,
            'seed_adj': SEEDAdjDataset,
            'seed_origin': SEEDDataset,
            'deap': DEAPDataset,
            'amigos': AMIGOSDataset,
            'seed_iv': SEEDIVDataset_spp,
            'seed_iv_adj': SEEDIVAdjDataset,
            'faced': FACEDataset,
        }
        return data_factory[self.args.dataset]

    def get_dataloader(self, is_train=True):
        r"""获取dataloader
        获取数据加载器:

        参数:

        is_train: 是否为训练集,默认为True

        功能:

        根据is_train变量切换训练/测试数据
        使用数据集类加载数据
        返回DataLoader加载器,包含batch处理及shuffle等操作

        返回:
        DataLoader对象,用于模型训练/测试数据的批量输入

        """

        if is_train:
            dataset = self.dataset(self.X_train, self.y_train)
        else:
            dataset = self.dataset(self.X_test, self.y_test)
        return DataLoader(dataset, num_workers=0, batch_size=int(self.args.batch_size), shuffle=True, drop_last=True)
    
    def get_label_dict(self,dataset = 'deap',label = 'valence'):
        label_dict = {
            'deap':{
                'valence':0, 
                'arousal':1, 
                'dominance':2,
                'liking':3
            }            
        }

        return label_dict[dataset][label]



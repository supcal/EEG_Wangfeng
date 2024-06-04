import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.DataSplit as DataSplit
from models import ATGRNet, DGCNN, SPP_GNN, Net, MAGCN, MAGCN_func,SVM
from utils.train_utils import EarlyStopping, Accuracy, MeanLoss, AccStd, TSNE, Confusion, DataSaver


class Trainer(object):

    def __init__(self, args):
        """
        构造函数:

        参数:
        - args: 参数对象

        功能:
        1. 初始化必要属性如数据加载器、模型、优化器等
        2. 打印日志信息
        """
        super(Trainer, self).__init__()

        self.args = args

        data_spliter = DataSplit.DataSplit(args)
        self.train_loader = data_spliter.train_loader
        self.test_loader = data_spliter.test_loader

        self.model = self.get_model()(args)
        self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=4, verbose=True)
        self.early_stopping = EarlyStopping(patience=10)
        self.mean_accuracy = Accuracy(args.nclass)
        self.mean_loss = MeanLoss(args.batch_size)
        self.acc_std = AccStd(args.batch_size)
        self.tsne_class = TSNE(args)
        self.tsne_domain = TSNE(args)
        self.confusion_matrix = Confusion(args)
        self.data_saver = DataSaver(args)

        print(args)
        logging.info(args)

    def run(self):
        """
        执行整个训练流程:

        功能:
        1. 初始化衡量指标
        2. 循环训练-验证过程
        3. 保存最优模型参数
        4. 返回最高精度结果
        """
        self.acc_std.reset()
        self.confusion_matrix.reset()
        for epoch in range(self.args.epochs):
            self.train()
            acc, mloss = self.validation(epoch)
            self.acc_std.update(acc)
            is_best, is_terminate = self.early_stopping(acc)
            self.tsne_class.update_best(is_best)
            if self.args.have_domain:
                self.tsne_domain.update_best(is_best)
            self.confusion_matrix.update_best(is_best)
            if is_terminate:
                break
            if is_best:
                state_dict = self.model.state_dict()
            self.lr_scheduler.step(mloss)
        max_acc, std = self.acc_std.compute()
        if self.args.save_tsne_cm == 1:
            if not self.args.have_domain:
                self.data_saver.save_c(
                    self.model, self.tsne_class.save())
            else:
                self.data_saver.save_cd(
                    self.model, self.tsne_class.save(), self.tsne_domain.save())

        print('max acc={} std={}'.format(max_acc, std))
        logging.info('max acc={} std={}'.format(max_acc, std))
        return max_acc, std

    def train(self):
        """
        训练一个epoch:

        功能: 
        1. 设置模型为训练模式
        2. 循环读取一个batch训练数据
        3. 前向传播计算损失
        4. 反向传播优化
        5. 更新其他辅助训练所需变量
        """
        self.model.train()
        # self.tsne.reset_train()
        self.confusion_matrix.reset_train()

        for step, (data, labels) in enumerate(self.train_loader):
            if isinstance(data, torch.Tensor):
                data, labels = data.cuda(), labels.cuda()
                logits = self.model(data)
                loss = self.model.loss(self.model, logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.tsne_class.update_train(
                    logits[1], logits[2], labels.cpu().numpy())
                probs = F.softmax(logits[0], dim=-1).cpu().detach().numpy()

                self.confusion_matrix.update_train(
                    probs, labels.cpu().numpy())

            else:
                data = [i.cuda() for i in data]
                labels = [j.cuda() for j in labels]
                logits = self.model(data)
                loss = self.model.loss(self.model, logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                probs = F.softmax(logits[0], dim=-1).cpu().detach().numpy()
                self.tsne_class.update_train(
                    logits[2], logits[3], labels[0].cpu().numpy())
                self.tsne_domain.update_train(
                    logits[2], logits[4], labels[1].cpu().numpy())
                self.confusion_matrix.update_train(
                    probs, labels[0].cpu().numpy())

    def validation(self, epoch):
        """
        验证模型:

        参数:
        - epoch: 当前epoch数

        功能:
        1. 设置模型评估模式
        2. 循环一个batch验证数据
        3. 计算损失和准确率  
        4. 返回本轮结果及损失  
        5. 更新其他辅助验证所需变量
        """
        self.model.eval()
        self.mean_loss.reset()
        self.mean_accuracy.reset()
        # self.tsne.reset_test()
        self.confusion_matrix.reset_test()

        with torch.no_grad():
            for step, (data, labels) in enumerate(self.test_loader):
                if isinstance(data, torch.Tensor):
                    data, labels = data.cuda(), labels.cuda()
                    # logits, tsne1, tsne2 = self.model(data)
                    logits = self.model(data)
                    # loss = self.criterion(logits, labels.cuda())
                    loss = self.model.loss(self.model, logits, labels)

                    probs = F.softmax(logits[0], dim=-1).cpu().detach().numpy()
                    labels = labels.cpu().numpy()
                    self.mean_loss.update(loss.cpu().detach().numpy())
                    self.mean_accuracy.update(probs, labels)
                    self.tsne_class.update_test(logits[1], logits[2], labels)
                    self.confusion_matrix.update_test(probs, labels)
                else:
                    data = [i.cuda() for i in data]
                    labels = [j.cuda() for j in labels]
                # logits, tsne1, tsne2 = self.model(data)
                    logits = self.model(data)
                    # loss = self.criterion(logits, labels.cuda())
                    loss = self.model.loss(self.model, logits, labels)

                    probs = F.softmax(logits[0], dim=-1).cpu().detach().numpy()
                    # labels = labels[0].cpu().numpy()
                    self.mean_loss.update(loss.cpu().detach().numpy())
                    self.mean_accuracy.update(probs, labels[0].cpu().numpy())
                    self.tsne_class.update_test(
                        logits[2], logits[3], labels[0].cpu().numpy())
                    self.tsne_domain.update_test(
                        logits[2], logits[4], labels[1].cpu().numpy())
                    self.confusion_matrix.update_test(
                        probs, labels[0].cpu().numpy())
                # logging.info(f"{np.argmax(probs, axis=1)},{labels}")

        acc = self.mean_accuracy.compute()
        mloss = self.mean_loss.compute()

        print(
            f"Validation Results - Epoch: {epoch} acc: {acc:.4f} loss: {mloss:.4f}")
        logging.info(
            f"Validation Results - Epoch: {epoch} acc: {acc:.4f} loss: {mloss:.4f}")

        return acc, mloss

    def test(self):
        """
        测试模型:

        功能: 
        1. 设置模型评估模式
        2. 循环测试数据计算准确率
        3.返回最终测试结果
        """
        self.model.eval()
        self.mean_accuracy.reset()
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].cuda(), batch[1]
                logits = self.model(data)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)
        acc = self.mean_accuracy.compute()
        print(f"Testing Results - acc: {acc:.4f}")
        logging.info(f"Testing Results - acc: {acc:.4f}")

    def get_model(self):
        r"""
        根据args的model参数获取对应的模型
        """

        model_dict = {
            'ATGRNet': ATGRNet,
            'DGCNN':  DGCNN,
            'SPP_GNN':  SPP_GNN,
            'Net': Net,
            'MAGCN': MAGCN,
            'MAGCN_func': MAGCN_func,
            'SVM': SVM

        }
        model = model_dict[self.args.model]
        return model

    def get_parameters_num(self):
        r"""
        获得模型参数量
        """
        total_parameters = 0
        for name, param in self.model.named_parameters():
            total_parameters += param.nelement()
            logging.info('{:15}\t{:25}\t{:5}'.format(
                name, str(param.shape), param.nelement()))
        print('Total parameters: {}'.format(total_parameters))
        logging.info('Total parameters: {}'.format(total_parameters))

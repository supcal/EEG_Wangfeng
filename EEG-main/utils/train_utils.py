import numpy as np
import torch
import os


class MeanAccuracy(object):
    def __init__(self, classes_num):
        super().__init__()
        self.classes_num = classes_num

    def reset(self):
        self._crt_counter = np.zeros(self.classes_num)
        self._gt_counter = np.zeros(self.classes_num)

    def update(self, probs, gt_y):
        pred_y = np.argmax(probs, axis=1)
        for pd_y, gt_y in zip(pred_y, gt_y):
            if pd_y == gt_y:
                self._crt_counter[pd_y] += 1
            self._gt_counter[gt_y] += 1

    def compute(self):
        self._gt_counter = np.maximum(
            self._gt_counter, np.finfo(np.float64).eps)
        accuracy = self._crt_counter / self._gt_counter
        mean_acc = np.mean(accuracy)
        return mean_acc


class Accuracy(object):
    def __init__(self, classes_num):
        super().__init__()
        self.classes_num = classes_num

    def reset(self):
        self._crt_counter = np.zeros(self.classes_num)
        self._gt_counter = np.zeros(self.classes_num)

    def update(self, probs, gt_y):
        pred_y = np.argmax(probs, axis=1)
        for pd_y, gt_y in zip(pred_y, gt_y):
            if pd_y == gt_y:
                self._crt_counter[pd_y] += 1
            self._gt_counter[gt_y] += 1

    def compute(self):
        self._crt_counter = np.sum(self._crt_counter)
        self._gt_counter = np.sum(self._gt_counter)
        acc = self._crt_counter / self._gt_counter
        return acc


class MeanLoss(object):
    def __init__(self, batch_size):
        super(MeanLoss, self).__init__()
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss):
        self._sum += loss * self._batch_size
        self._counter += self._batch_size

    def compute(self):
        return self._sum / self._counter


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        is_best, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_best = False
        else:
            self.best_score = score
            self.counter = 0
        return is_best, is_terminate


class AccStd(object):
    def __init__(self, batch_size):
        self.acc_array = []

    def reset(self):
        self.acc_array = []

    def update(self, acc):
        self.acc_array.append(acc)

    def compute(self):
        std = float(torch.FloatTensor(self.acc_array).std())*100
        max_acc = max(self.acc_array)
        return max_acc, std


class TSNE(object):
    def __init__(self, args):
        self.args = args
        self.tsne1_train = []
        self.tsne2_train = []
        self.tsne1_test = []
        self.tsne2_test = []

        self.bset_tsne1_train = []
        self.bset_tsne2_train = []

        self.bset_tsne1_test = []
        self.bset_tsne2_test = []

        self.label_train = []
        self.label_test = []
        self.best_label_train = []
        self.best_label_test = []

    def reset(self):
        self.reset_test()
        self.reset_train()
        self.bset_tsne1_train = []
        self.bset_tsne2_train = []

        self.bset_tsne1_test = []
        self.bset_tsne2_test = []

        self.best_label_train = []
        self.best_label_test = []

    def reset_train(self):
        self.tsne1_train = []
        self.tsne2_train = []
        self.label_train = []

    def reset_test(self):
        self.tsne1_test = []
        self.tsne2_test = []
        self.label_test = []

    def update_train(self, tsne1, tsne2, label):
        self.tsne1_train.append(tsne1)
        self.tsne2_train.append(tsne2)
        self.label_train.append(label)

    def update_test(self, tsne1, tsne2, label):
        self.tsne1_test.append(tsne1)
        self.tsne2_test.append(tsne2)
        self.label_test.append(label)

    def update_best(self, is_best):
        if is_best:
            self.bset_tsne1_train = self.tsne1_train
            self.bset_tsne2_train = self.tsne2_train
            self.bset_tsne1_test = self.tsne1_test
            self.bset_tsne2_test = self.tsne2_test
            self.best_label_train = self.label_train
            self.best_label_test = self.label_test

    def save(self):
        self.bset_tsne1_train = torch.cat(self.bset_tsne1_train, 0)
        self.bset_tsne2_train = torch.cat(self.bset_tsne2_train, 0)
        self.bset_tsne1_test = torch.cat(self.bset_tsne1_test, 0)
        self.bset_tsne2_test = torch.cat(self.bset_tsne2_test, 0)
        # print(self.best_label_train)
        self.best_label_train = torch.from_numpy(
            np.concatenate(self.best_label_train, 0))
        self.best_label_test = torch.from_numpy(
            np.concatenate(self.best_label_test, 0))

        return [self.bset_tsne1_train, self.bset_tsne2_train, self.best_label_train, self.bset_tsne1_test, self.bset_tsne2_test, self.best_label_test]


class Confusion(object):
    def __init__(self, args):
        self.args = args
        self.pre_train = []
        self.true_train = []
        self.pre_test = []
        self.true_test = []

        self.bset_pre_train = []
        self.bset_true_train = []

        self.bset_pre_test = []
        self.bset_true_test = []

    def reset(self):
        self.reset_test()
        self.reset_train()

        self.bset_pre_train = []
        self.bset_true_train = []

        self.bset_pre_test = []
        self.bset_true_test = []

    def reset_train(self):
        self.pre_train = []
        self.true_train = []

    def reset_test(self):
        self.pre_test = []
        self.true_test = []

    def update_train(self, pre, true):
        self.pre_train.append(np.argmax(pre, -1))
        self.true_train.append(true)

    def update_test(self, pre, true):
        self.pre_test.append(np.argmax(pre, -1))
        self.true_test.append(true)

    def update_best(self, is_best):
        if is_best:
            self.bset_pre_train = self.pre_train
            self.bset_true_train = self.true_train
            self.bset_pre_test = self.pre_test
            self.bset_true_test = self.true_test

    def save(self):
        self.bset_pre_train = np.concatenate(self.bset_pre_train, 0)
        self.bset_true_train = np.concatenate(self.bset_true_train, 0)
        self.bset_pre_test = np.concatenate(self.bset_pre_test, 0)
        self.bset_true_test = np.concatenate(self.bset_true_test, 0)

        return [self.bset_pre_train, self.bset_true_train, self.bset_pre_test, self.bset_true_test]


class DataSaver(object):
    def __init__(self, args):

        self.args = args
        self.save_path = args.save_path

    def save_c(self, model, tsne_class):

        if not os.path.exists(os.path.join(self.save_path, 'tsne_class')):
            os.makedirs(os.path.join(self.save_path, 'tsne_class'))
        torch.save(tsne_class, os.path.join(
            self.save_path, 'tsne_class', f'tsne_{self.args.cur_sub_index}.pt'))

    def save_cd(self, model, tsne_class, tsne_domain):

        if not os.path.exists(os.path.join(self.save_path, 'tsne_class')):
            os.makedirs(os.path.join(self.save_path, 'tsne_class'))
        if not os.path.exists(os.path.join(self.save_path, 'tsne_domain')):
            os.makedirs(os.path.join(self.save_path, 'tsne_domain'))

        torch.save(tsne_class, os.path.join(
            self.save_path, 'tsne_class', f'tsne_{self.args.cur_sub_index}.pt'))
        torch.save(tsne_domain, os.path.join(
            self.save_path, 'tsne_domain', f'tsne_{self.args.cur_sub_index}.pt'))

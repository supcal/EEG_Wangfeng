import logging
import numpy as np
import torch
from trainer import Trainer
from configparser import ConfigParser


def debug(args):
    r"""模型debug时使用, train_mode为loso, 但是固定测试为index为0的受试者"""

    args.cur_sub_index = 0

    trainer = Trainer(args)
    max_acc, std = trainer.run()

    print('debug is done, acc= {:.4f} std= {:.4f}'.format(max_acc, std))
    logging.info('debug is done, acc= {:.4f} std= {:.4f}'.format(max_acc, std))


def subject_independent(args):
    r"""受试者独立实验, 使用LOSO, 每个人轮流做测试集"""

    config = ConfigParser()
    config.read(args.config_path, encoding='UTF-8')

    acc_array = []
    sub_num = int(config[args.dataset]['sub_num'])
    if args.split_method == 'loso':
        for sub in range(sub_num):

            args.cur_sub_index = sub

            print(f'sub{sub} is start:')
            logging.info(f'sub{sub} is start:')

            trainer = Trainer(args)
            max_acc, std = trainer.run()

            print(
                'sub{} is done, acc= {:.4f} std= {:.4f}'.format(sub, max_acc, std))
            logging.info(
                'sub{} is done, acc= {:.4f} std= {:.4f}'.format(sub, max_acc, std))
            acc_array.append(max_acc)
            del trainer

        acc_array = np.array(acc_array)

        print('LOSO mean acc = {:.4f}, std = {:.4f}'.format(
            np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))
        logging.info('LOSO mean acc = {:.4f}, std = {:.4f}'.format(
            np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))

    elif args.split_method == 'k_fold':
        one_fold_nums = int(sub_num / args.k_fold_nums)
        for fold in range(int(args.k_fold_nums)):

            args.cur_sub_index = fold*one_fold_nums

            print(f'fold{fold} is start:')
            logging.info(f'fold{fold} is start:')

            trainer = Trainer(args)
            max_acc, std = trainer.run()

            print(
                'fold{} is done, acc= {:.4f} std= {:.4f}'.format(fold, max_acc, std))
            logging.info(
                'fold{} is done, acc= {:.4f} std= {:.4f}'.format(fold, max_acc, std))
            acc_array.append(max_acc)
            del trainer

        acc_array = np.array(acc_array)

        print('{}_fold mean acc = {:.4f}, std = {:.4f}'.format(args.k_fold_nums,
                                                               np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))
        logging.info('{}_fold mean acc = {:.4f}, std = {:.4f}'.format(args.k_fold_nums,
                                                                      np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))


def subject_dependent(args):
    if args.split_method == 'by_sess':
        subject_dependent_sess(args)
    if args.split_method == 'by_exp':
        subject_dependent_exp(args)


def subject_dependent_sess(args):
    r"""受试者依赖实验, 根据session分

    每个受试者的两个session做训练集, 另一个做测试集, 做三次取平均值

    仅限SEED使用
    """

    config = ConfigParser()
    config.read(args.config_path, encoding='UTF-8')

    acc_array = []

    for sub in range(int(config[args.dataset]['sub_num'])):

        args.cur_sub_index = sub

        print(f'sub{sub} is start:')
        logging.info(f'sub{sub} is start:')

        for session in range(int(config[args.dataset]['session_num'])):

            args.cur_session_index = session

            print(f'sub{sub} session{session} is start:')
            logging.info(f'sub{sub} session{session} is start:')

            trainer = Trainer(args)
            max_acc, std = trainer.run()

            print('sub{} session{} is done, acc= {:.4f} std= {:.4f}'.format(
                sub, session, max_acc, std))
            logging.info(
                'sub{} session{} is done, acc= {:.4f} std= {:.4f}'.format(sub, session, max_acc, std))
            acc_array.append(max_acc)
            del trainer

    acc_array = np.array(acc_array)

    print('SD by session mean acc = {:.4f}, std = {:.4f}'.format(
        np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))
    logging.info('SD by session mean acc = {:.4f}, std = {:.4f}'.format(
        np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))


def subject_dependent_exp(args):
    r"""受试者依赖实验, 根据实验分

    每个受试者的每个实验轮流做测试集, 取平均值
    """
    config = ConfigParser()
    config.read(args.config_path, encoding='UTF-8')

    acc_array = []

    # for sub in range(int(config[args.dataset]['sub_num'])):

    #     args.cur_sub_index = sub

    #     print(f'sub{sub} is start:')
    #     logging.info(f'sub{sub} is start:')

    for exp in range(int(config[args.dataset]['exp_num'])):

        args.cur_exp_index = exp

        # print(f'sub{sub} exp{exp} is start:')
        # logging.info(f'sub{sub} exp{exp} is start:')
        print(f' exp{exp} is start:')
        logging.info(f'exp{exp} is start:')

        trainer = Trainer(args)
        max_acc, std = trainer.run()

        # print('sub{} exp{} is done, acc= {:.4f} std= {:.4f}'.format(
        #     sub, exp, max_acc, std))
        # logging.info(
        #     'sub{} exp{} is done, acc= {:.4f} std= {:.4f}'.format(sub, exp, max_acc, std))
        print('exp{} is done, acc= {:.4f} std= {:.4f}'.format(
             exp, max_acc, std))
        logging.info(
            'exp{} is done, acc= {:.4f} std= {:.4f}'.format(exp, max_acc, std))
        
        acc_array.append(max_acc)
        del trainer

    acc_array = np.array(acc_array)

    print('SD by exp mean acc = {:.4f}, std = {:.4f}'.format(
        np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))
    logging.info('SD by exp mean acc = {:.4f}, std = {:.4f}'.format(
        np.mean(acc_array), float(torch.FloatTensor(acc_array).std())))

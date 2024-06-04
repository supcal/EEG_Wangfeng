import time
import logging
from utils.utils import arg_parse, setup_seed, setup_device, setup_save_path, get_time
from train_mode import debug, subject_dependent, subject_independent

if __name__ == "__main__":

    # 计时开始时间
    T1 = time.time()

    # 初始化各种参数
    args = arg_parse()  # 根据命令行和默认参数初始化各种参数
    setup_seed(args.seed)  # 设置随机种子
    setup_device(args)  # 设置device
    setup_save_path(args)  # 设置log文件的存储路径和模型保存的路径

    train_mode = {
        'debug': debug,
        'si': subject_independent,
        'sd': subject_dependent
    }

    train = train_mode[args.train_mode]
    train(args)

    T2 = time.time()
    logging.info(get_time(T1, T2))

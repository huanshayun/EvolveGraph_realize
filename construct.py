from utils.general import read_pickle, write_pickle
import config as cfg
from itertools import chain
import os
import random
import numpy as np
import torch


def get_filelist(path):
    files = [[(home, file) for file in files] for home, _, files in os.walk(path)]
    files = list(chain.from_iterable(files))
    return files


def construct(path):
    # 获取文件列表
    files = get_filelist(path)
    files = [os.path.join(home, file) for home, file in files]
    # cfg.segment：场比赛
    train_num = int(cfg.data_num * cfg.train)
    val_num = int(cfg.data_num * cfg.val)
    test_num = int(cfg.data_num * cfg.test)
    sizes = [size // cfg.segment for size in [0, train_num, val_num, test_num]]
    # size：每场比赛选取的样本数
    size = sum(sizes)
    # 训练集，验证集，测试集的分割点，如50000,60000,70000
    sizes = np.cumsum(sizes)
    # 随机选取cfg.segment场比赛
    files = random.sample(files, cfg.segment)
    data = [[] for _ in range(3)]
    for file in files:
        state, cat = read_pickle(file)
        # 从一场比赛中随机选取size个样本
        idx = np.random.choice(range(len(state)), size, replace=False)
        state, cat = state[idx], cat[idx]
        # 将所选样本按比例分配到训练集，验证集，测试集
        for i in range(3):
            data[i].append((state[sizes[i]:sizes[i + 1]], cat[sizes[i]:sizes[i + 1]]))
    data = [[np.concatenate(j) for j in zip(*i)] for i in data]
    return data



def construct_NBA(path):
    data = construct(path)
    # 原始数据中类别从-1开始，处理成从0开始
    data = [[state, cat[..., 0] + 1] for state, cat in data]
    # data = [[normalize(state), cat[..., 0] + 1] for state, cat in data]
    file = path.replace('raw', 'data_100seg.pkl')
    write_pickle(data, file)


def construct_H3D():
    pass


def construct_Stanford():
    pass


def normalize(x, a=100, b=50):
    x = torch.cat([x[..., [0]] / a, x[..., [1]] / b], dim=-1)
    x = x * 2 - 1
    return x


def denormalize(x, a=100, b=50):
    x = (x + 1) / 2
    x = torch.cat([x[..., [0]] * a, x[..., [1]] * b], dim=-1)
    return x


if __name__ == "__main__":
    path = 'data/NBA/raw'
    construct_NBA(path)
    data = read_pickle('data/NBA/data.pkl')
    # x = torch.rand(2, 4, 3, 2)
    pass

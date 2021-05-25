import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from datagen import gen_data

SIZE = 128
ITERATIONS = 5000
BATCH_SIZE = 8
SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 实现一个网络
        # conv 9x9,3->64, batchnorm, relu
        # conv 3x3,64->64, batchnorm, relu
        # conv 3x3,64->64, batchnorm, relu
        # conv 3x3,64->64, batchnorm, relu
        # conv 3x3,64->1, sigmoid
        pass
        

    def forward(self, x):
        # 实现
        pass


def data_generator():
    # 数据迭代器
    rnd = np.random.RandomState(SEED)
    while True:
        raw, norm = gen_data(rnd, BATCH_SIZE)
        yield torch.from_numpy(raw).permute(0, 3, 1, 2), \
              torch.from_numpy(norm).permute(0, 3, 1, 2)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------
    #  Model and Optimizer
    # ----------
    # 实例化模型和优化器（adam）
    
    # ----------
    #  Data
    # ----------
    dataIter = data_generator()

    # ----------
    #  Training
    # ----------
    for iteration in range(1, ITERATIONS + 1):
        # 以下为读取数据的过程
        rawData, normData = next(dataIter)

        rawData = rawData.to(DEVICE)
        normData = normData.to(DEVICE)

        # 实现训练过程 梯度置0，前向计算，计算loss，反向传播，优化器前进，输出loss值


if __name__ == "__main__":
    main()

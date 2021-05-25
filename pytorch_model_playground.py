import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from datagen import gen_data

SIZE = 128
ITERATIONS = 5
BATCH_SIZE = 8
SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, n_chans1=64, norm_layer=None):
        super(Net, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # 实现一个网络
        # conv 9x9,3->64, batchnorm, relu
        self.conv1 = nn.Conv2d(1, n_chans1, kernel_size=9, padding=4)
        self.bn1 = norm_layer(n_chans1)
        self.act1 = nn.ReLU()
        
        
        # conv 3x3,64->64, batchnorm, relu
        self.conv2 = nn.Conv2d(n_chans1, n_chans1, kernel_size=3, padding=1)
        self.bn2 = norm_layer(n_chans1)
        self.act2 = nn.ReLU()
        
        
        # conv 3x3,64->64, batchnorm, relu
        self.conv3 = nn.Conv2d(n_chans1, n_chans1, kernel_size=3, padding=1)
        self.bn3 = norm_layer(n_chans1)
        self.act3 = nn.ReLU()
        
        
        # conv 3x3,64->64, batchnorm, relu
        self.conv4 = nn.Conv2d(n_chans1, n_chans1, kernel_size=3, padding=1)
        self.bn4 = norm_layer(n_chans1)
        self.act4 = nn.ReLU()
        
        
        # conv 3x3,64->1, sigmoid
        self.conv5 = nn.Conv2d(n_chans1, 1, kernel_size=3, padding=1)
        self.act5 = nn.Sigmoid()

    def forward(self, x):
        # 实现
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out))
        out = self.act3(self.bn3(self.conv3(out)))
        out = self.act4(self.bn4(self.conv4(out)))
        out = self.act5(self.conv5(out))
        return out


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
    model = Net().to(DEVICE)
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0003,
            betas=(0.9, 0.99))
    
    
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
        optimizer.zero_grad()
        model.train()
        
        validity = model(rawData)
        
        loss_out = F.binary_cross_entropy_with_logits(validity, normData)
        
        loss = loss_out
        
        loss.backward()
        optimizer.step()
        
        print("[Train] [Iteration %d] [loss: %f] " % (
        iteration,
        loss.item(),
        ))


if __name__ == "__main__":
    main()

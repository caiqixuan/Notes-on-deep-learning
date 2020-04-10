import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from matplotlib import pyplot as plt

"""
核心函数，已知gamma和beta是要在module里进行更新的，所以不在这里写
"""


def batch_norm(is_training, X, eps, gamma, beta, running_mean, running_var, alpha):
    #
    assert len(X.shape) in (2, 4)
    if is_training:
        """ 
        Shape:
        - Input::math: `(N, C)`
        - Output::math: `(N, C)`(same
        shape as input)
        """
        # X [batch,n]或X [batch,n，L]
        if len(X.shape) <= 3:
            # 是对每一个channel的数据进行求均值和求方差，而对应输入的x数据，应该要映射到第一个维度上所以dim=0
            mean = X.mean(dim=0)
            var = X.var(dim=0, unbiased=False)


        else:
            """ 
            Shape:
            - Input::math: `(N, C, H, W)`
            - Output::math: `(N, C, H, W)`(same
            shape as input)
            """
            # X [batch,c,h,w]
            # 先把所有样本的均值求出，再算高度和宽度上的，收缩到每一个通道，方差同理
            N = X.shape[0] * X.shape[2] * X.shape[3]  # 统计样本数，便于后续做无偏估计
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var =  ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3,keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 求滑动平均值
        running_mean = alpha * mean + (1 - alpha) * running_mean
        # 求滑动方差
        running_var = alpha * var + (1 - alpha) * running_var
    # 如果是测试过程，就只将得到的数据带入计算
    else:
        X_hat = (X - running_mean) / torch.sqrt(running_var + eps)

    # print(gamma.shape,X_hat.shape,beta.shape)
    Y = gamma * X_hat + beta  #

    return Y, running_mean, running_var


class BatchNorm(nn.Module):
    def __init__(self, dimension_type, in_channels):
        super(BatchNorm, self).__init__()
        # 卷积层/全连接层归一化后的线性变换参数.
        if dimension_type == 2:
            # x:[batch,n]
            shape = (1, in_channels)
            self.gamma = nn.Parameter(torch.ones(shape))  # 是可学习的参数.反向传播时需要根据梯度更新,写入parameter方便更新.
            self.beta = nn.Parameter(torch.zeros(shape))  # 是可学习的参数.反向传播时需要根据梯度更新.
            self.running_mean = torch.zeros(shape)  # 不需要求梯度.在forward时候更新.
            self.running_var = torch.zeros(shape)  # 不需要求梯度.在forward时候更新.
        else:
            # x:[btach,c,h,w]
            shape = (1, in_channels, 1, 1)
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))
            self.running_mean = torch.zeros(shape)
            self.running_var = torch.zeros(shape)

        self.eps = 1e-5
        self.momentum = 0.9

    def forward(self, x):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        """
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        """
        # self.training继承自nn.Module,默认true,调用.eval()会设置成false
        if self.training:
            Y, self.running_mean, self.running_var = batch_norm(is_training=True, X=x, eps=self.eps, gamma=self.gamma,
                                                                beta=self.beta,
                                                                running_mean=self.running_mean,
                                                                running_var=self.running_var, alpha=self.momentum)
        else:
            Y, self.running_mean, self.running_var = batch_norm(is_training=False, X=x, eps=self.eps, gamma=self.gamma,
                                                                beta=self.beta,
                                                                running_mean=self.running_mean,
                                                                running_var=self.running_var, alpha=self.momentum)

        return Y

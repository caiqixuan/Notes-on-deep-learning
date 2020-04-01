import numpy as np
from collections import defaultdict

"""
最关键的是得知输入的尺度，形式
代码分为两个部分：前向传播和反向传播
前向传播是BN操作
反向传播为了更新参数
"""

"""
BN的前向传导，分为数据输入，训练，测试，传出
如果得到从上一层传来的x和相应的权重
BN_in的功能是为BN层提供输入
"""


class bn_layer:
    def __init__(self, train_state, learning_rate, forward_state, first, gamma=1, beta=0, eps=1e-8):
        self.train_state = train_state
        self.learning_rate = learning_rate
        self.forward_state = forward_state
        self.first = first  # 是否是第一次，即初始化
        self.N = x.shape[1]
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    """
    训练时前向传播函数
    cache为缓存，便于记录值，简化运算
    """

    def bn_forward_train(self, layer_in, d_cache):
        mean_now = np.mean(layer_in, axis=0)  # 计算均值
        N = self.N
        var_now = (N / N - 1) * np.var(layer_in, axis=0)  # 计算每一列的标准差的无偏估计

        if self.first:
            mean = mean_now
            var = var_now
            gamma = self.gamma
            beta = self.beta
            self.first = false
        else:
            gamma = d_cache['gamma']
            beta = d_cache['beta']
            mean = d_cache['mean']
            var = d_cache['var']
            mean = momentun * mean + (1 - momentun) * mean_now  # 计算滑动均值
            var = momentun * var + (1 - momentun) * var_now  # 计算滑动标准差

        x_m = x - mean
        v_l = 1 / sqrt(var + eps)

        x_norm = x_m / v_l
        x_out = gamma * x_norm + beta

        d_cache['mean'] = mean
        d_cache['var'] = var
        d_cache['gamma'] = gamma
        d_cache['N'] = N
        d_cache['x_m'] = x_m
        d_cache['v_l'] = v_l
        d_cache['x_norm'] = x_norm
        return x_out, d_cache

    """
    测试时前向传播函数
    用测试集所有数据算平均和标准差
    使用已经训练好的gamma和beta
    """

    def bn_forward_test(self, layer_in, d_cache):
        x_norm = (layer_in - np.mean(layer_in, axis=0)) / sqrt(np.var(layer_in, axis=0) + self.eps)
        x_last = d_cache['gamma'] * x_norm + d_cache['beta']
        return x_last

    """
    反向传播，主要通过导数进行参数更新
    :param dy是loss函数对y的求导，可以看出其他的求偏导都是和dy相关的
    :param dx是loss函数对x的求导
    :param dx_u是loss函数对x均的求导
    :param du是loss函数对均值u的求导
    :param dD是loss函数对方差的求导
    :param dr是loss函数对gamma的求导
    :param db是loss函数对beta的求导
    """

    def bn_backward_train(self, dy, d_cache):
        gamma = d_cache['gamma']
        N = d_cache['N']
        x_m = d_cache['x_m']
        v_l = d_cache['v_l']
        x_norm = d_cache['x_norm']

        dx_u = dy * gamma
        dr = np.sum(dy * x_norm, axis=0)
        db = np.sum(dy, axis=0)
        dD = np.sum(dx_u * x_m * (-0.5) * v_l ** 3, axis=0)
        du = np.sum(dx_u * (-v_l), axis=0) + dD * 1 / N * np.sum(-2 * x_m, axis=0)
        dx = dx_u * v_l + dD * 2 * x_m / N + du / N
        return dx, dr, db

    """
    输出函数,将几个函数的输出整合
    """

    def output(self, layer_in, d_cache, dy=None):
        if self.train_state and self.forward_state:
            x_out, d_cache = bn_forward_train(self, layer_in, d_cache)
            return x_out, d_cache
        if self.train_state and not self.forward_state:
            dx, dr, db = bn_backward_train(dy)
            # write_self(self, dr, db)
            return dx, d_cache
        if not self.train_state:
            x_out = bn_forward_test(self, layer_in)
            return x_out, d_cache


# x*w矩阵,x每一行对应着不同维度的一个数据，一列对应着一个维度的一组数据，w每一列对应着不同维度计算的权重，每一行对应着每一组数据，x的列数等于w的行数
def bn_in(x, w):
    layer_in = tf.matmul(x, w)
    return layer_in


# 假设net中有layer_in,train_state, learning_rate, forward_state,dy这几个参数
bn_operation: bn_layer = bn_layer(layer_in=layer_in, train_state=net.train_state, learning_rate=net.learning_rate,
                                  forward_state=net.forward_state, first=true)
# d_cache中记载上一层训练的一些参数，用于本次计算
# 如果是初始化，要先初始化缓存
if bn_operation.first:
    d_cache = defaultdict(list)
# 后续训练调用之前缓存区的内容帮助计算
x_out, d_cache = bn_operation.output(layer_in=net.layer_in, d_cache=d_cache, dy=net.dy)

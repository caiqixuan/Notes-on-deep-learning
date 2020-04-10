import torch
import torch.nn as nn
import BN_DIY as diy
X=torch.randn(1,3,2,2)
print("The original one:\n",nn.BatchNorm2d(3).forward(X))
print("Mine:\n",diy.BatchNorm(4,3).forward(X))

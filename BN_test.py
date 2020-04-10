import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import BN_DIY as bnd

# 准备数据
# Compose的意思是将多个transform组合在一起用，ToTensor 将像素转化为[0,1]的数字，Normalize则正则化变为 [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 构建卷积神经网络

# 常用的一些功能函数


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        # 最大值池化
        self.maxpool = nn.MaxPool2d(2, 2)
        # 均值池化
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        # BN层
        """
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.bn1 = bnd.BatchNorm(4, 64)
        self.bn2 = bnd.BatchNorm(4, 128)
        self.bn3 = bnd.BatchNorm(4, 256)
        # dropout层防止过拟合
        # self.dropout50 = nn.Dropout(0.5)
        # self.dropout10 = nn.Dropout(0.1)
        """
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        """
        x = self.bn1.forward(F.relu(self.conv1(x)))
        x = self.bn1.forward(F.relu(self.conv2(x)))
        """
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))

        x = self.maxpool(x)
        """
        #x = self.dropout10(x)
        x = self.bn2.forward(F.relu(self.conv3(x)))
        x = self.bn2.forward(F.relu(self.conv4(x)))        
        """
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.avgpool(x)
        """
        #x = self.dropout10(x)
        x = self.bn3.forward(F.relu(self.conv5(x)))
        x = self.bn3.forward(F.relu(self.conv6(x)))
        """
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.globalavgpool(x)
        # x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 实例化网络
net = Net()

# 损失函数定义
criterion = nn.CrossEntropyLoss()
# 优化器定义
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 检测是否有可使用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将网络存上去

net.to(device)
# 开始训练，10个epoch
for epoch in range(10):
    # 初始化一些参数
    running_loss = 0.
    batch_size = 100

    # 分别取出数据和下标
    for i, data in enumerate(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0),
                             0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 将梯度归0
        optimizer.zero_grad()
        # 带入网络，产生输出
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

    print('[%d] loss: %.4f' % (epoch + 1, loss.item()))

print('Finished Training')

torch.save(net, 'D:/cifar10.pkl')
net = torch.load('D:/cifar10.pkl')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

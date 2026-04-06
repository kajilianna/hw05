import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 经典LeNet-5结构适配MNIST（28x28输入）
        # 卷积层1：1输入通道，6输出通道，5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # padding=2保持28x28尺寸
        # 池化层1：2x2平均池化
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层2：6输入通道，16输出通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 池化层2：2x2平均池化
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层1：16*5*5 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2：120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 输出层：84 -> 10（10分类）
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 输入尺寸：[batch, 1, 28, 28]
        x = F.tanh(self.conv1(x))  # 输出：[batch, 6, 28, 28]
        x = self.pool1(x)          # 输出：[batch, 6, 14, 14]
        x = F.tanh(self.conv2(x))  # 输出：[batch, 16, 10, 10]
        x = self.pool2(x)          # 输出：[batch, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5) # 展平
        x = F.tanh(self.fc1(x))    # 输出：[batch, 120]
        x = F.tanh(self.fc2(x))    # 输出：[batch, 84]
        x = self.fc3(x)            # 输出：[batch, 10]
        return x

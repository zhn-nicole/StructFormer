'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from visual import plotter

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
#in_planes：输入通道数。 planes：残差块内部的通道数（conv1输出通道数）。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
#步幅不为 1，分辨率会变化/输入通道数与输出通道数不匹配，需要调整通道数
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  #残差连接要求 shortcut 和主路径的输出形状相同（分辨率和通道数）
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_nc=3, num_classes=10, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        block = BasicBlock  #残差块 得到卷积两次的结果
        num_blocks = [2, 2, 2, 2, 2]

        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        #num_blocks[0]=2 个 BasicBlock 输出：[B, 64, H, W]
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)#步幅2，[B, 128, H/2, W/2]
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#自适应平均池化，将特征图缩小到 1x1，输出：[B, 512, 1, 1]
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, num_classes)
#二层全连接层，输出类别预测。
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        fc7 = out.view(out.size(0), -1)
        out = F.relu(self.linear1(fc7))
        out = self.linear2(out)
        return out
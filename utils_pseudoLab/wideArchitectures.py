# Network from here:
# Interpolation Consistency Training (ICT) for Deep Semi-supervised Learning
# https://github.com/vikasverma1077/ICT/blob/master/networks/wide_resnet.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.nn.utils import weight_norm

act = torch.nn.LeakyReLU()

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_WN(nn.Module):
    def __init__(self, in_planes, planes,dropout_rate, stride=1):
        super(wide_WN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = weight_norm(nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = weight_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(act(self.bn1(x))))
        out = self.conv2(act(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet_WN(nn.Module):

    def __init__(self, block, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_WN, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = weight_norm(conv3x3(3,nStages[0]))
        self.layer1 = self._wide_layer(block, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(block, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(block, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = weight_norm(nn.Linear(nStages[3], num_classes))

    def _wide_layer(self, block, planes, num_blocks,dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup_hidden = False,  mixup_alpha = 0.1, layers_mix=None):
        #print x.shape
        out = x
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def WRN28_2_wn(num_classes=10, dropout = 0.0):
    model = Wide_ResNet_WN(wide_WN, depth =28, widen_factor =2, dropout_rate = dropout, num_classes = num_classes)
    return model


if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())

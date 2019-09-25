'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils import weight_norm


def conv3x3_wn(in_planes, out_planes, stride=1):
    return weight_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))


class PreActBlock_WNdrop(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(PreActBlock_WNdrop, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3_wn(in_planes, planes, stride)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_wn(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ResNet_wn(nn.Module):
    def __init__(self, block, num_blocks, drop_val = 0.0, num_classes=100):
        super(ResNet_wn, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3_wn(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.drop = drop_val
        self.layer1 = self._make_layer(block, 64, num_blocks[0], dropout_rate = self.drop, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], dropout_rate = self.drop, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], dropout_rate = self.drop, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], dropout_rate = self.drop, stride=2)
        self.linear = weight_norm(nn.Linear(512*block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

def PreactResNet18_WNdrop(drop_val = 0.5, num_classes = 100):
    return ResNet_wn(PreActBlock_WNdrop, [2,2,2,2],  drop_val = drop_val, num_classes = num_classes)

def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()

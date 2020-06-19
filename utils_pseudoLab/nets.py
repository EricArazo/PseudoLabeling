import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

from torch.nn.utils import weight_norm


########################### Debugging network ##############################

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear( 5*5*20, 80)
        self.fc2 = nn.Linear(80, self.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 5*5*20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

######################################################################

########################### Moon network ##############################

class moon_net(nn.Module):
    def __init__(self, num_classes=2):
        super(moon_net, self).__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.0)
        self.fc1 = nn.Linear( 2, 50)
        # self.fc2 = nn.Linear( 50, 10)
        self.fc_out = nn.Linear(50, self.num_classes)

    def forward(self, x):
        x = F.relu(self.drop(self.fc1(x)))
        # x = F.relu(self.drop(self.fc2(x)))
        x = self.fc_out(x)
        return x


class moon_net_wn(nn.Module):
    def __init__(self, num_classes=2):
        super(moon_net_wn, self).__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.0)
        self.fc1 = weight_norm(nn.Linear( 2, 50))
        # self.fc2 = nn.Linear( 50, 10)
        self.fc_out = weight_norm(nn.Linear(50, self.num_classes))

    def forward(self, x):
        x = F.relu(self.drop(self.fc1(x)))
        # x = F.relu(self.drop(self.fc2(x)))
        x = self.fc_out(x)
        return x


class moon_net_ICT(nn.Module):
    def __init__(self, num_classes=2):
        super(moon_net_ICT, self).__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.0)
        self.fc1 = nn.Linear( 2, 20)
        self.fc2 = nn.Linear( 20, 20)
        self.fc3 = nn.Linear( 20, 20)
        # self.fc2 = nn.Linear( 50, 10)
        self.fc_out = nn.Linear(20, self.num_classes)

    def forward(self, x):
        x = F.relu(self.drop(self.fc1(x)))
        x = F.relu(self.drop(self.fc2(x)))
        x = F.relu(self.drop(self.fc3(x)))
        # x = F.relu(self.drop(self.fc2(x)))
        x = self.fc_out(x)
        return x


######################################################################

############# Replicating TE network #################################
### Missing: mean-only BN

def conv3x3_TE(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



def conv1x1_TE(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlock_TE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock_TE, self).__init__()
        self.conv1 = conv3x3_TE(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum = 0.999)
        self.conv2 = conv3x3_TE(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum = 0.999)
        self.conv3 = conv3x3_TE(out_planes, out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes, momentum = 0.999)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.lrelu(self.bn3(self.conv3(out)))
        return out

class LastBlock_TE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(LastBlock_TE, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum = 0.999)
        self.conv2 = conv1x1_TE(out_planes, int(out_planes/2))
        self.bn2 = nn.BatchNorm2d(int(out_planes/2), momentum = 0.999)
        self.conv3 = conv1x1_TE(int(out_planes/2), int(out_planes/4))
        self.bn3 = nn.BatchNorm2d(int(out_planes/4), momentum = 0.999)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.lrelu(self.bn3(self.conv3(out)))
        return out

class TE_Net(nn.Module):
    def __init__(self, dropRatio = 0.0, num_classes=10):
        super(TE_Net, self).__init__()
        self.num_classes = num_classes
        self.block1 = BasicBlock_TE(3, 128)
        self.block2 = BasicBlock_TE(128, 256)
        self.block3 = LastBlock_TE(256, 512)
        self.fc = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(p=dropRatio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu') #leaky_relu
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout(F.max_pool2d(self.block1(x), 2))
        x = self.dropout(F.max_pool2d(self.block2(x), 2))
        x = self.block3(x)

        x = F.avg_pool2d(x, 6)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

##############################################################################

############### From "Label propagation..."" #################################


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet18_wndrop(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_wn(BasicBlock_wndrop, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_wn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return weight_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False))



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, isL2 = False):
        self.inplanes = 64
        self.isL2 = isL2


        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.isL2:
            x = F.normalize(x)
        c = self.fc(x)

        # return c , x
        return c

from IPython import embed
class BasicBlock_wndrop(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_wndrop, self).__init__()
        self.conv1 = conv3x3_wn(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_wn(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # embed()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet_wn(nn.Module):

    def __init__(self, block, layers, num_classes=1000, isL2 = False):
        self.inplanes = 64
        self.isL2 = isL2


        super(ResNet_wn, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = weight_norm(nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                weight_norm(conv1x1(self.inplanes, planes * block.expansion, stride)),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.isL2:
            x = F.normalize(x)
        c = self.fc(x)

        # return c , x
        return c




class GaussianNoise(nn.Module):

    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros_ = torch.zeros(x.size()).cuda()
        n = Variable(torch.normal(zeros_, std=self.std).cuda())
        return x + n



# For cifar_cnn
class CNN(nn.Module):
    """
    CNN from Mean Teacher paper
    """

    def __init__(self, num_classes=10, isL2 = False, dropRatio = 0.0):
        super(CNN, self).__init__()

        self.isL2 = isL2

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop1  = nn.Dropout(0.5)
        # self.drop1  = nn.Dropout(dropRatio)
        self.drop  = nn.Dropout(dropRatio)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop2  = nn.Dropout(0.5)
        # self.drop2  = nn.Dropout(dropRatio)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 =  weight_norm(nn.Linear(128, num_classes))
        self.fc2 =  weight_norm(nn.Linear(128, num_classes))

    def forward(self, x, debug=False):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        # x = self.drop1(x)
        x = self.drop(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        # x = self.drop2(x)
        x = self.drop(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)

        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)
        # return self.fc1(x), self.fc2(x), x
        return self.fc1(x)#, self.fc2(x), x

# For cifar_cnn
class CNN_gn(nn.Module):
    """
    CNN from Mean Teacher paper
    """

    def __init__(self, num_classes=10, isL2 = False, dropRatio = 0.0):
        super(CNN_gn, self).__init__()

        self.isL2 = isL2

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop1  = nn.Dropout(0.5)
        # self.drop1  = nn.Dropout(dropRatio)
        self.drop  = nn.Dropout(dropRatio)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop2  = nn.Dropout(0.5)
        # self.drop2  = nn.Dropout(dropRatio)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 =  weight_norm(nn.Linear(128, num_classes))
        self.fc2 =  weight_norm(nn.Linear(128, num_classes))

    def forward(self, x, debug=False):
        x = self.gn(x)
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        # x = self.drop1(x)
        x = self.drop(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        # x = self.drop2(x)
        x = self.drop(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)

        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)
        # return self.fc1(x), self.fc2(x), x
        return self.fc1(x)#, self.fc2(x), x

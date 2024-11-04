import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



__all__ = ['ResNet3D', 'resnet50_3d', 'resnet101_3d']

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x, clin=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3d_init(m):
    assert isinstance(m, nn.Conv3d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))

def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()


class ResNetGN3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetGN3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                conv3d_init(m)
        gn_init(self.bn1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )
            m = downsample[1]
            assert isinstance(m, nn.GroupNorm)
            gn_init(m)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
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
        x = self.fc(x)

        return x


class Net(nn.Module):
    def __init__(self, cf, logger):
        super(Net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.net = ResNetGn(input_shape=cf.input_size, input_channels=cf.input_channel, out_channels=cf.num_classes)
        self.compute_loss_func = nn.CrossEntropyLoss()

    def forward(self, image, labels, cli=None, phase='train'):
        if phase != 'test':
            if cli is None:
                cout = self.net(image, cli=None)
            else:
                cout = self.net(image, cli)

            loss = self.compute_loss_func(cout, torch.argmax(labels, dim=1))
            assert self.cf.num_classes == 2, "The calculations for tp, tn, fp, fn are limited to binary classification."

            predict = F.sigmoid(cout).squeeze(1).cpu().detach().numpy()
            labels = np.array(labels.cpu()) > 0
            th = 0.5
            predict = predict > th
            tp = np.sum((predict == True) & (labels == True))
            tn = np.sum((predict == False) & (labels == False))
            fp = np.sum((predict == True) & (labels == False))
            fn = np.sum((predict == False) & (labels == True))
            acc = (tp + tn) / (tp + tn + fp + fn)

            result_dict = {
                'loss': loss,
                'acc': acc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
            return result_dict
        else:
            if cli is None:
                cout = self.net(image)
            else:
                cout = self.net(image, cli)

            result_dict = {'outputs': cout}
            return result_dict
def resnet50_3d(**kwargs):
    model = ResNetGN3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    return model

def resnet101_3d(**kwargs):
    model = ResNetGN3D(Bottleneck3D, [3, 4, 23, 3], **kwargs)
    return model

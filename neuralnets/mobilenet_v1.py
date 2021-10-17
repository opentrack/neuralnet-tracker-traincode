# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk

Even more modified by M Welter
"""

import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from neuralnets.modelcomponents import GBN


__all__ = ['MobileNet']


class SequeezeAndExcitation(nn.Module):
    # https://arxiv.org/pdf/1709.01507.pdf

    def __init__(self, num_channels, middle_channels, momentum):
        super(SequeezeAndExcitation,self).__init__()
        assert middle_channels > 0
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(num_channels, middle_channels, 1, bias=False),
            GBN(middle_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, num_channels, 1, bias=False),
            GBN(num_channels, momentum=momentum),
        )

    def forward(self, x):
        y = self.layers(x)
        y = torch.sigmoid(y)
        return x.mul(y)

class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, momentum=0.1, sae = False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = GBN(inplanes, momentum=momentum)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = GBN(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            self.skip_layer = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.skip_layer = nn.Identity()
        sae_channels = planes // 16
        sae = sae and sae_channels > 1
        if sae:
            self.sae = SequeezeAndExcitation(inplanes, sae_channels, momentum)
        else:
            self.sae = nn.Identity()

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)
        out = self.conv_sep(out)

        out = self.sae(out)

        out = out + self.skip_layer(x)

        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, widen_factor=1.0, input_channel=1, momentum=0.01, dropout = 0.0, sae=False, return_only_featuremap=False):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = GBN(int(32 * widen_factor), momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, momentum=momentum)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, momentum=momentum)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, sae=sae, momentum=momentum)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, momentum=momentum)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, sae=sae, momentum=momentum)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, momentum=momentum)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, momentum=momentum)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, momentum=momentum)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, momentum=momentum)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, momentum=momentum)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, sae=sae, momentum=momentum)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, momentum=momentum)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, sae=sae, momentum=momentum)

        if not return_only_featuremap:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_classes:
            if dropout>0.:
                self.drop = nn.Dropout(p = dropout)
            else:
                self.drop = nn.Identity()

            self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)
        if not hasattr(self, 'avgpool'):
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if hasattr(self, 'fc'):
            x = self.drop(x)
            x = self.fc(x)

        return x

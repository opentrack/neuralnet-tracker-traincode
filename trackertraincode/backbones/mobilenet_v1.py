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

from typing import Optional
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import dataclasses

# In this paper "Making Convolutional Networks Shift-Invariant Again" from 2019 it was shown that it
# can be beneficial to blur the feature map before downsampling. The idea is to filter high frequency
# content in order to avoid aliasing artifacts. The output of such networks can be more stable w.r.t 
# shifts and even more accurate.
# Hence we add those layers here.
from trackertraincode.neuralnets.modelcomponents import BlurPool2D

NormalizationLayer = nn.BatchNorm2d
ActivationFunc = nn.ReLU

__all__ = ['MobileNet', 'InvMobileNet']


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, momentum=0.1, stochastic_depth=None, use_blurpool=True):
        super(DepthWiseBlock, self).__init__()
        assert stride in (1,2)
        self.inplanes, self.planes = inplanes, planes = int(inplanes), int(planes)
        if stride == 2 and use_blurpool:
            self.conv_dw = nn.Sequential(
                BlurPool2D(kernel_size=3,stride=2, channels=inplanes),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=1, groups=inplanes, bias=False))
        else:
            self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes, bias=False)
        self.bn_dw = NormalizationLayer(inplanes, momentum=momentum)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = NormalizationLayer(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.skip_connection = not (stride != 1 or inplanes != planes)
        self.stochastic_depth = None
        if stochastic_depth:
            self.stochastic_depth = stochastic_depth

    def forward(self, x):
        out = x

        out = self.conv_dw(out)
        out = self.bn_dw(out)
        out = self.relu(out)
        
        out = self.conv_sep(out)
        
        out = self.bn_sep(out)
        
        if self.skip_connection:
            out = self.stochastic_depth(out) if self.stochastic_depth is not None else out
            out = out + x

        out = self.relu(out)

        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, widen_factor=1.0, input_channel=1, momentum=0.1, dropout = 0.0, use_blurpool=True, return_only_featuremap=False):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()

        def block(inplanes, planes, stride=1):
            return DepthWiseBlock(inplanes*widen_factor, planes*widen_factor, stride=stride, momentum=momentum, use_blurpool=use_blurpool)

        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = NormalizationLayer(int(32 * widen_factor), momentum=momentum)
        self.relu = ActivationFunc(inplace=True)

        self.dw2_1 = block(32  , 64)
        self.dw2_2 = block(64  , 128, stride=2)
        self.dw3_1 = block(128 , 128)
        self.dw3_2 = block(128 , 256, stride=2)
        self.dw4_1 = block(256 , 256)
        self.dw4_2 = block(256 , 512, stride=2)
        self.dw5_1 = block(512 , 512)
        self.dw5_2 = block(512 , 512)
        self.dw5_3 = block(512 , 512)
        self.dw5_4 = block(512 , 512)
        self.dw5_5 = block(512 , 512)
        self.dw5_6 = block(512 , 1024, stride=2)
        self.dw6   = block(1024, 1024)

        if not return_only_featuremap:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_features = int(1024*widen_factor)
        self.num_intermediate_features = [
            int(c*widen_factor) for c in [ 64, 128, 256, 512, 1024 ]
        ]

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

        out1 = x = self.dw2_1(x)
        x = self.dw2_2(x)
        out2 = x = self.dw3_1(x)
        x = self.dw3_2(x)
        out3 = x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        out4 = x = self.dw5_5(x)
        x = self.dw5_6(x)
        out5 = x = self.dw6(x)
        if not hasattr(self, 'avgpool'):
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if hasattr(self, 'fc'):
            x = self.drop(x)
            x = self.fc(x)

        return x, [out1, out2, out3, out4, out5 ]

    def prepare_finetune(self):
        return [ [*ch.parameters()] for ch in self.children() ]



class UpsampleBlock(nn.Sequential):
    def __init__(self, inplanes, planes, momentum=0.1):
        inplanes, planes = int(inplanes), int(planes)
        conv_dw = nn.ConvTranspose2d(
            inplanes, inplanes, kernel_size=3, 
            padding=1, stride=2, groups=inplanes,
            bias=False)
        bn_dw = NormalizationLayer(inplanes, momentum=momentum)
        conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        bn_sep = NormalizationLayer(planes, momentum=momentum)
        relu = ActivationFunc(inplace=True)
        super(UpsampleBlock, self).__init__(
            conv_dw,
            bn_dw,
            relu,
            conv_sep,
            bn_sep,
            relu)

class InvMobileNet(nn.Sequential):
    def __init__(self, num_classes, widen_factor=1, momentum=0.1):
        
        block = DepthWiseBlock
        upblk = UpsampleBlock
        
        head = nn.ConvTranspose2d(int(32 * widen_factor), num_classes, kernel_size=3, stride=2, padding=1, bias=True)
        dw2_1 = block(64   * widen_factor, 32   * widen_factor, momentum=momentum)
        dw2_2 = upblk(128  * widen_factor, 64   * widen_factor, momentum=momentum)
        dw3_1 = block(128  * widen_factor, 128  * widen_factor, momentum=momentum)
        dw3_2 = upblk(256  * widen_factor, 128  * widen_factor, momentum=momentum)
        dw4_1 = block(256  * widen_factor, 256  * widen_factor, momentum=momentum)
        dw4_2 = upblk(512  * widen_factor, 256  * widen_factor, momentum=momentum)
        dw5_1 = block(512  * widen_factor, 512  * widen_factor, momentum=momentum)
        dw5_2 = block(512  * widen_factor, 512  * widen_factor, momentum=momentum)
        dw5_3 = block(512  * widen_factor, 512  * widen_factor, momentum=momentum)
        dw5_4 = block(512  * widen_factor, 512  * widen_factor, momentum=momentum)
        dw5_5 = block(512  * widen_factor, 512  * widen_factor, momentum=momentum)
        dw5_6 = upblk(1024 * widen_factor, 512  * widen_factor, momentum=momentum)
        dw6   = block(1024 * widen_factor, 1024 * widen_factor, momentum=momentum)
        super().__init__(*reversed((
            head,
            dw2_1,
            dw2_2,
            dw3_1,
            dw3_2,
            dw4_1,
            dw4_2,
            dw5_1,
            dw5_2,
            dw5_3,
            dw5_4,
            dw5_5,
            dw5_6,
            dw6
        )))
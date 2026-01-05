from typing import List, Literal, NamedTuple
import torchvision.models
import torchvision.models.regnet
from torchvision.models.convnext import LayerNorm2d
from torchvision.ops.misc import Conv2dNormActivation
from torch import nn
import torch
from functools import partial


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        net = torchvision.models.mobilenet_v2(weights=None)

        first_layer = Conv2dNormActivation(
            1, 32, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6
        )

        self._layers = net.features
        self._layers[0] = first_layer

        self.num_features = 1280

    def forward(self, x):
        z = self._layers(x)
        return torch.mean(z, dim=(-2, -1)), None


class MobileNetV3Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        net = torchvision.models.mobilenet_v3_small(weights=None)

        first_layer = Conv2dNormActivation(
            1, 16, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish
        )

        self._layers = net.features
        self._layers[0] = first_layer

        self.num_features = 576

    def forward(self, x):
        z = self._layers(x)
        return torch.mean(z, dim=(-2, -1)), None


class RegNetBackbone(nn.Module):
    def __init__(self, variant: Literal["400mf", "800mf", "1_6gf"]):
        super().__init__()

        ctors = {
            "400mf": torchvision.models.regnet_y_400mf,
            "800mf": torchvision.models.regnet_y_800mf,
            "1_6gf": torchvision.models.regnet_y_1_6gf,
        }
        num_features_by_size = {
            "400mf": 440,
            "800mf": 784,
            "1_6gf": 888,
        }

        net = ctors[variant](pretrained=False)

        self._stem = torchvision.models.regnet.SimpleStemIN(
            1, 32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU
        )
        self._trunk_output = net.trunk_output

        self.num_features = num_features_by_size[variant]

    def forward(self, x):
        z = self._stem(x)
        z = self._trunk_output(z)
        return torch.mean(z, dim=(-2, -1)), None


class ConvNextBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        net = torchvision.models.convnext_tiny(weights=None)

        first_layer = Conv2dNormActivation(
            1, 96, kernel_size=4, stride=4, norm_layer=LayerNorm2d, activation_layer=None
        )

        self._layers = net.features
        self._layers[0] = first_layer

        self.num_features = 768

    def forward(self, x):
        z = self._layers(x)
        return torch.mean(z, dim=(-2, -1)), None

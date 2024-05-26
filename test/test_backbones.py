# coding: utf-8
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytest

from trackertraincode.backbones.efficientnet import EfficientNetBackbone
from trackertraincode.backbones.mobilenet_v1 import MobileNet
from trackertraincode.backbones.resnet import resnet18

@pytest.fixture
def net(request): # Yeah, this parameter must be named "request" else it won't work ...
    net_class, args, kwargs = request.param
    return net_class(*args, **kwargs)


def generate_backbones():
    for kind in ['b0','b3','b4']:
        yield EfficientNetBackbone, (kind,), dict(input_channels=1, weights=None)
    yield MobileNet, (), dict(num_classes=None, input_channel=1)
    yield resnet18, (), {}

@pytest.mark.parametrize("net", generate_backbones(), indirect=True)
def test_backbone_output_api(net):
    #print (f"{type(net)} ... ",end='')
    y, zs = net(torch.zeros((1,1,129,129)))
    assert (y.shape == (1,net.num_features)), f"Bad output shape {y.shape} vs expected {1,net.num_features}"
    if zs is not None:
        expected_sizes = (65,33,17,9,5)
        for expected_feature_dim, expected_size, z in zip(net.num_intermediate_features, expected_sizes, zs):
            expected = (1,expected_feature_dim,expected_size, expected_size)
            assert (z.shape == expected), f"Bad intermediate tensor: {z.shape} vs expected {expected}"
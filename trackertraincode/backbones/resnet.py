from typing import List, NamedTuple, Optional, Callable
import torchvision.models.resnet
from torch import nn
import torch
from functools import partial

# BlurPool2D is the superior downsampling
from trackertraincode.neuralnets.modelcomponents import BlurPool2D


# class CustomBottleneck(torchvision.models.resnet.Bottleneck):
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ):
#         super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
#         width = int(planes * (base_width / 64.0)) * groups
#         self.conv2 = nn.Sequential(
#             BlurPool2D(kernel_size=3, channels=width, stride=stride),
#             torchvision.models.resnet.conv3x3(width, width, 1, groups, dilation)
#         )


class CustomBlock(torchvision.models.resnet.BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(
            inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer
        )
        self.conv1 = nn.Sequential(
            BlurPool2D(kernel_size=3, channels=inplanes, stride=stride),
            torchvision.models.resnet.conv3x3(inplanes, planes, 1, groups, dilation),
        )


class ResNetBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        use_blurpool = kwargs.pop("use_blurpool")

        kwargs["block"] = CustomBlock if use_blurpool else torchvision.models.resnet.BasicBlock

        resnet_factory = torchvision.models.resnet._resnet

        net: "torchvision.models.resnet.ResNet" = resnet_factory(*args, **kwargs)

        if use_blurpool:
            net.maxpool = BlurPool2D(kernel_size=3, channels=64, stride=2)
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        layers = [*net.children()][:-1]
        layers.append(nn.Flatten())

        self.layers = nn.Sequential(*layers)

        self.num_features = 512

        # def grab_output(dest : dict, name : str, model, input, output):
        #     dest[name] = output

        # self.dest = {}

        # self.layers[4].register_forward_hook(partial(grab_output,self.dest, 'z33'))
        # self.layers[5].register_forward_hook(partial(grab_output,self.dest, 'z17'))
        # self.layers[6].register_forward_hook(partial(grab_output,self.dest, 'z9'))
        # self.layers[7].register_forward_hook(partial(grab_output,self.dest, 'z5'))

        # self.num_intermediate_features = [ 64, 126, 256, 512 ]

    def forward(self, x):
        y = self.layers(x)
        # z = [ self.dest[k] for k in 'z33 z17 z9 z5'.split(' ') ]
        # self.dest.clear()
        return y, None


def resnet18(use_blurpool: bool = False):
    net = ResNetBackbone(
        layers=[2, 2, 2, 2],
        weights=None,
        progress=True,
        zero_init_residual=True,
        use_blurpool=use_blurpool,
    )
    return net

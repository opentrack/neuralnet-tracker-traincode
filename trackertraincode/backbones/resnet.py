from typing import List, NamedTuple, Optional, Callable
import torchvision.models.resnet
from torch import nn
import torch
from functools import partial

# BlurPool2D is the superior downsampling
from trackertraincode.neuralnets.modelcomponents import BlurPool2D, UpChannelsSlicingDeterministic


class StrangeCustomBlock(nn.Module):
    expansion: int = 1

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
        use_blurpool: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if use_blurpool:
            self.bp = BlurPool2D(kernel_size=3, channels=inplanes, stride=stride)
            self.conv1 = torchvision.models.resnet.conv3x3(inplanes, planes)
        else:
            self.bp = nn.Identity()
            self.conv1 = torchvision.models.resnet.conv3x3(inplanes, planes, stride=2)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = torchvision.models.resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.bp(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    block = StrangeCustomBlock

    def __init__(
        self,
        layers: List[int],
        input_channels: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        use_blurpool: bool = False,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self._use_blurpool = use_blurpool

        if use_blurpool:
            maxpool = BlurPool2D(kernel_size=3, channels=64, stride=2)
        else:
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        conv1 = nn.Conv2d(
            input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        bn1 = norm_layer(self.inplanes)
        relu = nn.ReLU(inplace=True)
        self.stem = nn.Sequential(conv1, bn1, relu, maxpool)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * self.block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, self.block) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        def grab_output(dest: dict, name: str, model, input, output):
            dest[name] = output

        self.dest = {}

        self.layer1.register_forward_hook(partial(grab_output, self.dest, "z33"))
        self.layer2.register_forward_hook(partial(grab_output, self.dest, "z17"))
        self.layer3.register_forward_hook(partial(grab_output, self.dest, "z9"))
        self.layer4.register_forward_hook(partial(grab_output, self.dest, "z5"))

        self.num_intermediate_features = [x * self.block.expansion for x in [64, 126, 256, 512]]

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * self.block.expansion,
                    kernel_size=1,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.block.expansion),
            )

        layers = []
        layers.append(
            self.block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                1,
                norm_layer,
            )
        )
        self.inplanes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                    use_blurpool=self._use_blurpool,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        z = [self.dest[k] for k in "z33 z17 z9 z5".split(" ")]
        assert z[-1].shape[2] == 5
        self.dest.clear()

        return x, z


def resnet18(use_blurpool: bool = False):
    net = ResNetBackbone(
        layers=[2, 2, 2, 2],
        zero_init_residual=True,
        use_blurpool=use_blurpool,
    )
    return net


def resnet34(use_blurpool: bool = False):
    net = ResNetBackbone(
        layers=[3, 4, 6, 3],
        zero_init_residual=True,
        use_blurpool=use_blurpool,
    )
    return net


def resnet(l1: int, l2: int, l3: int, l4: int, use_blurpool: bool = False):
    net = ResNetBackbone(
        layers=[l1, l2, l3, l4],
        zero_init_residual=True,
        use_blurpool=use_blurpool,
    )
    return net

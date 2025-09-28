from typing import List, NamedTuple
import torchvision.models.resnet
import torchvision.models.efficientnet
from torch import nn
import torch
from functools import partial


class EfficientNetBackbone(nn.Module):
    """
    kind must be in (b0,b3,b4)
    """

    def __init__(self, kind: str, input_channels, *args, return_only_featuremaps=False, **kwargs):
        super().__init__()

        Config = NamedTuple(
            "Config", [("feature_counts", List[int]), ("output_feature_count", int)]
        )
        configs = {
            "b0": Config([16, 24, 40, 112, 320], 1280),
            "b3": Config([24, 32, 48, 136, 384], 1536),
            "b4": Config([24, 32, 56, 160, 448], 1792),
        }
        assert kind in configs.keys()
        self.num_intermediate_features, self.num_features = configs[kind]
        self.return_only_featuremaps = return_only_featuremaps

        pretrained = kwargs.pop("weights", None)
        if pretrained == "pretrained":
            kwargs["weights"] = torchvision.models.efficientnet.EfficientNet_B3_Weights.DEFAULT
        else:
            assert pretrained is None

        ctor = getattr(torchvision.models.efficientnet, "efficientnet_" + kind)

        net: torchvision.models.efficientnet.EfficientNet = ctor(*args, **kwargs)

        self.layers = net.features

        if 0:
            self.layers[0] = torchvision.models.efficientnet.Conv2dNormActivation(
                input_channels,
                40,
                kernel_size=3,
                stride=2,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU,
            )
        else:
            self.to_3chn_input = nn.Conv2d(input_channels, 3, 1)
            self.to_3chn_input.bias.data.zero_()
            self.to_3chn_input.weight.data.fill_(1.0)

        def grab_output(dest: dict, name: str, model, input, output):
            dest[name] = output

        self.dest = {}

        self.layers[1].register_forward_hook(partial(grab_output, self.dest, "z65"))
        self.layers[2].register_forward_hook(partial(grab_output, self.dest, "z33"))
        self.layers[3].register_forward_hook(partial(grab_output, self.dest, "z17"))
        self.layers[5].register_forward_hook(partial(grab_output, self.dest, "z9"))
        self.layers[7].register_forward_hook(partial(grab_output, self.dest, "z5"))

    def forward(self, x):
        x = self.to_3chn_input(x)
        y = self.layers(x)
        z = [self.dest[k] for k in "z65 z33 z17 z9 z5".split(" ")]
        self.dest.clear()
        return y if self.return_only_featuremaps else (torch.mean(y, dim=(-2, -1)), z)

    def prepare_finetune(self) -> List[List[torch.nn.Parameter]]:
        """
        Returns parameters for lr rate scaling
        """
        return [list(c.parameters()) for c in self.layers] + [[*self.to_3chn_input.parameters()]]

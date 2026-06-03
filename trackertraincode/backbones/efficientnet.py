from typing import List, NamedTuple
import torchvision.models.efficientnet
from torch import nn
import torch
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation
from trackertraincode.neuralnets.modelcomponents import BlurPool2D
from trackertraincode.backbones.other_torchvision import _patch_downsampling_conv_norm_act


class EfficientNetBackbone(nn.Module):
    """
    kind must be in (b0,b3,b4)
    """

    def __init__(
        self,
        kind: str,
        input_channels,
        *args,
        return_only_featuremaps=False,
        use_blurpool=False,
        **kwargs,
    ):
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

        # Patch input stage
        tail = self.layers[0]
        assert isinstance(tail, Conv2dNormActivation)
        conv = tail[0]
        assert isinstance(conv, nn.Conv2d)
        assert conv.stride == (2, 2)
        assert conv.kernel_size == (3, 3)
        tail[0] = nn.Conv2d(
            input_channels,
            conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv.bias is not None,
        )
        if use_blurpool:
            tail.insert(1, BlurPool2D(kernel_size=3, stride=2, channels=conv.out_channels))

        if use_blurpool:
            _patch_downsampling_conv_norm_act(self.layers)

        def grab_output(dest: dict, name: str, model, input, output):
            dest[name] = output

        self.dest = {}

        self.layers[1].register_forward_hook(partial(grab_output, self.dest, "z65"))
        self.layers[2].register_forward_hook(partial(grab_output, self.dest, "z33"))
        self.layers[3].register_forward_hook(partial(grab_output, self.dest, "z17"))
        self.layers[5].register_forward_hook(partial(grab_output, self.dest, "z9"))
        self.layers[7].register_forward_hook(partial(grab_output, self.dest, "z5"))

    def forward(self, x):
        y = self.layers(x)
        z = [self.dest[k] for k in "z65 z33 z17 z9 z5".split(" ")]
        self.dest.clear()
        return y if self.return_only_featuremaps else (torch.mean(y, dim=(-2, -1)), z)

    def prepare_finetune(self) -> List[List[torch.nn.Parameter]]:
        """
        Returns parameters for lr rate scaling
        """
        return [list(c.parameters()) for c in self.layers] + [[*self.to_3chn_input.parameters()]]

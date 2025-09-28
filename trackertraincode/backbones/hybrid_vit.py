from typing import List, NamedTuple, Optional, Callable
import torchvision.models.resnet
from torch import nn
import torch
from torch import Tensor


class HybridVitBackbone(nn.Module):
    def __init__(self, num_heads):
        super().__init__()

        resnet = torchvision.models.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT
        )
        head = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        layers = [head, resnet.layer1, resnet.layer2]  # resnet.layer3, resnet.layer4 ]

        resnet = torchvision.models.resnet18(weights=None)
        layers += [resnet.layer3, resnet.layer4]

        self.convnet = nn.Sequential(*layers)
        _, C, H, W = self.convnet(torch.zeros((1, 1, 129, 129))).shape
        # print ("Backbone Channels: ", C, "FM Size: ", H)
        # print ("Final backbone layer: ", layers[-1])

        self.position_enc_dim = 8
        self.num_queries = num_heads
        self.transformer_dim = 256
        self.featuremap_size = H

        self.proj = nn.Sequential(
            nn.Conv2d(C, self.transformer_dim - self.position_enc_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.transformer_dim - self.position_enc_dim),
        )

        self.transformer = nn.Transformer(
            d_model=self.transformer_dim,
            nhead=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=2 * self.transformer_dim,
            dropout=0.1,
            batch_first=True,
        )

        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model = self.transformer_dim,
        #         nhead=4,
        #         dim_feedforward=2*self.transformer_dim,
        #         batch_first=True
        #     ),
        #     num_layers=3,
        #     norm = nn.LayerNorm(self.transformer_dim)
        # )
        # print (self.transformer)
        # self.norm_position_enc = nn.LayerNorm()
        # self.position_x = nn.Parameter(torch.randn((self.position_enc_dim//2,self.featuremap_size), requires_grad=True))
        # self.position_y = nn.Parameter(torch.randn((self.position_enc_dim//2,self.featuremap_size), requires_grad=True))
        self.position = nn.Parameter(
            torch.randn(
                (1, self.position_enc_dim, self.featuremap_size, self.featuremap_size),
                requires_grad=True,
            )
        )
        self.queries = nn.Parameter(
            torch.randn((1, self.num_queries, self.transformer_dim), requires_grad=True)
        )
        self.cls_token = nn.Parameter(torch.randn((1, 1, self.transformer_dim), requires_grad=True))

    def _add_position_encoding(self, z):
        B, C, H, W = z.shape
        # return torch.cat([
        #     z,
        #     self.position_x[None,:,None,:].expand(B,-1,H,W),
        #     self.position_y[None,:,:,None].expand(B,-1,H,W)
        # ], dim=1)
        return torch.cat([z, self.position.expand(B, -1, -1, -1)], dim=1)

    @property
    def num_features(self):
        return self.transformer_dim

    def forward(self, x: Tensor):
        z: Tensor = self.convnet(x)  # Shape with 129x129 input = 17x17 or 9x9
        z = self.proj(z)
        z = self._add_position_encoding(z)
        B, C, H, W = z.shape  # Update size
        z = z.view(B, C, H * W)
        z = z.moveaxis(1, -1)
        z = torch.cat([self.cls_token.expand(B, -1, -1), z], dim=-2)
        # Output shape (B,?,transformer_dim)
        output: Tensor = self.transformer(z, self.queries.expand(B, -1, -1))
        # output = output[:,0]
        return output, None


if __name__ == "__main__":
    x = torch.ones((7, 1, 129, 129))
    net = HybridVitBackbone(num_heads=5)
    print(net(x)[0].shape)

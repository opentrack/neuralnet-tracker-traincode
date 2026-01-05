from typing import List, NamedTuple, Optional, Callable
import torchvision.models.resnet
from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F

from trackertraincode.backbones.resnet import resnet


# Copy paste from https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, nheads=4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        expansion_factor = 2
        self.mha = MultiHeadAttention(
            embed_dim, embed_dim, embed_dim, embed_dim, nheads=nheads, dropout=0.1, bias=False
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, seq: Tensor) -> Tensor:
        z = self.layernorm1(seq)
        z = self.mha(z, z, z)
        z = z + seq
        seq = z
        z = self.layernorm2(seq)
        z = self.ffn(z)
        z = z + seq
        return z


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, nheads=4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        expansion_factor = 2
        self.mha_cross = MultiHeadAttention(
            embed_dim, embed_dim, embed_dim, embed_dim, nheads=nheads, dropout=0.1, bias=False
        )
        self.mha_self = MultiHeadAttention(
            embed_dim, embed_dim, embed_dim, embed_dim, nheads=nheads, dropout=0.1, bias=False
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        z = self.layernorm1(target)
        z = self.mha_self(z, z, z)
        z = z + target

        out = self.mha_cross(self.layernorm2(z), source, source)
        out = out + z

        z = self.layernorm3(out)
        z = self.ffn(z)
        out = z + out

        return out


class HybridVitBackbone(nn.Module):
    def __init__(self, num_heads):
        super().__init__()

        self.convnet = resnet(1, 2, 2, 2, use_blurpool=True)
        _, fms = self.convnet(torch.zeros((1, 1, 129, 129)))
        _, C, H, W = map(int, fms[-1].shape)
        # print("Backbone Channels: ", C, "FM Size: ", H)

        self.position_enc_dim = 8
        self.num_queries = num_heads
        self.transformer_dim = 128
        self.featuremap_size = H

        self.proj = nn.Sequential(
            nn.Conv2d(C, self.transformer_dim - self.position_enc_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.transformer_dim - self.position_enc_dim),
        )

        self.encoder1 = TransformerEncoderLayer(self.transformer_dim)
        self.encoder2 = TransformerEncoderLayer(self.transformer_dim)
        self.encoder_norm = nn.LayerNorm(self.transformer_dim)
        self.decoder1 = DecoderLayer(self.transformer_dim)
        self.decoder2 = DecoderLayer(self.transformer_dim)

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
        _, fms = self.convnet(x)  # Shape with 129x129 input = 17x17 or 9x9
        z = fms[-1]
        z = self.proj(z)
        z = self._add_position_encoding(z)
        B, C, H, W = z.shape  # Update size
        z = z.view(B, C, H * W)
        z = z.moveaxis(1, -1)
        z = torch.cat([self.cls_token.expand(B, -1, -1), z], dim=-2)
        # Output shape (B,?,transformer_dim)
        z = self.encoder1(z)
        z = self.encoder2(z)
        z = self.encoder_norm(z)
        qs = self.queries.expand(B, -1, -1)
        qs = self.decoder1(z, qs)
        qs = self.decoder2(z, qs)
        output: Tensor = qs
        # output = output[:,0]
        assert output.shape == (B, self.num_queries, self.transformer_dim), f"Got {output.shape}"
        return output, None


if __name__ == "__main__":
    x = torch.ones((7, 1, 129, 129))
    net = HybridVitBackbone(num_heads=5)
    print(net(x)[0].shape)

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
import torch.nn.functional as F



def build_pos_embeds(
    num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
    nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

    return pos_emb

def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        encoder_hidden_size: int,
        num_input_tokens: int,
        num_queries: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.num_input_tokens = num_input_tokens
        self.num_queries = num_queries
        self.hidden_size = 1024
        self.output_hidden_size = output_hidden_size

        # pos emb
        self.pos_emb = build_pos_embeds(num_input_tokens, encoder_hidden_size)
        # self.pos_emb = None

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.pos_emb is not None:
            x = x + self.pos_emb

        dtype = x.dtype
        # x = self._forward(x.to(torch.float32))  # (B, L, output_hidden_size)
        x = self._forward(x)

        return x.to(dtype)

class ConvProjector(Projector):
    def _forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x


class CAbstractor(ConvProjector):
    """C-Abstractor"""
    def build_net(self):
        encoder_hidden_size = self.encoder_hidden_size
        hidden_size = self.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = 3
        mlp_depth = 2

        n_queries = self.num_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)


class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


import math

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.output_size = shape  # store the target height and width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.interpolate(x.float(), size=self.output_size, mode='bilinear', align_corners=False).to(dtype=x.dtype).contiguous()
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x
    

class GroupedConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(GroupedConv2d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        self.group_in_dim = in_dim // groups
        self.group_out_dim = out_dim // groups
        
        # Create the linear layers for each group
        self.linears = nn.ModuleList([
            nn.Linear(self.group_in_dim * kernel_size * kernel_size, self.group_out_dim, bias=bias)
            for _ in range(groups)
        ])
        
    def forward(self, x):
        batch_size, in_dim, height, width = x.shape
        
        # Add padding
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # Unfold the input tensor to get sliding local blocks
        x_unfolded = F.unfold(x, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride)
        
        # Reshape to separate out the groups
        x_unfolded = x_unfolded.view(batch_size, self.groups, self.group_in_dim, self.kernel_size * self.kernel_size, -1)
        x_unfolded = x_unfolded.permute(0, 1, 4, 2, 3).contiguous()
        x_unfolded = x_unfolded.view(batch_size * self.groups, -1, self.group_in_dim * self.kernel_size * self.kernel_size)
        
        # Apply the linear layers to each group
        outputs = []
        for i in range(self.groups):
            output = self.linears[i](x_unfolded[i::self.groups])
            outputs.append(output)
        
        # Concatenate the outputs
        out = torch.cat(outputs, dim=1)
        
        # Reshape back to image dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.view(batch_size, self.out_dim, out_height, out_width)
        
        return out


class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            # nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
            GroupedConv2d(in_dim, out_dim, 3, stride=stride, padding=1, groups=out_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class LDPNetV2Projector(nn.Module):
    """Modified based on 
    https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py#L90

    """
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        out_side_shape = int(math.sqrt(config.image_token_len))
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((out_side_shape, out_side_shape))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x
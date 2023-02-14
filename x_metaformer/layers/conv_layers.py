import math
from collections.abc import Iterable
from itertools import repeat
import torch
import torch.nn as nn
from x_metaformer.layers.norm_layers import ConvLayerNorm


def _pair(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))


class Conv2dSame(nn.Conv2d):
    # logic taken from https://github.com/pytorch/pytorch/issues/3867#issuecomment-482711125
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input.shape
        H2 = math.ceil(H / self.stride[0])
        W2 = math.ceil(W / self.stride[1])
        Pr = (H2 - 1) * self.stride[0] + (self.kernel_size[0] - 1) * self.dilation[0] + 1 - H
        Pc = (W2 - 1) * self.stride[1] + (self.kernel_size[0] - 1) * self.dilation[1] + 1 - W
        x_pad = nn.ZeroPad2d((Pr//2, Pr - Pr//2, Pc//2, Pc - Pc//2))(input)
        x_out = self._conv_forward(x_pad, self.weight, self.bias)
        return x_out


class MBConv(nn.Module):
    def __init__(self, dim, expansion=2, kernel_size=3, act=nn.GELU, **kwargs):
        super().__init__()
        med_channels = int(expansion * dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, med_channels, (1, 1), padding=0, bias=False),  # pointwise 1x1 in
            act(),
            nn.Conv2d(med_channels, med_channels, kernel_size,            # depthwise
                      padding='same', groups=med_channels, bias=False),
            nn.Conv2d(med_channels, dim, (1, 1), padding=0, bias=False)   # pointwise 1x1 out
        )

    def forward(self, x):
        return self.conv(x)


class ConvDownsampling(Conv2dSame):
    def __init__(self, *args, norm=ConvLayerNorm, pre_norm=False, post_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_in  = norm(self.in_channels) if pre_norm else nn.Identity()
        self.norm_out = norm(self.out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        return self.norm_out(
            super().forward(
                self.norm_in(x)
            )
        )



if __name__ == '__main__':
    import torch
    import torch.nn as nn


    x = torch.randn(32, 126, 22, 22)
    us = ConvDownsampling(126, 126, kernel_size=3, stride=2)

    print(x.shape, '-->', us(x).shape)
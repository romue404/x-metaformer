import torch.nn as nn
from functools import partial


class MBConv(nn.Module):
    def __init__(self, dim, expansion=2, kernel_size=3, act=nn.GELU, feature_dim=1, **kwargs):
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


class LearnableUpDownsampling(nn.Module):
    UP, DOWN = 'up', 'down'

    def __init__(self, mode, in_channels, out_channels, norm, kernel_size=3, stride=2, pre_norm=False, post_norm=False):
        super(LearnableUpDownsampling, self).__init__()
        assert mode in [self.UP, self.DOWN], f'mode must either be "{self.UP}" or "{self.DOWN}"'
        kernel_size = (kernel_size, kernel_size) if not isinstance(kernel_size, tuple) else kernel_size
        stride = (stride, stride) if not isinstance(stride, tuple) else stride
        padding = (kernel_size[0] // stride[0], kernel_size[1] // stride[1])
        conv_cls = partial(nn.ConvTranspose2d, output_padding=padding) if mode == 'up' else nn.Conv2d
        self.conv = nn.Sequential(
            norm(in_channels=in_channels) if pre_norm else nn.Identity(),
            conv_cls(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            norm(in_channels=in_channels) if post_norm else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


Upsampling   = partial(LearnableUpDownsampling, LearnableUpDownsampling.UP)
Downsampling = partial(LearnableUpDownsampling, LearnableUpDownsampling.DOWN)

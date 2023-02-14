import numpy as np
import torch
import torch.nn as nn
from x_metaformer.layers.conv_layers import MBConv, ConvDownsampling
from x_metaformer.layers.mlp_layers import MLPConv
from x_metaformer.layers.attention_layers import AttentionConv
from x_metaformer.layers.act_layers import StarReLU, ReLUSquared, StarREGLU, DropPath, PatchMasking2D
from x_metaformer.layers.norm_layers import ConvLayerNorm, RMSNorm
from functools import partial
from abc import ABC, abstractmethod
from inspect import signature


class MetaFormerBlock(nn.Module):
    def __init__(self, dim, depth, mixer, drop_probs, norm=ConvLayerNorm,
                 mlp_act=ReLUSquared, mlp_expansion=4, mlp_dropout=0.1, **mixer_kwargs):
        super(MetaFormerBlock, self).__init__()
        self.depth = depth
        self.dim = dim
        self.drop_probs = drop_probs

        self.mixer = nn.ModuleList(
            [nn.Sequential(
                norm(dim),
                mixer(dim, **mixer_kwargs)
            ) for _ in range(depth)]
        )

        self.mlp = nn.ModuleList(
            [nn.Sequential(
                norm(dim),
                MLPConv(dim, mlp_expansion, mlp_dropout, mlp_act)
            ) for _ in range(depth)]
        )

        self.drop_paths      = nn.ModuleList([DropPath(p) for p in drop_probs])
        self.res_scale_mixer = nn.Parameter(torch.ones(depth, dim, 1, 1), requires_grad=True)
        self.res_scale_mlp   = nn.Parameter(torch.ones(depth, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        for i in range(self.depth):
            x = self.res_scale_mixer[i] * x + self.drop_paths[i](self.mixer[i](x))
            x = self.res_scale_mlp[i]   * x + self.drop_paths[i](self.mlp[i](x))
        return x


def posemb_sincos_2d(patches, temperature = 10000):
    b, dim, h, w, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype).view(1, dim, h, w)


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.kernel_size != (1, 1):
            nn.init.kaiming_normal_(m.weight)
        else:
            nn.init.trunc_normal_(m.weight, std=.02)


class MetaFormerABC(nn.Module, ABC):
    def __init__(self,
                 in_channels,
                 mixers,
                 depths=(3, 3, 9, 3),
                 dims=(64, 128, 256, 320),
                 norm='ln',
                 use_starreglu=False,
                 patchmasking_prob=0.0,
                 **kwargs
                 ):
        super().__init__()
        assert 2 <= len(dims) == len(depths) >= 2
        self.in_channels = in_channels
        self.mixers = mixers
        self.dims = dims
        self.depths = depths
        self.use_starreglu = use_starreglu
        self.act = StarReLU if not use_starreglu else StarREGLU
        self.patchmasking = PatchMasking2D(dims[0], 1, patchmasking_prob) \
            if patchmasking_prob > 0 else nn.Identity()

        self.norm_inner, self.norm_out = self.get_norm(norm)

        self.pooling: nn.ModuleList
        self.blocks: nn.ModuleList

        self.out_dim = dims[-1]
        self.norm_out = self.norm_out(self.out_dim)

    @abstractmethod
    def pool(self, x):
        pass

    def forward(self, x, return_embeddings=False):

        x = self.pooling[0](x)
        x = self.patchmasking(x)
        x = x + (posemb_sincos_2d(x) if self.use_pos_emb else 0)

        for i in range(1, len(self.blocks)):
            x = self.pooling[i](x)
            x = self.blocks[i](x)
        pooled = self.norm_out(self.pool(x))

        return pooled if not return_embeddings else (pooled, x)

    def get_norm(self, mode: str):
        if mode == 'ln':
            return ConvLayerNorm, nn.LayerNorm
        elif mode == 'rms':
            return partial(RMSNorm, dim=(-2, -1)), partial(RMSNorm, dim=(1, ))
        elif mode == 'bn':
            return nn.BatchNorm2d, nn.BatchNorm1d
        else:
            raise NotImplemented('Norm must be "ln", "rms" or "bn"')


class MetaFormer(MetaFormerABC):
    def __init__(self,
                 *args,
                 init_kernel_size=3,
                 init_stride=2,
                 drop_path_rate=0.3,
                 use_pos_emb=True,
                 use_dual_patchnorm=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.init_kernel_size = init_kernel_size
        self.init_stride = init_stride
        self.drop_path_rate = drop_path_rate
        self.use_pos_emb = use_pos_emb

        new_args = set(kwargs.keys()) - set(signature(super().__init__).parameters.keys())
        new_args = {k: v for k, v in kwargs.items() if k in new_args}
        self.mixer_kwargs = new_args  # every kwarg that is not already in the args of MetaFormerABC

        init_downsampling = ConvDownsampling(self.in_channels, self.dims[0],
                                             self.init_kernel_size, self.init_stride,
                                             norm=self.norm_inner,
                                             pre_norm=use_dual_patchnorm,
                                             post_norm=use_dual_patchnorm
                                             )

        dp_rates = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        dix = np.cumsum((0, *self.depths))

        self.pooling = nn.ModuleList([init_downsampling] + [
            ConvDownsampling(self.dims[i], self.dims[i+1],
                             kernel_size=3, stride=2,
                             pre_norm=True, norm=self.norm_inner) for i in range(len(self.dims)-1)
        ])

        self.blocks = nn.ModuleList([
            MetaFormerBlock(self.dims[i], depth=self.depths[i], mixer=self.mixers[i], norm=self.norm_inner,
                            mlp_act=self.act, act=StarReLU, drop_probs=dp_rates[dix[i]: dix[i+1]], **self.mixer_kwargs)
            for i in range(len(self.depths))
        ])

        self.apply(_init_weights)

    def pool(self, x):
        return x.mean([-2, -1])



class ConvFormer(MetaFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         mixers=(MBConv, MBConv, MBConv, MBConv),
                         **kwargs)


class CAFormer(MetaFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         mixers=(MBConv, MBConv, AttentionConv, AttentionConv),
                         **kwargs)


if __name__ == '__main__':
    x = torch.randn(64, 3, 64, 64)
    encoder = CAFormer(3, norm='ln', depths=(2, 2, 4, 2),
                       dims=(16, 32, 64, 128), init_kernel_size=3, init_stride=2, patchmasking_prob=0.2, dual_patchnorm=True)
    codes = encoder(x, return_embeddings=True)[-1]
    print('CODES', codes.shape)
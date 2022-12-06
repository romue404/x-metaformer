import numpy as np
import torch
import torch.nn as nn
from x_metaformer.layers.conv_layers import MBConv, Downsampling, Upsampling
from x_metaformer.layers.mlp_layers import MLPConv
from x_metaformer.layers.attention_layers import AttentionConv
from x_metaformer.layers.act_layers import StarReLU, ReLUSquared, DropPath
from x_metaformer.layers.norm_layers import ConvLayerNorm, RMSNorm
from functools import partial
from abc import ABC, abstractmethod


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
                 in_channels, mixers, depths, dims,
                 init_kernel_size, init_stride, drop_path_rate, norm, use_pos_emb
                 ):
        super().__init__()
        assert 2 <= len(dims) == len(depths) >= 2
        self.use_pos_emb = use_pos_emb
        self.in_channels = in_channels
        self.mixers = mixers
        self.dims = dims
        self.init_kernel_size = init_kernel_size
        self.init_stride = init_stride
        self.drop_path_rate = drop_path_rate
        self.depths = depths

        self.norm_inner, self.norm_out = self.get_norm(norm)

        self.pooling: nn.ModuleList
        self.blocks: nn.ModuleList

        self.out_dim = dims[-1]
        self.norm_out = self.norm_out(self.out_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    def get_norm(self, mode):
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
                 in_channels,
                 mixers,
                 depths=(3, 3, 9, 3),
                 dims=(64, 128, 320, 512),
                 init_kernel_size=3,
                 init_stride=2,
                 drop_path_rate=0.3,
                 norm='ln',
                 use_pos_emb=True,
                 **mixer_kwargs):
        super(MetaFormer, self).__init__(in_channels, mixers, depths, dims,
                                         init_kernel_size, init_stride,
                                         drop_path_rate, norm, use_pos_emb)
        self.mixer_kwargs = mixer_kwargs

        init_downsampling = Downsampling(in_channels, dims[0],
                                         init_kernel_size, init_stride,
                                         norm=self.norm_inner,
                                         pre_norm=False, post_norm=False
                                         )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dix = np.cumsum((0, *depths))

        self.pooling = nn.ModuleList([init_downsampling] + [
            Downsampling(dims[i], dims[i+1], pre_norm=True, norm=self.norm_inner) for i in range(len(dims)-1)
        ])

        self.blocks = nn.ModuleList([
            MetaFormerBlock(dims[i], depth=depths[i], mixer=mixers[i], norm=self.norm_inner,
                            act=StarReLU, drop_probs=dp_rates[dix[i]: dix[i+1]], **mixer_kwargs)
            for i in range(len(depths))
        ])

        self.apply(_init_weights)

    def pool(self, x):
        return x.mean([-2, -1])

    def forward(self, x, return_embeddings=False):
        for i in range(len(self.blocks)):
            x = self.pooling[i](x)
            if i == 0 and self.use_pos_emb:
                x = x + posemb_sincos_2d(x)
            x = self.blocks[i](x)
        pooled = self.norm_out(self.pool(x))
        return pooled if not return_embeddings else (pooled, x)

    def get_decoder(self):
        return MetaFormerDecoder(
            in_channels=self.dims[-1],
            out_channels=self.in_channels,
            mixers=self.mixers[::-1][1:],
            depths=self.depths[::-1][1:],
            dims=self.dims[::-1][1:],
            final_kernel_size=self.init_kernel_size,
            final_stride=self.init_stride,
            **self.mixer_kwargs
        )


MetaFormerEncoder = MetaFormer


class MetaFormerDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mixers,
                 depths,
                 dims,
                 final_kernel_size=3,
                 final_stride=2,
                 **mixer_kwargs
                 ):
        super(MetaFormerDecoder, self).__init__()

        first_upsampling = Upsampling(in_channels, dims[0], pre_norm=False, post_norm=False)
        final_upsampling = Upsampling(dims[-1], out_channels, final_kernel_size, final_stride, pre_norm=True)

        self.pooling = nn.ModuleList(
            [first_upsampling] +
            [Upsampling(dims[i], dims[i+1], pre_norm=True) for i in range(0, len(dims)-1)] +
            [final_upsampling]
        )

        self.blocks = nn.ModuleList([
            MetaFormerBlock(dims[i], depth=depths[i], mixer=mixers[i],
                            act=StarReLU, drop_probs=[0]*depths[i], **mixer_kwargs)
            for i in range(0, len(depths))
        ])

        self.out_dim = out_channels
        self.apply(_init_weights)

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.pooling[i](x)  # upsampling
            x = self.blocks[i](x)
        return self.pooling[-1](x)


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
    x = torch.randn(64, 3, 32, 32)
    encoder = CAFormer(3, norm='ln')
    codes = encoder(x, return_embeddings=True)[-1]
    decoder = encoder.get_decoder()
    print(f'{codes.shape} --> {decoder(codes).shape}')
    print('TinyTest successful')
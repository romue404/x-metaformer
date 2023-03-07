import torch
import torch.nn as nn
from functools import partial
from x_metaformer.layers.mlp_layers import GeneralizedLinear


def _get_agg_dims(channel_loc):
    return (2, 3) if channel_loc == 1 else (1, 2)


def _flatten(x, channel_loc):
    # assert channel_loc in [1, -1]
    x_shape = x.shape
    if len(x_shape) >= 4:
        s, e = _get_agg_dims(channel_loc)
        undo = nn.Unflatten(-channel_loc, x_shape[s: e+1])
        return x.flatten(s, e), undo, (s, e)
    return x, nn.Identity(), -channel_loc


class FNet(nn.Module):
    def __init__(self, channel_loc, *args, **kwargs):
        super().__init__()

        self.channel_dim = channel_loc

    def forward(self, x):
        x, unflatten, _ = _flatten(x, self.channel_dim)  # B C H W  or B H W C
        fft_hidden = torch.fft.fft(x, dim=self.channel_dim)
        fft_seq = torch.fft.fft(fft_hidden, dim=-self.channel_dim)
        out = fft_seq.real
        out = unflatten(out)
        return out


FNetConv = partial(FNet, 1)


class SeqPool(nn.Module):
    # https://arxiv.org/pdf/2104.05704.pdf
    def __init__(self, channel_loc, in_dim, *args, **kwargs):
        super().__init__()
        assert channel_loc in [1, -1]
        self.channel_dim = channel_loc
        self.score = GeneralizedLinear(in_features=in_dim, out_features=1, feature_dim=channel_loc)

    def forward(self, x):
        raw_scores, unflatten, dims = _flatten(self.score(x), self.channel_dim)
        weights = unflatten(raw_scores.softmax(self.channel_dim))
        pooled = (x * weights).sum(dims)
        return pooled


class MeanPool(nn.Module):
    def __init__(self, channel_loc, *args, **kwargs):
        super().__init__()
        self.dim = _get_agg_dims(channel_loc)

    def forward(self, x):
        return x.mean(self.dim)


SeqPoolConv  = partial(SeqPool,  1)
MeanPoolConv = partial(MeanPool, 1)


if __name__ == '__main__':
    seqpool = SeqPoolConv(32)
    batch = torch.randn(64, 32, 6, 6)
    print(seqpool(batch).shape)

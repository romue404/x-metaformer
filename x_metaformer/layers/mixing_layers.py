import torch
import torch.nn as nn
from functools import partial

class FNet(nn.Module):
    def __init__(self, channel_loc, *args, **kwargs):
        super().__init__()
        assert channel_loc in [1, -1]
        self.channel_dim = channel_loc

    def forward(self, x):
        x_shape = x.shape
        s, e = (2, 3) if self.channel_dim == 1 else (1, 2)
        x = x.flatten(s, e)  # B C H W  or B H W C
        fft_hidden = torch.fft.fft(x, dim=self.channel_dim)
        fft_seq = torch.fft.fft(fft_hidden, dim=-self.channel_dim)
        out = fft_seq.real
        out = out.unflatten(-self.channel_dim, x_shape[s:e+1])
        return out


FNetConv = partial(FNet, 1)
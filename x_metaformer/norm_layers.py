import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, shape=None, dim=(-1, -2), eps=1e-6):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(self.dim, keepdims=True) + self.eps)
        scale = self.scale.view_as(rms[0]).unsqueeze(0)
        x_bar = x.div(rms) * scale
        return x_bar


class ConvLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-6)
        self.ln.bias.requires_grad = False
        self.ln.bias.data.mul_(0)

    def forward(self, x):
        return self.norm(x)
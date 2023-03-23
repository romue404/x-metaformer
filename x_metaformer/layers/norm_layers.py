import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, shape=None, dim=(-1, -2), eps=1e-6):
        super().__init__()
        self.dim = dim
        self.shape = shape
        self.scale = nn.Parameter(torch.ones(shape if len(dim) >= 2 else 1))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(self.dim, keepdims=True) + self.eps)
        scale = self.scale.view_as(rms[0]).unsqueeze(0)
        x_bar = x.div(rms) * scale
        return x_bar


class ConvLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-6)
        self.norm.bias.requires_grad = False
        self.norm.bias.data.mul_(0)

    def forward(self, x):
        return self.norm(x)


class GRN(nn.Module):
    # https://arxiv.org/abs/2301.00808
    def __init__(self, feature_dim, dim):
        super().__init__()
        assert feature_dim in [-1, 1]
        self.feature_dim = feature_dim
        dims = [1]*4
        dims[feature_dim] = dim
        self.gamma = nn.Parameter(torch.zeros(*dims))
        self.beta = nn.Parameter(torch.zeros(*dims))

    def forward(self, x):
        dims = (-2, -1) if self.feature_dim == -1 else (1, 2)
        Gx = torch.norm(x, p=2, dim=dims, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-self.feature_dim, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


if __name__ == '__main__':
    norm = GRN(1, 64)
    batch = torch.randn(12, 64, 8, 8)
    out = norm(batch)
    print(out.shape)

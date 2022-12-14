import torch
import torch.nn as nn
from functools import partial


class ReLUSquared(nn.ReLU):
    """
        ReLUSquared: relu(x) ** 2
        Proposed in https://arxiv.org/abs/2109.08668v2
    """
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)**2


class StarReLU(ReLUSquared):
    """
        StarReLU: s * relu(x) ** 2 + b
        Proposed in https://arxiv.org/abs/2210.13452
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * super().forward(x) + self.bias


class GatedActFn(nn.Module):
    """
    Gated activation functions
    Proposed in https://arxiv.org/abs/2002.05202
    """
    def __init__(self, act, dim=-1):
        super(GatedActFn, self).__init__()
        self.act = act
        self.dim = dim

    def forward(self, x):
        x, gates = x.chunk(2, dim=self.dim)
        return x * self.act(gates)


GEGLU       = partial(GatedActFn, nn.GELU())
REGLU       = partial(GatedActFn, nn.ReLU())
StarREGLU   = partial(GatedActFn, StarReLU())
RSEGLU      = partial(GatedActFn, ReLUSquared())
SwishGLU    = partial(GatedActFn, nn.SiLU())


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchMasking2D(nn.Module):
    def __init__(self, dim, feature_dim, p):
        super().__init__()
        assert 0 <= p < 1, 'p does not satisfy 0 <= p < 1.'
        assert feature_dim in [1, -1], 'feature_dim must be either 1 or -1'
        self.p = p
        self.feature_dim = feature_dim
        self.mask_token = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        b, (r, c) = x.shape[0], x.shape[1:3] if self.feature_dim == -1 else x.shape[2:]
        num_patches_drop = max(1, int(r * c * self.p))

        b_rand, r_rand, c_rand = (
            torch.randint(low=0, high=high, size=(num_patches_drop, ), device=x.device) for high in (b, r, c)
        )

        if self.feature_dim == 1:
            x[b_rand, :, r_rand, c_rand] = self.mask_token
        elif self.feature_dim == -1:
            x[b_rand, r_rand, c_rand]    = self.mask_token

        return x

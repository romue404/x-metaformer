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

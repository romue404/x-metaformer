import torch.nn as nn


class GeneralizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, feature_dim=-1):
        super().__init__()
        assert feature_dim in [1, -1], 'Only first or last dimension supported'
        self.linear = nn.Linear(in_features, out_features, bias=bias) if feature_dim == -1 else \
                      nn.Conv2d(in_features, out_features, (1, 1), padding=0, bias=bias)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, dim, expansion=4, p_dropout=0.1, act=nn.GELU, bias=False, feature_dim=-1):
        super().__init__()
        med_channels = int(expansion * dim)
        self.mlp = nn.Sequential(
            GeneralizedLinear(dim, med_channels, bias, feature_dim),
            act(),
            nn.Dropout(p_dropout),
            GeneralizedLinear(dim, med_channels, bias, feature_dim),
            nn.Dropout(p_dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class MLPConv(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, feature_dim=1, **kwargs)


class FeatureRefiner(nn.Module):
    # https://arxiv.org/pdf/2210.05657.pdf
    def __init__(self, in_features, hidden_dim=64, act=nn.ReLU):
        super(FeatureRefiner, self).__init__()
        self.in_features = in_features
        self.hidde_dim = hidden_dim
        self.refiner = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        return self.refiner(x)

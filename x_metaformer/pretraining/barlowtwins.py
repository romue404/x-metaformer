import torch
import torch.nn as nn
from x_metaformer import MetaFormer
from x_metaformer.pretraining.mocov3 import MoCoV3


class BarlowTwins(nn.Module):
    def __init__(self,
                 backbone,
                 mlp_in_dim,
                 expansion=4
                 ):
        super().__init__()
        self.backbone = backbone
        self.mlp_in_dim = mlp_in_dim
        if isinstance(self.backbone, MetaFormer):
            self.backbone.norm_out = nn.Identity()

        mlp_hidden_dim = int(expansion * mlp_in_dim)
        self.projector = MoCoV3.make_head(3, mlp_in_dim, mlp_hidden_dim, mlp_hidden_dim, False)

        self.bn = nn.BatchNorm1d(mlp_hidden_dim, affine=False)

    def forward(self, x1, x2):

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # empirical cross-correlation matrix
        c = (self.bn(z1).T @ self.bn(z2)) / x1.shape[0]  # d x d
        n = c.shape[0]

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diag(c).pow_(2).sum()
        loss = (1.0 / n) * on_diag + (1.0 / (n*(n-1))) * off_diag
        return loss

    @classmethod
    def off_diag(cls, x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == '__main__':
    from x_metaformer import CAFormer
    caf = CAFormer(in_channels=1, dims=(16, 32, 54, 128))
    barlow = BarlowTwins(caf, mlp_in_dim=128)
    batch1 = torch.randn(16, 1, 64, 64)
    batch2 = torch.randn(16, 1, 64, 64)
    print(f'TinyTest successful - loss: {barlow(batch1, batch2).item():.3f}')



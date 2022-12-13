import torch
import torch.nn as nn
from x_metaformer.pretraining.barlowtwins import BarlowTwins
import torch.nn.functional as F


class VICReg(BarlowTwins):
    def __init__(self,
                 *args,
                 std_coeff=25.0,
                 sim_coeff=25.0,
                 cov_coeff=1.0,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.std_coeff = std_coeff
        self.sim_coeff = sim_coeff
        self.cov_coeff = cov_coeff

        self.bn = None

    def forward(self, x1, x2):

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        b, d = z1.shape

        var_term = 0.5 * F.relu(1.0 - z1.std(0)).mean() + \
                   0.5 * F.relu(z2.std(0)).mean()

        cov_var_term = self.off_diag(z1.cov()).pow_(2).sum().div(d) + \
                       self.off_diag(z2.cov()).pow_(2).sum().div(d)

        invariance_term = F.mse_loss(z1, z2)

        return self.std_coeff * var_term + self.sim_coeff * invariance_term + self.cov_coeff * cov_var_term


if __name__ == '__main__':
    from x_metaformer import CAFormer
    caf = CAFormer(in_channels=1, dims=(16, 32, 54, 128))
    barlow = VICReg(caf, mlp_in_dim=128)
    batch1 = torch.randn(16, 1, 64, 64)
    batch2 = torch.randn(16, 1, 64, 64)
    print(f'TinyTest successful - loss: {barlow(batch1, batch2).item():.3f}')
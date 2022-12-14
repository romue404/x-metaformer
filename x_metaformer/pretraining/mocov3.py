import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from x_metaformer import MetaFormer


class MoCoV3(nn.Module):
    def __init__(self,
                 student,
                 teacher,
                 mlp_in_dim,
                 temperature=1.0,
                 momentum=0.996,
                 mlp_hidden_dim=2048,
                 mlp_out_dim=256
                 ):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self.temperature = temperature
        self.momentum = momentum
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_out_dim = mlp_out_dim

        self.proj_s = self.make_head(3, mlp_in_dim, mlp_out_dim, mlp_hidden_dim, True)
        self.proj_t = self.make_head(3, mlp_in_dim, mlp_out_dim, mlp_hidden_dim, True)
        self.pred   = self.make_head(2, mlp_out_dim, mlp_out_dim, mlp_hidden_dim, False)

        self.ce = nn.CrossEntropyLoss()

        self.setup()

    @torch.inference_mode()
    def keys(self, x1, x2):
        k1, k2 = self.proj_t(self.teacher(x1)), self.proj_t(self.teacher(x2))
        return k1, k2

    def queries(self, x1, x2):
        q1, q2 = self.pred(self.proj_s(self.student(x1))), self.pred(self.proj_s(self.student(x2)))
        return q1, q2

    def forward(self, x1, x2):

        q1, q2 = self.keys(x1, x2)
        self.update()  # such that updates ar after .backward() call
        k1, k2 = self.keys(x1, x2)

        q1, q2, k1, k2 = (F.normalize(x, dim=-1, p=2) for x in (q1, q2, k1, k2))

        logits1 = (q1 @ k2.T) / self.temperature
        logits2 = (q2 @ k1.T) / self.temperature

        labels = torch.arange(0, logits1.shape[0], device=q1.device)

        l1, l2 = self.ce(logits1, labels), self.ce(logits2, labels)

        loss = 2.0 * self.temperature * (l1 + l2)

        return loss

    @torch.inference_mode()
    def update(self):
        for s, t in zip(chain(self.student.parameters(), self.proj_s.parameters()),
                        chain(self.teacher.parameters(), self.proj_t.parameters())):
            t.data = t.data * self.momentum + s.data * (1. - self.momentum)

    def setup(self):
        if isinstance(self.student, MetaFormer):
            self.teacher.norm_out = nn.Identity()
            self.student.norm_out = nn.Identity()
        for param_s, param_t in zip(chain(self.student.parameters(), self.proj_s.parameters()),
                                    chain(self.teacher.parameters(), self.proj_t.parameters())):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @classmethod
    def make_head(cls, n_layers, in_dim, out_dim, hidden_dim, bn_last=False):
        layers = [nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        for i in range(n_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                   nn.BatchNorm1d(out_dim, affine=False) if bn_last else nn.Identity()
                   ]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    from x_metaformer import CAFormer
    caf = CAFormer(in_channels=1, dims=(16, 32, 54, 128))
    caf2 = CAFormer(in_channels=1, dims=(16, 32, 54, 128))
    mocov3 = MoCoV3(caf, caf2, mlp_in_dim=128)
    batch1 = torch.randn(16, 1, 64, 64)
    batch2 = torch.randn(16, 1, 64, 64)
    print(f'TinyTest successful - loss: {mocov3(batch1, batch2).item():.3f}')
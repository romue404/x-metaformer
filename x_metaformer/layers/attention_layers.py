import torch
import torch.nn as nn
import torch.nn.functional as F
from x_metaformer.layers.mlp_layers import GeneralizedLinear
from functools import partial


class Attention(nn.Module):
    def __init__(self,
                 channel_loc,
                 dim,
                 head_dim=32,
                 num_heads=4,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 scale_value=1.0,
                 num_mem_vecs=0,
                 sparse_topk=0,
                 l2=False,
                 trainable_scale=False,
                 improve_locality=False,
                 **kwargs):
        super().__init__()
        assert channel_loc in [1, -1], 'Token dimension must be 1 or -1.]'
        assert scale_value > 0, 'scale_value must be > 0'
        self.num_heads       = num_heads
        self.trainable_scale = trainable_scale
        self.head_dim        = head_dim
        self.dim             = dim
        self.channel_dim  = channel_loc
        self.sparse_topk  = sparse_topk
        self.num_mem_vecs = num_mem_vecs
        self.improve_locality = improve_locality
        self.l2 = l2
        self.qkv = GeneralizedLinear(in_features=self.dim,
                                     out_features=3*self.head_dim*self.num_heads,
                                     feature_dim=channel_loc,
                                     bias=False
                                     )
        self.proj_out = GeneralizedLinear(in_features=self.head_dim*self.num_heads,
                                          out_features=self.dim,
                                          feature_dim=channel_loc,
                                          bias=False
                                          )
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1).mul_(scale_value).float(),
                                  requires_grad=trainable_scale)
        if num_mem_vecs > 0:
            self.mem_k = nn.Parameter(torch.randn(num_heads, num_mem_vecs, head_dim))
            self.mem_v = nn.Parameter(torch.randn(num_heads, num_mem_vecs, head_dim))

    def reshape_qkv(self, qkv):
        return qkv.reshape(3, qkv.shape[0], self.num_heads, -1, self.head_dim)

    def reshape_attention(self, attention, x_shape):
        if len(x_shape) > 3:  # initial shape of  B C H W or B H W C
            shape = (-1, *tuple(x_shape[2:])) if self.channel_dim == 1 else (*tuple(x_shape[1:3]), -1)
            return attention.transpose(1, 2).flatten(-2).reshape(x_shape[0], *shape)
        return attention

    def scale_scores(self, scores):
        return scores * (self.head_dim**-0.5 if not self.l2 else 1.0) * self.scale

    def compute_scores(self, q, k):
        if self.l2:
            q, k = (F.normalize(x, dim=-1, p=2) for x in (q, k))
        return q @ k.transpose(-1, -2)

    def cat_mem_vecs(self, k, v, batch_size):
        if self.num_mem_vecs > 0:
            k = torch.cat((k, self.mem_k.repeat(batch_size, 1, 1, 1)), dim=-2)
            v = torch.cat((v, self.mem_v.repeat(batch_size, 1, 1, 1)), dim=-2)
        return k, v

    def sparsify(self, scores):
        if self.sparse_topk and self.sparse_topk < scores.shape[-1]:
            top, _ = scores.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(scores)
            mask = scores < vk
            val = -torch.finfo(scores.dtype).max
            scores.masked_fill_(mask, val)  # inplace
        return scores

    def adjust_scores(self, scores):
        if self.improve_locality:
            mask = torch.eye(*scores.shape[-2:], device=scores.device).bool()
            mask_value = -torch.finfo(scores.dtype).max
            scores.masked_fill_(mask, mask_value)  # inplace
        return scores

    def forward(self, x):
        qkv = self.reshape_qkv(self.qkv(x))
        q, k, v = qkv.unbind(0)
        k, v = self.cat_mem_vecs(k, v, x.shape[0])
        scores = self.sparsify(self.adjust_scores(self.scale_scores(self.compute_scores(q, k))))
        attention = self.attn_dropout(scores.softmax(-1)) @ v
        out = self.reshape_attention(attention, x.shape)
        return self.proj_dropout(self.proj_out(out))


AttentionConv = partial(Attention, 1)


if __name__ == '__main__':
    for c, f, t in [(64, 16, 8), (128, 22, 44)]:
        test_batch = torch.randn(12, c, f, t)
        print(f'Attention:   {AttentionConv(c, l2=True, scale_value=10, trainable_scale=False)(test_batch).shape}')
    print('TinyTest1 successful')

    for c, f, t in [(64, 16, 8), (128, 22, 44)]:
        test_batch = torch.randn(12, f, t, c)
        print(f'Attention:   {Attention(-1, c, l2=True, scale_value=10, trainable_scale=False)(test_batch).shape}')
    print('TinyTest2 successful')

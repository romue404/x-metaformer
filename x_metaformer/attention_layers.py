from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class AttentionABC(nn.Module, ABC):
    def __init__(self, dim,
                 head_dim=32,
                 num_heads=4,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 scale_value=1.0,
                 num_mem_vecs=0,
                 sparse_topk=0,
                 trainable_scale=False,
                 improve_locality=False,
                 **kwargs):
        super().__init__()
        self.num_heads       = num_heads
        self.trainable_scale = trainable_scale
        self.head_dim        = head_dim
        self.dim             = dim
        self.sparse_topk = sparse_topk
        self.num_mem_vecs = num_mem_vecs
        self.improve_locality = improve_locality
        self.qkv:      Union[nn.Conv2d, nn.Linear]
        self.proj_out: Union[nn.Conv2d, nn.Linear]
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1).mul_(scale_value).float(),
                                  requires_grad=trainable_scale)
        if num_mem_vecs > 0:
            self.mem_k = nn.Parameter(torch.randn(num_heads, num_mem_vecs, head_dim))
            self.mem_v = nn.Parameter(torch.randn(num_heads, num_mem_vecs, head_dim))

    @abstractmethod
    def reshape_qkv(self, qkv):
        pass

    @abstractmethod
    def reshape_attention(self, attention, x_shape):
        pass

    def scale_scores(self, scores):
        return scores * self.head_dim**-0.5 * self.scale

    def compute_scores(self, q, k):
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
            return scores.masked_fill_(mask, val)  # inplace
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


class AttentionConv(AttentionABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv = nn.Conv2d(self.dim, 3*self.head_dim*self.num_heads, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(self.head_dim*self.num_heads, self.dim, kernel_size=1, bias=False)

    def reshape_qkv(self, qkv):
        return qkv.reshape(3, qkv.shape[0], self.num_heads, -1, self.head_dim)

    def reshape_attention(self, attention, x_shape):
        B, _, H, W = x_shape
        return attention.transpose(1, 2).flatten(-2).reshape(B, -1, H, W)


class AttentionConvL2(AttentionConv):
    def __init__(self, *args, scale_value=10, **kwargs):
        super().__init__(*args, scale_value=scale_value, **kwargs)

    def compute_scores(self, q, k):
        q_norm, k_norm = (F.normalize(x, dim=-1, p=2) for x in (q, k))
        scores = q_norm @ k_norm.transpose(-1, -2)
        return scores

    def scale_scores(self, scores):
        return scores * self.scale


if __name__ == '__main__':
    for c, f, t in [(64, 16, 8), (128, 22, 44), (256, 77, 100)]:
        test_batch = torch.randn(16, c, f, t)
        print(f'AttentionConvL2: {AttentionConvL2(c)(test_batch).shape}')
        print(f'AttentionConv:   {AttentionConv(c)(test_batch).shape}')
    print('TinyTest successful')

# 🥞 x-Metaformer

A PyTorch implementation of ["MetaFormer Baselines"](https://arxiv.org/abs/2210.13452) with optional extensions.  
We support various self-supervised pretraining approaches such as [BarlowTwins](https://arxiv.org/abs/2103.03230),
[MoCoV3](https://arxiv.org/abs/2104.02057) or [VICReg](https://arxiv.org/abs/2105.04906) (see ```x_metaformer.pretraining```).


## Setup
Simply run:
```pip install x-metaformer```

## Example
```py
import torch
from x_metaformer import CAFormer, ConvFormer


my_metaformer = CAFormer(
    in_channels=3,
    depths=(3, 3, 9, 3),
    dims=(64, 128, 320, 512),
    init_kernel_size=3,
    init_stride=2,
    drop_path_rate=0.5,
    norm='ln',  # ln, bn or rms (layernorm, batchnorm or rmsnorm)
    use_dual_patchnorm=False,  # norm on both sides for the patch embedding
    use_pos_emb=True,  # use 2d sinusodial positional embeddings
    head_dim=32,
    num_heads=4,
    attn_dropout=0.1,
    proj_dropout=0.1,
    patchmasking_prob=0.05,  # replace 5% of the initial tokens with a </mask> token
    scale_value=1.0, # scale attention logits by this value
    trainable_scale=False, # if scale can be trained
    num_mem_vecs=0, # additional memory vectors (in the attention layers)
    sparse_topk=0,  # sparsify - keep only top k values (in the attention layers)
    l2=False,   # l2 norm on tokens (in the attention layers) 
    improve_locality=False,  # remove attention on own token
    use_starreglu=False  # use gated StarReLU
)

x   = torch.randn(64, 3, 64, 64)  # B C H W
out = my_metaformer(x, return_embeddings=False)  # returns average pooled tokens
```
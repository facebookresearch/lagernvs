# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import xformers.ops as xops


def _get_flash_attention_ops():
    """Automatically detect GPU and return appropriate flash attention ops.

    Returns Flash Attention 3 ops for H100+ (compute capability >= 9.0),
    otherwise returns None to let xformers auto-dispatch to the best
    available backend (FA2, cutlass, etc.).
    """
    if not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()

    # Use Flash Attention 3 for H100 and newer (compute capability >= 9.0)
    if major >= 9:
        try:
            return (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)
        except AttributeError:
            pass

    # For all other GPUs, let xformers auto-dispatch to the best available
    # backend. This handles platforms where FA2 is not built (e.g. Windows)
    # by falling back to cutlass or other available operators.
    return None


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias=False,
        fc_bias=False,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        self.attn_dropout = attn_dropout

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Get appropriate flash attention ops based on GPU
        self.flash_attn_ops = _get_flash_attention_ops()

    def forward(self, q: torch.Tensor, kv=None) -> torch.Tensor:
        # attention block that supports non-query keys and values
        if kv is None:
            kv = q
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q, k, v = (
            einops.rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        x = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=self.attn_dropout if self.training else 0.0,
            op=self.flash_attn_ops,
        )

        x = einops.rearrange(x, "b n h d -> b n (h d)")

        x = self.attn_fc_dropout(self.proj(x))
        return x

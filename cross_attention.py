"""Dynamic cross-attention module."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCrossAttention(nn.Module):
    """Dynamic cross attention between spectral and image features."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        n_tokens: int = 8,
        use_topk: bool = True,
        topk_ratio: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.n_tokens = n_tokens
        self.use_topk = use_topk
        self.topk_ratio = topk_ratio

        self.tokenizer_s = nn.Linear(d_model, n_tokens * d_model)
        self.tokenizer_i = nn.Linear(d_model, n_tokens * d_model)
        self.q_s, self.k_i, self.v_i = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.q_i, self.k_s, self.v_s = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_i = nn.Linear(d_model, d_model)
        self.out = nn.Linear(2 * d_model, d_model)
        self.tau_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, 1))
        self.ln_in = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = x.shape
        return x.view(batch_size, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (scale / tau)
        if self.use_topk:
            k_top = max(1, int(self.topk_ratio * logits.size(-1)))
            topk_vals, topk_idx = torch.topk(logits, k_top, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            logits = mask
        attn = torch.softmax(logits, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, spec_feat: torch.Tensor, img_feat: torch.Tensor) -> Tuple[torch.Tensor, float]:
        batch_size, dim = spec_feat.shape
        spec = self.ln_in(spec_feat)
        img = self.ln_in(img_feat)
        spec_tokens = self.tokenizer_s(spec).view(batch_size, self.n_tokens, dim)
        img_tokens = self.tokenizer_i(img).view(batch_size, self.n_tokens, dim)

        qs, ki, vi = self._reshape_heads(self.q_s(spec_tokens)), self._reshape_heads(self.k_i(img_tokens)), self._reshape_heads(self.v_i(img_tokens))
        qi, ks, vs = self._reshape_heads(self.q_i(img_tokens)), self._reshape_heads(self.k_s(spec_tokens)), self._reshape_heads(self.v_s(spec_tokens))
        tau = F.softplus(self.tau_mlp(torch.cat([spec, img], dim=1))).view(batch_size, 1, 1, 1) + 1e-4

        zs = self._attend(qs, ki, vi, tau).transpose(1, 2).contiguous().view(batch_size, self.n_tokens, dim).mean(dim=1)
        zi = self._attend(qi, ks, vs, tau).transpose(1, 2).contiguous().view(batch_size, self.n_tokens, dim).mean(dim=1)
        hs = self.ln_out(spec + self.drop(self.proj_s(zs)))
        hi = self.ln_out(img + self.drop(self.proj_i(zi)))
        return self.out(torch.cat([hs, hi], dim=1)), tau.mean().item()


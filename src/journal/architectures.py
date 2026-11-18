#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backbones spanning a real capacity range.

Backbone capacity is a first-order variable in this setting rather than an
implementation detail: per-round energy cost scales with the parameter count, so
the choice of backbone determines whether an energy budget binds at all.

FT-Transformer follows Gorishniy, Rubachev, Khrulkov & Babenko,
"Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
(arXiv:2106.11959): a per-feature tokenizer + [CLS] token, a stack of standard
pre-norm Transformer blocks, and a head read off the [CLS] representation.
All features here are numerical, so no categorical embedding path is needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP with a selectable normalisation layer.

    ``norm="layer"`` is the default. Batch normalisation keeps running mean and
    variance as buffers, which federated averaging combines across clients
    holding different label distributions; the averaged statistics then describe
    no client's data. Layer normalisation normalises per sample and carries no
    cross-client state.
    """

    def __init__(self, d_in: int, n_cls: int, hidden=(256, 128, 64),
                 p: float = 0.3, norm: str = "layer"):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            nl = (nn.BatchNorm1d(h) if norm == "batch"
                  else nn.LayerNorm(h) if norm == "layer" else nn.Identity())
            layers += [nn.Linear(prev, h), nl, nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NumericalFeatureTokenizer(nn.Module):
    """x_j -> W_j * x_j + b_j, one learned vector per feature."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=d_token ** -0.5)
        nn.init.normal_(self.bias, std=d_token ** -0.5)

    def forward(self, x):                      # (B, F) -> (B, F, d)
        return self.weight.unsqueeze(0) * x.unsqueeze(-1) + self.bias.unsqueeze(0)


class FTBlock(nn.Module):
    """Pre-norm Transformer block (attention + feed-forward)."""

    def __init__(self, d: int, n_heads: int, d_ffn: int, p: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ffn), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(d_ffn, d))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    def __init__(self, d_in: int, n_cls: int, d_token: int = 192,
                 n_blocks: int = 6, n_heads: int = 8, d_ffn: int | None = None,
                 p: float = 0.1):
        super().__init__()
        d_ffn = d_ffn or int(d_token * 4 / 3)
        self.tokenizer = NumericalFeatureTokenizer(d_in, d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=d_token ** -0.5)
        self.blocks = nn.ModuleList(
            [FTBlock(d_token, n_heads, d_ffn, p) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_cls)

    def forward(self, x):
        t = self.tokenizer(x)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1)
        for b in self.blocks:
            t = b(t)
        return self.head(self.norm(t[:, 0]))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(name: str, d_in: int, n_cls: int) -> nn.Module:
    """Backbone registry spanning ~54 K to ~5 M parameters."""
    if name == "mlp":
        return MLP(d_in, n_cls, (256, 128, 64), norm="layer")
    if name == "mlp_tiny":
        # small enough for the microcontroller tier to afford
        # a participation at the binding allocation
        return MLP(d_in, n_cls, (64, 32, 16), norm="layer")
    if name == "mlp_bn":
        return MLP(d_in, n_cls, (256, 128, 64), norm="batch")
    if name == "mlp_mid":
        return MLP(d_in, n_cls, (512, 384, 256, 128), norm="layer")
    if name == "ft_small":
        return FTTransformer(d_in, n_cls, d_token=128, n_blocks=3, n_heads=8)
    if name == "ft":
        return FTTransformer(d_in, n_cls, d_token=192, n_blocks=6, n_heads=8)
    if name == "ft_large":
        return FTTransformer(d_in, n_cls, d_token=256, n_blocks=8, n_heads=8)
    raise ValueError(f"unknown backbone: {name}")

"""ViT-Lite (Hassani et al. 2021, "Escaping the Big Data Paradigm with Compact Transformers").

A Vision Transformer sized for 32×32 images. ViT-Lite-7/4 — patch 4, width 256, 7 blocks,
4 heads, MLP ratio 2, GELU — has 3,722,250 parameters for 10 classes.

Attention is written out explicitly (no fused kernels) so that double backpropagation —
needed by GAM and by the Hessian diagnostics — works on every backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.models.common import make_activation
from reglo.registry import OPTIMIZEES

__all__ = ["ViTLite"]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attention_dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.drop = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        weights = self.drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        return self.proj((weights @ v).transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, attention_dropout: float, activation: str):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            make_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTLite(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 7,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} is not a multiple of patch_size {patch_size}")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout, activation) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        return self.head(self.norm(self.blocks(tokens))[:, 0])


@OPTIMIZEES.register("vit_lite7")
def build_vit_lite7(cfg: DictConfig) -> ViTLite:
    return ViTLite(
        num_classes=int(cfg.num_classes),
        in_channels=int(cfg.in_channels),
        image_size=int(cfg.image_size),
        patch_size=int(cfg.patch_size),
        embed_dim=int(cfg.embed_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        mlp_ratio=float(cfg.mlp_ratio),
        dropout=float(cfg.dropout),
        attention_dropout=float(cfg.attention_dropout),
        activation=str(cfg.activation),
    )

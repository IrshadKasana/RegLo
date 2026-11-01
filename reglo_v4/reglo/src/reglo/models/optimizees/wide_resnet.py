"""Wide ResNet-d-k (Zagoruyko & Komodakis 2016), pre-activation blocks.

WRN-28-10 has 36.5 M parameters for 10 classes; WRN-28-4 has 5.8 M.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.models.common import make_activation
from reglo.registry import OPTIMIZEES

__all__ = ["WideResNet"]


class WideBlock(nn.Module):
    """BN → act → conv3×3 → BN → act → (dropout) → conv3×3, plus a (projected) shortcut."""

    def __init__(self, in_planes: int, planes: int, stride: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = make_activation(activation)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = make_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.act1(self.bn1(x))
        out = self.conv2(self.drop(self.act2(self.bn2(self.conv1(pre)))))
        return out + (x if self.shortcut is None else self.shortcut(pre))


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 10,
        in_channels: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WRN depth must be 6n + 4, got {depth}")
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv = nn.Conv2d(in_channels, widths[0], 3, stride=1, padding=1, bias=False)
        blocks, in_planes = [], widths[0]
        for planes, stride in zip(widths[1:], (1, 2, 2)):
            for i in range(n):
                blocks.append(WideBlock(in_planes, planes, stride if i == 0 else 1, dropout, activation))
                in_planes = planes
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(in_planes)
        self.act = make_activation(activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.blocks(self.conv(x))))
        return self.fc(torch.flatten(self.pool(out), 1))


@OPTIMIZEES.register("wrn28")
def build_wrn28(cfg: DictConfig) -> WideResNet:
    return WideResNet(
        depth=int(cfg.depth),
        widen_factor=int(cfg.widen_factor),
        num_classes=int(cfg.num_classes),
        in_channels=int(cfg.in_channels),
        dropout=float(cfg.dropout),
        activation=str(cfg.activation),
    )

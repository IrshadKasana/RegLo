"""ResNet-18 for 32×32 inputs (He et al. 2016; CIFAR variant: 3×3 stem, no max-pool).

11,173,962 parameters for 10 classes — the protocol's "≈11.2 M" — at the standard
``width`` of 64 (stages of 64, 128, 256, 512 channels). A smaller width keeps the same
layout for CPU smoke tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.models.common import make_activation
from reglo.registry import OPTIMIZEES

__all__ = ["ResNet18"]


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, activation: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = make_activation(activation)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, activation: str = "relu", width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            make_activation(activation),
        )
        blocks, in_planes = [], width
        for planes, stride in ((width, 1), (2 * width, 2), (4 * width, 2), (8 * width, 2)):
            for block_stride in (stride, 1):
                blocks.append(BasicBlock(in_planes, planes, block_stride, activation))
                in_planes = planes
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(self.blocks(self.stem(x))), 1))


@OPTIMIZEES.register("resnet18")
def build_resnet18(cfg: DictConfig) -> ResNet18:
    return ResNet18(num_classes=int(cfg.num_classes), in_channels=int(cfg.in_channels), activation=str(cfg.activation),
                    width=int(cfg.get("width", 64)))

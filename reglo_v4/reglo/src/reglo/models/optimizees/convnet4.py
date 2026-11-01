"""ConvNet-4: 3 × [Conv 3×3 → BatchNorm → activation → MaxPool 2] → Linear.

The network the LSTM optimizers are meta-trained on (Stage 1) and the Stage-2/3 optimizee.
With channels (32, 64, 128) on 32×32 inputs it has 113,962 parameters.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.models.common import make_activation
from reglo.registry import OPTIMIZEES

__all__ = ["ConvNet4"]


class ConvNet4(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 32,
        channels: Sequence[int] = (32, 64, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers, width = [], in_channels
        for out in channels:
            layers += [
                nn.Conv2d(width, out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                make_activation(activation),
                nn.MaxPool2d(2),
            ]
            width = out
        self.features = nn.Sequential(*layers)
        side = image_size
        for _ in channels:
            side //= 2
        if side < 1:
            raise ValueError(f"image_size {image_size} is too small for {len(channels)} pooling layers")
        self.classifier = nn.Linear(width * side * side, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


@OPTIMIZEES.register("convnet4")
def build_convnet4(cfg: DictConfig) -> ConvNet4:
    return ConvNet4(
        num_classes=int(cfg.num_classes),
        in_channels=int(cfg.in_channels),
        image_size=int(cfg.image_size),
        channels=list(cfg.channels),
        activation=str(cfg.activation),
    )

"""MLP: Flatten → [Linear → activation] × len(hidden) → Linear.

The fully connected optimizee: the source network of meta-training round B (an MLP on
Fashion-MNIST with two hidden layers of 128 ReLU units, the "FashionMnistRelu128x128" task
of the learned-optimizer literature) and an unseen optimizee of round A. With hidden
(128, 128) it has 118,282 parameters on 28×28 grayscale images and 411,146 on 32×32 RGB.
No normalisation and no dropout, so it carries no implicit regularizer of its own.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from reglo.models.common import make_activation
from reglo.registry import OPTIMIZEES

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        image_size: int = 28,
        hidden: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("an MLP needs at least one hidden layer")
        layers, width = [], in_channels * image_size * image_size
        for units in hidden:
            layers += [nn.Linear(width, int(units)), make_activation(activation)]
            width = int(units)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(torch.flatten(x, 1)))


@OPTIMIZEES.register("mlp")
def build_mlp(cfg: DictConfig) -> MLP:
    return MLP(
        num_classes=int(cfg.num_classes),
        in_channels=int(cfg.in_channels),
        image_size=int(cfg.image_size),
        hidden=list(cfg.hidden),
        activation=str(cfg.activation),
    )

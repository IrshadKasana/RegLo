"""Optimizee networks: MLP, ConvNet-4, ResNet-18.

All take 3 x 32 x 32 inputs and return log-probabilities (the base project trains with
``F.nll_loss`` on ``log_softmax`` outputs). Each model exposes

    features(x)  -> (N, feature_dim) representation (used by the phase-3 linear probe)
    classifier   -> nn.Linear(feature_dim, num_classes)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import INPUT_SHAPE


class Optimizee(nn.Module):
    feature_dim: int
    classifier: nn.Linear

    def features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.classifier(self.features(x)), dim=-1)


class MLP(Optimizee):
    """Fully connected optimizee. Defaults mirror the base `mlp` (one hidden layer of 20
    sigmoid units, N(0, 0.01) weights, zero biases)."""

    def __init__(self, num_classes: int, hidden: Sequence[int] = (20,), activation: str = "sigmoid"):
        super().__init__()
        acts = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}
        layers, prev = [], INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]
        for h in hidden:
            layers += [nn.Linear(prev, h), acts[activation]()]
            prev = h
        self.body = nn.Sequential(nn.Flatten(), *layers)
        self.feature_dim = prev
        self.classifier = nn.Linear(prev, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def features(self, x):
        return self.body(x)


class ConvNet4(Optimizee):
    """Conv-4: 4 x [conv3x3 -> BatchNorm -> ReLU -> maxpool2]; 32x32 -> 2x2 spatial."""

    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()
        blocks, cin = [], INPUT_SHAPE[0]
        for _ in range(4):
            blocks += [nn.Conv2d(cin, width, 3, padding=1), nn.BatchNorm2d(width),
                       nn.ReLU(inplace=False), nn.MaxPool2d(2)]
            cin = width
        self.body = nn.Sequential(*blocks, nn.Flatten())
        self.feature_dim = width * 2 * 2
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def features(self, x):
        return self.body(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(Optimizee):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool), [2, 2, 2, 2] basic blocks."""

    def __init__(self, num_classes: int, width: int = 64):
        super().__init__()
        self.in_planes = width
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        self.layer1 = self._make_layer(width, 2, 1)
        self.layer2 = self._make_layer(width * 2, 2, 2)
        self.layer3 = self._make_layer(width * 4, 2, 2)
        self.layer4 = self._make_layer(width * 8, 2, 2)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feature_dim = width * 8
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, planes, n_blocks, stride):
        layers = []
        for s in [stride] + [1] * (n_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.pool(x)


OPTIMIZEES = ("mlp", "convnet4", "resnet18")


def build_optimizee(name: str, num_classes: int, model_cfg=None) -> Optimizee:
    model_cfg = model_cfg or {}
    opts = dict(model_cfg.get(name, {}) or {})
    if name == "mlp":
        return MLP(num_classes, hidden=tuple(opts.get("hidden", (20,))), activation=opts.get("activation", "sigmoid"))
    if name == "convnet4":
        return ConvNet4(num_classes, width=int(opts.get("width", 32)))
    if name == "resnet18":
        return ResNet18(num_classes, width=int(opts.get("width", 64)))
    raise ValueError(f"Unknown optimizee '{name}'. Known: {OPTIMIZEES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

"""The networks being trained (θ): ConvNet-4, MLP, ResNet-18, WRN-28-k and ViT-Lite-7/4.

    model = build_optimizee(cfg.optimizee)
"""

from omegaconf import DictConfig
import torch.nn as nn

from reglo.models.optimizees import convnet4, mlp, resnet, vit, wide_resnet  # noqa: F401  (registers the builders)
from reglo.models.optimizees.convnet4 import ConvNet4
from reglo.models.optimizees.mlp import MLP
from reglo.models.optimizees.resnet import ResNet18
from reglo.models.optimizees.vit import ViTLite
from reglo.models.optimizees.wide_resnet import WideResNet
from reglo.registry import OPTIMIZEES

__all__ = ["ConvNet4", "MLP", "ResNet18", "ViTLite", "WideResNet", "build_optimizee"]


def build_optimizee(cfg: DictConfig) -> nn.Module:
    """The network described by ``configs/optimizees/<name>.yaml`` (as composed into ``cfg.optimizee``)."""
    return OPTIMIZEES.build(cfg.name, cfg)

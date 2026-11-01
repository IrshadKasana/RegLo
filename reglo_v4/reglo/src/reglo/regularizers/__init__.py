"""Regularizers — one file per YAML file in configs/regularizers/.

    none.yaml    base.py      NoRegularizer      M1, M3
    l1.yaml      l1.py        L1                 M2 and M4
    l2.yaml      l2.py        L2                 M2 and M4
    sam.yaml     sam.py       SAM                M2 and M4
    gam.yaml     gam.py       GAM                M2 and M4
    gsam.yaml    gsam.py      GSAM               M2 and M4
    smooth.yaml  smooth.py    SmoothnessPenalty  M4's meta-objective only

    reg = build_regularizer(cfg.regularizer, model)
    reg.explicit_step(model, x, y)               # M2: fills p.grad (1, 2 or 4 passes)
    reg.meta_penalty(fmodel.loss, θ_T, x, y)     # M4: L_reg(θ_T(φ)), differentiable into φ
"""

import torch.nn as nn
from omegaconf import DictConfig

from reglo.registry import REGULARIZERS
from reglo.regularizers.base import (
    ExplicitResult,
    MetaPenalty,
    NoRegularizer,
    Regularizer,
    frozen_bn_stats,
    global_norm,
)
from reglo.regularizers.gam import GAM
from reglo.regularizers.gsam import GSAM, gsam_direction
from reglo.regularizers.l1 import L1
from reglo.regularizers.l2 import L2
from reglo.regularizers.sam import SAM, sam_perturbation
from reglo.regularizers.smooth import SmoothnessPenalty

__all__ = [
    "GAM",
    "GSAM",
    "L1",
    "L2",
    "SAM",
    "ExplicitResult",
    "MetaPenalty",
    "NoRegularizer",
    "Regularizer",
    "SmoothnessPenalty",
    "build_regularizer",
    "frozen_bn_stats",
    "global_norm",
    "gsam_direction",
    "sam_perturbation",
]


def build_regularizer(cfg: DictConfig, model: nn.Module = None) -> Regularizer:
    """The regularizer described by ``cfg.regularizer`` (``model`` gives the L1/L2 penalty mask)."""
    return REGULARIZERS.build(cfg.name, cfg, model)

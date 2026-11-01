"""Helpers shared by every model: activations, parameter counting, and which weights a penalty touches."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch.nn as nn

__all__ = ["NORM_LAYERS", "count_parameters", "make_activation", "penalty_mask"]

NORM_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


def make_activation(name: str) -> nn.Module:
    """A fresh activation module (never in-place, so double backpropagation stays valid)."""
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'; choose from {sorted(_ACTIVATIONS)}") from None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def penalty_mask(model: nn.Module, exclude: Sequence[str] = ("bias", "norm")) -> List[bool]:
    """For each parameter, in ``model.named_parameters()`` order: does an L1/L2 penalty apply?

    ``exclude`` may contain:
        bias       — every parameter called ``bias``
        norm       — every parameter of BatchNorm / LayerNorm / GroupNorm layers
        embedding  — ViT class token and positional embedding
    Decided by module type and local parameter name — never by matching full names.
    """
    unknown = set(exclude) - {"bias", "norm", "embedding"}
    if unknown:
        raise ValueError(f"Unknown exclusion(s) {sorted(unknown)}; use bias, norm, embedding")
    owner: Dict[int, tuple] = {}
    for module in model.modules():
        for local_name, param in module.named_parameters(recurse=False):
            owner.setdefault(id(param), (module, local_name))
    mask = []
    for _, param in model.named_parameters():
        module, local_name = owner[id(param)]
        excluded = (
            ("norm" in exclude and isinstance(module, NORM_LAYERS))
            or ("bias" in exclude and local_name == "bias")
            or ("embedding" in exclude and local_name in ("cls_token", "pos_embed"))
        )
        mask.append(not excluded)
    return mask

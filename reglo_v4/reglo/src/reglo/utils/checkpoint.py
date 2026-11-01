"""Crash-safe checkpoints and random-number-generator state, for resuming after a PBS walltime."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

__all__ = ["load_checkpoint", "rng_state", "save_checkpoint", "set_rng_state"]

PathLike = Union[str, Path]


def save_checkpoint(path: PathLike, state: Dict[str, Any]) -> Path:
    """``torch.save`` to a temporary file, then rename: a killed job never leaves a corrupt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: PathLike, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

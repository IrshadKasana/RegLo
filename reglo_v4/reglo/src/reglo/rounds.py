"""Rounds of meta-training (configs/rounds.yaml): the pair each family of learned optimizers is
meta-trained on — its *source* — and what any other pair is to that family.

    A   ConvNet-4 / SVHN     the paper's protocol
    B   MLP / FMNIST         a second source, for cross-evaluation

A round's name is the ``meta_tag`` of its optimizers (artifacts/<profile>/meta_training/<round>/)
and of the M3/M4 runs that load them (run folders ``m3_lstm@<round>``, ``m4_lstm_reg-<r>@<round>``).
Composing a meta-training run with ``meta_tag=<round>`` selects the round's source pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import yaml

__all__ = [
    "ROLES",
    "ROLE_LABELS",
    "ROUNDS_FILE",
    "Round",
    "get_round",
    "load_rounds",
    "require_rounds",
    "rounds_file",
]

PathLike = Union[str, Path]
ROUNDS_FILE = "rounds.yaml"

#: what a pair is to a round, in reporting order
ROLES = ("seen", "unseen_optimizee", "unseen_dataset", "unseen_both")
ROLE_LABELS = {
    "seen": "seen",
    "unseen_optimizee": "unseen optimizee",
    "unseen_dataset": "unseen dataset",
    "unseen_both": "unseen optimizee + dataset",
}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")  # a folder-name token: no '@', '-', '/'


@dataclass(frozen=True)
class Round:
    name: str
    optimizee: str
    dataset: str
    description: str = ""

    @property
    def source_tag(self) -> str:
        return f"{self.optimizee}-{self.dataset}"

    def role(self, optimizee: str, dataset: str) -> str:
        """``seen`` · ``unseen_optimizee`` (same dataset) · ``unseen_dataset`` (same network) · ``unseen_both``."""
        if dataset == self.dataset:
            return "seen" if optimizee == self.optimizee else "unseen_optimizee"
        return "unseen_dataset" if optimizee == self.optimizee else "unseen_both"


def rounds_file(root: Optional[PathLike] = None) -> Path:
    from reglo.config import config_dir

    return Path(root or config_dir()) / ROUNDS_FILE


def load_rounds(root: Optional[PathLike] = None) -> Dict[str, Round]:
    """Every round of configs/rounds.yaml, in file order (no file: no rounds)."""
    from reglo.config import ConfigError, available

    path = rounds_file(root)
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as err:
        raise ConfigError(f"{path} is not valid YAML: {err}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must map round names to {{optimizee, dataset}}")
    optimizees, datasets = available("optimizee", root), available("dataset", root)
    rounds: Dict[str, Round] = {}
    sources: Dict[tuple, str] = {}
    for name, entry in data.items():
        name = str(name)
        if not _NAME.match(name):
            raise ConfigError(f"{path}: round name '{name}' must be a letter followed by letters, digits or '_'")
        if not isinstance(entry, dict) or set(entry) - {"optimizee", "dataset", "description"} \
                or not {"optimizee", "dataset"} <= set(entry):
            raise ConfigError(f"{path}: round {name} needs exactly optimizee, dataset (and optionally description)")
        optimizee, dataset = str(entry["optimizee"]), str(entry["dataset"])
        if optimizee not in optimizees:
            raise ConfigError(f"{path}: round {name}: unknown optimizee '{optimizee}' ({', '.join(optimizees)})")
        if dataset not in datasets:
            raise ConfigError(f"{path}: round {name}: unknown dataset '{dataset}' ({', '.join(datasets)})")
        if (optimizee, dataset) in sources:
            raise ConfigError(f"{path}: rounds {sources[(optimizee, dataset)]} and {name} have the same source pair")
        sources[(optimizee, dataset)] = name
        rounds[name] = Round(name, optimizee, dataset, str(entry.get("description", "")))
    return rounds


def get_round(name: Optional[str], root: Optional[PathLike] = None) -> Optional[Round]:
    """The round called ``name``, or None (a meta_tag that is not a round: an ad-hoc family)."""
    return load_rounds(root).get(str(name)) if name else None


def require_rounds(names: Sequence[str], where: str, root: Optional[PathLike] = None) -> List[Round]:
    """The rounds ``names`` refers to, in that order; an unknown name is an error."""
    from reglo.config import ConfigError

    known = load_rounds(root)
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise ConfigError(f"{where}: rounds is a list of round names, e.g. [A, B], got {names!r}")
    out = []
    for name in names:
        if str(name) not in known:
            raise ConfigError(f"{where}: unknown round '{name}'; configs/rounds.yaml defines "
                              f"{', '.join(known) or 'none'}")
        if known[str(name)] in out:
            raise ConfigError(f"{where}: round '{name}' is listed twice")
        out.append(known[str(name)])
    return out

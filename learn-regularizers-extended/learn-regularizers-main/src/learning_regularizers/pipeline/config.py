"""Configuration loading for the hierarchical pipeline.

The base YAML files under ``configs/dmo`` are read but never written. Each run's
arguments are a merge of (lowest -> highest precedence):

    base YAML (meta_train.yaml or optimizee_eval.yaml)
    smooth.yaml                    (meta-training only)
    <regularizer>.yaml             (no_reg.yaml when the run has no R)
    pipeline.yaml ``overrides`` blocks + task fields
    command-line ``--set key=value``
"""
from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG_DIR = CONFIG_DIR / "dmo" / "base"
REGULARIZER_CONFIG_DIR = CONFIG_DIR / "dmo" / "regularizers"
DEFAULT_PIPELINE_CONFIG = CONFIG_DIR / "pipeline" / "pipeline.yaml"

VALID_REGULARIZERS = ("l1", "l2", "sam", "gam", "gsam")
VALID_BASE_OPTIMIZERS = ("adam", "sgd")


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self):
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def load_yaml(path: Path | str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def train_arguments(path: Path | str) -> dict:
    """Return the ``train_arguments`` block of a base-style YAML."""
    return dict(load_yaml(path).get("train_arguments", {}))


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_set_overrides(items: Optional[Iterable[str]]) -> dict:
    """``["a.b=3", "c=[1,2]"]`` -> ``{"a": {"b": 3}, "c": [1, 2]}`` (values parsed as YAML)."""
    nested: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = nested
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


QUICK_PRESET = {
    # A tiny but complete configuration for smoke tests / debugging.
    "meta_training": {"overrides": {"max_epochs": 2, "train_period": 1, "optimizee_steps": 40,
                                    "truncated_bptt_step": 10, "if_cl": False, "eval_epochs": 1,
                                    "batch_size": 32}},
    "optimizee_training": {"eval_every": 10, "train_metric_max_samples": 256,
                           "overrides": {"optimizee_epochs": 20, "n_training": 1, "batch_size": 64}},
    "optimizee_evaluation": {"probe": {"epochs": 2, "max_train_samples": 512, "max_test_samples": 256}},
    "models": {"resnet18": {"width": 8}, "convnet4": {"width": 8}},
}


def load_pipeline_config(path: Optional[Path | str] = None, sets: Optional[Iterable[str]] = None,
                         quick: bool = False) -> Cfg:
    cfg = load_yaml(path or DEFAULT_PIPELINE_CONFIG)
    if quick:
        cfg = deep_update(cfg, QUICK_PRESET)
    cfg = deep_update(cfg, parse_set_overrides(sets))
    cfg = Cfg.wrap(cfg)
    resolve_task(cfg)
    validate_config(cfg)
    return cfg


def resolve_task(cfg: Cfg) -> None:
    """Make the meta-training task (meta_training.optimizee + meta_training.dataset) the single
    source of truth for what counts as "seen" in phases 2 and 3.

    Fields left as null follow the meta-training task:
        optimizer_evaluation.dataset         -> meta_training.dataset
        optimizer_evaluation.seen_optimizee  -> meta_training.optimizee (also added to .optimizees)
        optimizee_evaluation.seen_dataset    -> meta_training.dataset
        optimizee_evaluation.optimizees      -> [meta_training.optimizee]
        optimizee_evaluation.unseen_datasets -> every known dataset except the seen one
    An explicit value that contradicts the meta-training task raises an error.
    """
    from .data import DATASETS          # lazy: data.py imports REPO_ROOT from this module
    from .models import OPTIMIZEES

    mt, oe, ev = cfg.meta_training, cfg.optimizer_evaluation, cfg.optimizee_evaluation
    if mt.dataset not in DATASETS:
        raise ValueError(f"meta_training.dataset '{mt.dataset}' unknown; choose from {sorted(DATASETS)}")
    if mt.optimizee not in OPTIMIZEES:
        raise ValueError(f"meta_training.optimizee '{mt.optimizee}' unknown; choose from {list(OPTIMIZEES)}")

    def follow(section, name, field, value):
        current = section.get(field)
        if current is None:
            section[field] = value
        elif current != value:
            raise ValueError(f"{name}.{field} = '{current}' but the meta-training task uses '{value}'. "
                             f"Set it to null (follow meta_training) or change meta_training instead.")

    follow(oe, "optimizer_evaluation", "dataset", mt.dataset)
    follow(oe, "optimizer_evaluation", "seen_optimizee", mt.optimizee)
    follow(ev, "optimizee_evaluation", "seen_dataset", mt.dataset)

    optimizees = list(oe.get("optimizees") or [])
    if mt.optimizee not in optimizees:
        optimizees.insert(0, mt.optimizee)
    oe["optimizees"] = optimizees
    if not ev.get("optimizees"):
        ev["optimizees"] = [mt.optimizee]
    unseen = ev.get("unseen_datasets")
    if unseen is None:
        unseen = list(DATASETS)
    ev["unseen_datasets"] = [d for d in unseen if d != mt.dataset]

    for name in optimizees + list(ev.optimizees):
        if name not in OPTIMIZEES:
            raise ValueError(f"unknown optimizee '{name}'; choose from {list(OPTIMIZEES)}")
    for name in ev.unseen_datasets:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset '{name}'; choose from {sorted(DATASETS)}")


def validate_config(cfg: Cfg) -> None:
    for m in cfg.methods:
        if m not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"Unknown method id {m}; expected 1..6")
    for b in cfg.base_optimizers:
        if b not in VALID_BASE_OPTIMIZERS:
            raise ValueError(f"Unknown base optimizer '{b}'; expected one of {VALID_BASE_OPTIMIZERS}")
    for r in cfg.regularizers:
        if r not in VALID_REGULARIZERS:
            raise ValueError(f"Unknown regularizer '{r}'; expected one of {VALID_REGULARIZERS}")
        if r not in cfg.regularizer_configs:
            raise ValueError(f"No YAML mapped for regularizer '{r}' in regularizer_configs")
    if cfg.meta_training.meta_loss not in ("last", "sum"):
        raise ValueError("meta_training.meta_loss must be 'last' or 'sum'")
    if cfg.optimizee_evaluation.probe.optimizer not in ("adam", "method"):
        raise ValueError("optimizee_evaluation.probe.optimizer must be 'adam' or 'method'")


# --------------------------------------------------------------------------- run arguments

def regularizer_yaml(cfg: Cfg, regularizer: Optional[str]) -> Path:
    key = regularizer or "none"
    return REGULARIZER_CONFIG_DIR / cfg.regularizer_configs[key]


def meta_models_cfg(cfg: Cfg) -> Cfg:
    """Model settings used while meta-training: ``models`` updated by ``meta_training.models``.

    Lets the LO be meta-trained on a narrower proxy of the optimizee (e.g. ResNet-18 width 16)
    and then evaluated on the full model; the LO is coordinate-wise, so it transfers unchanged.
    """
    return Cfg.wrap(deep_update(dict(cfg.models), dict(cfg.meta_training.get("models") or {})))


def meta_training_args(cfg: Cfg, method, regularizer: Optional[str]) -> Namespace:
    """Arguments for meta-training a learned optimizer (methods 3-6)."""
    merged: dict[str, Any] = {}
    merged.update(train_arguments(BASE_CONFIG_DIR / "meta_train.yaml"))
    merged.update(train_arguments(REGULARIZER_CONFIG_DIR / "smooth.yaml"))
    merged.update(train_arguments(regularizer_yaml(cfg, regularizer if method.uses_reg else None)))

    mt = cfg.meta_training
    merged.update({
        "dataset": mt.dataset,
        "optimizee_model_name": mt.optimizee,
        "regularizer": regularizer if method.uses_reg else "none",
        "if_avreg": bool(method.uses_reg),
        "if_smooth": bool(method.uses_smooth),
        "meta_loss": mt.meta_loss,
        "meta_loss_scale": mt.meta_loss_scale,
        "meta_grad_clip": mt.meta_grad_clip,
        "smooth_every": int(mt.smooth_every),
        "seed": cfg.experiment.seed,
    })
    if mt.outer_optimizer != "yaml":
        merged["base_optimizer"] = mt.outer_optimizer
    merged.update(dict(mt.overrides or {}))
    merged["models"] = dict(meta_models_cfg(cfg).get(mt.optimizee) or {})   # recorded so a width change re-trains
    args = Namespace(**merged)
    if args.optimizee_steps % args.truncated_bptt_step != 0:
        raise ValueError("optimizee_steps must be divisible by truncated_bptt_step")
    return args


def optimizee_training_args(cfg: Cfg, base_optimizer: Optional[str], regularizer: Optional[str]) -> Namespace:
    """Arguments for training an optimizee with B (+R) or with a frozen learned optimizer.

    ``regularizer=None`` means no explicit regularizer and loads ``no_reg.yaml``.
    """
    merged: dict[str, Any] = {}
    merged.update(train_arguments(BASE_CONFIG_DIR / "optimizee_eval.yaml"))
    merged.update(train_arguments(regularizer_yaml(cfg, regularizer)))
    if base_optimizer is not None:
        merged["base_optimizer"] = base_optimizer
    merged["regularizer"] = regularizer or "none"
    ot = cfg.optimizee_training
    merged["eval_every"] = int(ot.eval_every)
    merged["train_metric_max_samples"] = ot.train_metric_max_samples
    merged.update(dict(ot.overrides or {}))
    return Namespace(**merged)


def namespace_to_dict(ns: Namespace) -> dict:
    return {k: v for k, v in vars(ns).items()}

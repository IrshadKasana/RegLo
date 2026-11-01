"""Where every run writes. Each profile has its own tree, ``cfg.paths.artifacts``::

    artifacts/                                                   $REGLO_ARTIFACTS: the root
    └── <profile>/                                               production · smoke · smoke_offline
        ├── meta_training/<round>/<m3_lstm | m4_lstm_reg-<r>>/          Stage 1: LSTM weights φ, logs
        ├── runs/<study>/<optimizee>-<dataset>/<method>[-<r>][@<round>]/seed<k>/
        │                                                               one optimizee run (Stages 1–3)
        ├── diagnostics/<study>/...                                     Stage 4 (supplementary)
        ├── manifests/                                                  run lists (cluster array jobs)
        ├── logs/                                                       per-run output of run_study.py
        ├── tables/<study>/                                             Stage 4: result cubes, tables
        └── figures/<study>/

``study`` (``main`` for the thesis) is a namespace: every (optimizee, dataset, method,
regularizer, round, seed) is trained once per study, and each evaluation setting of Stages 2–3
(seen/unseen optimizee, seen/unseen dataset — per round) reads the runs of its pairs from
that pool. ``meta_tag`` names the round of the Stage-1 optimizers (configs/rounds.yaml: A, B):
meta_train.py writes it and the checkpoint paths in ``configs/methods/m3_lstm.yaml`` /
``m4_lstm_reg.yaml`` read it. M3/M4 run folders end in ``@<round>`` — the same pair is trained
with the optimizers of every round — while M1/M2 runs, which use no learned optimizer, do not.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from reglo.config import NO_REGULARIZER, project_root

__all__ = [
    "artifacts_dir",
    "default_artifacts_root",
    "profile_artifacts",
    "set_artifacts_root",
    "default_data_root",
    "diagnostics_dir_for",
    "ensure_dir",
    "find_runs",
    "meta_run_dir",
    "meta_tag",
    "method_tag",
    "parse_method_tag",
    "regularizer_of",
    "run_dir",
    "subdir",
]


def default_artifacts_root() -> Path:
    """``$REGLO_ARTIFACTS`` or ``<repo>/artifacts`` — the root holding one folder per profile
    (the same rule as ``paths.artifacts_root`` in configs/default.yaml)."""
    env = os.environ.get("REGLO_ARTIFACTS")
    return Path(env).expanduser().resolve() if env else project_root() / "artifacts"


def set_artifacts_root(root: Optional[Union[str, Path]]) -> Path:
    """Make ``root`` the artifacts root of this process and of the runs it starts (``$REGLO_ARTIFACTS``)."""
    if root is not None:
        os.environ["REGLO_ARTIFACTS"] = str(Path(root).expanduser().resolve())
    return default_artifacts_root()


def profile_artifacts(profile: Optional[str] = None, root: Optional[Union[str, Path]] = None) -> Path:
    """``<root>/<profile>`` — where the runs of a profile (default: the active one) live."""
    from reglo.profiles import active_profile

    base = Path(root).expanduser() if root is not None else default_artifacts_root()
    return base / (profile or active_profile())


def default_data_root() -> Path:
    """``$REGLO_DATA`` or ``<repo>/datasets``."""
    env = os.environ.get("REGLO_DATA")
    return Path(env).expanduser().resolve() if env else project_root() / "datasets"


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_dir(cfg: DictConfig) -> Path:
    return Path(cfg.paths.artifacts)


def subdir(cfg: DictConfig, name: str) -> Path:
    """``artifacts/<name>`` — e.g. ``subdir(cfg, "tables")``."""
    return artifacts_dir(cfg) / name


def regularizer_of(cfg: DictConfig) -> str:
    return str(cfg.selected.get("regularizer", NO_REGULARIZER))


def method_tag(cfg: DictConfig) -> str:
    """``m1_adamw``, ``m2_adamw_reg-sam``, ``m3_lstm@A``, ``m4_lstm_reg-sam@B`` … (``@<round>``: the
    meta_tag of the learned optimizer an M3/M4 run loads)."""
    reg = regularizer_of(cfg)
    tag = cfg.method.name if reg == NO_REGULARIZER else f"{cfg.method.name}-{reg}"
    return f"{tag}@{cfg.meta_tag}" if cfg.method.get("optimizer_kind") == "learned" else tag


def parse_method_tag(folder: str) -> Dict[str, str]:
    """``m4_lstm_reg-sam@B`` → ``{"method": "m4_lstm_reg", "regularizer": "sam", "round": "B"}``
    (``round`` is ``""`` for M1/M2)."""
    name, _, round_name = folder.partition("@")
    method, _, reg = name.partition("-")
    return {"method": method, "regularizer": reg or NO_REGULARIZER, "round": round_name}


def run_dir(cfg: DictConfig) -> Path:
    """``artifacts/<profile>/runs/<study>/<optimizee>-<dataset>/<method>[-<r>][@<round>]/seed<k>``"""
    pair = f"{cfg.selected.optimizee}-{cfg.selected.dataset}"
    return artifacts_dir(cfg) / "runs" / str(cfg.study) / pair / method_tag(cfg) / f"seed{cfg.seed}"


def meta_tag(cfg: DictConfig) -> str:
    """``m3_lstm``, ``m4_lstm_reg-sam`` …"""
    reg = regularizer_of(cfg)
    return cfg.meta_training.name if reg == NO_REGULARIZER else f"{cfg.meta_training.name}-{reg}"


def meta_run_dir(cfg: DictConfig) -> Path:
    """``artifacts/<profile>/meta_training/<round>/<m3_lstm | m4_lstm_reg-<r>>`` — where M3/M4 read φ."""
    return artifacts_dir(cfg) / "meta_training" / str(cfg.meta_tag) / meta_tag(cfg)


def diagnostics_dir_for(run_path: Union[str, Path]) -> Path:
    """``<artifacts>/runs/<study>/…/seed<k>`` → ``<artifacts>/diagnostics/<study>/…/seed<k>``."""
    run_path = Path(run_path).resolve()
    if len(run_path.parents) < 5 or run_path.parents[3].name != "runs":
        raise ValueError(f"{run_path} is not a run folder "
                         "(artifacts/<profile>/runs/<study>/<optimizee>-<dataset>/<method>/seed<k>)")
    root = run_path.parents[4]
    return root / "diagnostics" / run_path.relative_to(root / "runs")


def find_runs(artifacts_root: Union[str, Path], study: str) -> List[Path]:
    """Every run folder of a study that has written its final.json, in a stable order."""
    base = Path(artifacts_root) / "runs" / study
    return sorted(p.parent for p in base.glob("*/*/seed*/final.json"))

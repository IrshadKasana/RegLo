"""Artifact layout for the pipeline (alongside the base ``artifacts/dmo``):

artifacts/pipeline/<experiment>/
    config_resolved.yaml
    checkpoints/lo/<run_id>.pt            learned-optimizer checkpoints (base-compatible state_dict)
    checkpoints/lo/<run_id>.json          meta-training history / arguments
    optimizee_runs/<key>.json|.pt         every optimizee training (cached, reused across phases)
    phase1/<run_id>.json                  phase-1 summary per trained entity
    phase2/<run_id>__<optimizee>.json     phase-2 records
    phase3/<run_id>__<optimizee>__<dataset>.json
    tables/                               3D result tables (npy / csv / md / png)
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import yaml

from .config import REPO_ROOT


def experiment_dir(cfg) -> Path:
    path = REPO_ROOT / "artifacts" / "pipeline" / cfg.experiment.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def subdir(cfg, *parts) -> Path:
    path = experiment_dir(cfg).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def lo_checkpoint_path(cfg, run_id: str) -> Path:
    return subdir(cfg, "checkpoints", "lo") / f"{run_id}.pt"


def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(_clean(data), fh, indent=1)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def save_resolved_config(cfg) -> Path:
    path = experiment_dir(cfg) / "config_resolved.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    return path

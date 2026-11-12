from __future__ import annotations

import os
from pathlib import Path


OLD_WORKSPACE_ROOT = Path("/nlsasfs/home/metalearning/nckris/Suraj_LSAIML")


def repo_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "datasets").exists() and (candidate / "L2O-CL").exists():
            return candidate

    raise RuntimeError(f"Could not infer repository root from {start}")


def experiment_dir(start: str | Path) -> Path:
    start_path = Path(start).resolve()
    return start_path.parent if start_path.is_file() else start_path


def remap_workspace_path(path_value: str | Path, start: str | Path) -> Path:
    path = Path(path_value)
    root = repo_root(start)

    if path.is_absolute() and str(path).startswith(str(OLD_WORKSPACE_ROOT)):
        return root / path.relative_to(OLD_WORKSPACE_ROOT)

    if path.is_absolute():
        return path

    return (experiment_dir(start) / path).resolve()


def datasets_dir(start: str | Path) -> Path:
    env_override = os.environ.get("LSAIML_DATASETS_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return repo_root(start) / "datasets"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

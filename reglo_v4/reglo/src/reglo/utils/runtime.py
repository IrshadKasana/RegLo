"""Seeding, device selection and an environment record for every run.

The environment record (GPU model, library versions, PBS / SLURM job id) is saved with
each run, so wall-clock comparisons can be checked for same-GPU fairness.
"""

from __future__ import annotations

import os
import platform
import random
import socket
from typing import Any, Dict, Optional

import numpy as np
import torch

__all__ = [
    "EXIT_CONFIG",
    "EXIT_FAILED",
    "EXIT_INTERRUPTED",
    "EXIT_LOCKED",
    "EXIT_MISSING",
    "EXIT_OK",
    "DeviceUnavailable",
    "describe_exit",
    "environment_info",
    "resolve_device",
    "seed_everything",
]

# Exit codes of the run scripts (meta_train.py, train_optimizee.py), read by run_study.py and the cluster jobs.
EXIT_OK = 0           # finished (a diverged training run is a finished run with status "diverged")
EXIT_FAILED = 1       # crashed
EXIT_CONFIG = 2       # invalid configuration
EXIT_MISSING = 3      # a prerequisite is missing: the Stage-1 optimizer, the dataset files, the GPU
EXIT_LOCKED = 4       # another process is running this run right now
EXIT_INTERRUPTED = 75  # stopped cleanly at max_hours; the same command continues it (EX_TEMPFAIL)

_EXIT_TEXT = {EXIT_OK: "done", EXIT_FAILED: "crashed", EXIT_CONFIG: "invalid configuration",
              EXIT_MISSING: "missing prerequisite", EXIT_LOCKED: "running elsewhere",
              EXIT_INTERRUPTED: "interrupted at max_hours (resumable)"}


def describe_exit(code: int) -> str:
    return _EXIT_TEXT.get(code, f"exit code {code}")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    Args:
        seed: the run's seed.
        deterministic: also force deterministic cuDNN/cuBLAS kernels. Slower, but
            repeated runs give bit-identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class DeviceUnavailable(RuntimeError):
    """The configuration asks for a GPU that this machine does not have."""


def resolve_device(spec: str = "auto") -> torch.device:
    """``auto`` -> the first GPU if there is one, else the CPU; ``cuda``/``cuda:1``/``cpu`` as given.

    The production profile asks for ``cuda``: on a machine without a usable GPU the run stops here
    instead of training for days on CPUs.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailable(
            f"device={spec} but no CUDA GPU is visible on {socket.gethostname()} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}). Production runs need a GPU; "
            f"on a laptop use --profile smoke (or device=auto / device=cpu)."
        )
    return device


def environment_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Versions, hardware and scheduler ids worth keeping with every result."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device) if device is not None else None,
    }
    if torch.cuda.is_available():
        index = device.index if device is not None and device.type == "cuda" and device.index is not None else 0
        props = torch.cuda.get_device_properties(index)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    return info

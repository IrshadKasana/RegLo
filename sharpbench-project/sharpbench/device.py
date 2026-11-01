"""
Device, CUDA, memory and determinism handling.

Everything that touches the GPU goes through this module so that behaviour is
identical across `train`, `sweep` and `check`, and so that a machine-specific
problem shows up in one place instead of five.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    name: str
    total_mb: float
    free_mb: float
    capability: tuple | None
    is_cuda: bool

    def __str__(self) -> str:
        if not self.is_cuda:
            return f"{self.name} (CPU)"
        cc = f"sm_{self.capability[0]}{self.capability[1]}" if self.capability else "?"
        return (f"{self.name} [{cc}] {self.free_mb:.0f}/{self.total_mb:.0f} MB free")


def resolve_device(spec: str = "auto", quiet: bool = False) -> torch.device:
    """
    spec: 'auto' | 'cpu' | 'cuda' | 'cuda:1' | '0'

    'auto' prefers CUDA and falls back to CPU with a loud warning, because a
    silent CPU fallback is the difference between a 3-hour sweep and a 6-hour
    one and you should never discover it from the wall clock.
    """
    spec = str(spec).strip().lower()

    if spec.isdigit():
        spec = f"cuda:{spec}"

    if spec == "cpu":
        return torch.device("cpu")

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if not quiet:
            warnings.warn(
                "\n" + "!" * 64 +
                "\n  CUDA NOT AVAILABLE - falling back to CPU."
                "\n  If you expected a GPU, you most likely installed the"
                "\n  CPU-only wheel. Reinstall with the CUDA index-url."
                "\n" + "!" * 64, stacklevel=2)
        return torch.device("cpu")

    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {spec!r} but torch.cuda.is_available() is False. "
                "Either install a CUDA build of PyTorch or pass --device cpu."
            )
        idx = int(spec.split(":")[1]) if ":" in spec else 0
        n = torch.cuda.device_count()
        if idx >= n:
            raise RuntimeError(
                f"Requested cuda:{idx} but only {n} CUDA device(s) visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        return torch.device(f"cuda:{idx}")

    raise ValueError(f"unrecognised device spec: {spec!r}")


def device_info(device: torch.device) -> DeviceInfo:
    if device.type != "cuda":
        import platform
        return DeviceInfo(device, platform.processor() or "cpu",
                          0.0, 0.0, None, False)
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return DeviceInfo(device, props.name, total / 1024**2, free / 1024**2,
                      (props.major, props.minor), True)


def require_memory(device: torch.device, need_mb: float) -> None:
    """Fail early and legibly rather than mid-sweep with a CUDA OOM."""
    if device.type != "cuda":
        return
    info = device_info(device)
    if info.free_mb < need_mb:
        raise RuntimeError(
            f"Need ~{need_mb:.0f} MB on {info.name} but only "
            f"{info.free_mb:.0f} MB is free. Close other GPU processes "
            f"(check `nvidia-smi`) or run fewer sweep shards."
        )


def cap_memory_fraction(device: torch.device, fraction: float | None) -> None:
    """
    Limit this process to a fraction of the card.

    Useful when running several sweep shards on one GPU: without a cap, the
    caching allocator in one process can grab memory it never returns, and a
    later shard OOMs even though the real usage is tiny.
    """
    if fraction is None or device.type != "cuda":
        return
    if not (0 < fraction <= 1.0):
        raise ValueError("memory fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)


def configure_backends(deterministic: bool = True, allow_tf32: bool = False,
                       cpu_threads: int | None = None) -> None:
    """
    Numerical policy for the whole study.

    TF32 defaults to OFF. On Ampere and newer, TF32 matmuls are enabled by
    default in many PyTorch builds and silently reduce mantissa precision.
    This study measures accuracy differences on the order of 0.2%, which is
    the same order as TF32 noise, so leaving it on would make results depend
    on which GPU generation you happened to run on. The model is far too small
    for TF32 to buy any speed here, so there is nothing to trade away.
    """
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cpu_threads:
        torch.set_num_threads(cpu_threads)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device.index or 0) / 1024**2


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index or 0)

"""Stage 4: FLOPs and wall-clock time per training step for all 12 variants, on one GPU.

Separate PBS jobs can land on different GPUs, so their wall-clock times are not comparable.
Here every variant of one optimizee × dataset pair — M1, M2 × 5, M3, M4 × 5 — is timed
back-to-back in a single process on a single device. Seconds per step × the steps of a run
then gives a same-hardware training time for the Pareto front (time vs. accuracy).

M3 and M4 use their Stage-1 checkpoints when present; otherwise an untrained LSTM of the
same architecture stands in — FLOPs and timing do not depend on φ's values.
"""

from __future__ import annotations

import functools

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from reglo.config import compose_run, load_group_file
from reglo.data.loaders import build_optimizee_data
from reglo.diagnostics.flops import count_flops, lstm_flops_per_coordinate, time_steps
from reglo.models.common import count_parameters
from reglo.models.optimizees import build_optimizee
from reglo.models.optimizers import FrozenLSTMOptimizer, build_lstm_optimizer
from reglo.training.methods import Method, build_method
from reglo.utils.runtime import environment_info, resolve_device

__all__ = ["PROTOCOL_VARIANTS", "profile_pair"]

REGULARIZERS = ("l1", "l2", "sam", "gam", "gsam")
PROTOCOL_VARIANTS: List[Tuple[str, str]] = (
    [("m1_adamw", "none")]
    + [("m2_adamw_reg", r) for r in REGULARIZERS]
    + [("m3_lstm", "none")]
    + [("m4_lstm_reg", r) for r in REGULARIZERS]
)


def _method_for_profiling(cfg, model, steps_per_epoch: int, device) -> Tuple[Method, Optional[str]]:
    try:
        method = build_method(cfg, model, steps_per_epoch, steps_per_epoch * int(cfg.training.epochs), device)
        return method, ("trained" if cfg.method.optimizer_kind == "learned" else None)
    except FileNotFoundError:
        lstm, features = build_lstm_optimizer(cfg.optimizer)
        optimizer = FrozenLSTMOptimizer(model.parameters(), lstm.to(device), features, chunk_size=int(cfg.optimizer.chunk_size))
        return Method(cfg, model, optimizer), "untrained"


def profile_pair(
    base_overrides: Sequence[str],
    variants: Sequence[Tuple[str, str]] = PROTOCOL_VARIANTS,
    warmup: int = 20,
    timed: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Time and count one training step of every variant on the pair named in ``base_overrides``."""
    rows: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}
    for method_name, reg in variants:
        args = [f"method={method_name}", *base_overrides] + ([f"regularizer={reg}"] if reg != "none" else [])
        cfg = compose_run(args)
        device = torch.device(device) if device is not None else resolve_device(str(cfg.device))
        data = build_optimizee_data(cfg, device)
        torch.manual_seed(int(cfg.seed))
        model = build_optimizee(cfg.optimizee).to(device)
        model.train()
        method, phi = _method_for_profiling(cfg, model, len(data.train), device)
        x, y = next(iter(data.train))
        step = functools.partial(method.step, x, y)
        flops = count_flops(step)
        timing = time_steps(step, device, warmup, timed)
        epochs = int(cfg.training.epochs)
        rows.append({
            "label": method.label,
            "method": method_name,
            "regularizer": reg,
            "passes": method.passes,
            "phi": phi,
            "flops_per_step": flops,
            "seconds_per_step": timing["seconds_per_step"],
            "steps_per_epoch": len(data.train),
            # training steps only: evaluation passes come on top (roughly 10–20 %)
            "estimated_hours_per_run": timing["seconds_per_step"] * len(data.train) * epochs / 3600.0,
        })
        if not info:
            lstm, _ = build_lstm_optimizer(cfg.optimizer if cfg.optimizer.kind == "learned"
                                           else load_group_file("optimizer", "lstm"))
            info = {
                "optimizee": str(cfg.selected.optimizee),
                "dataset": str(cfg.selected.dataset),
                "profile": str(cfg.profile),
                "artifacts": str(cfg.paths.artifacts),
                "batch_size": int(cfg.training.batch_size),
                "epochs": epochs,
                "num_params": count_parameters(model),
                "lstm_flops_per_parameter": lstm_flops_per_coordinate(lstm),
            }
        del step, method, model, data, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base = next((r for r in rows if r["method"] == "m1_adamw"), rows[0])
    for r in rows:
        r["flops_vs_m1"] = r["flops_per_step"] / base["flops_per_step"]
        r["time_vs_m1"] = r["seconds_per_step"] / base["seconds_per_step"]
    env = environment_info(device)
    return {**info, "device": str(device), "gpu": env.get("gpu"), "warmup_steps": warmup, "timed_steps": timed, "rows": rows}

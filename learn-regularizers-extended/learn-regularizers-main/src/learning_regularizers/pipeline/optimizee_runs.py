"""Cached optimizee trainings.

An optimizee training is identified by what actually determines it: the optimizer that drives it
(B with an *effective* regularizer, or a specific learned-optimizer checkpoint), the optimizee
architecture, the dataset, the seed and the resolved arguments. Identical trainings requested by
different phases (e.g. method 1 on ConvNet-4 + SVHN in phase 1 and phase 2, or method 2 with its
regularizer stripped in phase 2) are therefore computed once and reused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch

from . import DMO_DIR  # noqa: F401
from .artifacts import read_json, subdir, write_json
from .config import namespace_to_dict, optimizee_training_args
from .data import get_dataset, make_loader, num_classes, subsample
from .flops import FlopMeter
from .learned_optimizer import LSTMLearnedOptimizer
from .models import build_optimizee
from .trainers import make_baseline_step_fn, make_lo_step_fn, train_optimizee


@dataclass(frozen=True)
class OptimizerDescriptor:
    kind: str                              # "baseline" | "lo"
    base_optimizer: Optional[str] = None   # baseline only
    regularizer: Optional[str] = None      # effective R used while training this optimizee (baseline only)
    lo_run_id: Optional[str] = None        # lo only
    lo_checkpoint: Optional[str] = None    # lo only

    @property
    def tag(self) -> str:
        if self.kind == "baseline":
            return f"B-{self.base_optimizer}_R-{self.regularizer or 'none'}"
        return f"LO-{self.lo_run_id}"


def _file_digest(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:12]


def optimizee_seeds(cfg) -> List[int]:
    args = optimizee_training_args(cfg, None, None)
    return [int(args.seed) + i for i in range(int(args.n_training))]


def _run_key(cfg, desc: OptimizerDescriptor, optimizee: str, dataset: str, seed: int, args) -> str:
    payload = {
        "desc": {k: v for k, v in asdict(desc).items() if k != "lo_checkpoint"},
        "lo_digest": _file_digest(desc.lo_checkpoint) if desc.kind == "lo" else None,
        "optimizee": optimizee, "model_cfg": dict(cfg.models.get(optimizee, {}) or {}),
        "dataset": dataset, "seed": seed, "args": namespace_to_dict(args),
        "synthetic": bool(cfg.experiment.get("synthetic_data", False)),
        "meta_dataset": cfg.meta_training.dataset, "data_seed": int(cfg.experiment.seed),
        "gap": cfg.metrics.gap, "sharpness": dict(cfg.metrics.get("sharpness", {}) or {}),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{desc.tag}__{optimizee}__{dataset}__s{seed}__{digest}"


def run_optimizee_training(cfg, desc: OptimizerDescriptor, optimizee: str, dataset: str, seed: int, device,
                           lo: Optional[LSTMLearnedOptimizer] = None, log=print) -> Dict:
    """Train (or load from cache) one optimizee; returns its record (metrics + checkpoint path)."""
    args = optimizee_training_args(cfg, desc.base_optimizer if desc.kind == "baseline" else None,
                                   desc.regularizer if desc.kind == "baseline" else None)
    key = _run_key(cfg, desc, optimizee, dataset, seed, args)
    store = subdir(cfg, "optimizee_runs")
    json_path, ckpt_path = store / f"{key}.json", store / f"{key}.pt"
    if json_path.exists() and ckpt_path.exists():
        record = read_json(json_path)
        record["cached"] = True
        return record

    torch.manual_seed(seed)
    n_cls = num_classes(dataset)
    model = build_optimizee(optimizee, n_cls, cfg.models).to(device)
    train_ds = get_dataset(dataset, "optimizee_train", cfg)
    test_ds = get_dataset(dataset, "test", cfg)
    train_loader = make_loader(train_ds, args.batch_size, True, seed, cfg, drop_last=True)
    test_loader = make_loader(test_ds, max(256, int(args.batch_size)), False, seed, cfg)
    train_eval_loader = make_loader(subsample(train_ds, args.train_metric_max_samples, seed),
                                    max(256, int(args.batch_size)), False, seed, cfg)
    meter = FlopMeter(cfg.metrics.count_flops)
    steps = int(args.optimizee_epochs)
    if desc.kind == "baseline":
        step_fn = make_baseline_step_fn(model, args, steps, meter)
    else:
        if lo is None:
            raise ValueError("a learned optimizer is required for kind='lo'")
        step_fn = make_lo_step_fn(model, lo, meter, int(cfg.models.lo_chunk_size))

    prefix = f"[{desc.tag} | {optimizee} | {dataset} | seed {seed}]"
    log(f"{prefix} training {steps} steps")
    result = train_optimizee(model, step_fn, train_loader, test_loader, train_eval_loader, steps,
                             int(args.eval_every), device, meter, gap_kind=cfg.metrics.gap,
                             log_every=max(1, steps // 5), log_prefix=prefix)

    sharp_cfg = cfg.metrics.get("sharpness", {}) or {}
    if sharp_cfg.get("enabled", False):
        from analysis import max_grad_norm_PGD, max_loss_PGD  # base PGD neighbourhood analysis
        import torch.nn.functional as F
        model.zero_grad()
        x, y = next(iter(train_eval_loader))
        F.nll_loss(model(x.to(device)), y.to(device)).backward()   # max_grad_norm_PGD needs .grad
        eps, r_norm = float(sharp_cfg.get("eps", 0.1)), sharp_cfg.get("r_norm", "euc")
        result["final"]["sharpness"] = {
            "eps": eps,
            "max_loss_profile": max_loss_PGD(model, eps, train_eval_loader, device, r_norm=r_norm),
            "max_grad_norm_profile": max_grad_norm_PGD(model, eps, train_eval_loader, device, r_norm=r_norm),
        }

    torch.save(model.state_dict(), ckpt_path)
    record = {"key": key, "descriptor": asdict(desc), "optimizee": optimizee, "dataset": dataset, "seed": seed,
              "num_classes": n_cls, "final": result["final"], "curve": result["curve"],
              "checkpoint": str(ckpt_path), "args": namespace_to_dict(args), "cached": False}
    write_json(json_path, record)
    log(f"{prefix} done: test_acc {result['final']['test_acc']:.2f}  gap {result['final']['gen_gap']:.2f}  "
        f"GFLOPs {result['final']['train_flops'] / 1e9:.2f}")
    return record


def load_optimizee_from_record(cfg, record: Dict, device) -> torch.nn.Module:
    model = build_optimizee(record["optimizee"], int(record["num_classes"]), cfg.models).to(device)
    model.load_state_dict(torch.load(record["checkpoint"], map_location=device))
    return model

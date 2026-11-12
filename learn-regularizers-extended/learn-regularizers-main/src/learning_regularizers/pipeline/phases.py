"""The three pipeline phases.

Phase 1  Meta-training (seen task = meta_training.optimizee + meta_training.dataset)
    methods 3-6 : meta-train the learned optimizer            -> LO checkpoint
    methods 1-2 : no optimizer training; train the seen optimizee with B (+ R)
                  for every seed                               -> optimizee checkpoints

Phase 2  Optimizer evaluation (optimizer frozen, regularizers removed)
    every method trains fresh optimizees (seen + unseen architectures) on the seen dataset.
    learned optimizers: plain frozen LO, no R. Baselines: B only when
    strip_regularizers_for_baselines (so method 2 == method 1 here), else B + R.

Phase 3  Optimizee evaluation (optimizer and optimizee frozen)
    the method's seen-task optimizee (methods 1-2: the phase-1 optimizee trained with B (+ R);
    methods 3-6: the phase-2 optimizee trained by the frozen LO) is evaluated
      seen dataset   : directly (same label space)
      unseen datasets: frozen features + new linear head (linear probe)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .artifacts import lo_checkpoint_path, read_json, subdir, write_json
from .config import optimizee_training_args
from .data import DatasetUnavailable, get_dataset, make_loader, num_classes, subsample
from .evaluation import evaluate, linear_probe, summarize
from .flops import FlopMeter
from .meta_train import MetaTrainer, load_learned_optimizer, meta_fingerprint
from .methods import RunKey
from .optimizee_runs import OptimizerDescriptor, load_optimizee_from_record, optimizee_seeds, run_optimizee_training
from .trainers import make_baseline_step_fn, make_lo_step_fn

METRIC_KEYS = ("test_acc", "gen_gap", "flops", "train_acc", "train_loss", "test_loss", "gen_gap_acc",
               "gen_gap_loss")


def _aggregate(per_seed: List[Dict]) -> Dict:
    out_mean, out_std = {}, {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in per_seed if r.get(k) is not None and np.isfinite(r[k])]
        out_mean[k] = float(np.mean(vals)) if vals else None
        out_std[k] = float(np.std(vals)) if len(vals) > 1 else (0.0 if vals else None)
    return {"mean": out_mean, "std": out_std, "n_seeds": len(per_seed),
            "n_diverged": int(sum(bool(r.get("diverged")) for r in per_seed))}


def _descriptor(cfg, key: RunKey, phase: int) -> OptimizerDescriptor:
    if key.spec.uses_lo:
        return OptimizerDescriptor("lo", lo_run_id=key.run_id, lo_checkpoint=str(lo_checkpoint_path(cfg, key.run_id)))
    if phase == 2 and cfg.optimizer_evaluation.strip_regularizers_for_baselines:
        reg = None    # evaluation phase: regularizers removed
    else:
        reg = key.regularizer   # training phase (or original protocol): B + lambda R
    return OptimizerDescriptor("baseline", base_optimizer=key.base_optimizer, regularizer=reg)


def _train_seeds(cfg, key, desc, optimizee, dataset, device, log) -> List[Dict]:
    lo = load_learned_optimizer(cfg, key, device) if desc.kind == "lo" else None
    return [run_optimizee_training(cfg, desc, optimizee, dataset, seed, device, lo=lo, log=log)
            for seed in optimizee_seeds(cfg)]


# ============================================================================ phase 1

def phase1(cfg, keys: Iterable[RunKey], device, log=print, force: bool = False) -> None:
    mt = cfg.meta_training
    for key in keys:
        out = subdir(cfg, "phase1") / f"{key.run_id}.json"
        if key.spec.uses_lo:
            sidecar = lo_checkpoint_path(cfg, key.run_id).with_suffix(".json")
            saved = read_json(sidecar) if sidecar.exists() else None
            if saved and not force and saved.get("fingerprint") == meta_fingerprint(cfg, key):
                log(f"[phase 1] {key.run_id}: learned optimizer already meta-trained, skipping")
                summary = saved
            else:
                if saved and not force:
                    log(f"[phase 1] {key.run_id}: saved learned optimizer was trained with different settings; "
                        f"re-training")
                log(f"[phase 1] {key.run_id}: meta-training learned optimizer on {mt.optimizee} + {mt.dataset}")
                summary = MetaTrainer(cfg, key, device, log=log).fit()
            write_json(out, {"phase": 1, "run": key.as_dict(), "type": "meta_training",
                             "checkpoint": summary["checkpoint"], "best_eval_loss": summary["best_eval_loss"],
                             "meta_train_flops": summary["meta_train_flops"],
                             "meta_eval_flops": summary.get("meta_eval_flops"),
                             "wall_time_s": summary["wall_time_s"]})
        else:
            desc = _descriptor(cfg, key, phase=1)
            log(f"[phase 1] {key.run_id}: no optimizer to train; training {mt.optimizee} on {mt.dataset} "
                f"with {desc.tag}")
            runs = _train_seeds(cfg, key, desc, mt.optimizee, mt.dataset, device, log)
            per_seed = [dict(r["final"], flops=r["final"]["train_flops"]) for r in runs]
            write_json(out, {"phase": 1, "run": key.as_dict(), "type": "optimizee_training",
                             "descriptor": desc.__dict__, "optimizee": mt.optimizee, "dataset": mt.dataset,
                             "optimizee_runs": [r["key"] for r in runs], **_aggregate(per_seed)})


# ============================================================================ phase 2

def phase2(cfg, keys: Iterable[RunKey], device, log=print) -> None:
    oe = cfg.optimizer_evaluation
    for key in keys:
        desc = _descriptor(cfg, key, phase=2)
        for optimizee in oe.optimizees:
            setting = "seen_optimizee" if optimizee == oe.seen_optimizee else "unseen_optimizee"
            log(f"[phase 2] {key.run_id} | {optimizee} ({setting}) on {oe.dataset} with {desc.tag}")
            runs = _train_seeds(cfg, key, desc, optimizee, oe.dataset, device, log)
            per_seed = [dict(r["final"], flops=r["final"]["train_flops"]) for r in runs]
            write_json(subdir(cfg, "phase2") / f"{key.run_id}__{optimizee}.json", {
                "phase": 2, "run": key.as_dict(), "setting": setting, "optimizee": optimizee,
                "dataset": oe.dataset, "descriptor": desc.__dict__,
                "regularizers_stripped": desc.kind == "lo" or desc.regularizer is None,
                "optimizee_runs": [r["key"] for r in runs], "flops_definition": "optimizee training FLOPs",
                "per_seed": per_seed, **_aggregate(per_seed)})


# ============================================================================ phase 3

def _seen_task_optimizees(cfg, key: RunKey, optimizee: str, device, log) -> List[Dict]:
    """The optimizee each method produces on the seen dataset (phase 1 for 1-2, phase 2 for 3-6)."""
    desc = _descriptor(cfg, key, phase=1)  # training-time optimizer: B + R for method 2; frozen LO for 3-6
    return _train_seeds(cfg, key, desc, optimizee, cfg.meta_training.dataset, device, log)


def _probe_step_factory(cfg, key: RunKey, device):
    """probe.optimizer == 'method': train the head with the method's frozen optimizer, without R."""
    if cfg.optimizee_evaluation.probe.optimizer != "method":
        return None
    if key.spec.uses_lo:
        lo = load_learned_optimizer(cfg, key, device)
        return lambda meter: (lambda head, total: make_lo_step_fn(head, lo, meter, int(cfg.models.lo_chunk_size)))
    args = optimizee_training_args(cfg, key.base_optimizer, None)
    return lambda meter: (lambda head, total: make_baseline_step_fn(head, args, total, meter))


def phase3(cfg, keys: Iterable[RunKey], device, log=print) -> None:
    ev = cfg.optimizee_evaluation
    datasets = [ev.seen_dataset] + [d for d in ev.unseen_datasets if d != ev.seen_dataset]
    for key in keys:
        factory = _probe_step_factory(cfg, key, device)
        for optimizee in ev.optimizees:
            records = _seen_task_optimizees(cfg, key, optimizee, device, log)
            for dataset in datasets:
                setting = "seen_dataset" if dataset == ev.seen_dataset else "unseen_dataset"
                out = subdir(cfg, "phase3") / f"{key.run_id}__{optimizee}__{dataset}.json"
                try:
                    per_seed = [_evaluate_frozen(cfg, rec, dataset, setting, device, factory, log) for rec in records]
                except DatasetUnavailable as exc:
                    if not ev.skip_missing_datasets:
                        raise
                    log(f"[phase 3] skipping {dataset}: {exc}")
                    write_json(out, {"phase": 3, "run": key.as_dict(), "setting": setting, "optimizee": optimizee,
                                     "dataset": dataset, "status": "skipped", "reason": str(exc)})
                    continue
                write_json(out, {"phase": 3, "run": key.as_dict(), "setting": setting, "optimizee": optimizee,
                                 "dataset": dataset, "status": "ok",
                                 "protocol": "direct" if setting == "seen_dataset" else "linear_probe",
                                 "flops_definition": "optimizee training FLOPs + evaluation FLOPs",
                                 "source_optimizee_runs": [r["key"] for r in records],
                                 "per_seed": per_seed, **_aggregate(per_seed)})
                log(f"[phase 3] {key.run_id} | {optimizee} -> {dataset}: test_acc "
                    f"{np.mean([r['test_acc'] for r in per_seed]):.2f}")


def _evaluate_frozen(cfg, record: Dict, dataset: str, setting: str, device, factory, log) -> Dict:
    ev = cfg.optimizee_evaluation
    seed = int(record["seed"])
    model = load_optimizee_from_record(cfg, record, device)
    for p in model.parameters():
        p.requires_grad_(False)
    meter = FlopMeter(cfg.metrics.count_flops)
    if setting == "seen_dataset":
        train_ds = subsample(get_dataset(dataset, "optimizee_train", cfg),
                             cfg.optimizee_training.train_metric_max_samples, seed)
        test_ds = get_dataset(dataset, "test", cfg)
        tr_loss, tr_acc = evaluate(model, make_loader(train_ds, 512, False, seed, cfg), device, meter)
        te_loss, te_acc = evaluate(model, make_loader(test_ds, 512, False, seed, cfg), device, meter)
        res = summarize(tr_loss, tr_acc, te_loss, te_acc, cfg.metrics.gap)
    else:
        probe = ev.probe
        train_ds = subsample(get_dataset(dataset, "optimizee_train", cfg), probe.max_train_samples, seed)
        test_ds = subsample(get_dataset(dataset, "test", cfg), probe.max_test_samples, seed)
        res = linear_probe(model, make_loader(train_ds, 512, False, seed, cfg),
                           make_loader(test_ds, 512, False, seed, cfg), num_classes(dataset), device, probe,
                           cfg.metrics.gap, seed, meter, make_step_fn=factory(meter) if factory else None)
    res["eval_flops"] = int(meter.total)
    res["producing_train_flops"] = int(record["final"]["train_flops"])
    res["flops"] = res["eval_flops"] + res["producing_train_flops"]
    res["seed"] = seed
    res["diverged"] = bool(record["final"].get("diverged", False))
    return res

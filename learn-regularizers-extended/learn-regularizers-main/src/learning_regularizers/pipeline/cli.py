"""Command-line entry point:  python experiments/pipeline/run_pipeline.py [options]

Examples
--------
  # everything (phase 1 -> 2 -> 3 -> tables)
  python experiments/pipeline/run_pipeline.py

  # only meta-train the proposed method with GAM
  python experiments/pipeline/run_pipeline.py --phases 1 --methods 6 --regularizers gam

  # tiny end-to-end smoke test without downloads
  python experiments/pipeline/run_pipeline.py --quick --set experiment.synthetic_data=true

  # meta-train on ResNet-18 + ImageNet (seen/unseen columns follow automatically)
  python experiments/pipeline/run_pipeline.py --meta-optimizee resnet18 --meta-dataset imagenet

  # list the run grid (e.g. to build a SLURM array)
  python experiments/pipeline/run_pipeline.py --list
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from .artifacts import save_resolved_config
from .config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config
from .methods import run_grid
from .phases import phase1, phase2, phase3
from .report import build_report


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Meta-training -> optimizer evaluation -> optimizee evaluation")
    p.add_argument("--config", default=str(DEFAULT_PIPELINE_CONFIG), help="pipeline YAML")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any pipeline.yaml entry, e.g. --set meta_training.overrides.max_epochs=200")
    p.add_argument("--quick", action="store_true", help="tiny preset for smoke tests")
    p.add_argument("--phases", nargs="+", default=["1", "2", "3", "report"],
                   choices=["1", "2", "3", "report"])
    task = p.add_argument_group("task selection (shortcuts for --set; seen/unseen settings follow automatically)")
    task.add_argument("--meta-optimizee", default=None, choices=["mlp", "convnet4", "resnet18"],
                      help="optimizee the LO is meta-trained on (= the seen optimizee)")
    task.add_argument("--meta-dataset", default=None,
                      choices=["svhn", "mnist", "fmnist", "cifar10", "cifar100", "imagenet"],
                      help="dataset the LO is meta-trained on (= the seen dataset)")
    task.add_argument("--eval-optimizees", nargs="+", default=None, choices=["mlp", "convnet4", "resnet18"],
                      help="phase-2 optimizees (the seen one is always added)")
    task.add_argument("--unseen-datasets", nargs="+", default=None,
                      choices=["svhn", "mnist", "fmnist", "cifar10", "cifar100", "imagenet"],
                      help="phase-3 unseen datasets (default: all except the seen one)")
    p.add_argument("--methods", nargs="+", type=int, default=None, help="subset of methods (1-6)")
    p.add_argument("--base-optimizers", nargs="+", default=None, help="subset of B (adam sgd)")
    p.add_argument("--regularizers", nargs="+", default=None, help="subset of R (l1 l2 sam gam gsam)")
    p.add_argument("--run-ids", nargs="+", default=None,
                   help="restrict to these run ids (see --list), e.g. M6_LO_gam M2_sgd_sam")
    p.add_argument("--device", default=None, help="overrides experiment.device")
    p.add_argument("--force-meta", action="store_true", help="re-run meta-training even if a checkpoint exists")
    p.add_argument("--list", action="store_true", help="print the run grid and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sets = []
    if args.meta_optimizee:
        sets.append(f"meta_training.optimizee={args.meta_optimizee}")
    if args.meta_dataset:
        sets.append(f"meta_training.dataset={args.meta_dataset}")
    if args.eval_optimizees:
        sets.append(f"optimizer_evaluation.optimizees={args.eval_optimizees}")
    if args.unseen_datasets:
        sets.append(f"optimizee_evaluation.unseen_datasets={args.unseen_datasets}")
    sets += list(args.set)          # explicit --set wins
    if args.methods:
        sets.append(f"methods={args.methods}")
    if args.base_optimizers:
        sets.append(f"base_optimizers={args.base_optimizers}")
    if args.regularizers:
        sets.append(f"regularizers={args.regularizers}")
    cfg = load_pipeline_config(args.config, sets=sets, quick=args.quick)
    keys = list(run_grid(cfg.methods, cfg.base_optimizers, cfg.regularizers))
    if args.run_ids:
        unknown = set(args.run_ids) - {k.run_id for k in keys}
        if unknown:
            raise SystemExit(f"unknown run ids {sorted(unknown)}; see --list")
        keys = [k for k in keys if k.run_id in set(args.run_ids)]

    if args.list:
        for k in keys:
            print(k.run_id)
        return 0

    device = resolve_device(args.device or cfg.experiment.device)
    torch.manual_seed(int(cfg.experiment.seed))
    path = save_resolved_config(cfg)
    print(f"device: {device} | runs: {len(keys)} | resolved config: {path}", flush=True)

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    if "1" in args.phases:
        phase1(cfg, keys, device, log=log, force=args.force_meta)
    if "2" in args.phases:
        phase2(cfg, keys, device, log=log)
    if "3" in args.phases:
        phase3(cfg, keys, device, log=log)
    if "report" in args.phases:
        build_report(cfg, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())

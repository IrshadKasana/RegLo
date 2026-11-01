#!/usr/bin/env python
"""Stages 2–3 — train one optimizee with M1, M2, M3 or M4.

    python scripts/train_optimizee.py method=m1_adamw optimizee=convnet4 dataset=svhn
    python scripts/train_optimizee.py method=m2_adamw_reg regularizer=sam optimizee=resnet18 dataset=cifar10 seed=43
    python scripts/train_optimizee.py method=m3_lstm optimizee=resnet18 dataset=cifar10
    python scripts/train_optimizee.py method=m4_lstm_reg regularizer=sam optimizee=resnet18 dataset=cifar10 profile=smoke

M3 and M4 load the LSTM optimizer from Stage 1 (artifacts/<profile>/meta_training/...). Results go
to artifacts/<profile>/runs/<study>/<optimizee>-<dataset>/<method>[-<r>]/seed<k>/. Running the
same command again resumes an interrupted run, or returns at once if it already finished.
Exit codes: 0 finished · 1 crashed · 2 invalid config · 3 missing φ / data / GPU · 4 running
elsewhere · 75 stopped at max_hours (run again to continue).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError, compose_run  # noqa: E402
from reglo.training import CheckpointMismatch, OptimizeeTrainer  # noqa: E402
from reglo.utils.locking import RunLocked, run_lock  # noqa: E402
from reglo.utils.logging import safe_console, setup_logging  # noqa: E402
from reglo.utils.paths import run_dir  # noqa: E402
from reglo.utils.runtime import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_FAILED,
    EXIT_INTERRUPTED,
    EXIT_LOCKED,
    EXIT_MISSING,
    EXIT_OK,
    DeviceUnavailable,
    seed_everything,
)


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Stages 2–3: train one optimizee with M1, M2, M3 or M4.")
    parser.add_argument("overrides", nargs="*", help="e.g. method=m2_adamw_reg regularizer=sam optimizee=resnet18 dataset=cifar10")
    args = parser.parse_args(argv)
    try:
        cfg = compose_run(args.overrides)
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_CONFIG

    out = run_dir(cfg)
    try:
        with run_lock(out):  # a second job on the same run stops here instead of corrupting it
            logger = setup_logging(out / "train.log")
            seed_everything(int(cfg.seed), bool(cfg.deterministic))
            try:
                final = OptimizeeTrainer(cfg, logger=logger).run()
            except (CheckpointMismatch, FileNotFoundError, DeviceUnavailable) as err:
                logger.error("%s", err)
                return EXIT_MISSING
    except RunLocked as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_LOCKED
    status = final.get("status")
    if status == "interrupted":
        return EXIT_INTERRUPTED
    return EXIT_OK if status in ("completed", "diverged") else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())

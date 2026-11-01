#!/usr/bin/env python
"""Stage 1 — meta-train one LSTM optimizer (Algorithm 1).

    python scripts/meta_train.py meta_training=m3_lstm                      # φ for M3 (round A)
    python scripts/meta_train.py meta_training=m4_lstm_reg regularizer=sam  # φ*_sam for M4 (round A)
    python scripts/meta_train.py meta_training=m4_lstm_reg regularizer=sam meta_tag=B   # round B: MLP / FMNIST
    python scripts/meta_train.py meta_training=m4_lstm_reg regularizer=l1 profile=smoke

Writes artifacts/<profile>/meta_training/<round>/<m3_lstm | m4_lstm_reg-<r>>/ (meta_tag = the
round, default A; configs/rounds.yaml gives its source pair) — phi_best.pt is what the M3 / M4
training runs of that round load. Running the same
command again resumes from phi_last.pt, or returns at once if the run already finished.
Exit codes: 0 finished · 1 crashed · 2 invalid config · 3 missing data / GPU · 4 running
elsewhere · 75 stopped at max_hours (run again to continue).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError, compose_meta  # noqa: E402
from reglo.meta_training import MetaTrainer  # noqa: E402
from reglo.utils.locking import RunLocked, run_lock  # noqa: E402
from reglo.utils.logging import safe_console, setup_logging  # noqa: E402
from reglo.utils.paths import meta_run_dir, regularizer_of  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Stage 1: meta-train the LSTM optimizer for M3 or M4.")
    parser.add_argument("overrides", nargs="*", help="e.g. meta_training=m4_lstm_reg regularizer=sam")
    args = parser.parse_args(argv)
    try:
        cfg = compose_meta(args.overrides)
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_CONFIG

    out = meta_run_dir(cfg)
    try:
        with run_lock(out):  # a second job on the same arm stops here instead of corrupting it
            logger = setup_logging(out / "meta_train.log")
            logger.info(
                "Stage 1 — round %s: %s, regularizer %s, %s on %s, profile %s → %s",
                cfg.meta_tag, cfg.meta_training.name, regularizer_of(cfg), cfg.selected.optimizee,
                cfg.selected.dataset, cfg.profile, out,
            )
            seed_everything(int(cfg.seed), bool(cfg.deterministic))
            try:
                summary = MetaTrainer(cfg, logger=logger).fit()
            except (FileNotFoundError, DeviceUnavailable) as err:
                logger.error("%s", err)
                return EXIT_MISSING
    except RunLocked as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_LOCKED
    status = summary.get("status")
    if status == "interrupted":
        return EXIT_INTERRUPTED
    return EXIT_OK if status == "completed" else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())

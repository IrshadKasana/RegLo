#!/usr/bin/env python
"""Stage 4 — FLOPs and seconds per training step of all 12 variants, back-to-back on one GPU.

    python scripts/profile_compute.py --index 1                 # pair 1 of stage4_analysis.yaml's compute.pairs
    python scripts/profile_compute.py                           # every pair listed there, one after another
    python scripts/profile_compute.py optimizee=resnet18 dataset=cifar10
    python scripts/profile_compute.py --profile smoke           # at the smoke profile's scale

Writes artifacts/<profile>/diagnostics/compute/<optimizee>-<dataset>.json and prints a table
with an estimate of each run's training time (seconds per step × steps of the profile's
budget) — run it before the study to plan walltimes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError, load_experiment  # noqa: E402
from reglo.diagnostics.compute import profile_pair  # noqa: E402
from reglo.manifest import ManifestError, compute_overrides  # noqa: E402
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.utils.logging import safe_console, write_json  # noqa: E402
from reglo.utils.runtime import resolve_device  # noqa: E402


def print_table(result) -> None:
    print(f"\n{result['optimizee']} × {result['dataset']} on {result['gpu'] or result['device']}  "
          f"(profile {result['profile']}, batch {result['batch_size']}, {result['num_params']:,} parameters, "
          f"{result['epochs']} epochs)")
    print(f"{'variant':<22}{'passes':>7}{'GFLOPs/step':>13}{'×M1':>7}{'ms/step':>10}{'×M1':>7}{'h/run':>8}")
    for r in result["rows"]:
        name = r["label"] + ("" if r["regularizer"] == "none" else f" {r['regularizer']}")
        print(f"{name:<22}{r['passes']:>7}{r['flops_per_step'] / 1e9:>13.2f}{r['flops_vs_m1']:>7.2f}"
              f"{r['seconds_per_step'] * 1e3:>10.2f}{r['time_vs_m1']:>7.2f}{r['estimated_hours_per_run']:>8.2f}")
    print("h/run: training steps only (evaluation adds roughly 10–20 %); multiply by the seeds for a grid")


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Stage 4: profile every method variant on one GPU.")
    parser.add_argument("overrides", nargs="*", help="optimizee=<name> dataset=<name> [other overrides]")
    parser.add_argument("--experiment", default="stage4_analysis")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--index", type=int, help="profile only this pair of the experiment's compute.pairs")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--timed", type=int)
    parser.add_argument("--device", default=None, help="default: the profile's device (production: cuda)")
    args = parser.parse_args(argv)

    try:
        profile = set_active_profile(args.profile or active_profile())
        spec = load_experiment(args.experiment, profile=profile)
        block = spec["compute"]
        if args.overrides:
            jobs = [[*args.overrides, f"profile={profile}"]]
        else:
            indices = range(len(block["pairs"])) if args.index is None else [args.index]
            jobs = [compute_overrides(spec, i) for i in indices]
    except (ConfigError, ManifestError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    warmup = args.warmup if args.warmup is not None else int(block["warmup_steps"])
    timed = args.timed if args.timed is not None else int(block["timed_steps"])
    device = resolve_device(args.device) if args.device else None

    for overrides in jobs:
        result = profile_pair(overrides, warmup=warmup, timed=timed, device=device)
        out = Path(result["artifacts"]) / "diagnostics" / "compute" / f"{result['optimizee']}-{result['dataset']}.json"
        write_json(out, result)
        print_table(result)
        print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Run the whole study on this machine — Stage 1 → Stages 2–3 → Stage 4 → report → check.

    python scripts/run_study.py --profile smoke              # laptop: everything, in minutes (real datasets)
    python scripts/run_study.py --profile smoke_offline      # … without downloading anything
    python scripts/run_study.py --gpus all                   # a cloud GPU machine: the real study, every GPU busy
    python scripts/run_study.py --gpus 0,1 --per-gpu 2       # two runs per GPU (small optimizees)
    python scripts/run_study.py --profile smoke --jobs 4     # four CPU processes at a time
    python scripts/run_study.py --stages 1                   # only Stage 1
    python scripts/run_study.py --extended --pilot           # with the optional grids
    python scripts/run_study.py --regularizers sam           # the SAM block of every cube (with M1, M3)
    python scripts/run_study.py --only round=A               # round A first (M1 runs included), B later
    python scripts/run_study.py --only optimizee=convnet4    # filters as in make_manifest.py
    python scripts/run_study.py --dry-run                    # the plan, nothing runs

Without --gpus/--jobs the runs go one after another in this process, with their output here
(breakpoints work in VS Code). With --gpus each run is a child process on one GPU and writes
its output to artifacts/<profile>/logs/; M1 and M2 runs start at once, M3/M4 runs as soon as
the Stage-1 optimizer they load (their round's) is trained. Starting the same command again skips what has
finished and resumes what was interrupted. Exit code 0 when every run finished and every
connectivity check passed.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.checks import check_study  # noqa: E402
from reglo.config import ConfigError, compose_run  # noqa: E402
from reglo.manifest import ManifestError, parse_only  # noqa: E402
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.runner import Runner, audit_tasks, plan_study  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import profile_artifacts, set_artifacts_root  # noqa: E402


def gpu_slots(spec: str, per_gpu: int):
    if not spec:
        return []
    if spec == "all":
        import torch

        ids = [str(i) for i in range(torch.cuda.device_count())]
        if not ids:
            raise SystemExit("error: --gpus all, but no CUDA GPU is visible")
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
    return [i for i in ids for _ in range(max(per_gpu, 1))]


def device_check(profile: str) -> None:
    """Production asks for CUDA: say so before any run starts, not 150 times."""
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", f"profile={profile}"])
    if str(cfg.device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(f"error: profile '{profile}' runs on CUDA GPUs and none is visible here. "
                             f"On a laptop: --profile smoke (or smoke_offline).")


def data_check(tasks, profile: str) -> None:
    """Every dataset the plan trains on must be on disk (prepared once, on a machine with internet)."""
    from reglo.data import DatasetNotPrepared, load_image_data
    from reglo.data.datasets import is_standin

    names = {a.split("=", 1)[1] for t in tasks for a in t.line if a.startswith("dataset=")}
    for task in tasks:  # a meta_train line without dataset=: its round's source, or the base config's
        if task.line[0] == "meta_train" and not any(a.startswith("dataset=") for a in task.line):
            from reglo.config import compose_meta

            names.add(str(compose_meta(task.line[1:]).selected.dataset))
    missing = []
    for name in sorted(names):
        cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", f"dataset={name}", f"profile={profile}"])
        if is_standin(cfg.dataset):
            continue
        try:
            for train in (True, False):
                load_image_data(cfg.dataset, cfg.paths.data, train=train)  # builds the fast cache the first time
        except DatasetNotPrepared:
            missing.append(name)
    if missing:
        raise SystemExit(f"error: not downloaded yet: {', '.join(missing)}. Run once (needs internet):\n"
                         f"    python scripts/prepare_data.py {' '.join(missing)}\n"
                         f"or use --profile smoke_offline for a check without the datasets.")


def print_plan(tasks) -> None:
    by_stage = Counter(t.stage for t in tasks)
    print("plan: " + ", ".join(f"stage {s}: {n}" for s, n in by_stage.items()))
    for task in tasks:
        waits = f"   ← waits for {', '.join(sorted(task.needs))}" if task.needs else ""
        print(f"  [{task.stage:>5}] {task.title}{waits}")


def summarize(tasks, started: float) -> int:
    counts = Counter(t.status for t in tasks)
    minutes = (time.monotonic() - started) / 60.0
    print(f"\n{len(tasks)} runs in {minutes:.1f} min: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    for task in tasks:
        if task.status != "done":
            where = f" — log: {task.log}" if task.log else ""
            print(f"  {task.status:<11} stage {task.stage} · {task.title}{(' — ' + task.note) if task.note else ''}{where}")
    if counts.get("interrupted"):
        print("  interrupted runs continue where they stopped when this command is started again")
    return 0 if counts.get("done", 0) == len(tasks) else 1


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Run every stage of a study on this machine.")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--study", default="main")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--extended", action="store_true", help="also the optional extended grid (WRN-28, ViT, CIFAR-100)")
    parser.add_argument("--pilot", action="store_true", help="also the pilot (M1 vs M3, short)")
    parser.add_argument("--only", action="append", metavar="KEY=VALUE",
                        help="e.g. round=A or optimizee=convnet4 (repeatable; each key where it applies)")
    parser.add_argument("--regularizers", nargs="+", choices=["l1", "l2", "sam", "gam", "gsam"],
                        help="only these cube blocks: their M2/M4 runs and Stage-1 arms, plus M1 and M3")
    parser.add_argument("--gpus", default="", help="'all' or a list such as 0,1,2: one child process per GPU")
    parser.add_argument("--per-gpu", type=int, default=1, help="runs per GPU at a time (default 1)")
    parser.add_argument("--jobs", type=int, default=0, help="CPU child processes at a time (no GPU)")
    parser.add_argument("--artifacts", type=Path, default=None, help="artifacts root (default: $REGLO_ARTIFACTS or "
                                                                     "<repo>/artifacts)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--no-report", action="store_true", help="skip the tables and figures at the end")
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
        set_artifacts_root(args.artifacts)
        profile = set_active_profile(args.profile or active_profile())
        only = parse_only(args.only)
        tasks = plan_study(args.study, args.stages, args.extended, args.pilot, only, profile, write=not args.dry_run,
                           regularizers=args.regularizers)
    except (ConfigError, ManifestError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    folder = profile_artifacts(profile)
    print(f"profile {profile} → {folder}")
    earlier = sum(1 for t in tasks if t.status == "done")
    if earlier:
        print(f"{earlier} of {len(tasks)} runs finished earlier: skipped")
    if args.dry_run:
        print_plan(tasks)
        if 4 in args.stages:
            print("  [    4] + one audit per finished training run (planned after training)")
        return 0
    if not args.gpus:
        device_check(profile)
    data_check(tasks, profile)
    slots = gpu_slots(args.gpus, args.per_gpu) or (["cpu"] * args.jobs if args.jobs > 0 else [])
    threads = None
    if slots and slots[0] == "cpu":
        import os

        threads = max((os.cpu_count() or 2) // len(slots), 1)

    try:
        Runner(tasks, slots, folder / "logs", threads).run()
        if 4 in args.stages:
            audits = audit_tasks(args.study, profile, only=only, regularizers=args.regularizers)
            if audits:
                done = sum(1 for t in audits if t.status == "done")
                print(f"\n=== Stage 4 audits: {len(audits)} finished runs" + (f" ({done} audited earlier)" if done else ""))
                Runner(audits, slots, folder / "logs", threads).run()
                tasks += audits
    except KeyboardInterrupt:
        print("\nstopped — start the same command again to continue (finished runs are skipped, "
              "interrupted ones resume from their last checkpoint)")
        return 130
    code = summarize(tasks, started)

    if not args.no_report and 4 in args.stages:
        from reglo.reporting import build_report

        try:
            import matplotlib  # noqa: F401  (figures need the [analysis] extra)

            figures = True
        except ImportError:
            figures = False
            print("matplotlib is not installed: tables only (pip install -e \".[analysis]\" for the figures)")
        written = build_report(folder, None, None, figures=figures)
        summary = folder / "tables" / args.study / "summary.md"
        print(f"\nreport: {len(written['tables'])} tables, {len(written['figures'])} figures under {folder}")
        if summary.is_file():
            print(f"  → {summary}")
            for cube in sorted((folder / "tables" / args.study).glob("round_*/cube_*.md")):
                print(f"  → {cube}")
            for transfer in sorted((folder / "tables" / args.study).glob("transfer*.md")):
                print(f"  → {transfer}")
    report = check_study(args.study, None, profile)
    print("\n" + report.text())
    return code if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

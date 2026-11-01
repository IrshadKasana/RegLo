#!/usr/bin/env python
"""Stage 4 — audit finished optimizee runs (Hessian, sharpness tests, weight norms and sparsity).

    python scripts/run_diagnostics.py --run artifacts/production/runs/main/resnet18-svhn/m4_lstm_reg-sam/seed42
    python scripts/run_diagnostics.py                      # every finished run of the study
    python scripts/run_diagnostics.py --profile smoke      # … of the smoke profile
    python scripts/run_diagnostics.py --list               # the run folders, one per line
    python scripts/run_diagnostics.py --index 5            # only the 6th of them (an array element)

Settings and study: configs/experiments/stage4_analysis.yaml (or --experiment), with the
profile's numbers. Results: artifacts/<profile>/diagnostics/<study>/<optimizee>-<dataset>/
<method>[-<r>]/seed<k>/diagnostics.json. Runs that already have one are skipped unless --redo.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError, load_experiment  # noqa: E402
from reglo.diagnostics.audit import run_audit  # noqa: E402
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.utils.logging import safe_console, setup_logging  # noqa: E402
from reglo.utils.paths import diagnostics_dir_for, find_runs, profile_artifacts, set_artifacts_root  # noqa: E402
from reglo.utils.runtime import resolve_device  # noqa: E402


def source_runs(spec, artifacts_root=None) -> list:
    study = str(spec["diagnostics"].get("study", spec.get("study", "main")))
    return find_runs(profile_artifacts(spec["profile"], artifacts_root), study)


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Stage 4: audit finished optimizee runs.")
    parser.add_argument("--experiment", default="stage4_analysis", help="diagnostics settings and the study to audit")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--artifacts", type=Path, default=None, help="artifacts root (default: $REGLO_ARTIFACTS or "
                                                                     "<repo>/artifacts)")
    parser.add_argument("--run", type=Path, help="audit just this run folder")
    parser.add_argument("--list", action="store_true", help="print the run folders that would be audited")
    parser.add_argument("--index", type=int, help="audit only the run at this position of --list")
    parser.add_argument("--redo", action="store_true", help="audit again even if diagnostics.json exists")
    parser.add_argument("--device", default=None, help="default: the run's own device setting")
    args = parser.parse_args(argv)

    try:
        set_artifacts_root(args.artifacts)
        profile = set_active_profile(args.profile or active_profile())
        spec = load_experiment(args.experiment, profile=profile)
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if "diagnostics" not in spec:
        print(f"error: experiment '{args.experiment}' has no diagnostics block", file=sys.stderr)
        return 2
    if args.run is not None and not (args.run / "final.json").is_file():
        print(f"error: {args.run} is not a finished run (no final.json)", file=sys.stderr)
        return 2
    runs = [args.run] if args.run else source_runs(spec, args.artifacts)
    if args.list:
        for run in runs:
            print(Path(run).as_posix())
        return 0
    if args.index is not None:
        if not 0 <= args.index < len(runs):
            print(f"error: --index {args.index} is outside 0..{len(runs) - 1}", file=sys.stderr)
            return 2
        runs = [runs[args.index]]

    log = setup_logging()
    device = resolve_device(args.device) if args.device else None
    for run in runs:
        target = diagnostics_dir_for(run) / "diagnostics.json"
        if target.is_file() and not args.redo:
            log.info("skip (done): %s", run)
            continue
        log.info("auditing %s", run)
        result = run_audit(run, spec["diagnostics"], device)
        log.info("→ %s (%s)", target, result["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Turn one experiment grid into a run list (manifest) — or run the whole grid here.

    python scripts/make_manifest.py stage1_meta_training                     # 42 runs: 2 rounds × 6 arms + 30 M2
    python scripts/make_manifest.py stage1_meta_training --kind meta_train --only round=B   # round B's 6 arms
    python scripts/make_manifest.py stage2_optimizer_evaluation              # 117 runs
    python scripts/make_manifest.py stage3_optimizee_evaluation --only optimizee=resnet18
    python scripts/make_manifest.py stage4_analysis --kind diagnostics       # after Stages 2–3 finished
    python scripts/make_manifest.py stage2_optimizer_evaluation --profile smoke --run   # build and run here

Writes artifacts/<profile>/manifests/<experiment>[…].txt, one line per run; the cluster scripts
(cluster/slurm/submit.sh, cluster/pbs/submit.sh) submit it as one array job. Every line is
validated first: a typo or an invalid combination stops here, not on the cluster. For the
whole study in stage order use scripts/run_study.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError, experiment_names, load_experiment  # noqa: E402
from reglo.manifest import (  # noqa: E402
    KINDS,
    ManifestError,
    expand,
    manifest_path,
    parse_only,
    run_line,
    validate,
    write_manifest,
)
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import set_artifacts_root  # noqa: E402


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Experiment grid → manifest (one line per run).")
    parser.add_argument("experiment", help=f"one of: {', '.join(experiment_names())}")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--artifacts", type=Path, default=None, help="artifacts root (default: $REGLO_ARTIFACTS or "
                                                                     "<repo>/artifacts)")
    parser.add_argument("--kind", action="append", choices=KINDS, help="only these blocks (repeatable)")
    parser.add_argument("--only", action="append", metavar="KEY=VALUE",
                        help="filter runs, e.g. optimizee=resnet18 regularizer=sam seed=42 (repeatable)")
    parser.add_argument("--print-path", action="store_true", help="print only the manifest path (for scripts)")
    parser.add_argument("--no-validate", action="store_true", help="skip composing every run's config")
    parser.add_argument("--run", action="store_true", help="run every line here, block by block")
    args = parser.parse_args(argv)

    try:
        set_artifacts_root(args.artifacts)  # the runs started here write there too
        profile = set_active_profile(args.profile or active_profile())
        spec = load_experiment(args.experiment, profile=profile)
        only = parse_only(args.only)
        if args.run:
            return run_here(args.experiment, spec, args.kind, only, not args.no_validate, args.artifacts)
        lines = expand(spec, args.kind, only, args.artifacts)
        if not args.no_validate:
            validate(lines, spec)
    except (ConfigError, ManifestError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if "diagnostics" in spec and (not args.kind or "diagnostics" in args.kind):
        if not any(line[0] == "run_diagnostics" for line in lines):
            study = spec["diagnostics"].get("study", spec.get("study"))
            print(f"note: no finished runs in study '{study}' ({profile}) yet, so no diagnostics lines; build this "
                  "manifest again once Stages 2–3 have finished", file=sys.stderr)
    if not lines:
        print("error: no runs match (check --kind/--only, or that the source runs have finished)", file=sys.stderr)
        return 2

    path = write_manifest(lines, manifest_path(args.experiment, args.kind, only, args.artifacts, profile))
    if args.print_path:
        print(path)
    else:
        counts = {}
        for line in lines:
            counts[line[0]] = counts.get(line[0], 0) + 1
        options = " ".join(argv if argv is not None else sys.argv[1:])
        print(f"{len(lines)} runs ({', '.join(f'{n} {s}' for s, n in counts.items())}) → {path}")
        print(f"here:      python scripts/run_manifest.py {path} --all")
        print(f"SLURM:     bash cluster/slurm/submit.sh {options}")
        print(f"PBS:       bash cluster/pbs/submit.sh {options}")
    return 0


def run_here(experiment, spec, kinds, only, check, artifacts_root) -> int:
    """Block by block, so diagnostics see the runs the train block just finished."""
    failures = 0
    for kind in [k for k in KINDS if k in spec and (not kinds or k in kinds)]:
        lines = expand(spec, [kind], only, artifacts_root)
        if check:
            validate(lines, spec)
        path = write_manifest(lines, manifest_path(experiment, [kind], only, artifacts_root, spec["profile"]))
        print(f"\n=== {kind}: {len(lines)} runs ({path}) ===", flush=True)
        for number, line in enumerate(lines, start=1):
            print(f"--- [{number}/{len(lines)}] {' '.join(line)}", flush=True)
            code = run_line(line)
            if code != 0:
                failures += 1
                print(f"!!! exit code {code}", flush=True)
    print(f"\n{'all runs finished' if not failures else f'{failures} run(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

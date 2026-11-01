#!/usr/bin/env python
"""Stage 4 — tables and figures from everything a profile has finished; run it any time.

    python scripts/make_report.py                           # production: study "main" and its Stage-1 optimizers
    python scripts/make_report.py --profile smoke           # the smoke test
    python scripts/make_report.py --no-figures              # tables only (no matplotlib needed)

Writes artifacts/<profile>/tables/<study>/ — start with summary.md; round_<R>/cube_accuracy,
cube_gen_gap and cube_flops are the 3D tables of each round (method × regularizer × evaluation
setting), round_<R>/benchmark the Stage-1 comparison on its source pair (M1 against M2 with r),
transfer_* the cross-evaluation of the rounds — and artifacts/<profile>/figures/<study>/ (PNG +
PDF). Ball metrics (max loss, surrogate gap, max ‖∇L‖) are read at radius --radius, which must
be one of the radii the Stage-4 audit used.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError  # noqa: E402
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.reporting import build_report  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import profile_artifacts, set_artifacts_root  # noqa: E402


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Tables and figures from artifacts/<profile>/.")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--studies", nargs="+", help="default: every study with finished runs, except adhoc/debug")
    parser.add_argument("--meta-tags", nargs="*", help="Stage-1 rounds (default: every one found, except debug)")
    parser.add_argument("--radius", type=float, default=0.01, help="ε of the ball metrics (default 0.01)")
    parser.add_argument("--artifacts", type=Path, default=None, help="artifacts root (default: $REGLO_ARTIFACTS or "
                                                                     "<repo>/artifacts)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    try:
        set_artifacts_root(args.artifacts)
        profile = set_active_profile(args.profile or active_profile())
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    root = profile_artifacts(profile)
    written = build_report(root, args.studies, args.meta_tags, args.radius, figures=not args.no_figures)
    if not written["tables"] and not written["figures"]:
        print(f"nothing to report yet under {root} (profile {profile}; the smoke test: --profile smoke)")
        return 1
    for kind in ("tables", "figures"):
        files = written[kind]
        if files:
            print(f"{len(files)} {kind} files under {root / kind}")
    for path in written["tables"]:
        if path.name in ("summary.md", "overview.md") or (path.name.startswith("cube_") and path.suffix == ".md"):
            print(f"  → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

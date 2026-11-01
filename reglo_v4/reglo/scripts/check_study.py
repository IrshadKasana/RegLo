#!/usr/bin/env python
"""Does every stage of a study get what it needs from the stages before it — and how far is it?

    python scripts/check_study.py                       # the study "main", production profile
    python scripts/check_study.py --profile smoke       # the smoke test's folder
    python scripts/check_study.py pilot

Static checks (every run composes; every M3/M4 run reads a φ that Stage 1 writes; the meta
split is the same in Stage 1 and Stages 2/3; every evaluation setting's pair is trained exactly
once; complete regularizer blocks for M2 and M4; Stage 4 points at the study) and progress read
from artifacts/<profile>/. Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.checks import check_study  # noqa: E402
from reglo.config import ConfigError  # noqa: E402
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import set_artifacts_root  # noqa: E402


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Connectivity and progress of a study's stages.")
    parser.add_argument("study", nargs="?", default="main")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--artifacts", type=Path, default=None, help="artifacts root (default: $REGLO_ARTIFACTS or "
                                                                     "<repo>/artifacts)")
    args = parser.parse_args(argv)
    try:
        root = set_artifacts_root(args.artifacts)
        profile = set_active_profile(args.profile or active_profile())
        report = check_study(args.study, root, profile)
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(report.text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

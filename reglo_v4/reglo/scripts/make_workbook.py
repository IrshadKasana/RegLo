#!/usr/bin/env python
"""The experiment plan as an Excel workbook — every table of the study, built from the configs,
with blank cells for the results; --fill writes the finished results into them.

    python scripts/make_workbook.py                       # → artifacts/<profile>/workbook/reglo_experiments.xlsx
    python scripts/make_workbook.py --fill                # the same, then every finished result filled in
    python scripts/make_workbook.py --fill my_copy.xlsx   # fill an existing workbook in place (keeps your notes)
    python scripts/make_workbook.py --profile smoke --fill   # the smoke test's numbers (checks the formulas)
    python scripts/make_workbook.py --out plan.xlsx

Sheets: Read me · T1 Optimizees & Optimizers · T2 Regularizers · T3 Sweeps · T4 Experiments ·
Cubes · Cross-evaluation · Stage 1 meta-training · Runs · Metrics. Results go in the blue cells of
Runs and Stage 1 meta-training; means, spreads, differences to M1, the cubes and the
cross-evaluation are formulas, computed when the file is opened. Needs openpyxl.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.config import ConfigError  # noqa: E402
from reglo.profiles import active_profile, set_active_profile  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.paths import profile_artifacts, set_artifacts_root  # noqa: E402


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="The experiment plan as an Excel workbook.")
    parser.add_argument("--profile", default=None, help="production | smoke | smoke_offline (default: $REGLO_PROFILE "
                                                         "or production)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write (default: artifacts/<profile>/workbook/reglo_experiments.xlsx)")
    parser.add_argument("--fill", nargs="?", const="", default=None, metavar="XLSX",
                        help="write finished results into the workbook: a new one, or the existing file named")
    parser.add_argument("--artifacts", type=Path, default=None, help="artifacts root (default: $REGLO_ARTIFACTS or "
                                                                     "<repo>/artifacts)")
    args = parser.parse_args(argv)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("error: the workbook needs openpyxl:  pip install openpyxl", file=sys.stderr)
        return 2
    from reglo.reporting.workbook import build_workbook, fill_workbook

    try:
        set_artifacts_root(args.artifacts)
        profile = set_active_profile(args.profile or active_profile())
        folder = profile_artifacts(profile)
        if args.fill:  # an existing workbook: fill it in place
            target = Path(args.fill)
            if not target.is_file():
                print(f"error: {target} does not exist (without a file name, --fill builds a new workbook)",
                      file=sys.stderr)
                return 2
        else:
            target = args.out or folder / "workbook" / "reglo_experiments.xlsx"
            build_workbook(target, profile)
            print(f"workbook ({profile} plan) → {target}")
        if args.fill is not None:
            counts = fill_workbook(target, folder)
            print(f"filled from {folder}: {counts['runs']} finished runs, {counts['running']} running, "
                  f"{counts['meta']} Stage-1 optimizers → {target}")
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

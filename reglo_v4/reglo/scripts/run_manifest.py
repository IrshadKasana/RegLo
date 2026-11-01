#!/usr/bin/env python
"""Run one line of a manifest — what each cluster array element does — or all of them in order.

    python scripts/run_manifest.py artifacts/production/manifests/stage2_optimizer_evaluation.txt 7   # line 7 (0-based)
    python scripts/run_manifest.py artifacts/smoke/manifests/stage2_optimizer_evaluation.txt --all

Lines are parsed in Python (no shell expansion), so overrides such as
optimizee.channels=[8,8,16] reach the run exactly as written. The exit code is the run's
(75: stopped at max_hours — run the same line again to continue).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from reglo.manifest import read_manifest, run_line  # noqa: E402
from reglo.utils.logging import safe_console  # noqa: E402
from reglo.utils.runtime import describe_exit  # noqa: E402


def main(argv=None) -> int:
    safe_console()
    parser = argparse.ArgumentParser(description="Run manifest line(s).")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index", nargs="?", type=int, help="0-based line number (PBS_ARRAY_INDEX)")
    parser.add_argument("--all", action="store_true", help="run every line, one after another")
    args = parser.parse_args(argv)

    lines = read_manifest(args.manifest)
    if args.all:
        codes = []
        for number, line in enumerate(lines):
            print(f"[run_manifest] line {number}: {' '.join(line)}", flush=True)
            codes.append(run_line(line))
            if codes[-1] != 0:
                print(f"[run_manifest] line {number}: {describe_exit(codes[-1])}", flush=True)
        return 0 if all(code == 0 for code in codes) else 1
    if args.index is None:
        parser.error("give a line index or --all")
    if not 0 <= args.index < len(lines):
        print(f"error: index {args.index} outside 0..{len(lines) - 1} ({args.manifest})", file=sys.stderr)
        return 2
    print(f"[run_manifest] line {args.index}: {' '.join(lines[args.index])}", flush=True)
    return run_line(lines[args.index])


if __name__ == "__main__":
    sys.exit(main())

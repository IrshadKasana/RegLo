"""Thin shim: the manager's `report` phase entrypoint.
Runs the full uniform report for one experiment (or all)."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None,
                    help="expN (default: densest dataset/model combo)")
    ap.add_argument("--stage", default="at_completion")
    a = ap.parse_args()
    cmd = [sys.executable, str(ROOT / "generate_paper_plots.py"),
           "--stage", a.stage]
    if a.experiment:
        cmd += ["--experiment", a.experiment]
    raise SystemExit(subprocess.run(cmd, cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    main()

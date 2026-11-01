#!/usr/bin/env python3
"""
Convenience wrapper: `python experiments/dmo/generate_all_plots.py [...]`
forwards to the metadata-driven generate_paper_plots.py at the repo root.

The old root-level generate_all_plots.py / generate_baseline_plots.py were
RETIRED (extras/deprecated/legacy_plotters/) because they hardcoded the
2024 checkpoint names and crashed on design-v2 runs.

Examples:
  python experiments/dmo/generate_all_plots.py --experiment exp1
  python experiments/dmo/generate_all_plots.py --experiment exp4 --stage at_convergence
  python experiments/dmo/generate_all_plots.py            # densest cell, all runs
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.argv[0] = str(ROOT / "generate_paper_plots.py")
runpy.run_path(str(ROOT / "generate_paper_plots.py"), run_name="__main__")

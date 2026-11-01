Retired plot entry points.

generate_all_plots.py / generate_baseline_plots.py / plot_3bar_comparison.py
  hardcoded the 2024 checkpoint names and globbed results/json by
  filename -> crashed on design-v2 runs.
generate_experiment_report.py / experiments/dmo/generate_all_plots.py
  were thin shims that only forwarded to the real plotter.

THE SINGLE PLOTTER IS: generate_paper_plots.py (repo root).
  python generate_paper_plots.py --experiment exp2
  python generate_paper_plots.py --experiment exp2 --comparison-set gam

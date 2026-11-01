"""Stage 4 — tables and figures from everything under ``artifacts/`` (``scripts/make_report.py``).

    collect   artifacts → flat run records → seed statistics, per round (round_view)
    tables    per round: the result cubes (method × regularizer × evaluation setting, one per metric),
              the Stage-1 benchmark on the source pair, the per-pair tables behind them; across rounds:
              the cross-evaluation (transfer) tables; Stage 1, same-GPU timing — CSV, Markdown, LaTeX
    figures   the cubes as figures, training curves, accuracy vs. cost, point-of-convergence
              tests, weight distributions, Stage-1 curves

:func:`build_report` runs all of it and returns the files it wrote.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from reglo.reporting.collect import (
    REGULARIZERS,
    aggregate,
    collect_compute,
    collect_meta,
    collect_runs,
    discover_meta_tags,
    discover_studies,
    pairs_of,
    round_view,
)
from reglo.reporting.tables import (
    benchmark_table,
    comparison_table,
    compute_table,
    cube_table,
    meta_table,
    overview_table,
    settings_columns,
    study_summary,
    transfer_summary,
    transfer_table,
    write_cube_long,
    write_runs_csv,
)
from reglo.rounds import load_rounds
from reglo.study import METRICS, Pair, study_rounds, study_settings

__all__ = ["NOT_BY_DEFAULT", "build_report"]

NOT_BY_DEFAULT = {"adhoc", "debug", "test"}  # studies and meta_tags reported only when asked for by name
TRANSFER_METRICS = ("accuracy", "gen_gap")    # FLOPs do not depend on whose φ is used


def build_report(
    artifacts_root: Union[str, Path],
    studies: Optional[Sequence[str]] = None,
    meta_tags: Optional[Sequence[str]] = None,
    radius: float = 0.01,
    figures: bool = True,
) -> Dict[str, List[Path]]:
    """Write every table (and figure) the results so far allow; ``{"tables": [...], "figures": [...]}``.

    ``artifacts_root`` is one profile's folder (``artifacts/<profile>``). By default every study and
    Stage-1 round found there is reported, except hand-started and debugging runs
    (:data:`NOT_BY_DEFAULT`), which have to be named.
    """
    root = Path(artifacts_root)
    studies = list(studies) if studies else [s for s in discover_studies(root) if s not in NOT_BY_DEFAULT]
    if meta_tags is None:
        meta_tags = [t for t in discover_meta_tags(root) if t not in NOT_BY_DEFAULT]
    tables_dir, figures_dir = root / "tables", root / "figures"
    written: Dict[str, List[Path]] = {"tables": [], "figures": []}

    records = collect_runs(root, studies, radius)
    groups = aggregate(records)
    profiles = collect_compute(root)
    if figures:
        from reglo.reporting import figures as fig  # matplotlib only when figures are wanted

    for study in studies:
        study_records = [r for r in records if r["study"] == study]
        if not study_records:
            continue
        shutil.rmtree(tables_dir / study, ignore_errors=True)  # regenerated: no stale tables from earlier reports
        if figures:
            shutil.rmtree(figures_dir / study, ignore_errors=True)
        written["tables"].append(write_runs_csv(study_records, tables_dir / study / "runs.csv"))

        settings = study_settings(study)
        rounds = [r for r in study_rounds(study) if r.name in settings]
        cubes: Dict[str, Dict[str, object]] = {}
        benchmarks = {}
        pair_tables = {}
        for rnd in rounds:
            tdir, fdir = tables_dir / study / f"round_{rnd.name}", figures_dir / study / f"round_{rnd.name}"
            view = round_view(groups, rnd.name)  # M1, M2 (with r), the round's M3/M4

            # Stage 4: one 3D table (and figure) per metric — core pairs, then core + extended
            studied = {(k[1], k[2]) for k in view if k[0] == study}
            for extended in (False, True):
                columns = settings_columns(settings[rnd.name], extended)
                if extended:  # extended pairs only once they have results
                    columns = [c for c in columns if not c.extended or (c.pair.optimizee, c.pair.dataset) in studied]
                if not columns or (extended and not any(c.extended for c in columns)):
                    continue
                suffix = "_extended" if extended else ""
                for metric in METRICS:
                    table = cube_table(view, study, metric, columns, " extended" if extended else "", rnd)
                    if table is None:
                        continue
                    cubes.setdefault(rnd.name, {})[metric.key + suffix] = table
                    written["tables"] += table.write(tdir / f"cube_{metric.key}{suffix}")
                    if figures:
                        figure = fig.cube_figure(view, study, metric, columns, rnd)
                        if figure is not None:
                            written["figures"] += fig.save(figure, fdir / f"cube_{metric.key}{suffix}")
                long_csv = write_cube_long(view, study, columns, tdir / f"cube_long{suffix}.csv", rnd.name)
                if long_csv is not None:
                    written["tables"].append(long_csv)

            # Stage 1: the benchmark on the source pair (M2 with its regularizer on)
            bench = benchmark_table(view, study, rnd)
            if bench is not None:
                benchmarks[rnd.name] = bench
                written["tables"] += bench.write(tdir / "benchmark")

            # the detail behind the cubes, pair by pair
            pairs = [p for p in pairs_of(study_records, study) if any(
                sp.pair == p for per in settings[rnd.name].values() for sp in per)]
            for pair in pairs:
                found_any = False
                for reg in (*REGULARIZERS, "none"):
                    if reg == "none" and found_any:
                        break  # M1 vs M3 alone only for grids without regularizers (the pilot)
                    table = comparison_table(view, study, pair.optimizee, pair.dataset, reg, radius, rnd.name)
                    if table is not None:
                        found_any = True
                        pair_tables[(rnd.name, pair.optimizee, pair.dataset, reg)] = table
                        written["tables"] += table.write(tdir / pair.tag / reg)
                if figures:
                    stem = fdir / pair.tag
                    profile = profiles.get((pair.optimizee, pair.dataset))
                    title = f"{study} · round {rnd.name}"
                    for name, figure in (
                        ("training_curves", fig.curves_figure(view, study, pair, title)),
                        ("pareto_flops", fig.pareto_figure(view, study, pair.optimizee, pair.dataset, "flops",
                                                           title=title)),
                        ("pareto_time", fig.pareto_figure(view, study, pair.optimizee, pair.dataset, "time", profile,
                                                          title=title)),
                        ("poc", fig.poc_figure(view, study, pair.optimizee, pair.dataset, radius, title=title)),
                        ("weights", fig.weights_figure(records, study, pair.optimizee, pair.dataset, rnd.name)),
                    ):
                        if figure is not None:
                            written["figures"] += fig.save(figure, stem / name)

        # across rounds: the cross-evaluation on the core pairs every round evaluates
        transfer = []
        if len(rounds) > 1:
            core = [sp.pair for sp in settings_columns(settings[rounds[0].name], False)]
            core = list(dict.fromkeys(p for p in core if all(
                any(sp.pair == p and not sp.extended for per in settings[r.name].values() for sp in per) for r in rounds)))
            for metric in (m for m in METRICS if m.key in TRANSFER_METRICS):
                for maker, stem in ((transfer_table, "transfer"), (transfer_summary, "transfer_summary")):
                    table = maker(groups, study, rounds, core, metric)
                    if table is not None:
                        transfer.append(table)
                        written["tables"] += table.write(tables_dir / study / f"{stem}_{metric.key}")

        summary = tables_dir / study / "summary.md"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(study_summary(groups, records, study, settings, rounds, cubes, benchmarks, transfer,
                                         pair_tables, meta_tags), encoding="utf-8")
        written["tables"].append(summary)

    overview = overview_table(groups, studies)
    if overview is not None:
        written["tables"] += overview.write(tables_dir / "overview")

    known = load_rounds()
    for tag in meta_tags:  # Stage 1
        arms = collect_meta(root, tag)
        source = Pair(known[tag].optimizee, known[tag].dataset).label if tag in known else ""
        table = meta_table(arms, tag, source)
        if table is not None:
            written["tables"] += table.write(tables_dir / "meta_training" / tag)
        if figures and arms:
            shutil.rmtree(figures_dir / "meta_training" / tag, ignore_errors=True)
            for name, figure in fig.meta_figures(arms, tag, source).items():
                written["figures"] += fig.save(figure, figures_dir / "meta_training" / tag / name)

    if profiles:
        shutil.rmtree(tables_dir / "compute", ignore_errors=True)
    for (optimizee, dataset), profile in sorted(profiles.items()):
        written["tables"] += compute_table(profile).write(tables_dir / "compute" / f"{optimizee}-{dataset}")
    return written

"""Tables — Stage 4's result cubes and the detail behind them.

Every table is written three ways: CSV (raw numbers, for notebooks), Markdown (to read on the
cluster or GitHub) and LaTeX (booktabs, for the thesis)::

    tables/<study>/round_<R>/cube_{accuracy,gen_gap,flops}.*   one 3D table per round and metric:
                                                               method × regularizer × evaluation setting
                                                               (M2 = AdamW + r, trained with r on every pair)
    tables/<study>/round_<R>/cube_*_extended.*                 the same with the extended pairs
    tables/<study>/round_<R>/cube_long.csv                     every cube cell, one row each (tidy)
    tables/<study>/round_<R>/benchmark.*                       Stage 1 on the round's source pair: M1 against
                                                               M2(r) trained with r, beside M3 and M4(r)
    tables/<study>/round_<R>/<optimizee>-<dataset>/<r>.*       M1 · M2(r) · M3 · M4(r) on one pair, with the
                                                               property r targets and the cost
    tables/<study>/transfer_{accuracy,gen_gap}.*               cross-evaluation: every pair × every round's
                                                               optimizers, with each pair's role per round
    tables/<study>/transfer_summary_{accuracy,gen_gap}.*       advantage over M1 by role: seen → unseen
    tables/<study>/runs.csv                                    every run and seed, flat
    tables/<study>/summary.md                                  all of it, stage by stage
    tables/overview.{csv,md,tex}                               Δ top-1 of M4(r) against M1 and M3, per round
    tables/meta_training/<round>.{csv,md,tex}                  Stage 1 meta-training
    tables/compute/<optimizee>-<dataset>.{csv,md,tex}          same-GPU timing of all 12 variants

Cells read ``mean ± std`` over seeds (sample std; a single seed shows the mean alone).
Within each block of four methods, the best mean is bold where "better" is unambiguous.
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from reglo.profiles import active_profile
from reglo.reporting.collect import Group, GroupKey, RoundKey, Stat, pairs_of, round_view, to_float
from reglo.rounds import ROLE_LABELS, ROLES, Round
from reglo.study import (
    DATASET_NAMES,
    LABELS,
    LEARNED,
    METRICS,
    OPTIMIZEE_NAMES,
    REG_NAMES,
    REGULARIZERS,
    SETTINGS,
    STAGES,
    Metric,
    Pair,
    SettingPair,
    is_evaluation,
    study_experiments,
    training_pairs,
)

__all__ = [
    "Column",
    "Table",
    "benchmark_table",
    "comparison_table",
    "compute_table",
    "cube_table",
    "expected_runs",
    "fmt_number",
    "meta_table",
    "method_rows",
    "overview_table",
    "property_columns",
    "settings_columns",
    "study_summary",
    "transfer_summary",
    "transfer_table",
    "welch_p",
    "write_cube_long",
    "write_runs_csv",
]

DASH_MD, DASH_TEX = "–", "--"


# ───────────────────────────────────────────────────────────────── columns and cells

@dataclass(frozen=True)
class Column:
    key: str                      # record field
    header: str                   # Markdown / CSV header
    latex: str                    # LaTeX header
    scale: float = 1.0            # 100 → shown in %
    digits: int = 2
    better: Optional[str] = None  # "higher" | "lower": bold the best of M1–M4


CORE = (
    Column("test_top1", "Top-1 (%)", r"Top-1 (\%)", 100, 2, "higher"),
    Column("test_top5", "Top-5 (%)", r"Top-5 (\%)", 100, 2, "higher"),
    Column("gen_gap_top1", "Gap (pp)", "Gap (pp)", 100, 2, "lower"),
    Column("gen_gap", "Gap (loss)", "Gap (loss)", 1, 3),
)
COST = (
    Column("passes_per_step", "Passes", "Passes", 1, 0),
    Column("gflops_per_step", "GFLOPs/step", "GFLOPs/step", 1, 3),
    Column("train_minutes", "Time (min)", "Time (min)", 1, 1),
)


def property_columns(reg: str) -> Tuple[Column, ...]:
    """What regularizer ``reg`` is meant to change, measured on every method's weights."""
    if reg == "l1":
        return (Column("sparsity", "Sparsity (%)", r"Sparsity (\%)", 100, 1, "higher"),
                Column("l1", "‖θ‖₁", r"$\|\theta\|_1$", 1, 1, "lower"))
    if reg == "l2":
        return (Column("l2", "‖θ‖₂", r"$\|\theta\|_2$", 1, 2, "lower"),)
    if reg == "sam":
        return (Column("lambda_max", "λ_max", r"$\lambda_{\max}$", 1, 2, "lower"),
                Column("max_loss_converged", "Max loss (conv.)", r"$\max L_\epsilon$ (conv.)", 1, 3, "lower"),
                Column("max_loss_final", "Max loss (compl.)", r"$\max L_\epsilon$ (compl.)", 1, 3, "lower"))
    if reg == "gam":
        return (Column("hessian_trace", "Tr(H)", r"$\mathrm{Tr}(H)$", 1, 1, "lower"),
                Column("max_grad_norm_converged", "Max ‖∇L‖ (conv.)", r"$\max\|\nabla L\|_\epsilon$ (conv.)", 1, 3, "lower"),
                Column("max_grad_norm_final", "Max ‖∇L‖ (compl.)", r"$\max\|\nabla L\|_\epsilon$ (compl.)", 1, 3, "lower"))
    if reg == "gsam":
        return (Column("lambda_max", "λ_max", r"$\lambda_{\max}$", 1, 2, "lower"),
                Column("surrogate_gap_converged", "Surrogate gap (conv.)", r"$h_\epsilon$ (conv.)", 1, 4, "lower"),
                Column("surrogate_gap_final", "Surrogate gap (compl.)", r"$h_\epsilon$ (compl.)", 1, 4, "lower"))
    raise ValueError(f"unknown regularizer '{reg}'")


def fmt_number(value: Optional[float], digits: int = 2, signed: bool = False) -> str:
    """Fixed decimals; thousands separators from 10⁴; scientific below the shown precision."""
    if value is None or not math.isfinite(value):
        return DASH_MD
    sign = "+" if signed and value > 0 else ""
    if abs(value) >= 1e4:
        return f"{sign}{value:,.0f}"
    if value != 0 and abs(value) < 0.5 * 10.0 ** (-digits):
        return f"{sign}{value:.1e}"
    return f"{sign}{value:.{digits}f}"


def _cell(stat: Stat, col: Column, bold: bool = False, runs: Optional[int] = None,
          show_std: bool = True, suffix: str = "") -> Tuple[str, str]:
    """(markdown, latex) text of mean ± std; ``bold`` marks the mean. When fewer seeds than the
    row's runs carry this number (e.g. only some runs audited), the count is shown: "(n=1)"."""
    if stat.mean is None:
        return DASH_MD, DASH_TEX
    mean = fmt_number(stat.mean * col.scale, col.digits)
    md_mean, tex_mean = (f"**{mean}**", f"\\textbf{{{mean}}}") if bold else (mean, mean)
    note = f" (n={stat.n})" if runs is not None and stat.n != runs else ""
    tex_suffix = _tex_text(suffix)
    if stat.std is None or not show_std:
        return md_mean + suffix + note, tex_mean + tex_suffix + note
    std = fmt_number(stat.std * col.scale, col.digits)
    return f"{md_mean} ± {std}{suffix}{note}", f"{tex_mean} $\\pm$ {std}{tex_suffix}{note}"


def _best(stats: Sequence[Stat], col: Column) -> Set[int]:
    """Rows holding the best mean of a column. None when the column has no direction, fewer than two
    rows have a value, or a single-seed value would be ranked against means of several seeds."""
    if col.better is None:
        return set()
    means = [(i, s.mean) for i, s in enumerate(stats) if s.mean is not None and math.isfinite(s.mean)]
    if len(means) < 2:
        return set()
    counts = {stats[i].n for i, _ in means}
    if len(counts) > 1 and min(counts) < 2:
        return set()
    target = (max if col.better == "higher" else min)(m for _, m in means)
    return {i for i, m in means if m == target}


def welch_p(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided Welch t-test p-value (needs scipy and ≥ 2 seeds per side); None otherwise."""
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover - scipy is optional
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(stats.ttest_ind(list(a), list(b), equal_var=False).pvalue)
    return p if math.isfinite(p) else None


# ───────────────────────────────────────────────────────────────── a table, three formats

@dataclass
class Table:
    title: str                          # Markdown heading and LaTeX caption
    label: str                          # LaTeX \label
    headers: List[str]
    latex_headers: List[str]
    rows: List[List[str]]               # Markdown cells
    latex_rows: List[List[str]]
    csv_header: List[str]
    csv_rows: List[List[Any]]
    align: str                          # LaTeX column spec
    notes: List[str] = field(default_factory=list)
    latex_head: Optional[str] = None    # header lines replacing latex_headers (multi-level headers)
    latex_breaks: Set[int] = field(default_factory=set)  # \midrule after these row indices

    def markdown(self, level: int = 3) -> str:
        lines = [f"{'#' * level} {self.title}", ""] if level else []
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("|" + "|".join(":--" if c == "l" else "--:" for c in self.align) + "|")
        lines += ["| " + " | ".join(row) + " |" for row in self.rows]
        if self.notes:
            lines += [""] + [f"{note}  " for note in self.notes]
        return "\n".join(lines) + "\n"

    def latex(self) -> str:
        body = []
        for i, row in enumerate(self.latex_rows):
            body.append("    " + " & ".join(row) + r" \\")
            if i in self.latex_breaks and i != len(self.latex_rows) - 1:
                body.append("    \\midrule")
        caption = _tex_text(self.title) + ("." if not self.notes else ". " + " ".join(_tex_text(n) for n in self.notes))
        head = self.latex_head or f"    {' & '.join(self.latex_headers)} \\\\"
        return (
            "% generated by scripts/make_report.py — needs \\usepackage{booktabs, graphicx}\n"
            "\\begin{table}[t]\n  \\centering\n  \\small\n"
            f"  \\caption{{{caption}}}\n  \\label{{{self.label}}}\n"
            "  \\resizebox{\\linewidth}{!}{%\n"
            f"  \\begin{{tabular}}{{{self.align}}}\n    \\toprule\n"
            f"{head}\n    \\midrule\n" + "\n".join(body) + "\n    \\bottomrule\n"
            "  \\end{tabular}}\n\\end{table}\n"
        )

    def write(self, stem: Path) -> List[Path]:
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths = [Path(f"{stem}.{ext}") for ext in ("csv", "md", "tex")]  # not with_suffix: stems may contain dots
        with open(paths[0], "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.csv_header)
            writer.writerows([["" if v is None else v for v in row] for row in self.csv_rows])
        paths[1].write_text(self.markdown(level=2), encoding="utf-8")
        paths[2].write_text(self.latex(), encoding="utf-8")
        return paths


_TEX_SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "±": r"$\pm$", "×": r"$\times$", "−": "-",
                "ε": r"$\epsilon$", "λ": r"$\lambda$", "φ": r"$\phi$", "Δ": r"$\Delta$", "–": "--", "—": "---",
                "≥": r"$\geq$", "·": r"$\cdot$", "θ": r"$\theta$", "‖": r"$\|$", "∇": r"$\nabla$",
                "↑": r"$\uparrow$", "↓": r"$\downarrow$", "†": r"$^\dagger$",
                "<": r"$<$", ">": r"$>$", "^": r"\^{}", "~": r"\~{}", "$": r"\$", "{": r"\{", "}": r"\}",
                "\\": r"\textbackslash{}"}


def _tex_text(text: str) -> str:
    """Plain text (titles, notes, labels) → LaTeX-safe text."""
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in text)


# ───────────────────────────────────────────────────────────────── M1 · M2(r) · M3 · M4(r)

def method_rows(reg: str) -> List[Tuple[str, str, str]]:
    """(row label, method, regularizer) in table order; ``reg="none"``: M1 and M3 only (the pilot)."""
    if reg == "none":
        return [("M1 AdamW", "m1_adamw", "none"), ("M3 LSTM", "m3_lstm", "none")]
    name = REG_NAMES[reg]
    return [
        ("M1 AdamW", "m1_adamw", "none"),
        (f"M2 AdamW + {name}", "m2_adamw_reg", reg),
        ("M3 LSTM", "m3_lstm", "none"),
        (f"M4 LSTM-{name} (ours)", "m4_lstm_reg", reg),
    ]


def _latex_method(label: str) -> str:
    return _tex_text(label).replace("(ours)", r"\textit{(ours)}")


def comparison_table(
    groups: Mapping[GroupKey, Group], study: str, optimizee: str, dataset: str, reg: str, radius: float = 0.01,
    round_name: str = "",
) -> Optional[Table]:
    """The four methods for one regularizer on one pair, as round ``round_name`` sees it (``groups``:
    its :func:`round_view`); None when neither M2(r) nor M4(r) has a run.

    ``reg="none"`` gives M1 against M3, for grids without regularizers (the pilot).
    """
    rows_spec = method_rows(reg)
    found = [groups.get((study, optimizee, dataset, m, r)) for _, m, r in rows_spec]
    if reg == "none":
        if all(g is None for g in found):
            return None
    elif found[1] is None and found[3] is None:
        return None
    properties = (Column("l2", "‖θ‖₂", r"$\|\theta\|_2$", 1, 2),) if reg == "none" else property_columns(reg)
    columns = list(CORE) + list(properties) + list(COST)
    present = [g or Group() for g in found]

    per_column = []
    for col in columns:
        stats = [g.get(col.key) for g in present]
        best = _best(stats, col)
        per_column.append([_cell(s, col, bold=i in best, runs=g.completed)
                           for i, (s, g) in enumerate(zip(stats, present))])

    rows, latex_rows, csv_rows = [], [], []
    for i, ((label, method, r), group) in enumerate(zip(rows_spec, present)):
        runs = f"{group.completed}" + (f" (+{group.diverged} div.)" if group.diverged else "")
        rows.append([label, runs] + [per_column[j][i][0] for j in range(len(columns))])
        latex_rows.append([_latex_method(label), runs] + [per_column[j][i][1] for j in range(len(columns))])
        csv_row: List[Any] = [study, round_name, optimizee, dataset, reg, LABELS[method], method, r, group.completed,
                              group.diverged]
        for col in columns:
            stat = group.get(col.key)
            csv_row += [stat.mean, stat.std, stat.n]
        csv_rows.append(csv_row)

    csv_header = ["study", "round", "optimizee", "dataset", "table", "row", "method", "regularizer", "n", "diverged"]
    for col in columns:
        csv_header += [f"{col.key}_mean", f"{col.key}_std", f"{col.key}_n"]
    what = "M1 against M3 (no regularizer)" if reg == "none" else f"regularizer {REG_NAMES[reg]}"
    notes = [
        f"Mean ± std over seeds. Gap: train − test top-1 (pp) and test − train loss. "
        f"ε = {radius:g} for the ball metrics; conv. / compl. = at convergence / at completion. "
        f"Time: training wall-clock on whichever GPU the run got (same-GPU timings: compute tables).",
    ]
    where = f", round {round_name}" if round_name else ""
    return Table(
        title=f"{study}: {Pair(optimizee, dataset).label}, {what}{where}",
        label=f"tab:{study}-{round_name}-{optimizee}-{dataset}-{reg}" if round_name else f"tab:{study}-{optimizee}-{dataset}-{reg}",
        headers=["Method", "Runs"] + [c.header for c in columns],
        latex_headers=["Method", "Runs"] + [c.latex for c in columns],
        rows=rows, latex_rows=latex_rows, csv_header=csv_header, csv_rows=csv_rows,
        align="lr" + "r" * len(columns), notes=notes,
    )


# ───────────────────────────────────────────────────────────────── Stage 4: the result cubes

def _cube_regs(groups: Mapping[GroupKey, Group], study: str, columns: Sequence[SettingPair]) -> List[str]:
    return [r for r in REGULARIZERS
            if any((study, c.pair.optimizee, c.pair.dataset, m, r) in groups
                   for c in columns for m in ("m2_adamw_reg", "m4_lstm_reg"))]


def _column_head(c: SettingPair) -> str:
    return f"{c.setting.label}: {c.pair.label}" + (" ‡" if c.role == "unseen_both" else "") + (" †" if c.extended else "")


def cube_table(
    groups: Mapping[GroupKey, Group], study: str, metric: Metric, columns: Sequence[SettingPair], suffix: str = "",
    rnd: Optional[Round] = None,
) -> Optional[Table]:
    """One metric of Stage 4 for one round: blocks of M1 · M2(r) · M3 · M4(r), one block per
    regularizer r (the Z-axis), against the round's evaluation settings (the X-axis).

    ``groups`` is the round's view (:func:`~reglo.reporting.collect.round_view`): M2 trained with r,
    M3/M4 with the round's φ. M1 and M3 do not depend on r; they
    are repeated in every block so each block is a complete comparison. FLOPs are identical across
    seeds, so they show no ± but the ratio to M1.
    """
    regs = _cube_regs(groups, study, columns)
    if not regs or not columns:
        return None
    col = Column(metric.record, metric.label, metric.label, metric.scale, metric.digits, metric.better)
    m1 = [groups.get((study, c.pair.optimizee, c.pair.dataset, "m1_adamw", "none")) or Group() for c in columns]
    round_name = rnd.name if rnd else ""

    rows, latex_rows, csv_rows, breaks = [], [], [], set()
    for reg in regs:
        spec = method_rows(reg)
        block = [[groups.get((study, c.pair.optimizee, c.pair.dataset, m, r)) or Group() for c in columns] for _, m, r in spec]
        best = [_best([block[i][j].get(metric.record) for i in range(len(spec))], col) for j in range(len(columns))]
        for i, (label, method, r) in enumerate(spec):
            md_cells, tex_cells = [], []
            csv_row: List[Any] = [study, round_name, metric.key, reg, LABELS[method], method, r]
            for j in range(len(columns)):
                group, stat = block[i][j], block[i][j].get(metric.record)
                suffix_text = ""
                if not metric.seeds_vary:
                    ref = m1[j].get(metric.record).mean
                    if stat.mean is not None and ref:
                        suffix_text = f" ({stat.mean / ref:.2f}×)"
                bold = i in best[j]
                md, tex = _cell(stat, col, bold=bold, runs=group.completed if metric.seeds_vary else None,
                                show_std=metric.seeds_vary, suffix=suffix_text)
                md_cells.append(md)
                tex_cells.append(tex)
                mean = None if stat.mean is None else stat.mean * metric.scale
                std = None if stat.std is None else stat.std * metric.scale
                csv_row += [mean, std, stat.n]
            first = REG_NAMES[reg] if i == 0 else ""
            rows.append([f"**{first}**" if first else "", label] + md_cells)
            latex_rows.append([_tex_text(first), _latex_method(label)] + tex_cells)
            csv_rows.append(csv_row)
        breaks.add(len(latex_rows) - 1)

    # LaTeX: settings spanning their pairs, then the pairs
    spans: List[Tuple[str, int, int]] = []  # (setting label, first column, last column), 1-based
    for j, c in enumerate(columns, start=3):
        if spans and spans[-1][0] == c.setting.label:
            spans[-1] = (spans[-1][0], spans[-1][1], j)
        else:
            spans.append((c.setting.label, j, j))
    top = " & ".join(["", ""] + [f"\\multicolumn{{{b - a + 1}}}{{c}}{{{_tex_text(lbl)}}}" for lbl, a, b in spans])
    rules = " ".join(f"\\cmidrule(lr){{{a}-{b}}}" for _, a, b in spans)
    bottom = " & ".join(["Regularizer", "Method"] + [_tex_text(c.pair.label + (" ‡" if c.role == "unseen_both" else "")
                                                               + (" †" if c.extended else "")) for c in columns])
    latex_head = f"    {top} \\\\\n    {rules}\n    {bottom} \\\\"

    csv_header = ["study", "round", "metric", "block", "row", "method", "regularizer"]
    for c in columns:
        csv_header += [f"{c.setting.key}:{c.pair.tag}_{x}" for x in ("mean", "std", "n")]
    source = f" Round {round_name}: optimizers meta-trained on {Pair(rnd.optimizee, rnd.dataset).label}." if rnd else ""
    notes = [
        f"{metric.label} {metric.arrow} ({metric.unit}); mean ± std over seeds.{source} Rows: the four methods for each "
        "regularizer r (M1 and M3 do not use r and repeat in every block); columns: the evaluation settings of "
        "Stages 2 and 3. M2 computes r at every training step (AdamW + r); M3 and M4 train with the frozen Stage-1 "
        "LSTM and compute no r. On a round's source pair M2's runs are those of Stage 1.",
    ]
    if not metric.seeds_vary:
        notes.append("FLOPs per training step: forward + backward (+ r's extra passes for M2, the LSTM update for "
                     "M3/M4); in brackets the "
                     "ratio to M1 on the same pair. Meta-training (Stage 1) is a one-time cost, outside these numbers.")
    if any(c.role == "unseen_both" for c in columns):
        notes.append("‡ the network is unseen too (unseen optimizee + dataset).")
    if any(c.extended for c in columns):
        notes.append("† extended pairs, beyond the core design.")
    where = f"round {round_name}, " if round_name else ""
    return Table(
        title=f"{metric.label} {metric.arrow} — method × regularizer × evaluation setting ({where}{study}{suffix})",
        label=f"tab:{study}-{round_name}-cube-{metric.key}{suffix.replace(' ', '-')}" if round_name
        else f"tab:{study}-cube-{metric.key}{suffix.replace(' ', '-')}",
        headers=["Regularizer", "Method"] + [_column_head(c) for c in columns],
        latex_headers=["Regularizer", "Method"] + [_tex_text(_column_head(c)) for c in columns],
        rows=rows, latex_rows=latex_rows, csv_header=csv_header, csv_rows=csv_rows,
        align="ll" + "r" * len(columns), notes=notes, latex_head=latex_head, latex_breaks=breaks,
    )


def write_cube_long(groups: Mapping[GroupKey, Group], study: str, columns: Sequence[SettingPair], path: Path,
                    round_name: str = "") -> Optional[Path]:
    """Every cell of a round's cubes as one row: metric, setting, pair, role, regularizer block, method, mean, std, n."""
    regs = _cube_regs(groups, study, columns)
    if not regs:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["study", "round", "metric", "unit", "stage", "setting", "role", "optimizee", "dataset",
                         "extended", "block", "row", "method", "regularizer", "mean", "std", "n", "runs"])
        for metric in METRICS:
            for reg in regs:
                for _, method, r in method_rows(reg):
                    for c in columns:
                        group = groups.get((study, c.pair.optimizee, c.pair.dataset, method, r)) or Group()
                        stat = group.get(metric.record)
                        writer.writerow([
                            study, round_name, metric.key, metric.unit, c.setting.stage, c.setting.key, c.role,
                            c.pair.optimizee, c.pair.dataset, c.extended, reg, LABELS[method], method, r,
                            "" if stat.mean is None else stat.mean * metric.scale,
                            "" if stat.std is None else stat.std * metric.scale, stat.n, group.completed,
                        ])
    return path


# ───────────────────────────────────────────────────────────────── Stage 1: the benchmark on the source pair

def benchmark_table(groups: Mapping[GroupKey, Group], study: str, rnd: Round) -> Optional[Table]:
    """Stage 1's benchmark on the round's source pair: M1 (AdamW) against M2(r) (AdamW + r, the
    regularizer on) — what each regularizer does with a conventional optimizer — beside M3 and M4(r),
    which reach the same pair without computing r (their seen-setting runs, same images and seeds).

    ``groups``: the round's view (``round_view``).
    """
    pair = Pair(rnd.optimizee, rnd.dataset)
    key = lambda m, r: (study, pair.optimizee, pair.dataset, m, r)  # noqa: E731
    spec = [("M1 AdamW", "m1_adamw", "none")]
    spec += [(f"M2 AdamW + {REG_NAMES[r]}", "m2_adamw_reg", r) for r in REGULARIZERS if key("m2_adamw_reg", r) in groups]
    spec += [("M3 LSTM", "m3_lstm", "none")]
    spec += [(f"M4 LSTM-{REG_NAMES[r]} (ours)", "m4_lstm_reg", r) for r in REGULARIZERS if key("m4_lstm_reg", r) in groups]
    present = [groups.get(key(m, r)) for _, m, r in spec]
    if not any(g is not None for g, (_, m, _) in zip(present, spec) if m == "m2_adamw_reg"):
        return None
    m1 = groups.get(key("m1_adamw", "none")) or Group()
    columns = [CORE[0], CORE[2], CORE[3], COST[1], COST[0], COST[2]]
    headers = ["Method", "Runs", "Top-1 (%)", "Δ top-1 vs M1 (pp)", "Gap (pp)", "Δ gap vs M1 (pp)", "Gap (loss)",
               "GFLOPs/step", "Passes", "Time (min)"]
    latex_headers = ["Method", "Runs", r"Top-1 (\%)", r"$\Delta$ top-1 vs M1 (pp)", "Gap (pp)", r"$\Delta$ gap vs M1 (pp)",
                     "Gap (loss)", "GFLOPs/step", "Passes", "Time (min)"]
    rows, latex_rows, csv_rows = [], [], []
    for (label, method, r), group in zip(spec, present):
        group = group or Group()
        cells = []
        for col in columns:
            md, tex = _cell(group.get(col.key), col, runs=group.completed if col.key != "gflops_per_step" else None,
                            show_std=col.key not in ("gflops_per_step", "passes_per_step"))
            cells.append((md, tex))
        d_acc, p_acc = _delta(groups, key(method, r), key("m1_adamw", "none")) if method != "m1_adamw" else (None, None)
        d_gap, p_gap = (_delta(groups, key(method, r), key("m1_adamw", "none"), "gen_gap_top1")
                        if method != "m1_adamw" else (None, None))
        delta_acc = fmt_number(d_acc, 2, signed=True) + _p_text(p_acc) if d_acc is not None else DASH_MD
        delta_gap = fmt_number(d_gap, 2, signed=True) + _p_text(p_gap) if d_gap is not None else DASH_MD
        runs = f"{group.completed}" + (f" (+{group.diverged} div.)" if group.diverged else "")
        md_row = [label, runs, cells[0][0], delta_acc, cells[1][0], delta_gap, cells[2][0], cells[3][0], cells[4][0], cells[5][0]]
        rows.append(md_row)
        latex_rows.append([_latex_method(label), runs, cells[0][1], _tex_text(delta_acc).replace(DASH_MD, DASH_TEX),
                           cells[1][1], _tex_text(delta_gap).replace(DASH_MD, DASH_TEX), cells[2][1], cells[3][1],
                           cells[4][1], cells[5][1]])
        csv_row: List[Any] = [study, rnd.name, pair.optimizee, pair.dataset, LABELS[method], method, r, group.completed]
        for col in columns:
            stat = group.get(col.key)
            csv_row += [stat.mean, stat.std, stat.n]
        csv_row += [d_acc, p_acc, d_gap, p_gap]
        csv_rows.append(csv_row)
    csv_header = ["study", "round", "optimizee", "dataset", "row", "method", "regularizer", "n"]
    for col in columns:
        csv_header += [f"{col.key}_mean", f"{col.key}_std", f"{col.key}_n"]
    csv_header += ["delta_top1_vs_m1_pp", "delta_top1_p", "delta_gap_vs_m1_pp", "delta_gap_p"]
    return Table(
        title=f"Stage 1 benchmark, round {rnd.name}: {pair.label} — M1 against M2(r) with its regularizer, beside M3 and M4(r)",
        label=f"tab:{study}-{rnd.name}-benchmark",
        headers=headers, latex_headers=latex_headers, rows=rows, latex_rows=latex_rows,
        csv_header=csv_header, csv_rows=csv_rows, align="lr" + "r" * (len(headers) - 2),
        notes=[f"The round's source pair, {pair.label}: M2(r) trains with r at every step (Stage 1) — the explicit-"
               "regularization benchmark against M1; M3 and M4(r) are their seen-setting runs (Stage 2) on the same "
               "images and seeds, without computing r. Δ: mean difference to M1 (Welch p over seeds). "
               "Gap: train − test top-1. M1's and M2's runs are shared with the seen settings of Stages 2–3."],
    )


# ───────────────────────────────────────────────────────────────── cross-evaluation of the rounds

def _transfer_variants(groups: Mapping[RoundKey, Group], study: str) -> List[Tuple[str, str, str]]:
    """(label, method, regularizer) of the learned variants with runs, in table order."""
    out = []
    if any(k[0] == study and k[3] == "m3_lstm" for k in groups):
        out.append(("M3", "m3_lstm", "none"))
    out += [(f"M4-{REG_NAMES[r]}", "m4_lstm_reg", r) for r in REGULARIZERS
            if any(k[0] == study and k[3] == "m4_lstm_reg" and k[4] == r for k in groups)]
    return out


def transfer_table(groups: Mapping[RoundKey, Group], study: str, rounds: Sequence[Round], pairs: Sequence[Pair],
                   metric: Metric) -> Optional[Table]:
    """The cross-evaluation: every pair (rows) trained with M1 and with the optimizers of every round
    (columns), each pair's role for each round alongside. Reading along a row compares the rounds'
    optimizers on the same task; reading the source pairs, the seen round against the unseen one."""
    variants = _transfer_variants(groups, study)
    if len(rounds) < 2 or not variants or not pairs:
        return None
    col = Column(metric.record, metric.label, metric.label, metric.scale, metric.digits, metric.better)
    heads = ["M1"] + [f"{label}·{r.name}" for label, _, _ in variants for r in rounds]
    rows, latex_rows, csv_rows = [], [], []
    for pair in pairs:
        keys = [(study, pair.optimizee, pair.dataset, "m1_adamw", "none", "")]
        keys += [(study, pair.optimizee, pair.dataset, m, reg, r.name) for _, m, reg in variants for r in rounds]
        groups_row = [groups.get(k) or Group() for k in keys]
        if all(g.completed == 0 for g in groups_row):
            continue
        roles = [ROLE_LABELS[r.role(pair.optimizee, pair.dataset)] for r in rounds]
        cells = [_cell(g.get(metric.record), col, runs=g.completed) for g in groups_row]
        rows.append([pair.label] + roles + [c[0] for c in cells])
        latex_rows.append([_tex_text(pair.label)] + [_tex_text(x) for x in roles] + [c[1] for c in cells])
        csv_row: List[Any] = [study, pair.optimizee, pair.dataset] + [r.role(pair.optimizee, pair.dataset) for r in rounds]
        for g in groups_row:
            stat = g.get(metric.record)
            csv_row += [None if stat.mean is None else stat.mean * metric.scale,
                        None if stat.std is None else stat.std * metric.scale, stat.n]
        csv_rows.append(csv_row)
    if not rows:
        return None
    role_heads = [f"Role for {r.name}" for r in rounds]
    csv_header = ["study", "optimizee", "dataset"] + [f"role_{r.name}" for r in rounds]
    for h in heads:
        csv_header += [f"{h}_mean", f"{h}_std", f"{h}_n"]
    sources = "; ".join(f"{r.name} = {Pair(r.optimizee, r.dataset).label}" for r in rounds)
    return Table(
        title=f"Cross-evaluation — {metric.label} {metric.arrow} of every pair with every round's optimizers ({study})",
        label=f"tab:{study}-transfer-{metric.key}",
        headers=["Optimizee / dataset"] + role_heads + heads,
        latex_headers=["Optimizee / dataset"] + role_heads + [_tex_text(h) for h in heads],
        rows=rows, latex_rows=latex_rows, csv_header=csv_header, csv_rows=csv_rows,
        align="l" + "l" * len(rounds) + "r" * len(heads),
        notes=[f"{metric.label} ({metric.unit}), mean ± std over seeds. ·A, ·B …: the optimizers of that round "
               f"(sources: {sources}). The same pair, trained with the optimizers of a round that saw it and of one "
               "that did not, separates a harder task from a transfer gap."],
    )


def transfer_summary(groups: Mapping[RoundKey, Group], study: str, rounds: Sequence[Round], pairs: Sequence[Pair],
                     metric: Metric) -> Optional[Table]:
    """How far the learned optimizers carry their advantage over AdamW: for each variant and round,
    the mean of (variant − M1) over the pairs of each role, and on each round's source pair the
    difference between its own optimizers (seen) and the other rounds' (unseen there)."""
    variants = _transfer_variants(groups, study)
    if not variants or not rounds:
        return None
    rows, latex_rows, csv_rows = [], [], []
    for label, method, reg in variants:
        for rnd in rounds:
            per_role: Dict[str, List[float]] = {role: [] for role in ROLES}
            for pair in pairs:
                m1 = (groups.get((study, pair.optimizee, pair.dataset, "m1_adamw", "none", "")) or Group()).get(metric.record)
                v = (groups.get((study, pair.optimizee, pair.dataset, method, reg, rnd.name)) or Group()).get(metric.record)
                if m1.mean is not None and v.mean is not None:
                    per_role[rnd.role(pair.optimizee, pair.dataset)].append((v.mean - m1.mean) * metric.scale)
            means = [sum(v) / len(v) if v else None for v in (per_role[role] for role in ROLES)]
            # on this round's source: own optimizers minus the other rounds' (there the pair is unseen)
            own = (groups.get((study, rnd.optimizee, rnd.dataset, method, reg, rnd.name)) or Group()).get(metric.record)
            others = [(groups.get((study, rnd.optimizee, rnd.dataset, method, reg, o.name)) or Group()).get(metric.record)
                      for o in rounds if o.name != rnd.name]
            others = [o.mean for o in others if o.mean is not None]
            gap = None if own.mean is None or not others else (own.mean - sum(others) / len(others)) * metric.scale
            if all(m is None for m in means) and gap is None:
                continue
            counts = [len(per_role[role]) for role in ROLES]
            texts = [fmt_number(m, metric.digits, signed=True) + (f" (n={n})" if m is not None else "")
                     for m, n in zip(means, counts)]
            gap_text = fmt_number(gap, metric.digits, signed=True)
            rows.append([label, rnd.name] + texts + [gap_text])
            latex_rows.append([_tex_text(label), rnd.name] + [_tex_text(t) for t in texts] + [_tex_text(gap_text)])
            csv_rows.append([study, label, method, reg, rnd.name] + means + counts + [gap])
    if not rows:
        return None
    role_heads = [ROLE_LABELS[r] for r in ROLES]
    return Table(
        title=f"Transfer of the learned optimizers — {metric.label}: variant − M1 by role ({study})",
        label=f"tab:{study}-transfer-summary-{metric.key}",
        headers=["Variant", "Round"] + [f"{h} (vs M1)" for h in role_heads] + ["Source pair: own − other rounds"],
        latex_headers=["Variant", "Round"] + [_tex_text(h) for h in role_heads] + ["Source: own $-$ other"],
        rows=rows, latex_rows=latex_rows,
        csv_header=["study", "variant", "method", "regularizer", "round"] + [f"{r}_vs_m1" for r in ROLES]
        + [f"{r}_pairs" for r in ROLES] + ["source_own_minus_other"],
        csv_rows=csv_rows, align="ll" + "r" * (len(ROLES) + 1),
        notes=[f"{metric.label} ({metric.unit}). Columns 3–6: mean over the pairs with that role for the round of "
               "(variant − M1 on the same pair); n = pairs. Last column: on the round's own source pair, its optimizers "
               "minus the other rounds' — the cost of not having meta-trained on that task (a transfer gap)."],
    )


# ───────────────────────────────────────────────────────────────── summary, stage by stage

def _delta(groups, key_a: GroupKey, key_b: GroupKey, metric: str = "test_top1") -> Tuple[Optional[float], Optional[float]]:
    """Mean difference a − b (pp for accuracies) and its Welch p-value."""
    a, b = groups.get(key_a), groups.get(key_b)
    if a is None or b is None:
        return None, None
    sa, sb = a.get(metric), b.get(metric)
    if sa.mean is None or sb.mean is None:
        return None, None
    return 100 * (sa.mean - sb.mean), welch_p(sa.values, sb.values)


def _p_text(p: Optional[float]) -> str:
    return "" if p is None else f" (p = {p:.2g})"


RunId = Tuple[str, str, str, str, str, int]  # optimizee, dataset, method, regularizer, round ("" for M1/M2), seed


def expected_runs(study: str) -> Set[RunId]:
    """Every optimizee run the experiment files of ``study`` train (Stage 1's M2 and Stages 2–3)."""
    from reglo.config import default_meta_tag
    from reglo.manifest import expand

    out = set()
    default = default_meta_tag()
    for spec in study_experiments(study):
        if "train" not in spec:
            continue
        for line in expand(spec, ["train"]):
            arg = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in line[1:] if "=" in a}
            method = arg["method"]
            out.add((arg["optimizee"], arg["dataset"], method, arg.get("regularizer", "none"),
                     arg.get("meta_tag", default) if method in LEARNED else "", int(arg["seed"])))
    return out


def _run_id(r: Mapping[str, Any]) -> RunId:
    return (r["optimizee"], r["dataset"], r["method"], r["regularizer"],
            (r.get("round") or "") if r["method"] in LEARNED else "", int(r["seed"]))


def _inventory(study: str, records: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    found = {_run_id(r) for r in records}
    missing = []
    for o, d, m, r, rnd, s in sorted(expected_runs(study) - found):
        tag = (m if r == "none" else f"{m}-{r}") + (f"@{rnd}" if rnd else "")
        missing.append(f"{o}-{d}/{tag}/seed{s}")
    failed = [(str(r.get("status")), r["run"]) for r in records if r.get("status") != "completed"]
    return missing, failed


def _progress(expected: Set[RunId], records: Sequence[Mapping[str, Any]], pair: Pair, round_name: str) -> str:
    """Finished / expected evaluation runs on a pair as a round sees them: M1, M2, and the round's M3/M4."""
    def mine(o: str, d: str, m: str, rnd: str) -> bool:
        return (o, d) == (pair.optimizee, pair.dataset) and (m not in LEARNED or rnd == round_name)

    total = sum(1 for e in expected if mine(e[0], e[1], e[2], e[4]))
    done = sum(1 for r in records if r.get("status") == "completed" and mine(r["optimizee"], r["dataset"], r["method"],
                                                                            (r.get("round") or "")))
    return f"{done} / {total}" if total else str(done)


def study_summary(
    groups: Mapping[RoundKey, Group],
    records: Sequence[Mapping[str, Any]],
    study: str,
    settings: Mapping[str, Mapping[str, Sequence[SettingPair]]],
    rounds: Sequence[Round],
    cubes: Mapping[str, Mapping[str, Table]],
    benchmarks: Mapping[str, Table],
    transfer: Sequence[Table],
    pair_tables: Mapping[Tuple[str, str, str, str], Table],
    meta_tags: Sequence[str] = (),
) -> str:
    """Markdown report of one study, organised like the study design: Stages 1–4 (per round), the
    cross-evaluation of the rounds, then the detail."""
    study_records = [r for r in records if r["study"] == study]
    audited = sum(1 for r in study_records if r.get("audited"))
    profile = active_profile()
    expected = expected_runs(study)
    lines = [
        f"# Study `{study}` — results ({profile} profile)",
        "",
        f"{len(study_records)} runs ({audited} audited) · generated {_dt.datetime.now():%Y-%m-%d %H:%M} "
        f"by `scripts/make_report.py --profile {profile}`",
        "",
    ]
    if profile != "production":
        lines += [f"> **{profile} profile**: the numbers in configs/profiles/{profile}.yaml — a check that every "
                  "stage runs, not results. The thesis numbers come from the production profile.", ""]
    if rounds:
        lines += ["Rounds (configs/rounds.yaml): " + "; ".join(
            f"**{r.name}** — optimizers meta-trained on {Pair(r.optimizee, r.dataset).label}" for r in rounds) + ".", ""]
    if settings:
        lines += [f"## Stage 1 · {STAGES[1]}", ""]
        for rnd in rounds:
            lines += [f"### Round {rnd.name} — source {Pair(rnd.optimizee, rnd.dataset).label}", ""]
            if rnd.name in meta_tags:
                lines.append(f"LSTM optimizers φ (M3) and φ*_r (M4): `tables/meta_training/{rnd.name}.md`, curves in "
                             f"`figures/meta_training/{rnd.name}/`.")
            else:
                lines.append("No Stage-1 meta-training results for this round yet.")
            lines.append("")
            if rnd.name in benchmarks:
                lines += [benchmarks[rnd.name].markdown(level=0)]
        trained_by = {}
        for spec in study_experiments(study):
            if is_evaluation(spec):
                for pair in training_pairs(spec):
                    trained_by.setdefault(pair, spec["name"])
        for stage in (2, 3):
            entries = [sp for per in settings.values() for pairs in per.values() for sp in pairs if sp.setting.stage == stage]
            if not entries:
                continue
            lines += [f"## Stage {stage} · {STAGES[stage]}", "",
                      "| Round | Setting | Optimizee / dataset | Role | Finished runs (M1, M2 + round's M3/M4) | Trained by |",
                      "|:--|:--|:--|:--|--:|:--|"]
            for sp in entries:
                mark = " †" if sp.extended else ""
                lines.append(f"| {sp.round} | {sp.setting.label} | {sp.pair.label}{mark} | {ROLE_LABELS[sp.role]} | "
                             f"{_progress(expected, study_records, sp.pair, sp.round)} | {trained_by.get(sp.pair, '—')} |")
            lines += ["", "M2 trains with r at every step on every pair (on a round's source pair: its Stage-1 runs). "
                      "Training curves: `figures/" + study + "/round_<R>/<optimizee>-<dataset>/training_curves.png`.", ""]
        lines += [f"## Stage 4 · {STAGES[4]} — 3D tables per round (method × regularizer × evaluation setting)", ""]
        if cubes:
            for round_name, tables in cubes.items():
                for table in tables.values():
                    lines += [f"### {table.title}", "", table.markdown(level=0)]
        else:
            lines += ["No M4 results yet.", ""]

        d_lines = []
        for rnd in rounds:
            view = round_view(groups, rnd.name)
            for sp in [sp for pairs in settings.get(rnd.name, {}).values() for sp in pairs]:
                if any(line.startswith(f"- **{rnd.name} · {sp.pair.label}**") for line in d_lines):
                    continue
                k = lambda m, r: (study, sp.pair.optimizee, sp.pair.dataset, m, r)  # noqa: E731
                parts = []
                for reg in REGULARIZERS:
                    d1, p1 = _delta(view, k("m4_lstm_reg", reg), k("m1_adamw", "none"))
                    d2, p2 = _delta(view, k("m4_lstm_reg", reg), k("m2_adamw_reg", reg))
                    d3, p3 = _delta(view, k("m4_lstm_reg", reg), k("m3_lstm", "none"))
                    if d1 is not None or d2 is not None or d3 is not None:
                        parts.append(f"{REG_NAMES[reg]}: {fmt_number(d1, 2, signed=True)}{_p_text(p1)} / "
                                     f"{fmt_number(d2, 2, signed=True)}{_p_text(p2)} / "
                                     f"{fmt_number(d3, 2, signed=True)}{_p_text(p3)}")
                if parts:
                    d_lines.append(f"- **{rnd.name} · {sp.pair.label}** ({ROLE_LABELS[sp.role]}) — " + " · ".join(parts))
        if d_lines:
            lines += ["### M4 against M1, M2 and M3 — test top-1 differences "
                      "(pp: vs M1 / vs M2 with the same r / vs M3, Welch p)", ""] + d_lines + [""]

    if transfer:
        lines += ["## Cross-evaluation of the rounds — the transfer gap of the learned optimizers", "",
                  "Every pair is trained with the optimizers of every round. On a round's own source pair its "
                  "optimizers are 'seen'; elsewhere they meet an unseen network, dataset or both. Comparing the "
                  "rounds on the same pair separates a harder task from a transfer gap.", ""]
        for table in transfer:
            lines += [f"### {table.title}", "", table.markdown(level=0)]

    lines += ["## Detail per round and pair: M1 · M2(r) · M3 · M4(r), with the property each regularizer targets", ""]
    for rnd_name in [r.name for r in rounds] or [""]:
        for pair in pairs_of(study_records, study):
            tables = [pair_tables.get((rnd_name, pair.optimizee, pair.dataset, reg)) for reg in (*REGULARIZERS, "none")]
            if not any(tables):
                continue
            lines += [f"### Round {rnd_name}: {pair.label}" if rnd_name else f"### {pair.label}", ""]
            view = round_view(groups, rnd_name)
            for reg, table in zip((*REGULARIZERS, "none"), tables):
                if table is None:
                    continue
                lines.append(table.markdown(level=4))
                if reg == "none":
                    d31, p31 = _delta(view, (study, pair.optimizee, pair.dataset, "m3_lstm", "none"),
                                      (study, pair.optimizee, pair.dataset, "m1_adamw", "none"))
                    lines += [f"**Top-1 difference:** M3 − M1: {fmt_number(d31, 2, signed=True)} pp{_p_text(p31)}", ""]

    missing, failed = _inventory(study, study_records)
    if missing or failed:
        lines += ["## Missing or failed runs", ""]
        lines += [f"- missing: {m}" for m in missing]
        lines += [f"- {status}: {run}" for status, run in failed]
        lines.append("")
    return "\n".join(lines)


def write_runs_csv(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Every run (all seeds), one row each — the raw material of every table."""
    first = ["study", "optimizee", "dataset", "method", "label", "regularizer", "seed", "status"]
    keys = first + sorted({k for r in records for k in r} - set(first))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow({k: "" if r.get(k) is None else r.get(k) for k in keys})
    return path


# ───────────────────────────────────────────────────────────────── across studies

def overview_table(groups: Mapping[RoundKey, Group], studies: Sequence[str]) -> Optional[Table]:
    """Δ test top-1 of M4(r) against M1, M2(r) and M3, for every study × round × pair (pp)."""
    regs = [r for r in REGULARIZERS if any(k[3] == "m4_lstm_reg" and k[4] == r for k in groups)]
    if not regs:
        return None
    rows, latex_rows, csv_rows = [], [], []
    for study in studies:
        for rnd in sorted({k[5] for k in groups if k[0] == study and k[3] == "m4_lstm_reg"}):
            view = round_view(groups, rnd)
            pairs = sorted({(k[1], k[2]) for k in view if k[0] == study},
                           key=lambda p: (list(DATASET_NAMES).index(p[1]) if p[1] in DATASET_NAMES else 99,
                                          list(OPTIMIZEE_NAMES).index(p[0]) if p[0] in OPTIMIZEE_NAMES else 99))
            for o, d in pairs:
                cells, tex_cells, csv_row = [], [], [study, rnd, o, d]
                for reg in regs:
                    for other in (("m1_adamw", "none"), ("m2_adamw_reg", reg), ("m3_lstm", "none")):
                        delta, p = _delta(view, (study, o, d, "m4_lstm_reg", reg), (study, o, d, *other))
                        star = "*" if p is not None and p < 0.05 else ""
                        cells.append(fmt_number(delta, 2, signed=True) + star)
                        tex_cells.append((fmt_number(delta, 2, signed=True) if delta is not None else DASH_TEX)
                                         + ("$^*$" if star else ""))
                        csv_row += [delta, p]
                if any(c != DASH_MD for c in cells):
                    rows.append([study, rnd, Pair(o, d).label] + cells)
                    latex_rows.append([_tex_text(study), rnd, _tex_text(Pair(o, d).label)] + tex_cells)
                    csv_rows.append(csv_row)
    if not rows:
        return None
    headers = ["Study", "Round", "Optimizee / dataset"]
    latex_headers = ["Study", "Round", "Optimizee / dataset"]
    csv_header = ["study", "round", "optimizee", "dataset"]
    for reg in regs:
        name = REG_NAMES[reg]
        headers += [f"{name}: vs M1", f"{name}: vs M2", f"{name}: vs M3"]
        latex_headers += [f"{name} vs M1", f"{name} vs M2", f"{name} vs M3"]
        csv_header += [f"{reg}_vs_m1_pp", f"{reg}_vs_m1_p", f"{reg}_vs_m2_pp", f"{reg}_vs_m2_p",
                       f"{reg}_vs_m3_pp", f"{reg}_vs_m3_p"]
    return Table(
        title="Test top-1 of M4(r) minus M1, minus M2(r) and minus M3, in percentage points",
        label="tab:overview",
        headers=headers, latex_headers=latex_headers, rows=rows, latex_rows=latex_rows,
        csv_header=csv_header, csv_rows=csv_rows, align="lll" + "r" * (3 * len(regs)),
        notes=["* Welch t-test p < 0.05 over seeds. Positive: M4 is more accurate. Round: whose optimizers M3/M4 use. "
               "vs M2: against AdamW with the same regularizer r computed at every step."],
    )


# ───────────────────────────────────────────────────────────────── Stage 1

def _tail_mean(rows: Sequence[Mapping[str, Any]], key: str, fraction: float = 0.1) -> Optional[float]:
    values = [v for v in (to_float(r.get(key)) for r in rows) if v is not None and math.isfinite(v)]
    if not values:
        return None
    tail = values[-max(1, int(len(values) * fraction)):]
    return sum(tail) / len(tail)


def _median(values: List[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def meta_table(arms: Sequence[Mapping[str, Any]], meta_tag: str, source: str = "") -> Optional[Table]:
    """Stage 1: how each LSTM optimizer of one round (``meta_tag``) was meta-trained."""
    if not arms:
        return None
    headers = ["Optimizer", "Status", "Trajectories", "Meta-steps", "Best val. loss", "Best val. acc. (%)",
               "L_meta (end)", "L_smooth (end)", "L_reg (end)", "Median ‖∇φ‖", "Skipped", "Diverged", "Hours"]
    latex_headers = ["Optimizer", "Status", "Traj.", "Meta-steps", "Best val.\\ loss", "Best val.\\ acc.\\ (\\%)",
                     "$L_{\\mathrm{meta}}$ (end)", "$L_{\\mathrm{smooth}}$ (end)", "$L_{\\mathrm{reg}}$ (end)",
                     "Median $\\|\\nabla\\phi\\|$", "Skipped", "Diverged", "Hours"]
    rows, latex_rows, csv_rows = [], [], []
    for arm in arms:
        s, steps = arm["summary"], arm["steps"]
        reg_name = REG_NAMES.get(arm["regularizer"], arm["regularizer"])
        if arm["meta_training"] == "m3_lstm":
            name, latex_name = "M3 · φ", r"M3 $\cdot$ $\phi$"
        else:
            name, latex_name = f"M4 · φ*_{reg_name}", rf"M4 $\cdot$ $\phi^*_{{\mathrm{{{reg_name}}}}}$"
        uses_smooth = s.get("smooth", arm["meta_training"] != "m3_lstm")
        uses_reg = arm["regularizer"] != "none"
        values = [
            str(s.get("status", "?")),
            str(s.get("trajectories", max((r.get("trajectory", -1) for r in steps), default=-1) + 1)),
            str(s.get("meta_steps", sum(1 for r in steps if not r.get("diverged")))),
            fmt_number(to_float(s.get("best_val_loss")), 4),
            fmt_number(None if to_float(s.get("best_val_acc")) is None else 100 * to_float(s.get("best_val_acc")), 2),
            fmt_number(_tail_mean(steps, "l_meta"), 4),
            fmt_number(_tail_mean(steps, "l_smooth"), 4) if uses_smooth else DASH_MD,
            fmt_number(_tail_mean(steps, "l_reg"), 4) if uses_reg else DASH_MD,
            fmt_number(_median([to_float(r.get("grad_norm")) for r in steps]), 3),
            str(sum(1 for r in steps if r.get("skipped") and not r.get("diverged"))),
            str(sum(1 for r in steps if r.get("diverged"))),
            fmt_number(None if to_float(s.get("wall_clock_s")) is None else to_float(s.get("wall_clock_s")) / 3600, 2),
        ]
        rows.append([name] + values)
        latex_rows.append([latex_name] + [v.replace(DASH_MD, DASH_TEX) for v in values])
        csv_rows.append([arm["arm"], arm["meta_training"], arm["regularizer"]] + values)
    return Table(
        title=f"Stage 1 meta-training, round {meta_tag}" + (f" ({source})" if source else ""),
        label=f"tab:meta-{meta_tag}",
        headers=headers, latex_headers=latex_headers, rows=rows, latex_rows=latex_rows,
        csv_header=["arm", "meta_training", "regularizer"] + [h.lower().replace(" ", "_") for h in headers[1:]],
        csv_rows=csv_rows, align="ll" + "r" * (len(headers) - 2),
        notes=["(end): mean of the last 10% of meta-steps. Skipped: meta-steps with a non-finite meta-gradient. "
               "Diverged: truncations whose inner loss became non-finite."],
    )


# ───────────────────────────────────────────────────────────────── Stage 4: same-GPU timing

def compute_table(profile: Mapping[str, Any]) -> Table:
    """All twelve variants timed back-to-back on one GPU."""
    rows, latex_rows, csv_rows = [], [], []
    for r in profile["rows"]:
        name = r["label"] if r["regularizer"] == "none" else f"{r['label']} {REG_NAMES.get(r['regularizer'], r['regularizer'])}"
        values = [
            r.get("phi") or DASH_MD,
            str(r["passes"]),
            fmt_number(r["flops_per_step"] / 1e9, 3),
            fmt_number(r["flops_vs_m1"], 2) + "×",
            fmt_number(1e3 * r["seconds_per_step"], 2),
            fmt_number(r["time_vs_m1"], 2) + "×",
        ]
        rows.append([name] + values)
        latex_rows.append([name] + [_tex_text(v) for v in values])
        csv_rows.append([r["label"], r["method"], r["regularizer"], r.get("phi"), r["passes"], r["flops_per_step"],
                         r["flops_vs_m1"], r["seconds_per_step"], r["time_vs_m1"]])
    device = profile.get("gpu") or profile.get("device")
    pair = Pair(profile["optimizee"], profile["dataset"]).label
    return Table(
        title=f"Cost per training step, {pair} (batch {profile['batch_size']}, {profile['num_params']:,} parameters, {device})",
        label=f"tab:compute-{profile['optimizee']}-{profile['dataset']}",
        headers=["Variant", "φ", "Passes", "GFLOPs/step", "FLOPs × M1", "ms/step", "Time × M1"],
        latex_headers=["Variant", "$\\phi$", "Passes", "GFLOPs/step", "FLOPs $\\times$ M1", "ms/step", "Time $\\times$ M1"],
        rows=rows, latex_rows=latex_rows,
        csv_header=["label", "method", "regularizer", "phi", "passes", "flops_per_step", "flops_vs_m1",
                    "seconds_per_step", "time_vs_m1"],
        csv_rows=csv_rows, align="llrrrrr",
        notes=[f"FLOPs: forward + backward (+ the LSTM step for M3/M4, {profile.get('lstm_flops_per_parameter', '?')} "
               f"FLOPs per parameter). Time: mean of {profile.get('timed_steps', '?')} steps after "
               f"{profile.get('warmup_steps', '?')} warm-up steps. φ: trained = the Stage-1 checkpoint was found "
               "(untrained has the same cost)."],
    )


def settings_columns(settings: Mapping[str, Sequence[SettingPair]], extended: bool) -> List[SettingPair]:
    """The cube's X-axis for one round (``settings``: that round's setting → pairs): every setting's
    pairs, core only or core + extended."""
    return [sp for key in (s.key for s in SETTINGS) for sp in settings.get(key, []) if extended or not sp.extended]


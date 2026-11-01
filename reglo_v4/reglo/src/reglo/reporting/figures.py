"""Figures for the thesis — PNG (screen) and PDF (LaTeX), one style throughout.

    figures/<study>/round_<R>/cube_{accuracy,gen_gap,flops}           Stage 4: one round's 3D tables as figures —
                                                                      a panel per evaluation setting, regularizers
                                                                      along x, M1, M2(r), M3, M4(r)
    figures/<study>/round_<R>/<optimizee>-<dataset>/training_curves   training loss of M1 · M2(r) · M3 · M4(r)
    figures/<study>/round_<R>/<optimizee>-<dataset>/pareto_flops      accuracy vs. GFLOPs per step, one panel per r
    figures/<study>/round_<R>/<optimizee>-<dataset>/pareto_time       accuracy vs. time per step (same-GPU profile)
    figures/<study>/round_<R>/<optimizee>-<dataset>/poc               the property r targets, at convergence and completion
    figures/<study>/round_<R>/<optimizee>-<dataset>/weights           distribution of log10|w| (L1, L2)
    figures/meta_training/<round>/convergence                         Stage 1: L_meta, ‖∇φ‖, validation loss — M3 vs M4(r)
    figures/meta_training/<round>/terms                               Stage 1: L_reg and L_smooth of each M4(r)

Style rules: one fixed colour per method in every figure (M1 grey baseline, M2 orange,
M3 aqua, M4 blue — validated colour-blind safe as a set), with a marker shape as second
cue; a legend whenever there are two or more series, plus direct labels; text in ink
colours, never in series colours; one y-axis per panel (measures on different scales get
their own panel, never a second axis); hairline grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # files only; no display on the cluster
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from reglo.reporting.collect import Group, GroupKey, epoch_curves, to_float  # noqa: E402
from reglo.reporting.tables import method_rows  # noqa: E402
from reglo.rounds import Round  # noqa: E402
from reglo.study import LABELS, LEARNED, REG_NAMES, REGULARIZERS, Metric, Pair, SettingPair  # noqa: E402
from reglo.utils.logging import read_json  # noqa: E402

__all__ = [
    "METHOD_STYLE",
    "cube_figure",
    "curves_figure",
    "meta_figures",
    "pareto_figure",
    "poc_figure",
    "save",
    "weights_figure",
]

SURFACE = "#ffffff"
INK = "#0b0b0b"        # titles, values
INK_2 = "#52514e"      # labels, legends, ticks
MUTED = "#898781"
GRID = "#e1e0d9"       # hairline grid
AXIS = "#c3c2b7"       # baseline / axis line

METHOD_STYLE: Dict[str, Dict[str, str]] = {
    "M1": {"color": "#898781", "marker": "o", "legend": "M1 · AdamW"},
    "M2": {"color": "#eb6834", "marker": "s", "legend": "M2 · AdamW + r"},
    "M3": {"color": "#1baf7a", "marker": "^", "legend": "M3 · LSTM"},
    "M4": {"color": "#2a78d6", "marker": "D", "legend": "M4 · LSTM-r (ours)"},
}

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 8.5,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 9.5, "axes.titlelocation": "left",
    "axes.labelcolor": INK_2, "axes.labelsize": 8.5, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK_2,
    "lines.linewidth": 1.5, "lines.markersize": 6.5, "errorbar.capsize": 0,
    "pdf.fonttype": 42,  # editable text in the PDF
}

POC_METRIC = {  # regularizer → (record field prefix, axis label, scale)
    "l1": ("sparsity", "Share of |w| < 10⁻³ (%)", 100.0),
    "l2": ("l2", "‖θ‖₂", 1.0),
    "sam": ("max_loss", "Max loss in the ε-ball", 1.0),
    "gam": ("max_grad_norm", "Max ‖∇L‖ in the ε-ball", 1.0),
    "gsam": ("surrogate_gap", "Surrogate gap in the ε-ball", 1.0),
}


def save(fig, stem: Path, dpi: int = 200) -> List[Path]:
    """``stem``.png and ``stem``.pdf; closes the figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(f"{stem}.png"), Path(f"{stem}.pdf")]  # not with_suffix: stems may contain dots
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return paths


def _pair_title(optimizee: str, dataset: str) -> str:
    return Pair(optimizee, dataset).label


def _legend_handles(kind: str = "marker") -> List[Any]:
    if kind == "patch":
        return [Patch(facecolor=s["color"], edgecolor=SURFACE, label=s["legend"]) for s in METHOD_STYLE.values()]
    return [Line2D([], [], color=s["color"], marker=s["marker"], linestyle="none", markeredgecolor=SURFACE,
                   markeredgewidth=1.0, label=s["legend"]) for s in METHOD_STYLE.values()]


def _spread(values: Sequence[float], gap: float, lo: float = 0.04, hi: float = 0.96) -> List[float]:
    """Move label positions apart to at least ``gap``, as little as possible, inside [lo, hi] (order kept)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [min(max(values[i], lo), hi) for i in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(placed)):
            overlap = gap - (placed[k] - placed[k - 1])
            if overlap > 1e-12:
                placed[k - 1] -= overlap / 2
                placed[k] += overlap / 2
                moved = True
        if placed and placed[-1] > hi:  # pushed past the top: slide the whole stack down
            shift = placed[-1] - hi
            placed = [p - shift for p in placed]
            moved = True
        if placed and placed[0] < lo:
            shift = lo - placed[0]
            placed = [p + shift for p in placed]
            moved = moved or shift > 1e-12
        if not moved:
            break
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k]
    return out


def _direct_labels(ax, points: Sequence[Tuple[float, float, str]], gap: float = 0.085, dx: float = 7.0) -> None:
    """Label points to their right in ink colour, nudged vertically so labels never overlap."""
    if not points:
        return
    to_axes = ax.transData + ax.transAxes.inverted()
    fractions = [to_axes.transform((x, y))[1] for x, y, _ in points]
    for (x, y, text), frac in zip(points, _spread(fractions, gap)):
        x_axes = to_axes.transform((x, y))[0]
        ax.annotate(text, xy=(x_axes, frac), xycoords="axes fraction", xytext=(dx, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=INK_2)


def _label_when_settled(pending: Sequence[Tuple[Any, Sequence[Tuple[float, float, str]]]]) -> None:
    """Direct labels once every panel is drawn, so shared and autoscaled limits are final."""
    for ax, _ in pending:
        ax.autoscale_view()
        _ = ax.viewLim  # resolves pending autoscaling of shared axes
    for ax, points in pending:
        _direct_labels(ax, points)


def _labels_above(ax, points: Sequence[Tuple[float, float, str]], gap: float = 0.07) -> None:
    """Label peaks from above (the free side of a distribution), spread sideways so labels never overlap."""
    if not points:
        return
    to_axes = ax.transData + ax.transAxes.inverted()
    xy = [to_axes.transform((x, y)) for x, y, _ in points]
    for (x_frac, y_frac), x_new, (_, _, text) in zip(xy, _spread([p[0] for p in xy], gap, 0.03, 0.97), points):
        ax.annotate(text, xy=(x_new, y_frac), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK_2)


def _padded(lo: float, hi: float, minimum_span: float, pad: float = 0.18) -> Tuple[float, float]:
    span = max(hi - lo, minimum_span)
    mid = 0.5 * (lo + hi)
    return mid - span / 2 - pad * span, mid + span / 2 + pad * span


def _regs_with_runs(groups: Mapping[GroupKey, Group], study: str, optimizee: str, dataset: str) -> List[str]:
    return [r for r in REGULARIZERS
            if (study, optimizee, dataset, "m2_adamw_reg", r) in groups or (study, optimizee, dataset, "m4_lstm_reg", r) in groups]


# ───────────────────────────────────────────────────────────────── accuracy vs. compute

def _profiled_ms(profile: Optional[Mapping[str, Any]], method: str, reg: str) -> Optional[float]:
    for row in (profile or {}).get("rows", []):
        if row["method"] == method and row["regularizer"] == reg:
            return 1e3 * float(row["seconds_per_step"])
    return None


def pareto_figure(
    groups: Mapping[GroupKey, Group],
    study: str,
    optimizee: str,
    dataset: str,
    x: str = "flops",
    profile: Optional[Mapping[str, Any]] = None,
    title: Optional[str] = None,
):
    """Test top-1 against cost per step: M1 · M2(r) · M3 · M4(r), one panel per regularizer.

    ``x="flops"``: GFLOPs per step (from the runs). ``x="time"``: milliseconds per step —
    from the Stage-4 profile when there is one (every variant on the same GPU), else
    measured during training. Error bars: std of top-1 over seeds.
    """
    regs = _regs_with_runs(groups, study, optimizee, dataset)
    if not regs:
        return None
    panels: List[List[Tuple[str, float, float, float]]] = []
    for reg in regs:
        points = []
        for _, method, r in method_rows(reg):
            group = groups.get((study, optimizee, dataset, method, r))
            if group is None:
                continue
            acc = group.get("test_top1")
            if x == "flops":
                cost = group.get("gflops_per_step").mean
            else:
                cost = _profiled_ms(profile, method, r) if profile else group.get("ms_per_step").mean
            if acc.mean is None or cost is None:
                continue
            points.append((LABELS[method], cost, 100 * acc.mean, 100 * (acc.std or 0.0)))
        panels.append(points)
    if not any(panels):
        return None

    all_x = [p[1] for pts in panels for p in pts]
    all_lo = [p[2] - p[3] for pts in panels for p in pts]
    all_hi = [p[2] + p[3] for pts in panels for p in pts]
    xlim = _padded(min(all_x), max(all_x), 0.2 * max(all_x), pad=0.25)
    ylim = _padded(min(all_lo), max(all_hi), 2.0)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(regs), figsize=(2.35 * len(regs) + 0.7, 2.9), sharex=True, sharey=True,
                                 squeeze=False, layout="constrained")
        for ax, reg, points in zip(axes[0], regs, panels):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            for label, cost, mean, std in points:
                style = METHOD_STYLE[label]
                ax.errorbar(cost, mean, yerr=std, fmt=style["marker"], color=style["color"], ecolor=style["color"],
                            elinewidth=1.0, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
            _direct_labels(ax, [(cost, mean, label) for label, cost, mean, _ in points])
            ax.set_title(REG_NAMES[reg])
            ax.set_xlabel("GFLOPs per step" if x == "flops" else "ms per step")
        axes[0][0].set_ylabel("Test top-1 (%)")
        prefix = title or study
        if x == "flops":
            what = "accuracy against FLOPs"
        else:
            where = f"profiled on {profile.get('gpu') or profile.get('device')}" if profile else "measured during training"
            what = f"accuracy against time per step ({where})"
        fig.suptitle(f"{prefix} · {_pair_title(optimizee, dataset)} — {what}", x=0.01, ha="left", fontsize=10, color=INK)
        fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=4, handletextpad=0.3, columnspacing=1.4)
    return fig


# ───────────────────────────────────────────────────────────────── the paper's PoC tests

def poc_figure(groups: Mapping[GroupKey, Group], study: str, optimizee: str, dataset: str, radius: float = 0.01,
               title: Optional[str] = None):
    """The property each regularizer targets, at convergence and at completion, for M1–M4.

    Needs the Stage-4 audits; panels without any audited value are left out (None if all are).
    """
    regs = _regs_with_runs(groups, study, optimizee, dataset)
    panels = []
    for reg in regs:
        prefix, ylabel, scale = POC_METRIC[reg]
        bars = []
        for _, method, r in method_rows(reg):
            group = groups.get((study, optimizee, dataset, method, r)) or Group()
            bars.append((LABELS[method], [group.get(f"{prefix}_{snap}") for snap in ("converged", "final")]))
        if any(s.mean is not None for _, stats in bars for s in stats):
            panels.append((reg, ylabel, scale, bars))
    if not panels:
        return None

    width = 0.19
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(2.45 * len(panels) + 0.7, 2.9), squeeze=False,
                                 layout="constrained")
        for ax, (reg, ylabel, scale, bars) in zip(axes[0], panels):
            for i, (label, stats) in enumerate(bars):
                offset = (i - 1.5) * width
                for j, stat in enumerate(stats):
                    if stat.mean is None:
                        if any(other[j].mean is not None for _, other in bars):  # this method only: say so
                            ax.text(j + offset, 0.02, "n/a", transform=ax.get_xaxis_transform(), rotation=90,
                                    ha="center", va="bottom", fontsize=6.5, color=MUTED)
                        continue
                    ax.bar(j + offset, scale * stat.mean, width=width, color=METHOD_STYLE[label]["color"],
                           edgecolor=SURFACE, linewidth=1.0, zorder=2)
                    if stat.n > 1:
                        ax.errorbar(j + offset, scale * stat.mean, yerr=scale * stat.std, fmt="none",
                                    ecolor=INK_2, elinewidth=0.9, zorder=3)
            for j in range(2):
                if all(stats[j].mean is None for _, stats in bars):  # e.g. no run met the convergence criterion
                    ax.text(j, 0.5, "no snapshot", transform=ax.get_xaxis_transform(), ha="center", va="center",
                            fontsize=7.5, color=MUTED)
            ax.set_xlim(-0.55, 1.55)
            ax.set_xticks([0, 1], ["at convergence", "at completion"])
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel(ylabel)
            ax.set_title(REG_NAMES[reg])
            ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        fig.suptitle(f"{title or study} · {_pair_title(optimizee, dataset)} — point-of-convergence tests (ε = {radius:g})",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.legend(handles=_legend_handles("patch"), loc="outside lower center", ncol=4, handletextpad=0.4,
                   columnspacing=1.4)
    return fig


# ───────────────────────────────────────────────────────────────── weight distributions

def _mean_histogram(runs: Sequence[str]) -> Optional[Tuple[List[float], List[float]]]:
    """Seed-averaged share (%) of penalised weights per log10|w| bin, from each run's weights.json."""
    shares, centres = [], None
    for run in runs:
        path = Path(run) / "weights.json"
        if not path.is_file():
            continue
        hist = read_json(path).get("histogram") or {}
        edges, counts = hist.get("log10_edges"), hist.get("counts")
        if not edges or not counts or sum(counts) == 0:
            continue
        centres = [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]
        total = float(sum(counts))
        shares.append([100.0 * c / total for c in counts])
    if not shares:
        return None
    return centres, [sum(col) / len(col) for col in zip(*shares)]


def weights_figure(records: Sequence[Mapping[str, Any]], study: str, optimizee: str, dataset: str,
                   round_name: Optional[str] = None, threshold: float = 1e-3):
    """Where the weights sit after training, for the norm regularizers (L1, L2): M1–M4 (M3/M4 with the
    optimizers of ``round_name``; M2 with r)."""
    panels = []
    for reg in ("l1", "l2"):
        lines = []
        for _, method, r in method_rows(reg):
            runs = [rec["run"] for rec in records
                    if rec["study"] == study and rec["optimizee"] == optimizee and rec["dataset"] == dataset
                    and rec["method"] == method and rec["regularizer"] == r and rec.get("status") == "completed"
                    and (round_name is None or method not in LEARNED or (rec.get("round") or "") == round_name)]
            hist = _mean_histogram(runs)
            if hist is not None:
                lines.append((LABELS[method], *hist))
        if any(label in ("M2", "M4") for label, *_ in lines):
            panels.append((reg, lines))
    if not panels:
        return None

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 0.6, 2.8), squeeze=False, sharey=True,
                                 layout="constrained")
        pending = []
        for ax, (reg, lines) in zip(axes[0], panels):
            label_points = []
            for label, centres, share in lines:
                style = METHOD_STYLE[label]
                ax.plot(centres, share, drawstyle="steps-mid", color=style["color"], linewidth=1.5, zorder=3)
                peak = max(range(len(share)), key=share.__getitem__)
                ax.plot(centres[peak], share[peak], marker=style["marker"], color=style["color"],
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
                label_points.append((centres[peak], share[peak], label))
            ax.axvline(math.log10(threshold), color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
            pending.append((ax, label_points))
            ax.set_title(f"{REG_NAMES[reg]}: M1 · M2-{REG_NAMES[reg]} · M3 · M4-{REG_NAMES[reg]}")
            ax.set_xlabel("log₁₀ |w|, penalised weights")
        axes[0][0].set_ylabel("Share of weights (%)")
        top = max(max(share) for _, lines in panels for _, _, share in lines)
        axes[0][0].set_ylim(0, 1.3 * top)  # headroom: the labels sit above the peaks
        for ax, points in pending:
            _labels_above(ax, points)
        fig.suptitle(f"{study}{f' · round {round_name}' if round_name else ''} · {_pair_title(optimizee, dataset)} — weight magnitudes at completion "
                     f"(dashed: sparsity threshold {threshold:g})", x=0.01, ha="left", fontsize=10, color=INK)
        fig.legend(handles=[Line2D([], [], color=s["color"], marker=s["marker"], markeredgecolor=SURFACE,
                                   label=s["legend"]) for s in METHOD_STYLE.values()],
                   loc="outside lower center", ncol=4, handletextpad=0.4, columnspacing=1.4)
    return fig


# ───────────────────────────────────────────────────────────────── Stage 1

def _ema(values: Sequence[Optional[float]], span: int) -> List[float]:
    alpha, out, current = 2.0 / (span + 1.0), [], None
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(math.nan)
            continue
        current = v if current is None else alpha * v + (1 - alpha) * current
        out.append(current)
    return out


def _series(rows: Sequence[Mapping[str, Any]], key: str, x_key: str) -> Tuple[List[float], List[Optional[float]]]:
    pairs = [(to_float(r.get(x_key)), to_float(r.get(key))) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None]
    return [x for x, _ in pairs], [y for _, y in pairs]


def _meta_panel(ax, arm, key: str, x_key: str, label: str, log: bool = False,
                smooth: bool = True) -> Optional[Tuple[float, float, str]]:
    """Plot one arm's series (an EMA when long and noisy); return its last point for a direct label."""
    rows = arm["validation"] if x_key == "trajectory" else arm["steps"]
    xs, ys = _series(rows, key, x_key)
    if log:
        ys = [y if y is not None and y > 0 else None for y in ys]
    if not xs or all(y is None or not math.isfinite(y) for y in ys):
        return None
    style = METHOD_STYLE[label]
    raw = [math.nan if y is None else y for y in ys]
    if smooth and len(xs) > 10:
        shown = _ema(ys, max(5, len(xs) // 40))
        ax.plot(xs, shown, color=style["color"], linewidth=1.5, zorder=3)
    else:
        shown = raw
        ax.plot(xs, shown, color=style["color"], linewidth=1.5, marker=style["marker"], markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
    if log:
        ax.set_yscale("log")
    last = next(((x, y) for x, y in zip(reversed(xs), reversed(shown)) if math.isfinite(y)), None)
    return None if last is None else (last[0], last[1], label)


def meta_figures(arms: Sequence[Mapping[str, Any]], meta_tag: str, source: str = "") -> Dict[str, Any]:
    """{'convergence': fig, 'terms': fig} for one round of Stage-1 optimizers (missing ones omitted)."""
    where = f"round {meta_tag}" + (f", {source}" if source else "")
    m3 = next((a for a in arms if a["meta_training"] == "m3_lstm"), None)
    m4s = [a for a in arms if a["meta_training"] == "m4_lstm_reg"]
    columns = m4s or ([m3] if m3 else [])
    if not columns:
        return {}
    figures: Dict[str, Any] = {}
    rows = [("l_meta", "meta_step", "L_meta per truncation (EMA)", False),
            ("grad_norm", "meta_step", "‖∇φ‖ before clipping (EMA)", True),
            ("val_loss", "trajectory", "Validation loss", False)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(rows), len(columns), figsize=(2.6 * len(columns) + 0.8, 6.2), squeeze=False,
                                 sharex="row", layout="constrained")
        pending = []
        for c, arm in enumerate(columns):
            reg_name = REG_NAMES.get(arm["regularizer"], arm["regularizer"])
            for r, (key, x_key, ylabel, log) in enumerate(rows):
                ax = axes[r][c]
                ends = []
                if m3 is not None:
                    ends.append(_meta_panel(ax, m3, key, x_key, "M3", log, smooth=x_key != "trajectory"))
                if arm is not m3:
                    ends.append(_meta_panel(ax, arm, key, x_key, "M4", log, smooth=x_key != "trajectory"))
                ends = [e for e in ends if e is not None]
                if len(ends) > 1:
                    pending.append((ax, ends))
                if not ends:
                    ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                if r == 0:
                    ax.set_title(f"M3 vs M4-{reg_name}" if arm is not m3 else "M3")
                if c == 0:
                    ax.set_ylabel(ylabel)
                ax.set_xlabel("trajectory" if x_key == "trajectory" else "meta-step")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        _label_when_settled(pending)
        fig.suptitle(f"Stage 1 ({where}) — meta-training of the LSTM optimizers", x=0.01, ha="left",
                     fontsize=10, color=INK)
        handles = [Line2D([], [], color=METHOD_STYLE[k]["color"], label=lbl)
                   for k, lbl in (("M3", "M3 · φ (L_meta only)"), ("M4", "M4 · φ*_r (L_meta + λ·L_smooth + λ·L_reg)"))]
        fig.legend(handles=handles if m4s and m3 else handles[:1] if m3 else handles[1:],
                   loc="outside lower center", ncol=2, handletextpad=0.5, columnspacing=1.6)
    figures["convergence"] = fig

    if m4s:
        term_rows = [("l_reg", "L_reg at θ_T (EMA)", False), ("l_smooth", "L_smooth (EMA)", False)]
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(len(term_rows), len(m4s), figsize=(2.6 * len(m4s) + 0.8, 4.2), squeeze=False,
                                     sharex=True, layout="constrained")
            for c, arm in enumerate(m4s):
                for r, (key, ylabel, log) in enumerate(term_rows):
                    ax = axes[r][c]
                    if _meta_panel(ax, arm, key, "meta_step", "M4", log) is None:
                        ax.text(0.5, 0.5, "not used", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                    if r == 0:
                        ax.set_title(f"M4-{REG_NAMES.get(arm['regularizer'], arm['regularizer'])}")
                    if c == 0:
                        ax.set_ylabel(ylabel)
                    if r == len(term_rows) - 1:
                        ax.set_xlabel("meta-step")
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            fig.suptitle(f"Stage 1 ({where}) — regularization terms inside M4's meta-objective", x=0.01,
                         ha="left", fontsize=10, color=INK)
        figures["terms"] = fig
    return figures


# ───────────────────────────────────────────────────────────────── Stage 4: the result cubes

def _cube_regs(groups: Mapping[GroupKey, Group], study: str, columns: Sequence[SettingPair]) -> List[str]:
    return [r for r in REGULARIZERS
            if any((study, c.pair.optimizee, c.pair.dataset, m, r) in groups
                   for c in columns for m in ("m2_adamw_reg", "m4_lstm_reg"))]


def cube_figure(groups: Mapping[GroupKey, Group], study: str, metric: Metric, columns: Sequence[SettingPair],
                rnd: Optional[Round] = None):
    """One round's metric as small multiples: a panel per evaluation setting (and pair), the
    regularizers along x, M2(r) and M4(r) as points with seed error bars side by side, and M1 and
    M3 — which do not use r — as horizontal reference lines with a ± std band."""
    regs = _cube_regs(groups, study, columns)
    if not regs or not columns:
        return None
    handles = [
        Line2D([], [], color=METHOD_STYLE["M1"]["color"], linestyle=(0, (4, 2)), label="M1 · AdamW"),
        Line2D([], [], color=METHOD_STYLE["M2"]["color"], marker="s", linestyle="none", markeredgecolor=SURFACE,
               label="M2 · AdamW + r"),
        Line2D([], [], color=METHOD_STYLE["M3"]["color"], label="M3 · LSTM (no r)"),
        Line2D([], [], color=METHOD_STYLE["M4"]["color"], marker="D", linestyle="none", markeredgecolor=SURFACE,
               label="M4 · LSTM-r (ours)"),
    ]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(columns), figsize=(max(2.35 * len(columns) + 0.9, 4.6), 3.2), squeeze=False,
                                 layout="constrained")
        for ax, c in zip(axes[0], columns):
            key = lambda m, r: (study, c.pair.optimizee, c.pair.dataset, m, r)  # noqa: E731
            right = len(regs) - 0.4
            references, extent = [], []
            for label, method in (("M1", "m1_adamw"), ("M3", "m3_lstm")):
                stat = (groups.get(key(method, "none")) or Group()).get(metric.record)
                if stat.mean is None:
                    continue
                y, spread = stat.mean * metric.scale, (stat.std or 0.0) * metric.scale
                color = METHOD_STYLE[label]["color"]
                ax.axhline(y, color=color, linewidth=1.5, linestyle=(0, (4, 2)) if label == "M1" else "-", zorder=2)
                if spread:
                    ax.axhspan(y - spread, y + spread, color=color, alpha=0.12, linewidth=0, zorder=1)
                references.append((right, y, label))
                extent += [y - spread, y + spread]
            firsts = []
            for label, method, offset in (("M2", "m2_adamw_reg", -0.16), ("M4", "m4_lstm_reg", 0.16)):
                style = METHOD_STYLE[label]
                for i, reg in enumerate(regs):
                    stat = (groups.get(key(method, reg)) or Group()).get(metric.record)
                    if stat.mean is None:
                        continue
                    y, spread = stat.mean * metric.scale, (stat.std or 0.0) * metric.scale
                    ax.errorbar(i + offset, y, yerr=spread, fmt=style["marker"], color=style["color"],
                                ecolor=style["color"], elinewidth=1.0, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
                    extent += [y - spread, y + spread]
                    if not any(f[2] == label for f in firsts):
                        firsts.append((i + offset, y, label))
            ax.set_xticks(range(len(regs)), [REG_NAMES[r] for r in regs])
            ax.set_xlim(-0.6, right)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{c.setting.label}\n{c.pair.label}{' ‡' if c.role == 'unseen_both' else ''}"
                         f"{' †' if c.extended else ''}", fontsize=9)
            if extent:  # final limits before any label is placed
                lo, hi = min(extent), max(extent)
                span = (hi - lo) or (abs(hi) * 0.1) or 1.0
                ax.set_ylim(lo - 0.18 * span, hi + 0.18 * span)
            for x, y, text in firsts:  # the methods that do use r, named once beside their first point
                ax.annotate(text, xy=(x, y), xytext=(-6 if text == "M2" else 6, 0), textcoords="offset points",
                            ha="right" if text == "M2" else "left", va="center", fontsize=7.5, color=INK_2)
            _direct_labels(ax, references, gap=0.08, dx=2.0)
        axes[0][0].set_ylabel(f"{metric.label} ({metric.unit})")
        where = f" · round {rnd.name} ({Pair(rnd.optimizee, rnd.dataset).label})" if rnd else ""
        fig.suptitle(f"{study}{where} · {metric.label} {metric.arrow} — method × regularizer × evaluation setting",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.legend(handles=handles, loc="outside lower center", ncol=4, handletextpad=0.5, columnspacing=1.6)
    return fig


# ───────────────────────────────────────────────────────────────── Stages 2–3: training curves

def curves_figure(groups: Mapping[GroupKey, Group], study: str, pair: Pair, title: Optional[str] = None):
    """How each optimizer trains this optimizee: training loss and test accuracy per epoch (seed
    means) of M1 · M2(r) · M3 · M4(r), one column per regularizer — Stage 2's question of whether
    the frozen LSTM still optimizes well on an unseen architecture, and Stage 3's on unseen data."""
    regs = _regs_with_runs(groups, study, pair.optimizee, pair.dataset)
    panels = regs or (["none"] if any((study, pair.optimizee, pair.dataset, m, "none") in groups
                                      for m in ("m1_adamw", "m3_lstm")) else [])
    if not panels:
        return None
    rows = [("train_loss_running", "Training loss (epoch mean)", 1.0, True), ("test_top1", "Test accuracy (%)", 100.0, False)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(rows), len(panels), figsize=(2.5 * len(panels) + 0.9, 5.0), squeeze=False,
                                 sharex=True, sharey="row", layout="constrained")
        pending, drawn = [], False
        for c, reg in enumerate(panels):
            for r, (key, ylabel, scale, log) in enumerate(rows):
                ax = axes[r][c]
                ends = []
                for _, method, regularizer in method_rows(reg):
                    group = groups.get((study, pair.optimizee, pair.dataset, method, regularizer))
                    curve = epoch_curves(group.runs, key) if group else None
                    if curve is None:
                        continue
                    xs, ys = curve[0], [y * scale for y in curve[1]]
                    label = LABELS[method]
                    style = METHOD_STYLE[label]
                    ax.plot(xs, ys, color=style["color"], linewidth=1.5, zorder=3,
                            linestyle=(0, (4, 2)) if label == "M1" else "-",
                            marker=style["marker"] if len(xs) <= 12 else None, markersize=4,
                            markeredgecolor=SURFACE, markeredgewidth=0.6)
                    ends.append((xs[-1], ys[-1], label))
                    drawn = True
                if log and ends and all(e[1] > 0 for e in ends):
                    lows = [y for line in ax.get_lines() for y in line.get_ydata() if y > 0]
                    if lows and max(lows) / min(lows) > 20:  # log only when the loss spans orders of magnitude
                        ax.set_yscale("log")
                if ends:
                    pending.append((ax, ends))
                if r == 0:
                    ax.set_title("M1 vs M3" if reg == "none" else f"{REG_NAMES[reg]}: M1 · M2 · M3 · M4")
                if c == 0:
                    ax.set_ylabel(ylabel)
                if r == len(rows) - 1:
                    ax.set_xlabel("epoch")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if not drawn:
            plt.close(fig)
            return None
        _label_when_settled(pending)
        fig.suptitle(f"{title or study} · {pair.label} — training curves (seed means)", x=0.01, ha="left", fontsize=10,
                     color=INK)
        fig.legend(handles=[Line2D([], [], color=s["color"], linestyle=(0, (4, 2)) if k == "M1" else "-", label=s["legend"])
                            for k, s in METHOD_STYLE.items()],
                   loc="outside lower center", ncol=4, handletextpad=0.5, columnspacing=1.6)
    return fig

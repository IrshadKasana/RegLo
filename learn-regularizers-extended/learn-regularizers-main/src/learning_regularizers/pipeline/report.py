"""Reported results: one 3D table per metric (and per base optimizer B).

    axis 0 (Z)  regularizer      L1, L2, SAM, GAM, GSAM
    axis 1 (Y)  method           1 .. 6
    axis 2 (X)  evaluation       seen optimizee | unseen optimizee | seen dataset | unseen dataset

Cell definitions
    seen optimizee    phase 2, optimizee == optimizer_evaluation.seen_optimizee
    unseen optimizee  phase 2, mean over the other optimizees (MLP, ResNet-18)
    seen dataset      phase 3, dataset == seen dataset (mean over phase-3 optimizees)
    unseen dataset    phase 3, mean over the unseen datasets that were available
Metrics
    test_acc  (%)  higher is better
    gen_gap        lower is better (train_acc - test_acc by default; see metrics.gap)
    flops (GFLOPs) lower is better; phase 2: optimizee-training FLOPs,
                   phase 3: FLOPs to train the evaluated optimizee + evaluation/probe FLOPs
Methods without R (1, 3, 4) repeat the same value in every R slice; methods 3-6 do not depend
on B and repeat in both B tables.

Outputs (artifacts/pipeline/<experiment>/tables/):
    results_long.csv              every phase-2/3 record (mean, std per metric)
    <metric>__B-<b>.npy           the 3D array [R, M, S] (NaN where missing)
    <metric>__B-<b>.csv           the same, flattened (one row per (R, method))
    <metric>__B-<b>.png           one heat-map per R slice (method x setting)
    breakdown.md                  per-optimizee / per-dataset values behind the aggregated cells
    summary.md                    all 3D tables as markdown slices
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .artifacts import read_json, subdir, write_json
from .methods import METHODS, cell_run_key

SETTINGS = ("seen_optimizee", "unseen_optimizee", "seen_dataset", "unseen_dataset")
SETTING_LABELS = ("Seen optimizee", "Unseen optimizee", "Seen dataset", "Unseen dataset")
TABLE_METRICS = {
    "test_acc": ("Test Accuracy (%)", "higher is better", 1.0),
    "gen_gap": ("Generalization Gap", "lower is better", 1.0),
    "flops": ("FLOPs (GFLOPs)", "lower is better", 1e-9),
}


def load_records(cfg) -> List[Dict]:
    records = []
    for phase in ("phase2", "phase3"):
        for path in sorted(subdir(cfg, phase).glob("*.json")):
            rec = read_json(path)
            if rec.get("status", "ok") == "ok":
                records.append(rec)
    return records


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def cell_value(index, cfg, run_id: str, setting: str, metric: str) -> float:
    oe, ev = cfg.optimizer_evaluation, cfg.optimizee_evaluation
    recs = index.get((run_id, setting), [])
    if setting == "seen_optimizee":
        recs = [r for r in recs if r["optimizee"] == oe.seen_optimizee]
    elif setting == "unseen_optimizee":
        recs = [r for r in recs if r["optimizee"] != oe.seen_optimizee and r["optimizee"] in oe.optimizees]
    elif setting == "seen_dataset":
        recs = [r for r in recs if r["dataset"] == ev.seen_dataset and r["optimizee"] in ev.optimizees]
    else:
        recs = [r for r in recs if r["dataset"] in ev.unseen_datasets and r["optimizee"] in ev.optimizees]
    return _mean([r["mean"].get(metric) for r in recs])


def build_tables(cfg, records) -> Dict:
    index = defaultdict(list)
    for r in records:
        index[(r["run"]["run_id"], r["setting"])].append(r)
    regs, methods = list(cfg.regularizers), list(cfg.methods)
    tables = {}
    for metric, (_, _, scale) in TABLE_METRICS.items():
        for b in cfg.base_optimizers:
            arr = np.full((len(regs), len(methods), len(SETTINGS)), np.nan)
            for i, r in enumerate(regs):
                for j, m in enumerate(methods):
                    run_id = cell_run_key(m, b, r).run_id
                    for k, s in enumerate(SETTINGS):
                        arr[i, j, k] = cell_value(index, cfg, run_id, s, metric) * scale
            tables[(metric, b)] = arr
    return tables


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if abs(v) >= 1000:
        return f"{v:.3g}"
    return f"{v:.2f}"


def _write_long_csv(path, records):
    metrics = ("test_acc", "gen_gap", "flops", "train_acc", "train_loss", "test_loss")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "run_id", "method", "base_optimizer", "regularizer", "setting", "optimizee", "dataset",
                    "n_seeds", "n_diverged"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")])
        for r in records:
            run = r["run"]
            row = [r["phase"], run["run_id"], run["method"], run["base_optimizer"] or "", run["regularizer"] or "",
                   r["setting"], r["optimizee"], r["dataset"], r.get("n_seeds"), r.get("n_diverged")]
            for m in metrics:
                row += [r["mean"].get(m), r["std"].get(m)]
            w.writerow(row)


def _write_table_csv(path, arr, regs, methods):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regularizer", "method", "method_label"] + list(SETTINGS))
        for i, r in enumerate(regs):
            for j, m in enumerate(methods):
                w.writerow([r, m, METHODS[m].label] + [None if np.isnan(v) else float(v) for v in arr[i, j]])


def _plot(path, arr, regs, methods, title, better) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    n = len(regs)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 1, 3.6), squeeze=False)
    finite = arr[np.isfinite(arr)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    cmap = "viridis" if better.startswith("higher") else "viridis_r"
    for i, r in enumerate(regs):
        ax = axes[0, i]
        ax.imshow(np.ma.masked_invalid(arr[i]), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"R = {r.upper()}", fontsize=10)
        ax.set_xticks(range(len(SETTINGS)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in SETTING_LABELS], fontsize=7)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([f"M{m}" for m in methods] if i == 0 else [], fontsize=8)
        for j in range(len(methods)):
            for k in range(len(SETTINGS)):
                v = arr[i, j, k]
                frac = (v - vmin) / (vmax - vmin) if np.isfinite(v) and vmax > vmin else 0.0
                bright = frac > 0.6 if cmap == "viridis" else frac < 0.4
                ax.text(k, j, _fmt(v), ha="center", va="center", fontsize=6.5,
                        color="black" if bright else "white")
    fig.suptitle(f"{title} ({better}) — Regularizer (Z) x Method (Y) x Evaluation setting (X)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _breakdown(cfg, records) -> str:
    lines = ["# Per-optimizee / per-dataset breakdown", "",
             "Mean over seeds (± std). Phase 2 rows: optimizer evaluation; phase 3 rows: optimizee evaluation.", ""]
    for metric, (label, better, scale) in TABLE_METRICS.items():
        lines += [f"## {label} — {better}", ""]
        cols = sorted({(r["phase"], r["optimizee"], r["dataset"]) for r in records})
        header = "| run | " + " | ".join(f"P{p}: {o} / {d}" for p, o, d in cols) + " |"
        lines += [header, "|" + "---|" * (len(cols) + 1)]
        by_run = defaultdict(dict)
        for r in records:
            by_run[r["run"]["run_id"]][(r["phase"], r["optimizee"], r["dataset"])] = r
        for run_id in sorted(by_run):
            cells = []
            for c in cols:
                rec = by_run[run_id].get(c)
                if rec is None or rec["mean"].get(metric) is None:
                    cells.append("–")
                else:
                    mu, sd = rec["mean"][metric] * scale, (rec["std"].get(metric) or 0.0) * scale
                    cells.append(f"{_fmt(mu)} ± {_fmt(sd)}")
            lines.append(f"| {run_id} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _phase1_section(cfg) -> List[str]:
    """Meta-training cost is not part of the evaluation tables; report it separately."""
    rows = []
    for path in sorted(subdir(cfg, "phase1").glob("*.json")):
        r = read_json(path)
        if r.get("type") == "meta_training":
            rows.append(f"| {r['run']['run_id']} | meta-train LO | {_fmt((r.get('meta_train_flops') or 0) * 1e-9)} | "
                        f"{_fmt(r.get('best_eval_loss'))} | {_fmt(r.get('wall_time_s'))} |")
        else:
            m = r.get("mean", {})
            rows.append(f"| {r['run']['run_id']} | train {r['optimizee']} on {r['dataset']} | "
                        f"{_fmt((m.get('flops') or 0) * 1e-9)} | test acc {_fmt(m.get('test_acc'))} | – |")
    if not rows:
        return []
    return ["## Phase 1 (meta-training) cost", "",
            "| run | what | GFLOPs | best LO eval loss / result | wall time (s) |", "|---|---|---|---|---|",
            *rows, ""]


def build_report(cfg, log=print) -> Dict:
    out = subdir(cfg, "tables")
    records = load_records(cfg)
    if not records:
        log("[report] no phase-2/3 records found; run phases 2 and 3 first")
        return {}
    regs, methods = list(cfg.regularizers), list(cfg.methods)
    tables = build_tables(cfg, records)
    _write_long_csv(out / "results_long.csv", records)

    md = ["# Reported results — one 3D table per metric", "",
          "Axes: **Regularizer (Z)** x **Method (Y)** x **Evaluation setting (X)**. "
          "B in {Adam, SGD} is evaluated separately (one table per B).", "",
          "Methods: " + "; ".join(f"M{m} = {METHODS[m].label}" for m in methods), "",
          "Notes: methods without R (1, 3, 4) repeat across R slices; methods 3-6 are independent of B. "
          "Phase 2/3 run all learned optimizers without explicit regularizers"
          + (" and baselines without R as well (so M2 == M1 in the optimizer-evaluation columns)."
             if cfg.optimizer_evaluation.strip_regularizers_for_baselines else "; baselines keep B + λR.")
          + " Unseen-dataset cells use a frozen backbone + linear probe.", ""]
    for metric, (label, better, _) in TABLE_METRICS.items():
        md += [f"## {label} — {better}", ""]
        for b in cfg.base_optimizers:
            arr = tables[(metric, b)]
            stem = f"{metric}__B-{b}"
            np.save(out / f"{stem}.npy", arr)
            _write_table_csv(out / f"{stem}.csv", arr, regs, methods)
            plotted = _plot(out / f"{stem}.png", arr, regs, methods, f"{label}, B = {b}", better)
            md += [f"### B = {b}" + (f"  (figure: `{stem}.png`)" if plotted else ""), ""]
            for i, r in enumerate(regs):
                md += [f"**R = {r.upper()}**", "", "| Method | " + " | ".join(SETTING_LABELS) + " |",
                       "|---|" + "---|" * len(SETTINGS)]
                for j, m in enumerate(methods):
                    md.append(f"| M{m} {METHODS[m].key} | " + " | ".join(_fmt(v) for v in arr[i, j]) + " |")
                md.append("")
    md += _phase1_section(cfg)
    (out / "summary.md").write_text("\n".join(md))
    (out / "breakdown.md").write_text(_breakdown(cfg, records))
    write_json(out / "tables_index.json", {
        "axes": {"regularizer": regs, "method": methods, "setting": list(SETTINGS)},
        "metrics": {k: {"label": v[0], "direction": v[1]} for k, v in TABLE_METRICS.items()},
        "files": sorted(p.name for p in out.iterdir())})
    log(f"[report] wrote 3D tables to {out}")
    return tables

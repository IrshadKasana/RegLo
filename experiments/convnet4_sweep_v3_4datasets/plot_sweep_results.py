#!/usr/bin/env python
"""
plot_sweep_results.py -- PNG figures for the five-method ConvNet-4 sweep over four datasets
(70:20:10 split, ID test + cross-dataset OOD between every ordered pair).

Reads  results/<preset>/best_config.json  and  results/<preset>/all_cells.csv
Writes results/<preset>/plots/*.png

    python plot_sweep_results.py [--preset full] [--results-dir results] [--dpi 200]

    01_summary_<source>_to_<target>.png   12 tables (one per ordered pair): selected config in separate columns, val / train /
                                          ID test acc, ID AUROC, OOD acc, OOD AUROC (± = std. error), cells tried
    02_id_vs_ood.png                      ID test accuracy, ID AUROC and OOD AUROC (every target) of all methods, per source
    03_m2_<source>.png                    M2: validation accuracy heat maps lr x lambda (default rho) and rho x lambda (best lr)
    04_m3_m4a.png                         M3 meta_lr curves and M4a meta_lr x lambda_smooth heat maps, per source
    05_m5_<source>.png                    M5 lambda_smooth x lambda_reg heat maps per regulariser (+ selected cell)
    06_m5_meta_rho.png                    M5 meta_lr and rho sweeps at the best cell, per source
    07_training_curves.png                train loss / validation accuracy vs optimizee step, selected configs
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

REGS = ("l1", "l2", "sam", "gsam", "gam")
RHO_REGS = ("sam", "gsam", "gam")
RHO_DEFAULT = {"sam": 0.001, "gsam": 0.005, "gam": 0.04}
RL = {"l1": "L1", "l2": "L2", "sam": "SAM", "gsam": "GSAM", "gam": "GAM"}
NAME = {"svhn": "SVHN", "cifar10": "CIFAR-10", "mnist": "MNIST", "fmnist": "FashionMNIST"}
COL = {"l1": "#2a78d6", "l2": "#eb6834", "sam": "#1baf7a", "gsam": "#eda100", "gam": "#e87ba4"}
MCOL = {"M1": "#52514e", "M2": "#2a78d6", "M3": "#898781", "M4a": "#6250d6", "M5": "#eb6834"}   # method colours
SURFACE, INK, INK2, MUTED, GRID_C, AXIS_C = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
LAB = {"M1": "M1 AdamW", "M2": "M2 AdamW+R", "M3": "M3 LO", "M4a": "M4a LO+smooth", "M5": "M5 LO+smooth+R"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS_C, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def g(v):
    return f"{v:g}"


def pct(x):
    return 100.0 * x


def isnum(m):
    return m is not None and not (isinstance(m, float) and math.isnan(m))


def fmt(ms, scale=100.0, nd=2):
    m, s = ms
    if not isnum(m):
        return "-"
    return f"{scale * m:.{nd}f}" + (f" ± {scale * s:.{nd}f}" if isnum(s) else "")


def save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def read_cells(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            if k in ("method", "source", "reg", "seed"):
                continue
            r[k] = float(v) if v not in ("", None) else float("nan")
        rows.append(r)
    return rows


def targets_of(B, d):
    return list(next(iter(B["results"][d].values()))["ood"].keys())


def same(a, b):
    return isinstance(a, float) and isinstance(b, float) and not math.isnan(a) and not math.isnan(b) and abs(a - b) < 1e-12


def cfg_cols(key, e):
    """lr, meta_lr, lambda_smooth, lambda_reg / lambda (M2), rho  as strings ('-' when not applicable)."""
    c, m = e["config"], key.split(":")[0]
    f = lambda v: "-" if v in (None, "") else g(v)
    if m == "M1":
        return [f(c["adamw_lr"]), "-", "-", "-", "-"]
    if m == "M2":
        return [f(c["adamw_lr"]), "-", "-", f(c["lam"]), f(c.get("rho"))]
    if m == "M3":
        return ["-", f(c["meta_lr"]), "-", "-", "-"]
    if m == "M4a":
        return ["-", f(c["meta_lr"]), f(c["ls"]), "-", "-"]
    return ["-", f(c["meta_lr"]), f(c["ls"]), f(c["lam_reg"]), f(c.get("rho"))]


# ------------------------------------------------------------------ 01 tables
def fig_tables(B, out, dpi):
    head = ["method", "R", "lr", "meta_lr", "λ_smooth", "λ_reg / λ", "ρ", "val acc %", "train acc %", "ID test acc %",
            "ID AUROC", "OOD acc %", "OOD AUROC", "cells tried"]
    for d, R in B["results"].items():
        for t in targets_of(B, d):
            rows, vals = [], []
            for key, e in R.items():
                m, _, reg = key.partition(":")
                o = e["ood"][t]
                rows.append([LAB[m], RL.get(reg, "-")] + cfg_cols(key, e) +
                            [f"{pct(e['val']):.2f}", fmt(e["train_acc"]), fmt(e["test_acc"]), fmt(e["id_auroc"], 1, 3),
                             fmt(o["acc"]), fmt(o["auroc_msp"], 1, 3), str(e.get("n_cells", "-"))])
                vals.append(e["val"])
            best = int(np.argmax(vals))
            fig, ax = plt.subplots(figsize=(24, 0.37 * (len(rows) + 1) + 0.9))
            ax.axis("off")
            tb = ax.table(cellText=rows, colLabels=head, bbox=[0, 0, 1, 1], cellLoc="center",
                          colWidths=[0.095, 0.04, 0.05, 0.055, 0.06, 0.07, 0.05, 0.06, 0.09, 0.095, 0.085, 0.09, 0.09, 0.07])
            tb.auto_set_font_size(False)
            tb.set_fontsize(9)
            for (i, j), cell in tb.get_celld().items():
                cell.set_edgecolor(GRID_C)
                cell.set_linewidth(0.8)
                if i == 0:
                    cell.set_facecolor("#f0efec")
                    cell.set_text_props(weight="bold", color=INK)
                else:
                    cell.set_facecolor("#e8f1fc" if i - 1 == best else SURFACE)
                    cell.set_text_props(color=INK, weight="bold" if i - 1 == best else "normal")
            ax.set_title(f"{NAME[d]} → {NAME[t]}:  trained on {NAME[d]}, ID = {NAME[d]} test split, OOD = {NAME[t]}", loc="left", pad=22,
                         fontsize=13)
            fig.text(0.005, -0.012,
                     "Selection on the source validation split (seed 0) only; highlighted row = highest validation accuracy of the table. "
                     "ID AUROC = macro one-vs-rest AUROC (ID test). OOD AUROC = ID test vs OOD by max-softmax confidence (higher = better). "
                     "OOD acc ≈ chance (unrelated labels). ± = std. error (inits: M1/M2, meta-seeds: M3/M4a/M5). "
                     "'cells tried' = grid cells searched for that pick. val = seed-0 selection value.",
                     fontsize=8, color=MUTED, wrap=True)
            save(fig, out / f"01_summary_{d}_to_{t}.png", dpi)


# ------------------------------------------------------------------ 02 ID vs OOD
def bar_items(R):
    keys = ["M1"] + [f"M2:{r}" for r in REGS] + ["M3", "M4a"] + [f"M5:{r}" for r in REGS]
    return [k for k in keys if k in R]


def fig_id_ood(B, out, dpi):
    ds = list(B["results"])
    tmax = max(len(targets_of(B, d)) for d in ds)
    fig, axes = plt.subplots(len(ds), 2 + tmax, figsize=(5.2 * (2 + tmax), 3.9 * len(ds)), squeeze=False)
    for row, d in enumerate(ds):
        R = B["results"][d]
        keys = bar_items(R)
        panels = [("ID test accuracy (%)", lambda e: e["test_acc"], 100.0), ("ID AUROC", lambda e: e["id_auroc"], 1.0)]
        for t in targets_of(B, d):
            panels.append((f"OOD AUROC → {NAME[t]}", (lambda tt: lambda e: e["ood"][tt]["auroc_msp"])(t), 1.0))
        for col in range(2 + tmax):
            ax = axes[row, col]
            if col >= len(panels):
                ax.axis("off")
                continue
            title, getter, sc = panels[col]
            v = np.array([getter(R[k])[0] * sc for k in keys])
            s = np.array([0 if not isnum(getter(R[k])[1]) else getter(R[k])[1] * sc for k in keys])
            cols = [COL.get(k.partition(":")[2], MCOL[k.split(":")[0]]) for k in keys]
            for xi, (k, vi, si, ci) in enumerate(zip(keys, v, s, cols)):
                ax.bar(xi, vi, color=ci, width=0.68, hatch="//" if k.split(":")[0] in ("M3", "M4a", "M5") else None,
                       edgecolor=SURFACE if k.split(":")[0] in ("M3", "M4a", "M5") else ci, linewidth=0)
                ax.errorbar(xi, vi, yerr=si, color=INK2, capsize=2, lw=1)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([k.replace(":", "\n") for k in keys], fontsize=6.5)
            lo = float(np.nanmin(v - s))
            ax.set_ylim(max(0, lo - (4 if sc == 100 else 0.04)), float(np.nanmax(v + s)) + (1.5 if sc == 100 else 0.02))
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{NAME[d]} — {title}", fontsize=10)
    handles = [Patch(color=COL[r], label=RL[r]) for r in REGS] + [Patch(color=MCOL["M4a"], label="M4a (no R)"),
                                                                 Patch(facecolor="#898781", hatch="//", edgecolor=SURFACE,
                                                                       label="hatched = learned optimizer")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out / "02_id_vs_ood.png", dpi)


# ------------------------------------------------------------------ heat-map helper
def heat(ax, cells, xkey, ykey, xs, ys, mark=None, title="", vmin=None, vmax=None):
    Z = np.full((len(ys), len(xs)), np.nan)
    for c in cells:
        for xi, xv in enumerate(xs):
            if same(c[xkey], xv):
                for yi, yv in enumerate(ys):
                    if same(c[ykey], yv):
                        Z[yi, xi] = pct(c["val_acc"])
    ax.grid(False)
    im = ax.imshow(Z, origin="lower", cmap=SEQ, vmin=vmin if vmin is not None else np.nanmin(Z),
                   vmax=vmax if vmax is not None else np.nanmax(Z), aspect="auto")
    ax.set_facecolor("#f0efec")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([g(v) for v in xs], fontsize=6, rotation=70)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([g(v) for v in ys], fontsize=6)
    if mark and mark[0] in xs and mark[1] in ys:
        ax.plot(xs.index(mark[0]), ys.index(mark[1]), marker="s", ms=11, mfc="none", mec="#eb6834", mew=2)
    ax.set_title(title, fontsize=10)
    return im


def uniq(cells, key):
    return sorted({c[key] for c in cells if not math.isnan(c[key])})


def fig_m2(B, cells, out, dpi):
    for d in B["results"]:
        c2 = [c for c in cells if c["source"] == d and c["method"] == "M2"]
        if not c2:
            continue
        fig, axes = plt.subplots(2, 5, figsize=(24, 9.4), squeeze=False)
        allv = [pct(c["val_acc"]) for c in c2 if c["val_acc"] > 0.5 * max(x["val_acc"] for x in c2)]
        vmax = max(pct(c["val_acc"]) for c in c2)
        vmin = min(allv)
        for ci, r in enumerate(REGS):
            cr = [c for c in c2 if c["reg"] == r and (r not in RHO_REGS or same(c["rho"], RHO_DEFAULT[r]))]
            sel = B["results"][d].get(f"M2:{r}")
            mark = (sel["config"]["lam"], sel["config"]["adamw_lr"]) if sel else None
            im = heat(axes[0, ci], cr, "lam", "adamw_lr", uniq(cr, "lam"), uniq(cr, "adamw_lr"), mark,
                      f"{RL[r]}: lr × λ" + (f" (ρ={RHO_DEFAULT[r]:g})" if r in RHO_REGS else ""), vmin, vmax)
            axes[0, ci].set_xlabel("λ")
            axes[0, ci].set_ylabel("AdamW lr" if ci == 0 else "")
            if r in RHO_REGS and sel:
                lr_b = sel["config"]["adamw_lr"]
                cs = [c for c in c2 if c["reg"] == r and same(c["adamw_lr"], lr_b)]
                if len({c["rho"] for c in cs}) > 1:
                    heat(axes[1, ci], cs, "lam", "rho", uniq(cs, "lam"), uniq(cs, "rho"), (sel["config"]["lam"], sel["config"]["rho"]),
                         f"{RL[r]}: ρ × λ at lr={g(lr_b)}", vmin, vmax)
                    axes[1, ci].set_xlabel("λ")
                    axes[1, ci].set_ylabel("ρ" if ci == 2 else "")
                    continue
            axes[1, ci].axis("off")
        fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01).set_label("validation acc (%)")
        fig.suptitle(f"M2 AdamW + λ·R — {NAME[d]}: validation accuracy over the full lr × λ grid (top) and ρ × λ at the best lr (bottom); "
                     "square = selected; grey = not run", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.0)
        save(fig, out / f"03_m2_{d}.png", dpi)


# ------------------------------------------------------------------ 04 M3 / M4a
def fig_m3_m4a(B, cells, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(2, len(ds), figsize=(6.2 * len(ds), 9), squeeze=False)
    for i, d in enumerate(ds):
        c = sorted([x for x in cells if x["method"] == "M3" and x["source"] == d and x["seed"] == "0"], key=lambda x: x["meta_lr"])
        ax = axes[0, i]
        if c:
            xs = [x["meta_lr"] for x in c]
            ax.plot(xs, [pct(x["val_acc"]) for x in c], color=MCOL["M3"], marker="o", label="validation (selection)")
            ax.plot(xs, [pct(x["test_acc"]) for x in c], color=MCOL["M5"], marker="s", ls="--", lw=1.4, label="ID test (not used)")
            sel = B["results"][d].get("M3")
            if sel:
                ax.plot([sel["config"]["meta_lr"]], [pct(sel["val"])], marker="o", ms=12, mfc="none", mec=INK, mew=2)
            lo = min(pct(x["val_acc"]) for x in c)
            ax.set_ylim(max(0, np.percentile([pct(x["val_acc"]) for x in c], 25) - 5), max(pct(x["val_acc"]) for x in c) + 1)
        ax.set_xscale("log")
        ax.set_xlabel("meta_lr (Adam rate of the learned optimizer's weights)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"M3 — {NAME[d]}")
        ax.legend(fontsize=8)
        c4 = [x for x in cells if x["method"] == "M4a" and x["source"] == d and x["seed"] == "0"]
        ax = axes[1, i]
        if c4:
            sel = B["results"][d].get("M4a")
            im = heat(ax, c4, "ls", "meta_lr", uniq(c4, "ls"), uniq(c4, "meta_lr"),
                      (sel["config"]["ls"], sel["config"]["meta_lr"]) if sel else None, f"M4a — {NAME[d]}: meta_lr × λ_smooth")
            ax.set_xlabel("λ_smooth")
            ax.set_ylabel("meta_lr")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("val acc (%)")
    fig.tight_layout()
    save(fig, out / "04_m3_m4a.png", dpi)


# ------------------------------------------------------------------ 05 M5 heat maps
def fig_m5(B, cells, out, dpi):
    for d in B["results"]:
        c5 = [c for c in cells if c["source"] == d and c["method"] == "M5" and c["seed"] == "0"]
        if not c5:
            continue
        mlr4 = (B["results"][d].get("M4a") or {}).get("config", {}).get("meta_lr")
        fig, axes = plt.subplots(1, len(REGS), figsize=(5.2 * len(REGS), 5.2), squeeze=False)
        base = [c for c in c5 if mlr4 is None or same(c["meta_lr"], mlr4)]
        vmax = max(pct(c["val_acc"]) for c in base)
        vmin = np.percentile([pct(c["val_acc"]) for c in base], 5)
        im = None
        for ax, r in zip(axes[0], REGS):
            cr = [c for c in base if c["reg"] == r and (r not in RHO_REGS or same(c["rho"], RHO_DEFAULT[r]))]
            if not cr:
                ax.axis("off")
                continue
            sel = B["results"][d].get(f"M5:{r}")
            im = heat(ax, cr, "ls", "lam_reg", uniq(cr, "ls"), uniq(cr, "lam_reg"),
                      (sel["config"]["ls"], sel["config"]["lam_reg"]) if sel else None, f"{RL[r]} — {len(cr)} cells", vmin, vmax)
            ax.set_xlabel("λ_smooth")
            ax.set_ylabel("λ_reg" if r == REGS[0] else "")
        if im is not None:
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01).set_label("validation acc (%)")
        fig.suptitle(f"M5 — {NAME[d]}: validation accuracy over λ_smooth × λ_reg at M4a's meta_lr"
                     f"{' = ' + g(mlr4) if mlr4 else ''} (default ρ); square = selected cell; grey = not run",
                     x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
        save(fig, out / f"05_m5_{d}.png", dpi)


# ------------------------------------------------------------------ 06 M5 meta_lr / rho sweeps
def best_group(cells, reg, vary):
    """The (ls, lr[, mlr]) group of M5 cells of `reg` that holds the most distinct values of `vary`."""
    groups = {}
    for c in cells:
        if c["reg"] != reg or c["seed"] != "0":
            continue
        key = (c["ls"], c["lam_reg"]) + ((c["meta_lr"],) if vary == "rho" else ())
        groups.setdefault(key, []).append(c)
    if not groups:
        return []
    return max(groups.values(), key=lambda v: len({c[vary] for c in v if not math.isnan(c[vary])}))


def fig_m5_sweeps(B, cells, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(len(ds), 2, figsize=(13, 4.0 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        c5 = [c for c in cells if c["source"] == d and c["method"] == "M5"]
        for r in REGS:
            grp = sorted(best_group(c5, r, "meta_lr"), key=lambda c: c["meta_lr"])
            if len(grp) > 2:
                axes[i, 0].plot([c["meta_lr"] for c in grp], [pct(c["val_acc"]) for c in grp], color=COL[r], marker="o", label=RL[r])
            if r in RHO_REGS:
                grp = sorted([c for c in best_group(c5, r, "rho") if not math.isnan(c["rho"])], key=lambda c: c["rho"])
                if len(grp) > 2:
                    axes[i, 1].plot([c["rho"] for c in grp], [pct(c["val_acc"]) for c in grp], color=COL[r], marker="o", label=RL[r])
        axes[i, 0].set_xscale("log")
        axes[i, 1].set_xscale("log")
        axes[i, 0].set_title(f"M5 meta_lr sweep at the best (λ_smooth, λ_reg) — {NAME[d]}")
        axes[i, 1].set_title(f"M5 ρ sweep at the best (λ_smooth, λ_reg, meta_lr) — {NAME[d]}")
        axes[i, 0].set_xlabel("meta_lr")
        axes[i, 1].set_xlabel("ρ")
        for a in axes[i]:
            a.set_ylabel("validation accuracy (%)")
        axes[i, 0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    save(fig, out / "06_m5_meta_rho.png", dpi)


# ------------------------------------------------------------------ 07 curves
def fig_curves(B, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(len(ds), 3, figsize=(19, 3.9 * len(ds)), squeeze=False)
    for row, d in enumerate(ds):
        R = B["results"][d]
        picks = {}
        for m in ("M1", "M2", "M3", "M4a", "M5"):
            c = [(k, e) for k, e in R.items() if k.split(":")[0] == m and e.get("curve")]
            if c:
                picks[m] = max(c, key=lambda ke: ke[1]["val"])
        for m, (k, e) in picks.items():
            lab = m if ":" not in k else f"{m} ({RL[k.split(':')[1]]})"
            cv = e["curve"]
            st = [p["step"] for p in cv]
            axes[row, 0].plot(st, [p["train_subset_loss"] for p in cv], color=MCOL[m], label=lab)
            axes[row, 1].plot(st, [pct(p["train_subset_acc"]) for p in cv], color=MCOL[m], label=lab)
            axes[row, 2].plot(st, [pct(p["val_acc"]) for p in cv], color=MCOL[m], label=lab)
        for col, t in enumerate(("training loss (train subset)", "training accuracy (train subset)", "validation accuracy")):
            axes[row, col].set_title(f"{NAME[d]} — {t}")
            axes[row, col].set_xlabel("optimizee step (step 0 = before training)")
        axes[row, 0].set_ylabel("loss")
        axes[row, 1].set_ylabel("accuracy (%)")
        axes[row, 2].set_ylabel("accuracy (%)")
        axes[row, 2].legend(fontsize=8, loc="lower right")
    fig.text(0.01, -0.005, "Selected configuration of each method (mean over inits; for M2/M5 the best R by validation).",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, out / "07_training_curves.png", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    res = Path(a.results_dir) / a.preset
    B = json.load(open(res / "best_config.json"))
    cells = read_cells(res / "all_cells.csv")
    out = res / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig_tables(B, out, a.dpi)
    fig_id_ood(B, out, a.dpi)
    fig_m2(B, cells, out, a.dpi)
    fig_m3_m4a(B, cells, out, a.dpi)
    fig_m5(B, cells, out, a.dpi)
    fig_m5_sweeps(B, cells, out, a.dpi)
    fig_curves(B, out, a.dpi)


if __name__ == "__main__":
    main()

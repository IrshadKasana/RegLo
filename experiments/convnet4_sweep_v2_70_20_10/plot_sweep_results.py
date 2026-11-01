#!/usr/bin/env python
"""
plot_sweep_results.py -- PNG figures for the five-method ConvNet-4 sweep (70:20:10, ID + cross-dataset OOD).

Reads  results/<preset>/best_config.json  and  results/<preset>/all_cells.csv
Writes results/<preset>/plots/*.png

    python plot_sweep_results.py [--preset full] [--results-dir results] [--dpi 200]

    01_summary_<source>_to_<target>.png   table: config, val, train, ID test, OOD acc / AUROC (source -> target)
    02_id_vs_ood.png                      ID test accuracy and OOD AUROC of all methods, per source dataset
    03_m2_sweeps.png                      M2: validation accuracy vs lambda (best rho at each lambda), per R
    04_m3_meta_lr.png                     M3: validation / test accuracy vs meta_lr
    05_m4a_m5_axes.png                    M4a lambda_smooth axis and M5 lambda_reg axes
    06_m5_block_<source>.png              M5 joint (lambda_smooth x lambda_reg) block, validation accuracy
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REGS = ("l1", "l2", "sam", "gsam", "gam")
RL = {"l1": "L1", "l2": "L2", "sam": "SAM", "gsam": "GSAM", "gam": "GAM"}
NAME = {"svhn": "SVHN", "cifar10": "CIFAR-10"}
COL = {"l1": "#2a78d6", "l2": "#eb6834", "sam": "#1baf7a", "gsam": "#eda100", "gam": "#e87ba4"}
MCOL = {"M1": "#52514e", "M2": "#2a78d6", "M3": "#898781", "M4a": "#6250d6", "M5": "#eb6834"}   # method colours (fig 07)
SURFACE, INK, INK2, MUTED, GRID_C, AXIS_C = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

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


def fmt(ms):
    m, s = ms
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "-"
    return f"{pct(m):.2f}" + (f" ± {pct(s):.2f}" if s is not None and not math.isnan(s) else "")


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


def symlog_x(ax):
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 1])
    ax.set_xticklabels(["1e-3", "0.01", "0.1", "1"])
    ax.minorticks_off()


def targets_of(B, d):
    r = B["results"][d]
    k = next(iter(r))
    return list(r[k]["ood"].keys())


def entry_cfg(key, e):
    c, m = e["config"], key.split(":")[0]
    if m == "M1":
        return f"lr={g(c['adamw_lr'])}"
    if m == "M2":
        return f"λ={g(c['lam'])}" + (f", ρ={g(c['rho'])}" if c.get("rho") else "")
    if m == "M3":
        return f"meta_lr={g(c['meta_lr'])}"
    if m == "M4a":
        return f"λ_smooth={g(c['ls'])}, meta_lr={g(c['meta_lr'])}"
    return f"λ_smooth={g(c['ls'])}, λ_reg={g(c['lam_reg'])}, meta_lr={g(c['meta_lr'])}"


# ------------------------------------------------------------------ 01 tables
def fig_tables(B, out, dpi):
    LAB = {"M1": "M1 AdamW", "M2": "M2 AdamW+R", "M3": "M3 LO", "M4a": "M4a LO+smooth", "M5": "M5 LO+smooth+R"}
    for d, R in B["results"].items():
        for t in targets_of(B, d):
            rows = []
            for key, e in R.items():
                m, _, reg = key.partition(":")
                o = e["ood"][t]
                rows.append([LAB[m], RL.get(reg, "-"), entry_cfg(key, e), f"{pct(e['val']):.2f}", fmt(e["train_acc"]),
                             fmt(e["test_acc"]), fmt(o["acc"]), f"{o['auroc_msp'][0]:.3f}"])
            head = ["method", "R", "selected config", "val acc %", "train acc %", f"ID test acc % ({NAME[d]})",
                    f"OOD acc % ({NAME[t]})", "OOD AUROC"]
            fig, ax = plt.subplots(figsize=(18, 0.37 * (len(rows) + 1) + 0.2))
            ax.axis("off")
            tb = ax.table(cellText=rows, colLabels=head, bbox=[0, 0, 1, 1], cellLoc="center",
                          colWidths=[0.11, 0.045, 0.385, 0.065, 0.09, 0.12, 0.11, 0.075])
            tb.auto_set_font_size(False)
            tb.set_fontsize(9)
            best = max(range(len(rows)), key=lambda i: float(rows[i][5].split()[0]))
            for (i, j), cell in tb.get_celld().items():
                cell.set_edgecolor(GRID_C)
                cell.set_linewidth(0.8)
                if i == 0:
                    cell.set_facecolor("#f0efec")
                    cell.set_text_props(weight="bold", color=INK)
                else:
                    cell.set_facecolor("#e8f1fc" if i - 1 == best else SURFACE)
                    cell.set_text_props(color=INK, weight="bold" if i - 1 == best else "normal")
            ax.set_title(f"{NAME[d]} → {NAME[t]}:  λ / lr selected on {NAME[d]} validation (20 %);  ID test = {NAME[d]} 10 %;  "
                         f"OOD = {NAME[t]} 20 % partition;  ± = std. error (inits: M1/M2, meta-seeds: M3/M4a/M5)",
                         loc="left", pad=10, fontsize=10)
            save(fig, out / f"01_summary_{d}_to_{t}.png", dpi)


# ------------------------------------------------------------------ 02 ID vs OOD
def bar_items(R):
    keys = ["M1"] + [f"M2:{r}" for r in REGS] + ["M3", "M4a"] + [f"M5:{r}" for r in REGS]
    return [k for k in keys if k in R]


def fig_id_ood(B, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(len(ds), 2, figsize=(15, 4.6 * len(ds)), squeeze=False)
    for row, d in enumerate(ds):
        R = B["results"][d]
        t = targets_of(B, d)[0]
        keys = bar_items(R)
        for col, (title, getter, yl) in enumerate((
                (f"{NAME[d]} — ID test accuracy (10 % split)", lambda e: e["test_acc"], "accuracy (%)"),
                (f"{NAME[d]} → {NAME[t]} — OOD detection AUROC (max-softmax)", lambda e: e["ood"][t]["auroc_msp"], "AUROC"))):
            ax = axes[row, col]
            x = np.arange(len(keys))
            v = np.array([getter(R[k])[0] for k in keys]) * (100 if col == 0 else 1)
            s = np.array([0 if math.isnan(getter(R[k])[1]) else getter(R[k])[1] for k in keys]) * (100 if col == 0 else 1)
            cols = [COL.get(k.partition(":")[2], MCOL[k.split(":")[0]]) for k in keys]
            bars = ax.bar(x, v, width=0.68, color=cols, edgecolor=SURFACE, linewidth=1.5, yerr=s,
                          error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2.5))
            for b, k in zip(bars, keys):
                if k.split(":")[0] in ("M3", "M4a", "M5"):
                    b.set_hatch("//")
            for xi, vi, si in zip(x, v, s):
                ax.text(xi, vi + si + (0.1 if col == 0 else 0.003), f"{vi:.1f}" if col == 0 else f"{vi:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.set_xticks(x)
            ax.set_xticklabels([k.replace(":", "\n").upper().replace("M4A", "M4a") for k in keys], fontsize=7.5)
            lo = max(0, (math.floor(v.min() - 3) if col == 0 else math.floor((v.min() - 0.05) * 20) / 20))
            ax.set_ylim(lo, v.max() + s.max() + (1.5 if col == 0 else 0.03))
            ax.set_ylabel(yl)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
    h = [Line2D([0], [0], color=COL[r], lw=8, label=RL[r]) for r in REGS]
    h += [Patch(facecolor=MCOL["M4a"], label="M4a (no R)"), Patch(facecolor=MUTED, hatch="//", label="hatched = learned optimizer")]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out / "02_id_vs_ood.png", dpi)


# ------------------------------------------------------------------ 03 M2
def fig_m2(B, cells, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(1, len(ds), figsize=(6.5 * len(ds), 4.6), squeeze=False)
    for ax, d in zip(axes[0], ds):
        for r in REGS:
            c = [x for x in cells if x["method"] == "M2" and x["source"] == d and x["reg"] == r]
            if not c:
                continue
            best = {}
            for x in c:
                best[x["lam"]] = max(best.get(x["lam"], -1), x["val_acc"])
            xs = sorted(best)
            ax.plot(xs, [pct(best[v]) for v in xs], color=COL[r], marker="o", label=RL[r])
            sel = B["results"][d].get(f"M2:{r}")
            if sel:
                ax.plot([sel["config"]["lam"]], [pct(sel["val"])], marker="o", ms=11, mfc="none", mec=COL[r], mew=2)
        m1 = B["results"][d].get("M1")
        if m1:
            ax.axhline(pct(m1["val"]), color=MUTED, ls="--", lw=1.3)
        symlog_x(ax)
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel("validation accuracy (%)")
        if m1:
            ax.set_ylim(pct(m1["val"]) - 8, None)
        ax.set_title(f"M2 — {NAME[d]}")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.01, -0.1, "At each λ the best ρ is shown (SAM/GSAM/GAM). Ring = selected. Dashed = M1 (best lr). "
             "Curves below the y-range are clipped.", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, out / "03_m2_sweeps.png", dpi)


# ------------------------------------------------------------------ 04 M3
def fig_m3(B, cells, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(1, len(ds), figsize=(6.5 * len(ds), 4.4), squeeze=False)
    for ax, d in zip(axes[0], ds):
        c = sorted([x for x in cells if x["method"] == "M3" and x["source"] == d and x["seed"] == "0"], key=lambda x: x["meta_lr"])
        if c:
            xs = [x["meta_lr"] for x in c]
            ax.plot(xs, [pct(x["val_acc"]) for x in c], color=MCOL["M3"], marker="o", label="validation (selection)")
            ax.plot(xs, [pct(x["test_acc"]) for x in c], color=MCOL["M5"], marker="s", ls="--", lw=1.4, label="ID test (not used)")
            sel = B["results"][d].get("M3")
            if sel:
                ax.plot([sel["config"]["meta_lr"]], [pct(sel["val"])], marker="o", ms=12, mfc="none", mec=INK, mew=2)
        ax.set_xscale("log")
        ax.set_xlabel("meta_lr (Adam rate of the learned optimizer's weights)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"M3 — {NAME[d]}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "04_m3_meta_lr.png", dpi)


# ------------------------------------------------------------------ 05 axes
def fig_axes(B, cells, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(len(ds), 2, figsize=(13, 4.2 * len(ds)), squeeze=False)
    for row, d in enumerate(ds):
        mlr = (B["results"][d].get("M3") or {}).get("config", {}).get("meta_lr")
        c = [x for x in cells if x["source"] == d and x["seed"] == "0" and (mlr is None or x["meta_lr"] == mlr)]
        ax = axes[row, 0]
        m3 = [x for x in c if x["method"] == "M3"]
        m4 = sorted([x for x in c if x["method"] == "M4a"], key=lambda x: x["ls"])
        if m4:
            ax.plot([x["ls"] for x in m4], [pct(x["val_acc"]) for x in m4], color=MCOL["M4a"], marker="o", label="M4a")
        if m3:
            ax.axhline(pct(m3[0]["val_acc"]), color=MUTED, ls="--", lw=1.3)
            ax.annotate("M3 (no smooth term)", (m4[0]["ls"] if m4 else 0.001, pct(m3[0]["val_acc"])), xytext=(6, -12), textcoords="offset points",
                        fontsize=8, color=INK2, va="top")
        sel = B["results"][d].get("M4a")
        if sel:
            ax.plot([sel["config"]["ls"]], [pct(sel["val"])], marker="o", ms=11, mfc="none", mec=INK, mew=2)
        symlog_x(ax)
        ax.set_xlabel("λ_smooth (no regularizer)")
        ax.set_ylabel("validation accuracy (%)")
        ax.set_title(f"M4a λ_smooth axis — {NAME[d]}")
        ax = axes[row, 1]
        for r in REGS:
            ls_star = (B["results"][d].get("M4a") or {}).get("config", {}).get("ls")
            a = sorted([x for x in c if x["method"] == "M5" and x["reg"] == r and x["ls"] == ls_star], key=lambda x: x["lam_reg"])
            if a:
                ax.plot([x["lam_reg"] for x in a], [pct(x["val_acc"]) for x in a], color=COL[r], marker="o", label=RL[r])
        if m3:
            ax.axhline(pct(m3[0]["val_acc"]), color=MUTED, ls="--", lw=1.3)
        ys_ = [pct(x["val_acc"]) for x in c if x["method"] == "M5" and x["ls"] == ls_star] + ([pct(m3[0]["val_acc"])] if m3 else [])
        if ys_:
            ax.set_ylim(min(ys_) - 0.5, max(ys_) + 0.5)
        symlog_x(ax)
        ax.set_xlabel("λ_reg (at the λ_smooth selected for M4a)")
        ax.set_ylabel("validation accuracy (%)")
        ax.set_title(f"M5 λ_reg axis at M4a λ_smooth — {NAME[d]}")
        if row == 0:
            ax.legend(ncol=5, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, out / "05_m4a_m5_axes.png", dpi)


# ------------------------------------------------------------------ 06 M5 block
def fig_blocks(B, cells, out, dpi):
    for d in B["results"]:
        mlr = (B["results"][d].get("M3") or {}).get("config", {}).get("meta_lr")
        c = [x for x in cells if x["source"] == d and x["method"] == "M5" and x["seed"] == "0" and (mlr is None or x["meta_lr"] == mlr)]
        if not c:
            continue
        fig, axes = plt.subplots(1, len(REGS), figsize=(4.6 * len(REGS), 4.4))
        allv = [pct(x["val_acc"]) for x in c]
        vmin, vmax = min(allv), max(allv)
        im = None
        for ax, r in zip(axes, REGS):
            cr = [x for x in c if x["reg"] == r]
            ax.grid(False)
            if not cr:
                ax.axis("off")
                continue
            xs = sorted({x["ls"] for x in cr})
            ys = sorted({x["lam_reg"] for x in cr})
            Z = np.full((len(ys), len(xs)), np.nan)
            for x in cr:
                Z[ys.index(x["lam_reg"]), xs.index(x["ls"])] = pct(x["val_acc"])
            im = ax.imshow(Z, origin="lower", cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
            for iy in range(len(ys)):
                for ix in range(len(xs)):
                    z = Z[iy, ix]
                    if not np.isnan(z):
                        ax.text(ix, iy, f"{z:.1f}", ha="center", va="center", fontsize=6.5,
                                color="#fcfcfb" if (z - vmin) / max(vmax - vmin, 1e-9) > 0.55 else INK)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([g(v) for v in xs], fontsize=7, rotation=60)
            ax.set_yticks(range(len(ys)))
            ax.set_yticklabels([g(v) for v in ys], fontsize=7)
            ax.set_facecolor("#f0efec")
            sel = B["results"][d].get(f"M5:{r}")
            if sel and sel["config"]["ls"] in xs and sel["config"]["lam_reg"] in ys:
                ax.plot(xs.index(sel["config"]["ls"]), ys.index(sel["config"]["lam_reg"]), marker="s", ms=18, mfc="none",
                        mec="#eb6834", mew=2.2)
            ax.set_xlabel("λ_smooth")
            if r == REGS[0]:
                ax.set_ylabel("λ_reg")
            ax.set_title(f"{RL[r]} — {len(cr)} cells", fontsize=11)
        if im is not None:
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01).set_label("validation acc (%)")
        fig.suptitle(f"M5 joint block — {NAME[d]} (λ_smooth > 0, λ_reg > 0; ring = selected; grey = not run)",
                     x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
        save(fig, out / f"06_m5_block_{d}.png", dpi)


# ------------------------------------------------------------------ 07 curves
def fig_curves(B, out, dpi):
    ds = list(B["results"])
    fig, axes = plt.subplots(len(ds), 2, figsize=(13, 4.2 * len(ds)), squeeze=False)
    for row, d in enumerate(ds):
        R = B["results"][d]
        picks = {}
        for m in ("M1", "M2", "M3", "M4a", "M5"):
            c = [(k, e) for k, e in R.items() if k.split(":")[0] == m and e.get("curve")]
            if c:
                picks[m] = max(c, key=lambda ke: ke[1]["val"])
        for m, (k, e) in picks.items():
            lab = m if ":" not in k else f"{m} ({RL[k.split(':')[1]]})"
            st = [p["step"] for p in e["curve"]]
            axes[row, 0].plot(st, [p["train_loss"] for p in e["curve"]], color=MCOL[m], label=lab)
            axes[row, 1].plot(st, [pct(p["val_acc"]) for p in e["curve"]], color=MCOL[m], label=lab)
        axes[row, 0].set_title(f"{NAME[d]} — training loss (mean of the last window)")
        axes[row, 1].set_title(f"{NAME[d]} — validation accuracy")
        axes[row, 0].set_ylabel("loss")
        axes[row, 1].set_ylabel("accuracy (%)")
        for a in axes[row]:
            a.set_xlabel("optimizee step")
        axes[row, 1].legend(fontsize=8, loc="lower right")
    fig.text(0.01, -0.01, "Selected configuration of each method (mean over inits; for M2/M5 the best R by validation).",
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
    fig_m3(B, cells, out, a.dpi)
    fig_axes(B, cells, out, a.dpi)
    fig_blocks(B, cells, out, a.dpi)
    fig_curves(B, out, a.dpi)


if __name__ == "__main__":
    main()

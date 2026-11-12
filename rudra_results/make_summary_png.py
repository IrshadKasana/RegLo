#!/usr/bin/env python
"""Render the best-pick summary of the sweep as one PNG table per (source -> OOD target) pair: 4 sources x 3 targets = 12 PNGs.

    python make_summary_png.py [full_dir]     # default: ./full  ->  full/summary_png_pairs/summary_<source>_to_<target>.png

Reads <full_dir>/best_config.json only. Columns: method, reg, one column per hyperparameter, cells tried, train acc,
ID test acc, OOD acc, ID AUROC, OOD AUROC. A hyperparameter that does not apply to a method shows "-".
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "full")
OUT = D / "summary_png_pairs"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID, BAND, HEAD = "#1f2328", "#59636e", "#d1d9e0", "#f6f8fa", "#eaeef2"
NAMES = {"M1": "M1  AdamW", "M2": "M2  AdamW+R", "M3": "M3  LO", "M4a": "M4a  LO+smooth", "M5": "M5  LO+smooth+R"}
TITLES = {"svhn": "SVHN", "cifar10": "CIFAR-10", "mnist": "MNIST", "fmnist": "Fashion-MNIST"}
# (header, config key, width as a fraction of the figure width)
HP = [("AdamW lr", "adamw_lr", 0.062), ("λ", "lam", 0.052), ("ρ", "rho", 0.052),
      ("λ_smooth", "ls", 0.062), ("λ_reg", "lam_reg", 0.056), ("meta lr", "meta_lr", 0.058)]
METRICS = ["Train acc %", "ID test acc %", "OOD acc %", "ID AUROC", "OOD AUROC"]
W_METHOD, W_REG, W_CELLS, W_METRIC = 0.130, 0.050, 0.058, 0.098
RANKED = [True, True, False, True, True]          # OOD acc is chance by construction: not ranked


def g(x):
    return "-" if x is None else f"{x:g}"


def rows(res, target):
    out = []
    for key, r in res.items():
        m = key.split(":")[0]
        c = r["config"]
        o = r["ood"][target]
        na = {"M3": ("ls", "lam_reg"), "M4a": ("lam_reg",)}.get(m, ())        # fixed at 0, not tuned for this method
        out.append(dict(method=m, name=NAMES[m], reg=key.split(":")[1] if ":" in key else "-", hp=[g(None if k in na else c.get(k)) for _, k, _ in HP],
                        cells=r["n_cells"], vals=[r["train_acc"], r["test_acc"], o["acc"], r["id_auroc"], o["auroc_msp"]]))
    return out


def fmt(i, mu, se):
    return f"{mu * 100:.2f} ± {se * 100:.2f}" if i < 3 else f"{mu:.3f} ± {se:.3f}"


def OOD_ACC_NOTE(src, tgt):
    if {src, tgt} == {"svhn", "mnist"}:
        return "both are digits with the same label ids, so it is a real transfer accuracy."
    return "the label sets are unrelated, so about 10% (chance) is expected."


def draw(src, tgt, res):
    R = rows(res, tgt)
    n = len(R)
    best = {i: max(r["vals"][i][0] for r in R) for i, up in enumerate(RANKED) if up}
    widths = [W_METHOD, W_REG] + [w for _, _, w in HP] + [W_CELLS] + [W_METRIC] * len(METRICS)
    heads = ["Method", "Reg"] + [h for h, _, _ in HP] + ["Cells tried"] + METRICS
    xs = np.concatenate([[0], np.cumsum(widths)])
    xs = xs / xs[-1]
    rh = 0.36
    fig, ax = plt.subplots(figsize=(18.5, 2.05 + rh * (n + 1)))
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 2.4, -1.45)
    ax.axis("off")
    ax.text(0, -1.25, f"{TITLES[src]} (source)  →  {TITLES[tgt]} (OOD target) — best-pick configuration per method", fontsize=14,
            fontweight="bold", color=INK, va="center")
    ax.text(0, -0.72, f"ID = {TITLES[src]} test split.  OOD = the same model, no retraining, on {TITLES[tgt]}.  "
            "Selected on source validation accuracy (seed 0).", fontsize=8.8, color=MUTED, va="center")
    ax.add_patch(plt.Rectangle((0, -0.35), 1, 1.0, color=HEAD, lw=0))
    first_num = 2 + len(HP)                                   # first column that is right-aligned data
    for j, h in enumerate(heads):
        left = j < 2
        ax.text(xs[j] + 0.006 if left else xs[j + 1] - 0.006, 0.15, h, fontsize=9.4, fontweight="bold", color=INK,
                ha="left" if left else "right", va="center")
    prev, band = None, False
    for i, r in enumerate(R):
        y = i + 1.15
        if r["method"] != prev:
            band = not band
            if prev is not None:
                ax.plot([0, 1], [y - 0.5, y - 0.5], color=GRID, lw=0.8)
        prev = r["method"]
        if band:
            ax.add_patch(plt.Rectangle((0, y - 0.5), 1, 1, color=BAND, lw=0, zorder=0))
        ax.text(xs[0] + 0.006, y, r["name"], fontsize=9.4, color=INK, va="center")
        ax.text(xs[1] + 0.006, y, r["reg"], fontsize=9.4, color=INK, va="center")
        for k, v in enumerate(r["hp"]):
            ax.text(xs[2 + k + 1] - 0.006, y, v, fontsize=9.2, color=INK if v != "-" else MUTED, ha="right", va="center",
                    family="DejaVu Sans Mono")
        ax.text(xs[first_num + 1] - 0.006, y, f"{r['cells']:,}", fontsize=9.4, color=INK, ha="right", va="center")
        for k, (mu, se) in enumerate(r["vals"]):
            dp = 4 if k < 2 else 3                            # a tie = equal at the displayed precision
            is_best = k in best and round(mu, dp) == round(best[k], dp)
            ax.text(xs[first_num + 2 + k] - 0.006, y, fmt(k, mu, se), fontsize=9.4, color=INK, ha="right", va="center",
                    fontweight="bold" if is_best else "normal")
    yb = n + 0.65
    ax.plot([0, 1], [yb, yb], color=GRID, lw=1)
    ax.text(0, yb + 0.55, "Cells tried = hyperparameter combinations searched for that row (the best one is shown).  Bold = best in column among the 13 picks "
            "(train acc, ID test acc, ID AUROC, OOD AUROC).  ± = std. error over 3 inits (M1/M2) or 3 meta-seeds (M3/M4a/M5).\n"
            f"OOD acc is accuracy under {TITLES[tgt]}'s own label ids: {OOD_ACC_NOTE(src, tgt)}  It is not ranked.  "
            "OOD AUROC = ID-test vs OOD separation by max-softmax confidence.  λ: M2 regularisation weight;  λ_reg: M5 regularisation weight.",
            fontsize=8.2, color=MUTED, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.015)
    p = OUT / f"summary_{src}_to_{tgt}.png"
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    results = json.load(open(D / "best_config.json"))["results"]
    for src in ["svhn", "cifar10", "mnist", "fmnist"]:
        for tgt in results[src]["M1"]["ood"]:
            draw(src, tgt, results[src])

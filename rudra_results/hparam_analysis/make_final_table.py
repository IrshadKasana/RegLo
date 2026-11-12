"""Final recommended hyperparameters (consensus over the 4 Rudra datasets) -> CSV + PNG. Values are the result of final_pick.py,
with the shared meta_lr override for M5 sam/gsam/gam (0.002 = lowest mean regret of all six meta_lr values for those three)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "full"
H = ["Method", "Reg", "AdamW lr", "λ", "ρ", "λ_smooth", "λ_reg", "meta lr", "Val-acc loss vs own best (mean / worst of 4 datasets, pp)", "Confidence", "Note"]
D = "-"
R = [
 ["M1 AdamW", D, "0.001", D, D, D, D, D, "0.23 / 0.42", "High", "lowest grid value; 0.001-0.003 all within 0.5 pp on average (0.002 worst 1.2)"],
 ["M2 AdamW+R", "l1", "0.002", "0.001", D, D, D, D, "0.81 / 2.08", "Medium", "λ at grid minimum; CIFAR-10 loses 2.1 pp"],
 ["M2 AdamW+R", "l2", "0.001", "0.001", D, D, D, D, "0.00 / 0.00", "High", "best cell on all 4 datasets; both are the smallest values tested, so the optimum may be lower (λ=0.001 is already tiny)"],
 ["M2 AdamW+R", "sam", "0.001", "0.4", "0.05", D, D, D, "0.15 / 0.22", "High", "flat: λ 0.3-0.5 with ρ 0.04-0.06 all < 0.4 pp mean loss"],
 ["M2 AdamW+R", "gsam", "0.001", "0.4", "0.05", D, D, D, "0.23 / 0.29", "High", "flat: same region as sam"],
 ["M2 AdamW+R", "gam", "0.001", "0.01", "0.001", D, D, D, "0.40 / 0.57", "Low", "landscape flat (many cells within 0.8 pp); λ/ρ barely matter"],
 ["M3 LO", D, D, D, D, D, D, "0.005", "0.27 / 0.41", "Medium", "0.001 is nearly as good (0.33 / 0.64); single meta-seed per cell"],
 ["M4a LO+smooth", D, D, D, D, "0.004", D, "0.001", "0.70 / 1.00", "Medium", ""],
 ["M5 LO+smooth+R", "l1", D, D, D, "0.005", "0.001", "0.001", "1.28 / 2.63", "Low", "unstable: wrong meta_lr/λ_reg costs up to 40+ pp; λ_reg at grid minimum"],
 ["M5 LO+smooth+R", "l2", D, D, D, "0.3", "0.001", "0.001", "0.83 / 1.20", "Low", "unstable in meta_lr (worst 6 pp at 0.001, 44 pp at 0.003); λ_reg at grid minimum"],
 ["M5 LO+smooth+R", "sam", D, D, "0.02", "0.5", "0.002", "0.002", "0.90 / 1.60", "Medium", "λ_smooth at grid maximum; interior alternative 0.02 / 0.3"],
 ["M5 LO+smooth+R", "gsam", D, D, "0.02", "0.02", "0.01", "0.002", "0.88 / 1.75", "Medium", ""],
 ["M5 LO+smooth+R", "gam", D, D, "0.004", "0.01", "0.007", "0.002", "1.09 / 1.54", "Low", "ρ weakly determined: best 3 values lose 0.8-1.3 pp on average"],
]
with open(OUT / "final_hyperparameters.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(H); w.writerows(R)

INK, MUTED, GRID, BAND, HEAD = "#1f2328", "#59636e", "#d1d9e0", "#f6f8fa", "#eaeef2"
W = [0.105, 0.04, 0.058, 0.042, 0.042, 0.055, 0.050, 0.052, 0.098, 0.062, 0.296]
xs = [0]
for w_ in W: xs.append(xs[-1] + w_)
xs = [x / xs[-1] for x in xs]
n = len(R); rh = 0.40
fig, ax = plt.subplots(figsize=(19, 2.1 + rh * (n + 1)))
ax.set_xlim(0, 1); ax.set_ylim(n + 2.6, -1.5); ax.axis("off")
ax.text(0, -1.3, "Recommended hyperparameters for production runs — consensus over SVHN, CIFAR-10, MNIST, Fashion-MNIST (Rudra sweep)", fontsize=14, fontweight="bold", color=INK, va="center")
ax.text(0, -0.78, "Chosen to minimise mean + worst-dataset validation-accuracy loss inside the flat region of the grid. Starting points for a short re-tune, not final values for ResNet-18 / ImageNet.", fontsize=9, color=MUTED, va="center")
ax.add_patch(plt.Rectangle((0, -0.4), 1, 1.2, color=HEAD, lw=0))
for j, h in enumerate(H):
    h = h.replace(" (mean / worst of 4 datasets, pp)", "\nmean / worst (pp)").replace("Val-acc loss vs own best", "Val-acc loss vs own best")
    num = 2 <= j <= 7
    ax.text(xs[j + 1] - 0.005 if num else xs[j] + 0.005, 0.2, h, fontsize=9, fontweight="bold", color=INK, ha="right" if num else "left", va="center")
prev, band = None, False
CONF = {"High": "#1a7f37", "Medium": "#9a6700", "Low": "#cf222e"}
for i, r in enumerate(R):
    y = i + 1.3
    if r[0] != prev:
        band = not band
        if prev is not None: ax.plot([0, 1], [y - 0.5, y - 0.5], color=GRID, lw=0.8)
    prev = r[0]
    if band: ax.add_patch(plt.Rectangle((0, y - 0.5), 1, 1, color=BAND, lw=0, zorder=0))
    for j, v in enumerate(r):
        num = 2 <= j <= 7
        kw = dict(fontsize=9.4, va="center", color=INK if v != D else MUTED)
        if j == 9: kw.update(color=CONF[v], fontweight="bold")
        if j == 10: kw.update(fontsize=8.3, color=MUTED)
        if num: kw.update(family="DejaVu Sans Mono", fontweight="bold" if v != D else "normal")
        ax.text(xs[j + 1] - 0.005 if num else xs[j] + 0.005, y, v, ha="right" if num else "left", **kw)
yb = n + 0.8
ax.plot([0, 1], [yb, yb], color=GRID, lw=1)
ax.text(0, yb + 0.5, "λ = M2 regulariser weight; λ_reg = M5 regulariser weight; ρ = SAM/GSAM/GAM perturbation radius; meta lr = learned-optimizer Adam lr.  '-' = not applicable.\n"
        "Loss = validation-accuracy points lost vs each dataset's own best cell, mean and worst over the 4 datasets (M5: on the λ_smooth × λ_reg grid; its meta_lr and ρ stages add to this).\n"
        "Grid: 20 values from 0.001 to 0.5. Single seed per cell (3 inits for M1/M2, 1 meta-seed for M3/M4a/M5), 1000 steps, ConvNet-4 on 32×32 images.", fontsize=8.2, color=MUTED, va="top", linespacing=1.5)
fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.015)
fig.savefig(OUT / "final_hyperparameters.png", dpi=150, facecolor="white")
print("wrote", OUT / "final_hyperparameters.csv", OUT / "final_hyperparameters.png")

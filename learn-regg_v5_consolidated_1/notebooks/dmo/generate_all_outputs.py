"""
generate_all_outputs.py
=======================
One-shot results generator. Reads the three evaluation JSONs (No Reg / SAM / L2),
produces every chart and a summary text file, then bundles everything into
a single ZIP for download.

What it produces in <outdir>:
  - 01_test_accuracy_bar.png
  - 02_train_loss_bar.png
  - 03_test_loss_bar.png
  - 04_max_loss_neighbourhood.png
  - 05_max_grad_norm_neighbourhood.png
  - 06_loss_curves.png
  - 07_accuracy_curve.png
  - 08_summary_panel.png            (the main paper figure)
  - summary_table.txt               (numeric tables)
  - results_bundle.zip              (everything zipped for easy download)

Usage:
  python notebooks/dmo/generate_all_outputs.py \
      --noreg <noreg.json> --sam <sam.json> --l2 <l2.json> \
      --outdir artifacts/dmo/results/plots/
"""

from __future__ import annotations
import argparse, json, os, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")    # headless — no GUI needed
import matplotlib.pyplot as plt


METHODS = ["No Reg", "SAM", "L2"]
COLORS  = {"No Reg": "#2E7D32", "SAM": "#7CB342", "L2": "#9575CD"}


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------
def load(path): return json.load(open(path))

def final_values(payload, key):
    finals = [r[-1] for r in payload.get(key, []) if r]
    if not finals: return float("nan"), 0.0
    a = np.asarray(finals, dtype=float)
    return float(a.mean()), float(a.std())

def all_curves(payload, key):
    rows = payload.get(key, [])
    if not rows: return np.array([])
    maxlen = max(len(r) for r in rows)
    out = np.full((len(rows), maxlen), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out

def sharpness_by_eps(payload, key):
    pooled = {}
    for instance in payload.get(key, []):
        for _bi, eps_dict in instance.items():
            for e_str, val in eps_dict.items():
                prof = val[0] if isinstance(val, (list, tuple)) else val
                if isinstance(prof, (list, tuple)) and prof:
                    pooled.setdefault(float(e_str), []).append(float(max(prof)))
    return {e: (float(np.mean(v)), float(np.std(v))) for e, v in sorted(pooled.items())}


# ----------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------
def bar_final(ax, vals, title, ylabel, fmt="{:.4f}"):
    means = [vals[m][0] for m in METHODS]
    stds  = [vals[m][1] for m in METHODS]
    x = np.arange(len(METHODS))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[COLORS[m] for m in METHODS], edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontweight='bold')
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    headroom = (max(means) if max(means) else 1) * 0.05
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + headroom,
                fmt.format(v), ha='center', fontsize=10)


def bar_grouped_eps(ax, by_method, title, ylabel):
    all_eps = sorted({e for d in by_method.values() for e in d})
    if not all_eps:
        ax.set_title(f"{title} (no data)"); return
    xs = np.arange(len(all_eps)); width = 0.25
    for i, m in enumerate(METHODS):
        means = [by_method[m].get(e, (np.nan, 0.0))[0] for e in all_eps]
        stds  = [by_method[m].get(e, (np.nan, 0.0))[1] for e in all_eps]
        ax.bar(xs + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=m, color=COLORS[m], edgecolor='black')
    ax.set_xticks(xs); ax.set_xticklabels([str(e) for e in all_eps])
    ax.set_xlabel("Perturbation radius ε"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)


def line_curves(ax, curves_by_method, title, ylabel):
    for m in METHODS:
        c = curves_by_method[m]
        if c.size == 0: continue
        mean = np.nanmean(c, axis=0)
        std  = np.nanstd(c, axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=m, color=COLORS[m], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std,
                        color=COLORS[m], alpha=0.15)
    ax.set_xlabel("Step"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)


# ----------------------------------------------------------------------
# Save individual figures
# ----------------------------------------------------------------------
def save_single(plot_fn, fname, outdir, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    plt.tight_layout()
    path = outdir / fname
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Summary text
# ----------------------------------------------------------------------
def write_summary(path, acc, ltr, lte, sharp_loss, sharp_grad):
    with open(path, "w") as f:
        f.write("=" * 78 + "\n")
        f.write(f"{'Method':<10} {'TestAcc':>16} {'TrainLoss':>16} {'TestLoss':>16}\n")
        f.write("-" * 78 + "\n")
        for m in METHODS:
            a_mu, a_sd = acc[m]; tr_mu, tr_sd = ltr[m]; te_mu, te_sd = lte[m]
            f.write(f"{m:<10} {a_mu:>7.4f} ± {a_sd:.4f}  "
                    f"{tr_mu:>7.4f} ± {tr_sd:.4f}  "
                    f"{te_mu:>7.4f} ± {te_sd:.4f}\n")
        f.write("=" * 78 + "\n\n")

        f.write("Max loss in ε-neighbourhood (sharpness):\n")
        all_eps = sorted({e for d in sharp_loss.values() for e in d})
        f.write(f"{'eps':>6}" + "".join(f"{m:>20}" for m in METHODS) + "\n")
        for e in all_eps:
            row = f"{e:>6}"
            for m in METHODS:
                mu, sd = sharp_loss[m].get(e, (np.nan, 0.0))
                row += f"   {mu:>8.4f} ± {sd:.4f}"
            f.write(row + "\n")

        f.write("\nMax gradient norm in ε-neighbourhood:\n")
        f.write(f"{'eps':>6}" + "".join(f"{m:>20}" for m in METHODS) + "\n")
        for e in all_eps:
            row = f"{e:>6}"
            for m in METHODS:
                mu, sd = sharp_grad[m].get(e, (np.nan, 0.0))
                row += f"   {mu:>8.4f} ± {sd:.4f}"
            f.write(row + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(noreg_path, sam_path, l2_path, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    payloads = {"No Reg": load(noreg_path),
                "SAM":    load(sam_path),
                "L2":     load(l2_path)}

    acc       = {m: final_values(p, "instances_acc_ts")   for m, p in payloads.items()}
    ltr       = {m: final_values(p, "instances_loss_tr")  for m, p in payloads.items()}
    lte       = {m: final_values(p, "instances_loss_ts")  for m, p in payloads.items()}

    sharp_loss = {m: sharpness_by_eps(p, "max_loss_nghb_list")      for m, p in payloads.items()}
    sharp_grad = {m: sharpness_by_eps(p, "max_grad_norm_nghb_list") for m, p in payloads.items()}

    acc_curves   = {m: all_curves(p, "instances_acc_ts")  for m, p in payloads.items()}
    train_curves = {m: all_curves(p, "instances_loss_tr") for m, p in payloads.items()}
    test_curves  = {m: all_curves(p, "instances_loss_ts") for m, p in payloads.items()}

    # ---- individual figures ----
    save_single(lambda a: bar_final(a, acc, "Test accuracy (final)", "Accuracy"),
                "01_test_accuracy_bar.png", outdir)
    save_single(lambda a: bar_final(a, ltr, "Train loss (final)", "Loss"),
                "02_train_loss_bar.png", outdir)
    save_single(lambda a: bar_final(a, lte, "Test loss (final)", "Loss"),
                "03_test_loss_bar.png", outdir)
    save_single(lambda a: bar_grouped_eps(a, sharp_loss, "Max loss in ε-neighbourhood", "Max loss"),
                "04_max_loss_neighbourhood.png", outdir, figsize=(8, 5))
    save_single(lambda a: bar_grouped_eps(a, sharp_grad, "Max ‖∇L‖ in ε-neighbourhood", "Max grad norm"),
                "05_max_grad_norm_neighbourhood.png", outdir, figsize=(8, 5))
    save_single(lambda a: line_curves(a, test_curves, "Test loss over steps", "Loss"),
                "06_loss_curves.png", outdir, figsize=(8, 5))
    save_single(lambda a: line_curves(a, acc_curves, "Test accuracy over steps", "Accuracy"),
                "07_accuracy_curve.png", outdir, figsize=(8, 5))

    # ---- combined paper figure (2 rows × 3 cols) ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("3-way comparison — No Reg vs SAM vs L2", fontsize=14, fontweight='bold')
    bar_final(axes[0, 0], acc, "Test accuracy (final)", "Accuracy")
    bar_final(axes[0, 1], ltr, "Train loss (final)",     "Loss")
    bar_final(axes[0, 2], lte, "Test loss (final)",      "Loss")
    bar_grouped_eps(axes[1, 0], sharp_loss, "Max loss in ε-nghb",   "Max loss")
    bar_grouped_eps(axes[1, 1], sharp_grad, "Max ‖∇L‖ in ε-nghb",   "Max grad norm")
    line_curves(axes[1, 2], test_curves, "Test loss over steps",   "Loss")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    panel_path = outdir / "08_summary_panel.png"
    plt.savefig(panel_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ---- summary text ----
    summary_path = outdir / "summary_table.txt"
    write_summary(summary_path, acc, ltr, lte, sharp_loss, sharp_grad)

    # ---- zip everything ----
    bundle_path = outdir / "results_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # include the source JSONs and all generated outputs
        for src in [noreg_path, sam_path, l2_path]:
            zf.write(src, arcname=f"input_json/{Path(src).name}")
        for png in sorted(outdir.glob("*.png")):
            zf.write(png, arcname=f"plots/{png.name}")
        zf.write(summary_path, arcname="summary_table.txt")

    # write the location of the bundle as the only thing printed
    print(str(bundle_path))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--noreg", required=True)
    p.add_argument("--sam",   required=True)
    p.add_argument("--l2",    required=True)
    p.add_argument("--outdir", default="artifacts/dmo/results/plots/")
    args = p.parse_args()
    main(args.noreg, args.sam, args.l2, args.outdir)

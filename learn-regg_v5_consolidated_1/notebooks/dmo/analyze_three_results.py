"""
analyze_three_results.py
========================
Comprehensive analysis of the 3-way comparison (No Reg / SAM / L2):

  1. Final test accuracy + std across instances  -> table + bar chart
  2. Final train loss / test loss                 -> table + bar chart
  3. Max loss in epsilon-neighbourhood            -> grouped bar chart
  4. Max gradient norm in epsilon-neighbourhood   -> grouped bar chart
  5. Loss & accuracy curves over training steps   -> line plots
  6. A side-by-side summary panel                 -> the main figure for the paper

Usage:
    python notebooks/dmo/analyze_three_results.py \
        --noreg <noreg.json> --sam <sam.json> --l2 <l2.json> \
        --outdir artifacts/dmo/results/plots/
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
METHODS = ["No Reg", "SAM", "L2"]
COLORS  = {"No Reg": "#2E7D32", "SAM": "#7CB342", "L2": "#9575CD"}


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------
def load(path):
    with open(path) as f:
        return json.load(f)


def final_values(payload, key):
    """Mean and std of the last value of each instance's array."""
    rows = payload.get(key, [])
    finals = [r[-1] for r in rows if r]
    if not finals:
        return float("nan"), 0.0
    arr = np.asarray(finals, dtype=float)
    return float(arr.mean()), float(arr.std())


def all_curves(payload, key):
    """List of full per-instance curves, padded to max length with NaN."""
    rows = payload.get(key, [])
    if not rows:
        return np.array([])
    maxlen = max(len(r) for r in rows)
    out = np.full((len(rows), maxlen), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def sharpness_by_eps(payload, key):
    """
    For each eps in the PGD sweep, return mean and std of (max loss in
    the perturbation trajectory) across all instances.
    Returns {eps: (mean, std)}.
    """
    pooled = {}
    for instance in payload.get(key, []):
        for _best_iter, eps_dict in instance.items():
            for eps_str, value in eps_dict.items():
                profile = value[0] if isinstance(value, (list, tuple)) else value
                if isinstance(profile, (list, tuple)) and len(profile):
                    pooled.setdefault(float(eps_str), []).append(float(max(profile))))
    return {e: (float(np.mean(v)), float(np.std(v))) for e, v in sorted(pooled.items())}


# ----------------------------------------------------------------------
# Plot blocks
# ----------------------------------------------------------------------
def panel_final_metric(ax, vals, title, ylabel, fmt="{:.4f}"):
    means = [vals[m][0] for m in METHODS]
    stds  = [vals[m][1] for m in METHODS]
    x = np.arange(len(METHODS))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[COLORS[m] for m in METHODS], edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontweight='bold')
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2,
                v + (max(means) if max(means) else 1) * 0.02,
                fmt.format(v), ha='center', fontsize=9)


def panel_grouped_eps(ax, by_method, title, ylabel):
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


def panel_curves(ax, curves_by_method, title, ylabel, log=False):
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
    if log: ax.set_yscale('log')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(noreg_path, sam_path, l2_path, outdir):
    payloads = {"No Reg": load(noreg_path),
                "SAM":    load(sam_path),
                "L2":     load(l2_path)}

    # ---------- Gather scalar metrics ----------
    acc_final  = {m: final_values(p, "instances_acc_ts") for m, p in payloads.items()}
    loss_train = {m: final_values(p, "instances_loss_tr") for m, p in payloads.items()}
    loss_test  = {m: final_values(p, "instances_loss_ts") for m, p in payloads.items()}

    sharp_loss = {m: sharpness_by_eps(p, "max_loss_nghb_list")     for m, p in payloads.items()}
    sharp_grad = {m: sharpness_by_eps(p, "max_grad_norm_nghb_list") for m, p in payloads.items()}

    acc_curves   = {m: all_curves(p, "instances_acc_ts")   for m, p in payloads.items()}
    train_curves = {m: all_curves(p, "instances_loss_tr") for m, p in payloads.items()}
    test_curves  = {m: all_curves(p, "instances_loss_ts") for m, p in payloads.items()}

    # ---------- Print summary table ----------
    print("=" * 78)
    print(f"{'Method':<10} {'TestAcc':>14} {'TrainLoss':>14} {'TestLoss':>14}")
    print("-" * 78)
    for m in METHODS:
        a_mu, a_sd = acc_final[m]
        tr_mu, tr_sd = loss_train[m]
        te_mu, te_sd = loss_test[m]
        print(f"{m:<10} "
              f"{a_mu:>6.4f} ± {a_sd:.4f}  "
              f"{tr_mu:>6.4f} ± {tr_sd:.4f}  "
              f"{te_mu:>6.4f} ± {te_sd:.4f}")
    print("=" * 78)

    print("\nSharpness (max loss in neighbourhood) by epsilon:")
    all_eps = sorted({e for d in sharp_loss.values() for e in d})
    header = "eps " + " ".join(f"{m:>14}" for m in METHODS)
    print(header)
    for e in all_eps:
        row = f"{e:>5}"
        for m in METHODS:
            mu, sd = sharp_loss[m].get(e, (np.nan, 0.0))
            row += f"  {mu:>6.3f} ± {sd:.3f}"
        print(row)

    # ---------- Make the figures ----------
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1 — main summary (the one for your paper)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("3-way comparison: No Reg vs SAM vs L2", fontsize=14, fontweight='bold')

    panel_final_metric(axes[0, 0], acc_final,  "Test Accuracy (final)",  "Accuracy")
    panel_final_metric(axes[0, 1], loss_train, "Train Loss (final)",     "Loss")
    panel_final_metric(axes[0, 2], loss_test,  "Test Loss (final)",      "Loss")
    panel_grouped_eps (axes[1, 0], sharp_loss, "Max Loss in ε-nghb",     "Max loss")
    panel_grouped_eps (axes[1, 1], sharp_grad, "Max ‖∇L‖ in ε-nghb",     "Max grad norm")
    panel_curves      (axes[1, 2], test_curves,"Test Loss curve",        "Loss")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = outdir / "summary_3way.png"
    plt.savefig(out1, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f"\nSaved main summary -> {out1}")

    # Figure 2 — accuracy & loss curves over training steps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panel_curves(axes[0], train_curves, "Train loss vs step",    "Loss")
    panel_curves(axes[1], test_curves,  "Test loss vs step",     "Loss")
    panel_curves(axes[2], acc_curves,   "Test accuracy vs step", "Accuracy")
    plt.tight_layout()
    out2 = outdir / "curves_3way.png"
    plt.savefig(out2, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f"Saved curves         -> {out2}")

    # Figure 3 — sharpness comparison alone (Figure-4-style)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panel_grouped_eps(axes[0], sharp_loss, "Max Loss in ε-nghb", "Max loss")
    panel_grouped_eps(axes[1], sharp_grad, "Max ‖∇L‖ in ε-nghb", "Max grad norm")
    plt.tight_layout()
    out3 = outdir / "sharpness_3way.png"
    plt.savefig(out3, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f"Saved sharpness plot -> {out3}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--noreg", required=True)
    p.add_argument("--sam",   required=True)
    p.add_argument("--l2",    required=True)
    p.add_argument("--outdir", default="artifacts/dmo/results/plots/")
    args = p.parse_args()
    main(args.noreg, args.sam, args.l2, args.outdir)

# -*- coding: utf-8 -*-
"""
generate_baseline_plots.py
==========================
Produces the reviewer-requested 3-way comparison:

    AdamW + SAM   (explicit sharpness-aware training, NO learned optimizer)
    No Reg        (your learned optimizer, no regularization)
    with SAM      (your learned optimizer, SAM-regularized)

Reads:
    - AdamW+SAM JSON: evaluation_base_optimizer_*AdamW_SAM*_mlp_mnist_*.json
    - No Reg JSON:    evaluation_learned_optimizer_*DMO_adam_280424*.json
    - SAM LO JSON:    evaluation_learned_optimizer_*AdSAM_sgd_LO_140424*.json

Outputs to artifacts/dmo/results/plots/:
    baseline_01_test_accuracy.png
    baseline_02_sharpness.png        (max loss in eps-nghb)
    baseline_03_grad_norm.png
    baseline_04_summary_panel.png
    baseline_summary_table.txt
"""

from __future__ import annotations
import json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.path.expanduser("~/learn-reg"))
JSON_DIR = ROOT / "artifacts" / "dmo" / "results" / "json"
OUT_DIR = ROOT / "artifacts" / "dmo" / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Display label -> (filename glob fragment)
METHODS = ["AdamW + SAM", "No Reg", "with SAM"]
GLOBS = {
    "AdamW + SAM": ["*base_optimizer*AdamW_SAM*", "*base_optimizer*AdamW*SAM*", "*AdamW_SAM*"],
    "No Reg":      ["*learned_optimizer*DMO_adam_280424*"],
    "with SAM":    ["*learned_optimizer*AdSAM_sgd_LO_140424*"],
}
COLORS = {
    "AdamW + SAM": "#D17A22",   # orange - the explicit baseline
    "No Reg":      "#2E7D32",   # green
    "with SAM":    "#7CB342",   # light green
}


def latest_json(method):
    for frag in GLOBS[method]:
        matches = glob.glob(str(JSON_DIR / (frag + ".json")))
        if matches:
            return Path(max(matches, key=os.path.getmtime))
    return None


def load_all():
    print(f"Looking for eval JSONs in: {JSON_DIR}")
    payloads = {}
    for m in METHODS:
        p = latest_json(m)
        if p is None:
            print(f"  ! {m:14s} -> NOT FOUND (run its eval first)")
            continue
        print(f"  {m:14s} -> {p.name}")
        payloads[m] = json.load(open(p))
    return payloads


def _arr(payload, key):
    aliases = {
        "acc":  ["instances_acc_ts", "instances_acc"],
        "loss_tr": ["instances_loss_tr", "instances_loss_train_ts"],
        "loss_ts": ["instances_loss_ts", "instances_loss_test_ts"],
    }
    for c in aliases.get(key, [key]):
        if c in payload:
            return payload[c]
    return None


def final_val(payload, key):
    inst = _arr(payload, key)
    if inst is None:
        return float("nan"), 0.0
    finals = [r[-1] for r in inst if r]
    a = np.asarray(finals, float)
    return float(a.mean()), float(a.std())


def sharpness_by_eps(payload, key):
    out = {}
    for instance in payload.get(key, []):
        for _, eps_dict in instance.items():
            for eps_str, value in eps_dict.items():
                profile = value[0] if isinstance(value, (list, tuple)) else value
                if isinstance(profile, (list, tuple)) and len(profile) > 0:
                    out.setdefault(float(eps_str), []).append(float(max(profile)))
    return {e: (float(np.mean(v)), float(np.std(v))) for e, v in sorted(out.items())}


def main():
    payloads = load_all()
    methods = [m for m in METHODS if m in payloads]
    if not methods:
        print("ERROR: no JSONs found. Run the evals first.")
        return
    n = len(methods)

    # ---- Test accuracy bar ----
    acc = {m: final_val(payloads[m], "acc") for m in methods}
    fig, ax = plt.subplots(figsize=(max(6, n*1.6), 5))
    x = np.arange(n)
    means = [acc[m][0] for m in methods]
    stds  = [acc[m][1] for m in methods]
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[COLORS[m] for m in methods], edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontweight='bold')
    ax.set_ylabel('Test accuracy (final)')
    ax.set_title('Baseline comparison - Final Test Accuracy')
    ax.grid(axis='y', alpha=0.3)
    top = max(v for v in means if not np.isnan(v))
    for b, v in zip(bars, means):
        if not np.isnan(v):
            ax.text(b.get_x()+b.get_width()/2, v+top*0.02, f'{v:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_01_test_accuracy.png", dpi=160, bbox_inches='tight')
    plt.close()

    # ---- Sharpness (grouped bars across eps) ----
    sharp = {m: sharpness_by_eps(payloads[m], "max_loss_nghb_list") for m in methods}
    grads = {m: sharpness_by_eps(payloads[m], "max_grad_norm_nghb_list") for m in methods}
    all_eps = sorted({e for d in sharp.values() for e in d})

    def grouped(ax, data, ylabel, title):
        if not all_eps:
            ax.text(0.5, 0.5, "no eps data", ha='center', transform=ax.transAxes); return
        xs = np.arange(len(all_eps)); w = 0.8 / n
        for i, m in enumerate(methods):
            vals = [data[m].get(e, (np.nan, 0))[0] for e in all_eps]
            errs = [data[m].get(e, (np.nan, 0))[1] for e in all_eps]
            ax.bar(xs + (i - (n-1)/2)*w, vals, w, yerr=errs, capsize=3,
                   label=m, color=COLORS[m], edgecolor='black')
        ax.set_xticks(xs); ax.set_xticklabels([str(e) for e in all_eps])
        ax.set_xlabel(r'Perturbation radius $\varepsilon$')
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped(ax, sharp, 'Max loss', r'Max loss in $\varepsilon$-neighbourhood')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_02_sharpness.png", dpi=160, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped(ax, grads, 'Max grad norm', r'Max $\|\nabla L\|$ in $\varepsilon$-neighbourhood')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_03_grad_norm.png", dpi=160, bbox_inches='tight')
    plt.close()

    # ---- Combined panel ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Reviewer baseline - AdamW+SAM vs No Reg vs with SAM",
                 fontsize=14, fontweight='bold')
    bars = axes[0].bar(x, means, yerr=stds, capsize=6,
                       color=[COLORS[m] for m in methods], edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, fontweight='bold')
    axes[0].set_ylabel('Test accuracy'); axes[0].set_title('Final test accuracy')
    axes[0].grid(axis='y', alpha=0.3)
    for b, v in zip(bars, means):
        if not np.isnan(v):
            axes[0].text(b.get_x()+b.get_width()/2, v+top*0.02, f'{v:.4f}', ha='center', fontsize=9)
    grouped(axes[1], sharp, 'Max loss', r'Sharpness (max loss in $\varepsilon$-nghb)')
    grouped(axes[2], grads, 'Max grad norm', r'Max $\|\nabla L\|$ in $\varepsilon$-nghb')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_DIR / "baseline_04_summary_panel.png", dpi=160, bbox_inches='tight')
    plt.close()

    # ---- Summary table ----
    lines = ["=" * 80,
             f"{'Method':<16} {'TestAcc':<22} {'TestLoss':<22}",
             "-" * 80]
    for m in methods:
        am, asd = final_val(payloads[m], "acc")
        tm, tsd = final_val(payloads[m], "loss_ts")
        lines.append(f"{m:<16} {am:.4f} +/- {asd:.4f}     {tm:.4f} +/- {tsd:.4f}")
    lines.append("=" * 80)
    if all_eps:
        lines.append("\nMax loss in epsilon-neighbourhood:")
        lines.append("   eps      " + "".join(f"{m:<22}" for m in methods))
        for e in all_eps:
            row = f"   {e:<8}"
            for m in methods:
                mn, sd = sharp[m].get(e, (float('nan'), 0))
                row += f"{mn:.4f} +/- {sd:.4f}    "
            lines.append(row)
    table = "\n".join(lines)
    (OUT_DIR / "baseline_summary_table.txt").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
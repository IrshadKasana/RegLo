"""
plot_3bar_comparison.py
=======================
Produces the 3-way bar-graph comparison requested by the reviewer:

    No Reg   vs   SAM (optimizee + LO + LR_sam)   vs   L2 (optimizee + L2)

Reads three evaluation JSONs produced by `evaluate_optimizee.py`:
    artifacts/dmo/results/json/evaluation_learned_optimizer_*_DMO_adam_280424_*.json   <- No Reg
    artifacts/dmo/results/json/evaluation_learned_optimizer_*_AdSAM_sgd_LO_140424_*.json <- SAM
    artifacts/dmo/results/json/evaluation_learned_optimizer_*_L2_adam_LO_140424_*.json   <- L2

For each JSON it extracts:
    - test accuracy at completion
    - max-loss-in-neighbourhood across epsilon sweep (sharpness)
    - max-gradient-norm-in-neighbourhood across epsilon sweep

And produces a Figure-4-style multi-panel bar chart.

Usage
-----
    python notebooks/dmo/plot_3bar_comparison.py \
        --noreg path/to/noreg.json \
        --sam   path/to/sam.json   \
        --l2    path/to/l2.json    \
        --out   artifacts/dmo/results/plots/three_bar_comparison.png

If only one JSON per method is given, the script plots single instance values.
If multiple instances are present in `instances_*` arrays, it averages them
and shows the standard deviation as an error bar (matching the paper style).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["No Reg", "SAM", "L2"]
COLORS = {"No Reg": "#2E7D32", "SAM": "#7CB342", "L2": "#9575CD"}


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _load(path: str | os.PathLike) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _final_test_accuracy(payload: dict) -> tuple[float, float]:
    """Return (mean, std) of last test accuracy across instances."""
    instances = payload.get("instances_acc_ts", [])
    finals = [row[-1] for row in instances if row]
    if not finals:
        return float("nan"), 0.0
    arr = np.asarray(finals, dtype=float)
    return float(arr.mean()), float(arr.std())


def _sharpness_by_eps(payload: dict, key: str) -> dict[float, tuple[float, float]]:
    """
    For each epsilon in the PGD sweep, return the maximum loss/grad-norm
    encountered along the perturbation trajectory, averaged across instances.

    `key` must be 'max_loss_nghb_list' or 'max_grad_norm_nghb_list'.
    Returns {eps: (mean_of_max, std_of_max)}.
    """
    out: dict[float, list[float]] = {}
    for instance in payload.get(key, []):
        # `instance` is dict keyed by best_iter -> dict[eps -> (profile_list, baseline)]
        for _best_iter, eps_dict in instance.items():
            for eps_str, value in eps_dict.items():
                # value is [profile_list, baseline_scalar]
                profile = value[0] if isinstance(value, (list, tuple)) else value
                if isinstance(profile, (list, tuple)) and len(profile) > 0:
                    out.setdefault(float(eps_str), []).append(float(max(profile)))
    return {eps: (float(np.mean(v)), float(np.std(v))) for eps, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_three_bar_comparison(noreg_path: str, sam_path: str, l2_path: str,
                              out_path: str = "three_bar_comparison.png",
                              rho_label: str = "0.01") -> None:
    payloads = {
        "No Reg": _load(noreg_path),
        "SAM":    _load(sam_path),
        "L2":     _load(l2_path),
    }

    # ---- gather metrics ----
    acc_mean = {m: _final_test_accuracy(p)[0] for m, p in payloads.items()}
    acc_std  = {m: _final_test_accuracy(p)[1] for m, p in payloads.items()}

    sharp = {m: _sharpness_by_eps(p, "max_loss_nghb_list")     for m, p in payloads.items()}
    grads = {m: _sharpness_by_eps(p, "max_grad_norm_nghb_list") for m, p in payloads.items()}

    all_eps = sorted({e for d in sharp.values() for e in d})
    if not all_eps:
        raise RuntimeError("No epsilon sweep found in any JSON. "
                           "Make sure `hyper_eps: True` in optimizee_eval.yaml.")

    # ---- figure layout: 1 row x 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"3-way comparison  —  No Reg vs SAM vs L2  (ρ = {rho_label})",
                 fontsize=13, fontweight='bold')

    # ----- panel 1: final test accuracy -----
    ax = axes[0]
    x = np.arange(len(METHODS))
    means = [acc_mean[m] for m in METHODS]
    stds  = [acc_std[m]  for m in METHODS]
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[COLORS[m] for m in METHODS], edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontweight='bold')
    ax.set_ylabel('Test accuracy (final)')
    ax.set_title('Test accuracy at completion')
    ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + max(means) * 0.02,
                f'{v:.3f}', ha='center', fontsize=9)

    # ----- panel 2: max loss in neighbourhood vs epsilon -----
    ax = axes[1]
    width = 0.25
    xs = np.arange(len(all_eps))
    for i, m in enumerate(METHODS):
        vals = [sharp[m].get(e, (np.nan, 0.0))[0] for e in all_eps]
        errs = [sharp[m].get(e, (np.nan, 0.0))[1] for e in all_eps]
        ax.bar(xs + (i - 1) * width, vals, width,
               yerr=errs, capsize=3,
               label=m, color=COLORS[m], edgecolor='black')
    ax.set_xticks(xs); ax.set_xticklabels([str(e) for e in all_eps])
    ax.set_xlabel('Perturbation radius ε')
    ax.set_ylabel('Max loss in neighbourhood')
    ax.set_title('Sharpness: Max loss vs ε')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ----- panel 3: max grad norm in neighbourhood vs epsilon -----
    ax = axes[2]
    for i, m in enumerate(METHODS):
        vals = [grads[m].get(e, (np.nan, 0.0))[0] for e in all_eps]
        errs = [grads[m].get(e, (np.nan, 0.0))[1] for e in all_eps]
        ax.bar(xs + (i - 1) * width, vals, width,
               yerr=errs, capsize=3,
               label=m, color=COLORS[m], edgecolor='black')
    ax.set_xticks(xs); ax.set_xticklabels([str(e) for e in all_eps])
    ax.set_xlabel('Perturbation radius ε')
    ax.set_ylabel('Max ‖∇L‖ in neighbourhood')
    ax.set_title('Max gradient norm vs ε')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved 3-bar comparison plot to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--noreg", required=True, help="JSON for No-Reg LO evaluation")
    p.add_argument("--sam",   required=True, help="JSON for SAM-regularized LO evaluation")
    p.add_argument("--l2",    required=True, help="JSON for L2-regularized LO evaluation")
    p.add_argument("--out",   default="artifacts/dmo/results/plots/three_bar_comparison.png",
                   help="Output PNG path")
    p.add_argument("--rho",   default="0.01", help="ρ label for the title")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_three_bar_comparison(args.noreg, args.sam, args.l2,
                              out_path=args.out, rho_label=args.rho)
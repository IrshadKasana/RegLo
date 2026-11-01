#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_paper_plots.py
=======================
One script for the full revision figure set. Reads every eval JSON in
artifacts/dmo/results/json/ (new metadata-rich format written by the updated
optimizee_train.py; old-format JSONs still work for the basic plots) and
writes into artifacts/dmo/results/plots/paper/:

    01_test_accuracy.png        final test accuracy bars
    02_final_losses.png         final train + test loss bars
    03_generalization_gap.png   acc-gap and loss-gap bars   (NEW metric)
    04_loss_curves.png          train/test loss vs step, mean +/- std
    05_accuracy_curves.png      train/test accuracy vs step (NEW: train curve)
    06_sharpness.png            max loss in eps-nghb, grouped by eps
    07_grad_norm.png            max ||grad L|| in eps-nghb
    08_hessian_spectrum.png     lambda_max bars + top-k spectrum  (NEW)
    09_efficiency.png           ms/step, peak GPU mem, acc-vs-time Pareto (NEW)
    10_crossdataset_heatmap.png meta-train dataset x eval dataset acc  (NEW)
    11_sweep_heatmap.png        smooth_lamb x reg-strength acc heatmap (NEW)
    12_summary_panel.png        headline panel
    paper_summary_table.txt

Usage:
    python generate_paper_plots.py                     # everything it can find
    python generate_paper_plots.py --dataset cifar10 --model resnet18
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.path.expanduser("~/learn-reg"))
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "artifacts" / "dmo" / "results" / "json"
CKPT_DIR = ROOT / "artifacts" / "dmo" / "checkpoints"
OUT_DIR = ROOT / "artifacts" / "dmo" / "results" / "plots" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# EDIT ME: pretty labels for your LO checkpoints (fallback = raw name)
# ------------------------------------------------------------------
LO_LABELS = {
    "DMO_adam_280424":      "LO (No Reg)",
    "AdSAM_sgd_LO_140424":  "LO + SAM",
    "L2_adam_LO_140424":    "LO + L2",
    "AdSAMopzee_sgd_LO_140424": "LO + SAM(opzee)",
    "testpGAM_sgd_LO_300424":   "LO + GAM",
    "testpGSAM_sgd_LO_300424":  "LO + GSAM",
}

PALETTE = ["#2E7D32", "#7CB342", "#9575CD", "#D17A22", "#1565C0",
           "#C62828", "#00838F", "#6D4C41", "#546E7A", "#AD1457"]


# ================= loading & labeling =================

def load_records():
    records = []
    for p in sorted(glob.glob(str(JSON_DIR / "*.json"))):
        try:
            payload = json.load(open(p))
        except Exception as exc:
            print(f"  ! skipping unreadable {Path(p).name}: {exc}")
            continue
        meta = payload.get("metadata", {})
        if not meta:  # legacy JSON: infer what we can from the filename
            name = Path(p).name
            meta = {"legacy": True, "dataset": _guess(name, ["mnist", "fmnist", "cifar100", "cifar10"]),
                    "optimizee_model": _guess(name, ["mlp_Relu", "big_conv", "resnet18", "resnet8",
                                                     "vgg16", "vgg11", "conv", "mlp"]),
                    "if_normal_optim": "base_optimizer" in name,
                    "saved_model": _legacy_saved_model(name),
                    "regularizer": _guess(name, ["GSAM", "GAM", "SAM", "noREG"]) or "?",
                    "lo_train_dataset": None, "eval_tag": None}
        records.append({"path": Path(p), "meta": meta, "payload": payload})
    print(f"Loaded {len(records)} result JSONs from {JSON_DIR}")
    return records


def _guess(name, options):
    for o in options:
        if o in name:
            return o
    return None


def _legacy_saved_model(name):
    for k in LO_LABELS:
        if k in name:
            return k
    return None


def label_of(rec):
    m = rec["meta"]
    if m.get("if_normal_optim"):
        reg = (m.get("regularizer") or "").replace("adamw_sam_baseline", "SAM").upper()
        base = (m.get("base_optimizer") or "SGD")
        lbl = f"{base}+{reg}" if reg not in ("NO_REG", "NOREG", "?") else base
    else:
        sm = m.get("saved_model")
        lbl = LO_LABELS.get(sm, f"LO:{sm}")
        if m.get("lo_train_dataset") and m.get("lo_train_dataset") != m.get("dataset"):
            lbl += f" [{m['lo_train_dataset']}->{m['dataset']}]"
    return lbl


def is_sweep(rec):
    tag = rec["meta"].get("eval_tag") or ""
    return tag.startswith("sw_")


# ================= metric extraction =================

def series(payload, key):
    aliases = {
        "acc_ts": ["instances_acc_ts", "instances_acc"],
        "acc_tr": ["instances_acc_tr"],
        "loss_tr": ["instances_loss_tr", "instances_loss_train_ts"],
        "loss_ts": ["instances_loss_ts", "instances_loss_test_ts"],
    }
    for c in aliases.get(key, [key]):
        if c in payload:
            return payload[c]
    return None


def final_stat(payload, key):
    inst = series(payload, key)
    if not inst:
        return np.nan, 0.0, []
    finals = [r[-1] for r in inst if r]
    a = np.asarray(finals, float)
    return float(a.mean()), float(a.std()), finals


def mean_curve(payload, key):
    inst = series(payload, key)
    if not inst:
        return np.array([]), np.array([])
    valid = [r for r in inst if r]
    if not valid:
        return np.array([]), np.array([])
    n = min(len(r) for r in valid)
    arr = np.asarray([r[:n] for r in valid], float)
    return arr.mean(axis=0), arr.std(axis=0)


def sharpness_by_eps(payload, key):
    out = defaultdict(list)
    for instance in payload.get(key, []):
        if not isinstance(instance, dict):
            continue
        for _, eps_dict in instance.items():
            if not isinstance(eps_dict, dict):
                continue
            for eps_str, value in eps_dict.items():
                profile = value[0] if isinstance(value, (list, tuple)) else value
                if isinstance(profile, (list, tuple)) and profile:
                    out[float(eps_str)].append(float(max(profile)))
    return {e: (float(np.mean(v)), float(np.std(v))) for e, v in sorted(out.items())}


def gen_gap(payload):
    gg = payload.get("generalization_gap")
    if not gg:
        return None
    acc = np.asarray([g["acc_gap"] for g in gg], float)
    lss = np.asarray([g["loss_gap"] for g in gg], float)
    return (acc.mean(), acc.std(), lss.mean(), lss.std())


def hessian_lmax(payload):
    """eigen_values: list per instance of {best_iter: [eigvals...]}"""
    lmaxes, spectra = [], []
    for inst in payload.get("eigen_values", []) or []:
        if not isinstance(inst, dict):
            continue
        for _, ev in inst.items():
            ev = sorted([float(v) for v in ev], reverse=True)
            if ev:
                lmaxes.append(ev[0])
                spectra.append(ev)
    if not lmaxes:
        return None
    return float(np.mean(lmaxes)), float(np.std(lmaxes)), spectra


def efficiency(payload):
    return payload.get("efficiency") or None


# ================= plotting helpers =================

def colors_for(labels):
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labels)}


def bar(ax, labels, means, stds, cmap, ylabel, title, fmt="{:.4f}"):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[cmap[l] for l in labels], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    finite = [v for v in means if np.isfinite(v)]
    top = max(finite) if finite else 1.0
    for b, v in zip(bars, means):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v + abs(top) * 0.02,
                    fmt.format(v), ha="center", fontsize=8)


def grouped_eps(ax, labels, data, cmap, ylabel, title):
    all_eps = sorted({e for d in data.values() for e in d})
    if not all_eps:
        ax.text(0.5, 0.5, "no eps data", ha="center", transform=ax.transAxes)
        return
    xs = np.arange(len(all_eps))
    w = 0.8 / max(len(labels), 1)
    for i, l in enumerate(labels):
        vals = [data[l].get(e, (np.nan, 0))[0] for e in all_eps]
        errs = [data[l].get(e, (np.nan, 0))[1] for e in all_eps]
        ax.bar(xs + (i - (len(labels) - 1) / 2) * w, vals, w, yerr=errs,
               capsize=2, label=l, color=cmap[l], edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(e) for e in all_eps])
    ax.set_xlabel(r"Perturbation radius $\varepsilon$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / name}")


# ================= main figure builders =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="filter main comparison to this eval dataset")
    ap.add_argument("--model", default=None, help="filter main comparison to this optimizee model")
    cli = ap.parse_args()

    records = load_records()
    if not records:
        print("No result JSONs found - run evaluations first.")
        return

    main_recs = [r for r in records if not is_sweep(r)]

    # pick the (dataset, model) combo to compare on, unless user specified
    if cli.dataset is None or cli.model is None:
        counts = defaultdict(int)
        for r in main_recs:
            counts[(r["meta"].get("dataset"), r["meta"].get("optimizee_model"))] += 1
        (ds, mdl), _ = max(counts.items(), key=lambda kv: kv[1])
        ds = cli.dataset or ds
        mdl = cli.model or mdl
    else:
        ds, mdl = cli.dataset, cli.model
    print(f"Main comparison on: dataset={ds}, model={mdl}")

    comp = [r for r in main_recs
            if r["meta"].get("dataset") == ds and r["meta"].get("optimizee_model") == mdl]
    # newest JSON per label wins
    by_label = {}
    for r in sorted(comp, key=lambda r: r["path"].stat().st_mtime):
        by_label[label_of(r)] = r
    labels = list(by_label)
    cmap = colors_for(labels)
    P = {l: by_label[l]["payload"] for l in labels}

    if labels:
        # ---- 01 accuracy ----
        acc = {l: final_stat(P[l], "acc_ts") for l in labels}
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        bar(ax, labels, [acc[l][0] for l in labels], [acc[l][1] for l in labels],
            cmap, "Test accuracy (final)", f"Final test accuracy - {mdl} / {ds}")
        save(fig, "01_test_accuracy.png")

        # ---- 02 final losses ----
        ltr = {l: final_stat(P[l], "loss_tr") for l in labels}
        lts = {l: final_stat(P[l], "loss_ts") for l in labels}
        fig, axes = plt.subplots(1, 2, figsize=(max(11, len(labels) * 2.6), 5))
        bar(axes[0], labels, [ltr[l][0] for l in labels], [ltr[l][1] for l in labels],
            cmap, "Loss", "Train loss (final)")
        bar(axes[1], labels, [lts[l][0] for l in labels], [lts[l][1] for l in labels],
            cmap, "Loss", "Test loss (final)")
        save(fig, "02_final_losses.png")

        # ---- 03 generalization gap ----
        gaps = {l: gen_gap(P[l]) for l in labels}
        gl = [l for l in labels if gaps[l] is not None]
        if gl:
            fig, axes = plt.subplots(1, 2, figsize=(max(11, len(gl) * 2.6), 5))
            bar(axes[0], gl, [gaps[l][0] for l in gl], [gaps[l][1] for l in gl],
                cmap, r"$\mathrm{Acc}_{train} - \mathrm{Acc}_{test}$",
                "Generalization gap (accuracy)")
            bar(axes[1], gl, [gaps[l][2] for l in gl], [gaps[l][3] for l in gl],
                cmap, r"$L_{test} - L_{train}$", "Generalization gap (loss)")
            save(fig, "03_generalization_gap.png")

        # ---- 04 loss curves ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for l in labels:
            for ax, key, ttl in ((axes[0], "loss_tr", "Train loss"),
                                 (axes[1], "loss_ts", "Test loss")):
                m, s = mean_curve(P[l], key)
                if m.size:
                    xs = np.arange(m.size)
                    ax.plot(xs, m, label=l, color=cmap[l])
                    ax.fill_between(xs, m - s, m + s, color=cmap[l], alpha=0.15)
        for ax, ttl in zip(axes, ("Train loss over steps", "Test loss over steps")):
            ax.set_xlabel("Step"); ax.set_ylabel("Loss")
            ax.set_title(ttl); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        save(fig, "04_loss_curves.png")

        # ---- 05 accuracy curves (train + test) ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for l in labels:
            for ax, key in ((axes[0], "acc_tr"), (axes[1], "acc_ts")):
                m, s = mean_curve(P[l], key)
                if m.size:
                    xs = np.arange(m.size)
                    ax.plot(xs, m, label=l, color=cmap[l])
                    ax.fill_between(xs, m - s, m + s, color=cmap[l], alpha=0.15)
        for ax, ttl in zip(axes, ("Train accuracy over steps", "Test accuracy over steps")):
            ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
            ax.set_title(ttl); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        save(fig, "05_accuracy_curves.png")

        # ---- 06 / 07 sharpness ----
        sharp = {l: sharpness_by_eps(P[l], "max_loss_nghb_list") for l in labels}
        gnorm = {l: sharpness_by_eps(P[l], "max_grad_norm_nghb_list") for l in labels}
        fig, ax = plt.subplots(figsize=(9, 5))
        grouped_eps(ax, labels, sharp, cmap, "Max loss",
                    r"Sharpness at convergence: $\max_{\|\epsilon\|\leq\rho} L(\theta+\epsilon)$")
        save(fig, "06_sharpness.png")
        fig, ax = plt.subplots(figsize=(9, 5))
        grouped_eps(ax, labels, gnorm, cmap, "Max grad norm",
                    r"$\max_{\|\epsilon\|\leq\rho}\|\nabla L(\theta+\epsilon)\|$")
        save(fig, "07_grad_norm.png")

        # ---- 08 Hessian spectrum ----
        hs = {l: hessian_lmax(P[l]) for l in labels}
        hl = [l for l in labels if hs[l] is not None]
        if hl:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            bar(axes[0], hl, [hs[l][0] for l in hl], [hs[l][1] for l in hl],
                cmap, r"$\lambda_{max}(H)$", "Top Hessian eigenvalue at convergence",
                fmt="{:.2f}")
            for l in hl:
                spec = np.mean([s[:min(len(x) for x in hs[l][2])]
                                for s in [np.asarray(x) for x in hs[l][2]]], axis=0)
                axes[1].plot(np.arange(1, spec.size + 1), spec, marker="o",
                             ms=3, label=l, color=cmap[l])
            axes[1].set_xlabel("Eigenvalue index (descending)")
            axes[1].set_ylabel("Eigenvalue")
            axes[1].set_title("Top-k Hessian spectrum")
            axes[1].set_yscale("symlog")
            axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
            save(fig, "08_hessian_spectrum.png")
        else:
            print("  (no Hessian data - rerun eval with --set calc_hessian=True)")

        # ---- 09 efficiency ----
        eff = {l: efficiency(P[l]) for l in labels}
        el = [l for l in labels if eff[l] and eff[l].get("mean_step_time_ms")]
        if el:
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            bar(axes[0], el, [eff[l]["mean_step_time_ms"] for l in el],
                [eff[l].get("std_step_time_ms", 0) or 0 for l in el],
                cmap, "ms / step", "Wall-clock per optimization step", fmt="{:.1f}")
            mems = [np.mean(eff[l].get("peak_gpu_mem_mb") or [np.nan]) for l in el]
            bar(axes[1], el, mems, [0] * len(el), cmap, "Peak GPU memory (MB)",
                "Peak GPU memory", fmt="{:.0f}")
            for l in el:
                wc = np.mean(eff[l].get("total_wallclock_sec_per_instance") or [np.nan])
                a, astd, _ = final_stat(P[l], "acc_ts")
                axes[2].errorbar(wc, a, yerr=astd, fmt="o", ms=9,
                                 color=cmap[l], label=l, capsize=4)
            axes[2].set_xlabel("Total wall-clock per run (s)")
            axes[2].set_ylabel("Final test accuracy")
            axes[2].set_title("Accuracy vs compute (Pareto)")
            axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
            save(fig, "09_efficiency.png")

        # ---- 12 summary panel ----
        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        fig.suptitle(f"Summary - {mdl} / {ds}", fontsize=15, fontweight="bold")
        bar(axes[0, 0], labels, [acc[l][0] for l in labels],
            [acc[l][1] for l in labels], cmap, "Accuracy", "Test accuracy (final)")
        bar(axes[0, 1], labels, [ltr[l][0] for l in labels],
            [ltr[l][1] for l in labels], cmap, "Loss", "Train loss (final)")
        bar(axes[0, 2], labels, [lts[l][0] for l in labels],
            [lts[l][1] for l in labels], cmap, "Loss", "Test loss (final)")
        grouped_eps(axes[1, 0], labels, sharp, cmap, "Max loss",
                    r"Max loss in $\varepsilon$-nghb")
        grouped_eps(axes[1, 1], labels, gnorm, cmap, "Max grad norm",
                    r"Max $\|\nabla L\|$ in $\varepsilon$-nghb")
        for l in labels:
            m, s = mean_curve(P[l], "loss_ts")
            if m.size:
                xs = np.arange(m.size)
                axes[1, 2].plot(xs, m, label=l, color=cmap[l])
        axes[1, 2].set_title("Test loss over steps")
        axes[1, 2].set_xlabel("Step"); axes[1, 2].set_ylabel("Loss")
        axes[1, 2].legend(fontsize=7); axes[1, 2].grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT_DIR / "12_summary_panel.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {OUT_DIR / '12_summary_panel.png'}")

        # ---- summary table ----
        lines = ["=" * 110,
                 f"{'Method':<28}{'TestAcc':<20}{'TestLoss':<20}{'AccGap':<16}{'ms/step':<12}{'PeakMB':<10}",
                 "-" * 110]
        for l in labels:
            a, asd, _ = acc[l]
            t, tsd, _ = lts[l]
            g = gaps.get(l)
            e = eff.get(l)
            gap_s = "{:+.4f}".format(g[0]) if g else "  n/a "
            ms_s = "{:.1f}".format(e["mean_step_time_ms"]) if e and e.get("mean_step_time_ms") else "n/a"
            mem_s = "{:.0f}".format(np.mean(e["peak_gpu_mem_mb"])) if e and e.get("peak_gpu_mem_mb") else "n/a"
            lines.append(
                f"{l:<28}{a:.4f} +/- {asd:.4f}   {t:.4f} +/- {tsd:.4f}   "
                f"{gap_s:<16}{ms_s:<12}{mem_s:<10}")
        lines.append("=" * 110)
        (OUT_DIR / "paper_summary_table.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))

    # ---- 10 cross-dataset generalization heatmap ----
    xrecs = [r for r in main_recs
             if not r["meta"].get("if_normal_optim") and r["meta"].get("lo_train_dataset")]
    if xrecs:
        by_lo = defaultdict(list)
        for r in xrecs:
            by_lo[LO_LABELS.get(r["meta"].get("saved_model"),
                                str(r["meta"].get("saved_model")))].append(r)
        n = len(by_lo)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        for ax, (lo, recs) in zip(axes[0], sorted(by_lo.items())):
            rows = sorted({r["meta"]["lo_train_dataset"] for r in recs})
            cols = sorted({r["meta"]["dataset"] for r in recs})
            mat = np.full((len(rows), len(cols)), np.nan)
            for r in recs:
                a, _, _ = final_stat(r["payload"], "acc_ts")
                mat[rows.index(r["meta"]["lo_train_dataset"]),
                    cols.index(r["meta"]["dataset"])] = a
            im = ax.imshow(mat, cmap="YlGn", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
            ax.set_xlabel("Meta-test dataset"); ax.set_ylabel("Meta-train dataset")
            ax.set_title(f"Cross-dataset transfer: {lo}")
            for i in range(len(rows)):
                for j in range(len(cols)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                                fontsize=9, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
        save(fig, "10_crossdataset_heatmap.png")
    else:
        print("  (no cross-dataset runs found - use --lo_train_dataset at eval time)")

    # ---- 11 sweep heatmap ----
    sweeps = [r for r in records if is_sweep(r)]
    if sweeps:
        pat = re.compile(r"sw_(?P<reg>[a-z0-9_]+)_sl(?P<sl>[0-9pm]+)_(?P<key>[a-z_]+?)(?P<rv>[0-9pm]+)$")
        by_reg = defaultdict(dict)
        for r in sweeps:
            m = pat.match(r["meta"].get("eval_tag") or "")
            if not m:
                continue
            dec = lambda s: float(s.replace('p', '.').replace('m', '-'))
            a, _, _ = final_stat(r["payload"], "acc_ts")
            by_reg[(m["reg"], m["key"])][(dec(m["sl"]), dec(m["rv"]))] = a
        n = len(by_reg)
        if n:
            fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5), squeeze=False)
            for ax, ((reg, key), grid) in zip(axes[0], sorted(by_reg.items())):
                sls = sorted({k[0] for k in grid})
                rvs = sorted({k[1] for k in grid})
                mat = np.full((len(sls), len(rvs)), np.nan)
                for (sl, rv), a in grid.items():
                    mat[sls.index(sl), rvs.index(rv)] = a
                im = ax.imshow(mat, cmap="viridis")
                ax.set_xticks(range(len(rvs))); ax.set_xticklabels(rvs)
                ax.set_yticks(range(len(sls))); ax.set_yticklabels(sls)
                ax.set_xlabel(key); ax.set_ylabel(r"$\lambda_{smooth}$")
                ax.set_title(f"Sweep: {reg} (test acc)")
                for i in range(len(sls)):
                    for j in range(len(rvs)):
                        if np.isfinite(mat[i, j]):
                            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                                    fontsize=8, color="white", fontweight="bold")
                fig.colorbar(im, ax=ax, fraction=0.046)
            save(fig, "11_sweep_heatmap.png")
    else:
        print("  (no sweep runs found - use run_sweep.py)")

    print(f"\nAll figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()

"""
generate_paper_plots.py (v2 — metadata-driven)
==============================================
One complete, uniform report per experiment (or per dataset/model combo),
built from result METADATA — never from filename patterns (reviewer item 4).

What it reads
-------------
* every evaluation JSON in  artifacts/dmo/results/json/   (LO rows AND the
  AdamW+SAM benchmark rows — identical structure, so the baseline appears
  in EVERY figure automatically)
* every timing/FLOPs benchmark in  results/benchmark/*.json
  (benchmark_compute.py output — ingested into the efficiency figure and
  the summary table; reviewer STEP 11)

Figures (fixed method ordering everywhere; reviewer STEPs 2, 3, 7)
------------------------------------------------------------------
01 final test accuracy          02 final test loss
03 test-accuracy curves         04 test-loss curves
05 validation curves            06 sharpness (max loss)  vs radius
07 max grad-norm vs radius      08 Hessian lambda_max + spectra
09 efficiency (+ benchmark)     10 cross-DATASET heatmaps
11 sweep heatmap                12 summary panel  (the reviewer-image layout)
13 convergence (train loss)     14 cross-ARCHITECTURE heatmaps
15 summary table (csv+md)       16 CIFAR-10-C robustness

Usage
-----
    python generate_paper_plots.py --experiment exp1
    python generate_paper_plots.py --dataset mnist --model mlp
    python generate_paper_plots.py            # auto-pick densest combo
"""
import argparse
import csv
import glob
import json
import os
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
BENCH_DIR = ROOT / "results" / "benchmark"
OUT_BASE = ROOT / "artifacts" / "dmo" / "results" / "plots" / "paper"

# ---------------- fixed labels & ordering (STEPs 2+3) ----------------
# The 8 methods of the common comparison setup, in fixed figure order.
METHOD_ORDER = ["AdamW", "AdamW + SAM", "AdamW + GAM", "AdamW + GSAM",
                "No-Reg LO", "SAM LO", "GAM LO", "GSAM LO",
                "SGD + SAM", "SAM-opzee LO"]

# "Compare corresponding pairs only": each set = LO No-Reg vs LO+R vs AdamW+R
COMPARISON_SETS = {
    "sam":  ["No-Reg LO", "SAM LO", "AdamW + SAM"],
    "gam":  ["No-Reg LO", "GAM LO", "AdamW + GAM"],
    "gsam": ["No-Reg LO", "GSAM LO", "AdamW + GSAM"],
}
LO_REG_LABEL = {"no_reg": "No-Reg LO", "sam": "SAM LO", "gam": "GAM LO",
                "gsam": "GSAM LO", "sam_opzee": "SAM-opzee LO"}
BENCH_METHOD_LABEL = {"adamw_sam": "AdamW + SAM", "no_reg": "No-Reg LO",
                      "sam": "SAM LO", "gam": "GAM LO", "gsam": "GSAM LO"}
LO_LABELS = {  # legacy checkpoint names (2024 paper artifacts)
    "DMO_adam_280424": "No-Reg LO", "AdSAM_sgd_LO_140424": "SAM LO",
    "L2_adam_LO_140424": "L2 LO", "AdSAMopzee_sgd_LO_140424": "SAM-opzee LO",
    "testpGAM_sgd_LO_300424": "GAM LO", "testpGSAM_sgd_LO_300424": "GSAM LO",
}
PALETTE = {"AdamW + SAM": "#D17A22", "SGD + SAM": "#B8860B",
           "No-Reg LO": "#2E7D32", "SAM LO": "#7CB342", "GAM LO": "#1565C0",
           "GSAM LO": "#9575CD", "L2 LO": "#00838F", "SAM-opzee LO": "#6D4C41"}
FALLBACK = ["#C62828", "#546E7A", "#AD1457", "#4E342E"]


def order_labels(labels):
    known = [m for m in METHOD_ORDER if m in labels]
    rest = sorted(l for l in labels if l not in METHOD_ORDER)
    return known + rest


def colors_for(labels):
    out, i = {}, 0
    for l in labels:
        base = next((PALETTE[k] for k in PALETTE if l.startswith(k)), None)
        if base is None:
            base = FALLBACK[i % len(FALLBACK)]; i += 1
        out[l] = base
    return out


# ---------------- loading & labelling ----------------

def load_records():
    records = []
    for p in sorted(glob.glob(str(JSON_DIR / "*.json"))):
        if Path(p).name.startswith("partial_"):
            continue
        try:
            payload = json.load(open(p))
        except Exception as exc:
            print(f"  ! skipping unreadable {Path(p).name}: {exc}")
            continue
        if "metadata" not in payload and "completed" in payload:
            # walltime PARTIAL dump (ev_pt_*_sN.json): resumable state, not
            # a finished result — the old 'partial_' prefix check never
            # matched this naming, so partials leaked into the "all" pool
            # as legacy records (fixed 2026-07-31).
            continue
        meta = payload.get("metadata") or _legacy_meta(Path(p).name)
        records.append({"path": Path(p), "meta": meta, "payload": payload})
    print(f"Loaded {len(records)} evaluation JSONs from {JSON_DIR}")
    return records


def _legacy_meta(name):
    def guess(opts):
        return next((o for o in opts if o in name), None)
    return {"legacy": True,
            "dataset": guess(["fmnist", "cifar100", "cifar10", "mnist"]),
            "optimizee_model": guess(["mlp_Relu", "resnet18", "resnet8",
                                      "vgg16", "mlp"]),
            "if_normal_optim": "base_optimizer" in name,
            "base_optimizer": "AdamW" if "AdamW" in name else "sgd",
            "saved_model": next((k for k in LO_LABELS if k in name), None),
            "regularizer": "adamw_sam" if "base_optimizer" in name else None,
            "lo_regularizer": None, "train_dataset": None,
            "test_dataset": guess(["fmnist", "cifar100", "cifar10", "mnist"]),
            "train_architecture": None, "test_architecture": None,
            "eval_tag": None, "experiment_name": None}


def label_of(rec):
    m = rec["meta"]
    if m.get("if_normal_optim"):
        base = (m.get("base_optimizer") or "sgd")
        reg = (m.get("regularizer") or "")
        r = reg.lower()
        # order matters: 'gsam' contains 'sam' and 'gam' contains 'am'
        for key, pretty in (("gsam", "GSAM"), ("gam", "GAM"), ("sam", "SAM")):
            if key in r:
                return (f"AdamW + {pretty}" if base.lower() == "adamw"
                        else f"{base.upper()} + {pretty}")
        # method 8: plain AdamW. Accept the family name AND the raw config
        # name (older JSONs recorded 'adamw_no_reg_baseline' before the
        # family normalisation landed).
        if r in ("no_reg", "none", "") or "no_reg" in r:
            return "AdamW" if base.lower() == "adamw" else base.upper()
        return f"{base}+{reg}"
    lr = (m.get("lo_regularizer") or "").lower()
    if lr in LO_REG_LABEL:
        return LO_REG_LABEL[lr]
    sm = m.get("saved_model")
    return LO_LABELS.get(sm, f"LO:{sm}")


def is_sweep(rec):
    return (rec["meta"].get("eval_tag") or "").startswith("sw_")


def is_cross(rec):
    m = rec["meta"]
    ds_x = m.get("train_dataset") and m.get("test_dataset") and \
        m["train_dataset"] != m["test_dataset"]
    ar_x = m.get("train_architecture") and m.get("test_architecture") and \
        m["train_architecture"] != m["test_architecture"]
    return bool(ds_x or ar_x)


def pick_per_label(recs):
    """One record per label: most instances wins, then newest (item 52)."""
    by = defaultdict(list)
    for r in recs:
        by[label_of(r)].append(r)
    out = {}
    for l, rs in by.items():
        def key(r):
            n = len(r["payload"].get("instance_seeds") or
                    r["payload"].get("instances_loss_ts") or [])
            return (n, r["path"].stat().st_mtime)
        rs.sort(key=key)
        best = rs[-1]
        ns = {key(r)[0] for r in rs}
        if len(ns) > 1:
            print(f"  [warn] label '{l}': mixed n_training {sorted(ns)} — "
                  f"using the run with {key(best)[0]} instances")
        out[l] = best
    return out


# ---------------- metric extraction ----------------

def series(payload, key):
    aliases = {"acc_ts": ["instances_acc_ts"], "acc_tr": ["instances_acc_tr"],
               "loss_tr": ["instances_loss_tr"], "loss_ts": ["instances_loss_ts"],
               "loss_val": ["instances_loss_val"], "acc_val": ["instances_acc_val"]}
    for c in aliases.get(key, [key]):
        if c in payload and payload[c]:
            return payload[c]
    return None


def _ffill(row):
    row = list(row)
    last = None
    for i, v in enumerate(row):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[i] = last if last is not None else np.nan
        else:
            last = row[i]
    return row


def mean_curve(payload, key, ffill=False):
    inst = series(payload, key)
    if not inst:
        return np.array([]), np.array([])
    valid = [(_ffill(r) if ffill else r) for r in inst if r]
    if not valid:
        return np.array([]), np.array([])
    n = min(len(r) for r in valid)
    arr = np.asarray([r[:n] for r in valid], float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def final_stat(payload, key):
    inst = series(payload, key)
    if not inst:
        return np.nan, 0.0
    finals = [r[-1] for r in inst if r]
    a = np.asarray(finals, float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def _norm_eps_node(node, default_eps):
    """node: {eps:(profile,actual)} | (profile,actual) | [profile,actual].
    Yield (eps, max_profile) pairs — handles hyper_eps AND single-eps
    shapes (Change.MD item 49)."""
    if isinstance(node, dict):
        for e, v in node.items():
            prof = v[0] if isinstance(v, (list, tuple)) and len(v) == 2 and \
                isinstance(v[0], (list, tuple)) else v
            if isinstance(prof, (list, tuple)) and prof:
                yield float(e), float(max(prof))
    elif isinstance(node, (list, tuple)) and node:
        prof = node[0] if isinstance(node[0], (list, tuple)) else node
        if isinstance(prof, (list, tuple)) and prof and default_eps is not None:
            yield float(default_eps), float(max(prof))


def sharpness_by_eps(rec, which, stage="at_completion"):
    """which: 'max_loss' | 'max_grad_norm'. Prefers the two-stage
    stage_properties_list records (items 37/50); falls back to the legacy
    payload dicts. Returns {eps: (mean, std)} across instances."""
    payload, meta = rec["payload"], rec["meta"]
    default_eps = (meta.get("args") or {}).get("eps") or \
        (meta.get("lo_hparams") or {}).get("rho")
    per_eps = defaultdict(list)
    used_stage = False
    for sp in payload.get("stage_properties_list") or []:
        node = (sp or {}).get(stage, {}).get(which)
        if node is not None:
            used_stage = True
            for e, v in _norm_eps_node(node, default_eps):
                per_eps[e].append(v)
    if not used_stage:   # legacy fallback
        legacy_key = {"max_loss": "max_loss_nghb_list",
                      "max_grad_norm": "max_grad_norm_nghb_list"}[which]
        for inst in payload.get(legacy_key) or []:
            if isinstance(inst, dict):
                for _, node in inst.items():
                    for e, v in _norm_eps_node(node, default_eps):
                        per_eps[e].append(v)
    return {e: (float(np.mean(v)), float(np.std(v)))
            for e, v in sorted(per_eps.items())}


def hessian_lmax(payload):
    """Mean/std of the top Hessian eigenvalue + the spectra.

    Returns None when there is nothing usable — e.g. --profile smoke sets
    calc_hessian=False, so every eigen_values entry is None. Previously this
    did `sorted([float(v) for v in ev])` on that None and every report job
    died with `TypeError: 'NoneType' object is not iterable`.
    """
    lmaxes, spectra = [], []
    for inst in payload.get("eigen_values") or []:
        if not isinstance(inst, dict):
            continue
        for _, ev in inst.items():
            if ev is None:                 # calc_hessian=False
                continue
            try:
                ev = sorted([float(v) for v in ev], reverse=True)
            except (TypeError, ValueError):
                continue                   # unusable record; skip, don't crash
            if ev:
                lmaxes.append(ev[0])
                spectra.append(ev)
    if not lmaxes:
        return None
    return float(np.mean(lmaxes)), float(np.std(lmaxes)), spectra

def gen_gap(payload):
    gg = payload.get("generalization_gap")
    if not gg:
        return None
    acc = np.asarray([g["acc_gap"] for g in gg], float)
    return float(acc.mean()), float(acc.std())


def opt_seconds(payload):
    st = payload.get("instances_step_times") or []
    tot = [sum(r) for r in st if r]
    return (float(np.mean(tot)), float(np.std(tot))) if tot else (np.nan, 0.0)


def peak_mem(payload):
    pm = payload.get("instances_peak_mem_mb") or []
    return (float(np.mean(pm)), float(np.std(pm))) if pm else (np.nan, 0.0)


def load_bench_rows(exp=None, ds=None, mdl=None):
    rows = []
    for p in sorted(glob.glob(str(BENCH_DIR / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if exp and Path(p).stem not in (exp,):
            continue
        for r in d.get("rows", []):
            if ds and r.get("dataset") != ds:
                continue
            if mdl and r.get("model") != mdl:
                continue
            r["label"] = BENCH_METHOD_LABEL.get(r["method"], r["method"])
            rows.append(r)
    return rows


# ---------------- plotting primitives ----------------

def bar(ax, labels, means, stds, cmap, ylabel, title, fmt="{:.4f}"):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[cmap[l] for l in labels], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8,
                       fontweight="bold")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    finite = [v for v in means if np.isfinite(v)]
    top = max(finite) if finite else 1.0
    for b, v in zip(bars, means):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v + abs(top) * 0.02,
                    fmt.format(v), ha="center", fontsize=8)


def grouped_eps(ax, labels, data, cmap, ylabel, title):
    all_eps = sorted({e for d in data.values() for e in d})
    if not all_eps:
        ax.text(0.5, 0.5, "no radius data", ha="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    xs = np.arange(len(all_eps))
    w = 0.8 / max(len(labels), 1)
    for i, l in enumerate(labels):
        vals = [data[l].get(e, (np.nan, 0))[0] for e in all_eps]
        errs = [data[l].get(e, (np.nan, 0))[1] for e in all_eps]
        ax.bar(xs + (i - (len(labels) - 1) / 2) * w, vals, w, yerr=errs,
               capsize=2, label=l, color=cmap[l], edgecolor="black")
    ax.set_xticks(xs); ax.set_xticklabels([str(e) for e in all_eps])
    ax.set_xlabel(r"Perturbation radius $\varepsilon$")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)


def heat(ax, mat, rows, cols, title, fmt="{:.3f}"):
    im = ax.imshow(mat, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30,
                                                        ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


# ---- academic / camera-ready defaults (extended 2026-07-31) ----------
# TrueType (Type-42) fonts so text stays EDITABLE/searchable in the PDF
# (many venues reject Type-3), STIX mathtext, subtle dotted grid, no
# top/right spines, capped error bars, 300 dpi raster + vector PDF.
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.axisbelow": True,
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "errorbar.capsize": 3,
    "legend.frameon": False,
    "savefig.dpi": 300,
})


def save(fig, out_dir, name):
    """PNG for quick viewing + PDF (vector) for the paper."""
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    fig.savefig((out_dir / name).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / name} (+.pdf)")


# ---------------- main ----------------

PAPER_KEEP_PREFIXES = (
    "01_", "02_", "03_", "04_", "06b", "07b", "09_", "10_", "13_",
    "15_", "15b", "20_", "20b", "21_", "22_", "23_", "index",
    "24_", "25_", "26_", "27_",   # efficiency table + FLOPs figs (2026-07-31)
)


def stage_loss(payload, stage):
    """Unperturbed test loss at a stage, for the surrogate gap
    (h = max-neighbourhood-loss - loss, the GSAM quantity)."""
    import numpy as _np
    curves = payload.get("instances_loss_ts") or []
    if not curves:
        return None
    arr = _np.array([c for c in curves if c], dtype=float)
    if arr.size == 0:
        return None
    mean_curve = _np.nanmean(arr, axis=0)
    if stage == "at_convergence":
        return float(_np.nanmin(mean_curve))
    return float(mean_curve[-1])


def paper_metric_bars(R, labels, cmap, out_dir, which, fname, ylabel, meta_eps):
    """The user's-paper-style figure: side-by-side 'at convergence' and
    'at completion' bars for one sharpness metric at a fixed rho."""
    stages = [("at_convergence", "at convergence"),
              ("at_completion", "at completion")]
    data = {}
    for st, _ in stages:
        for l in labels:
            d = sharpness_by_eps(R[l], "max_loss" if which == "surrogate_gap"
                                 else which, st) or {}
            if not d:
                continue
            eps = meta_eps if meta_eps in d else sorted(d)[0]
            m, sd = d[eps][0], (d[eps][1] if len(d[eps]) > 1 else 0.0)
            if which == "surrogate_gap":
                base = stage_loss(R[l]["payload"], st)
                if base is None:
                    continue
                m = m - base
            data[(st, l)] = (m, sd, eps)
    if not data:
        print(f"  ({fname} skipped: no two-stage sharpness data)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.2), sharey=False)
    for ax, (st, title) in zip(axes, stages):
        ls = [l for l in labels if (st, l) in data]
        if not ls:
            ax.set_visible(False)
            continue
        vals = [data[(st, l)][0] for l in ls]
        errs = [data[(st, l)][1] for l in ls]
        ax.bar(range(len(ls)), vals, yerr=errs, capsize=4,
               color=[cmap[l] for l in ls], edgecolor="black")
        ax.set_xticks(range(len(ls)))
        ax.set_xticklabels(ls, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    eps_used = next(iter(data.values()))[2]
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"rho = {eps_used}", y=0.02, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{fname}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / fname}.png (+.pdf)")


PACS_DOMS = ["art_painting", "cartoon", "photo", "sketch"]
METAREG_REF = {   # Balaji et al. 2018, Table 2 (ImageNet-pretrained ResNet-18)
    "MetaReg paper: Baseline (SGD)": [79.9, 75.1, 95.2, 69.5],
    "MetaReg paper: MetaReg":        [83.7, 77.2, 95.5, 70.3],
}


def pacs_generalization_table(records, out_dir):
    """exp6: leave-one-domain-out accuracy per method x held-out domain,
    with the MetaReg paper's reported numbers as reference rows."""
    import numpy as _np
    rows = {}
    for r in records:
        m = r["meta"]
        if (m.get("dataset") != "pacs") and not str(
                m.get("eval_tag", "")).startswith("exp6"):
            continue
        dom = m.get("pacs_holdout")
        if dom not in PACS_DOMS:
            _k = str(m.get("experiment_name") or m.get("eval_tag") or "")
            dom = dict(zip("abcd", PACS_DOMS)).get(_k[4:5])
        if dom not in PACS_DOMS:
            continue
        accs = [c[-1] for c in (r["payload"].get("instances_acc_ts") or []) if c]
        if not accs:
            continue
        lbl = label_of(r)
        rows.setdefault(lbl, {})[dom] = 100.0 * float(_np.mean(accs))
    if not rows:
        return False
    lines = ["method," + ",".join(PACS_DOMS) + ",Avg"]
    md = ["# Experiment 6 — PACS leave-one-domain-out (vs MetaReg Table 2)",
          "", "| method | " + " | ".join(PACS_DOMS) + " | Avg |",
          "|" + "---|" * 6]
    _ordered = order_labels(list(rows))
    _ordered += [l for l in rows if l not in _ordered]   # e.g. plain SGD
    for lbl in _ordered:
        vals = [rows[lbl].get(d) for d in PACS_DOMS]
        got = [v for v in vals if v is not None]
        avg = sum(got) / len(got) if got else float("nan")
        cells = [f"{v:.1f}" if v is not None else "-" for v in vals]
        lines.append(f"{lbl}," + ",".join(cells) + f",{avg:.1f}")
        md.append("| " + lbl + " | " + " | ".join(cells) + f" | {avg:.1f} |")
    for name, vals in METAREG_REF.items():
        avg = sum(vals) / 4
        lines.append(name + "," + ",".join(f"{v:.1f}" for v in vals)
                     + f",{avg:.1f}")
        md.append("| " + name + " | "
                  + " | ".join(f"{v:.1f}" for v in vals) + f" | {avg:.1f} |")
    md += ["", "Reference rows are REPORTED numbers (Balaji et al. 2018, "
              "Table 2, ImageNet-pretrained ResNet-18, their protocol); ours "
              "are directly measured under this repo's protocol — compare "
              "trends, not decimals."]
    (out_dir / "23_pacs_generalization.csv").write_text("\n".join(lines) + "\n")
    (out_dir / "23_pacs_generalization.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {out_dir / '23_pacs_generalization.csv'} (+.md)")
    return True




def _lstm_param_count(meta):
    """Parameter count of the coordinate-wise LSTM optimizer itself
    (bias-free 2x20 LSTM + linear head; overridden by traininfo args)."""
    a = (meta or {}).get("args") or {}
    H = int(a.get("hidden_size") or 20)
    L = int(a.get("num_layers") or 2)
    n_in = 2
    p = 4 * H * (n_in + H)                       # layer 1 (bias=False)
    p += (L - 1) * 4 * H * (2 * H)               # deeper layers
    p += H                                        # output head (bias=False)
    return p


def efficiency_flops_report(P, R, labels, cmap, out_dir, bench=None):
    import numpy as _np
    brow = {r["label"]: r for r in (bench or [])}
    cols = ["method", "time_per_step_ms", "std_ms", "steps_per_sec",
            "samples_per_sec", "passes_per_step", "optimizee_flops_per_step",
            "lstm_flops_per_step", "total_flops_per_step",
            "peak_gpu_mem_mb", "n_params_optimizee", "lstm_params",
            "meta_train_flops"]
    rows = []
    for l in labels:
        e = (P[l].get("efficiency") or {})
        b = brow.get(l, {})
        pk = e.get("peak_gpu_mem_mb") or []
        pk = float(_np.mean(pk)) if pk else b.get("peak_gpu_mem_mb")
        rows.append(dict(
            method=l,
            time_per_step_ms=e.get("mean_step_time_ms") or b.get("mean_step_ms"),
            std_ms=e.get("std_step_time_ms") or 0.0,
            steps_per_sec=e.get("throughput_steps_per_sec")
                          or b.get("throughput_steps_per_s"),
            samples_per_sec=e.get("throughput_samples_per_sec")
                            or b.get("throughput_samples_per_s"),
            passes_per_step=e.get("passes_per_step") or b.get("passes_per_step"),
            optimizee_flops_per_step=e.get("optimizee_flops_per_step")
                                     or b.get("optimizee_flops_per_step"),
            lstm_flops_per_step=e.get("lstm_flops_per_step")
                                or b.get("lstm_flops_per_step") or 0,
            total_flops_per_step=e.get("total_flops_per_step")
                                 or b.get("total_flops_per_step"),
            peak_gpu_mem_mb=pk,
            n_params_optimizee=e.get("n_params_optimizee") or e.get("n_params")
                               or b.get("n_params"),
            lstm_params=_lstm_param_count(R[l]["meta"])
                        if "LO" in l else 0,
            meta_train_flops=e.get("meta_train_flops") or 0,
        ))
    if not rows:
        return
    fmt = lambda v: ("-" if v is None else
                     f"{v:,.0f}" if isinstance(v, (int, float)) and
                     abs(v) >= 1000 else
                     f"{v:.2f}" if isinstance(v, float) else str(v))
    csv_lines = [",".join(cols)] + [
        ",".join("" if r[c] is None else str(r[c]) for c in cols) for r in rows]
    md = ["# Efficiency analysis (poster panel 6)", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    md += ["", "meta_train_flops is the ONE-TIME meta-training investment "
              "(AdamW/SGD baselines honestly report 0); per-step FLOPs use "
              "the shared flops_utils accounting (bwd = 2x fwd; LSTM cost "
              "= n_params x per-coordinate cell cost)."]
    (out_dir / "24_efficiency_table.csv").write_text("\n".join(csv_lines) + "\n")
    (out_dir / "24_efficiency_table.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {out_dir / '24_efficiency_table.csv'} (+.md)")

    have_flops = [r for r in rows if r["total_flops_per_step"]]
    if have_flops:
        ls = [r["method"] for r in have_flops]
        opt = _np.array([r["optimizee_flops_per_step"] or 0 for r in have_flops],
                        float)
        lo = _np.array([r["lstm_flops_per_step"] or 0 for r in have_flops], float)
        x = _np.arange(len(ls))
        fig, ax = plt.subplots(figsize=(max(6, len(ls) * 1.5), 4.6))
        ax.bar(x, opt, color=[cmap.get(l, "#888") for l in ls],
               label="optimizee fwd+bwd")
        ax.bar(x, lo, bottom=opt, color="#37474F", hatch="//",
               label="coordinate-wise LSTM")
        for xi, (o, g) in enumerate(zip(opt, lo)):
            ax.annotate(f"{(o + g)/1e6:,.0f}M", (xi, o + g),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(ls, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("FLOPs per optimization step (log)")
        ax.set_title("Per-step compute, split into optimizee vs LSTM cost")
        ax.legend()
        save(fig, out_dir, "25_flops_per_step.png")

    mt = [(r["method"], r["meta_train_flops"] or 0) for r in rows]
    if any(v > 0 for _, v in mt):
        ls = [m for m, _ in mt]; vs = [max(v, 1) for _, v in mt]
        fig, ax = plt.subplots(figsize=(max(6, len(ls) * 1.5), 4.4))
        ax.bar(range(len(ls)), vs, color=[cmap.get(l, "#888") for l in ls])
        for xi, (m, v) in enumerate(mt):
            ax.annotate("0" if v == 0 else f"{v:.2e}", (xi, max(v, 1)),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ls)))
        ax.set_xticklabels(ls, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Meta-training FLOPs (one-time, log)")
        ax.set_title("Upfront meta-training investment "
                     "(baselines pay none)")
        save(fig, out_dir, "26_meta_train_flops.png")

    pkrows = [(r["method"], r["peak_gpu_mem_mb"]) for r in rows
              if r["peak_gpu_mem_mb"]]
    if pkrows:
        ls = [m for m, _ in pkrows]; vs = [v for _, v in pkrows]
        fig, ax = plt.subplots(figsize=(max(6, len(ls) * 1.5), 4.2))
        bar(ax, ls, vs, [0] * len(ls), cmap, "Peak GPU memory (MB)",
            "Peak GPU memory per method", fmt="{:.0f}")
        ax.axhline(32 * 1024, color="crimson", ls="--", lw=1.2)
        ax.annotate("V100 32 GB", (0.99, 32 * 1024), xycoords=("axes fraction",
                    "data"), ha="right", va="bottom", fontsize=8,
                    color="crimson")
        save(fig, out_dir, "27_peak_gpu_memory.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None,
                    help="filter by metadata.experiment_name (e.g. exp1); "
                         "figures land in plots/paper/<experiment>/")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--figures", choices=["paper", "all"], default="paper",
                    help="'paper' (default) keeps ONLY the reviewer-essential "
                         "set: accuracy/loss bars, train+test curves, "
                         "sharpness & grad-norm vs epsilon, the paper-style "
                         "'at convergence | at completion' metric bars "
                         "(max-loss for SAM sets, max-grad-norm for GAM, "
                         "surrogate gap for GSAM), convergence, FLOPs/"
                         "efficiency, summary+significance tables, and the "
                         "cross-dataset heatmap where applicable. "
                         "'all' restores every legacy figure.")
    ap.add_argument("--comparison-set", default=None,
                    choices=sorted(COMPARISON_SETS),
                    help="restrict figures to one 3-method comparison set "
                         "(LO No-Reg vs LO+R vs AdamW+R) — 'compare "
                         "corresponding pairs only'. Output goes to "
                         "<out_dir>/set_<R>/")
    ap.add_argument("--stage", default="at_completion",
                    choices=["at_completion", "at_convergence"])
    cli = ap.parse_args()

    out_dir = OUT_BASE / (cli.experiment or "all")
    if cli.comparison_set:                    # compare corresponding pairs only
        out_dir = out_dir / f"set_{cli.comparison_set}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records()
    if not records:
        print("No result JSONs found — run evaluations first.")
        return

    non_sweep = [r for r in records if not is_sweep(r)]
    if cli.experiment:
        pool = [r for r in non_sweep
                if r["meta"].get("experiment_name") == cli.experiment]
        if not pool:
            # prefix fallback (2026-07-31): `--experiment exp6` aggregates
            # exp6a-exp6d (and `exp4` -> exp4a/exp4b) into ONE report, e.g.
            # the full 4-domain PACS table with a single command.
            pool = [r for r in non_sweep
                    if str(r["meta"].get("experiment_name") or ""
                           ).startswith(cli.experiment)]
            if pool:
                print(f"  ({cli.experiment}: exact match empty — "
                      f"aggregating prefix matches: "
                      f"{sorted({r['meta']['experiment_name'] for r in pool})})")
        if not pool:
            print(f"No records tagged experiment_name={cli.experiment}")
            return
        ds = pool[0]["meta"].get("test_dataset") or pool[0]["meta"].get("dataset")
        mdl = pool[0]["meta"].get("test_architecture") or \
            pool[0]["meta"].get("optimizee_model")
    else:
        counts = defaultdict(int)
        for r in non_sweep:
            if not is_cross(r):
                counts[(r["meta"].get("dataset"),
                        r["meta"].get("optimizee_model"))] += 1
        (ds, mdl), _ = max(counts.items(), key=lambda kv: kv[1])
        ds = cli.dataset or ds
        mdl = cli.model or mdl
        pool = [r for r in non_sweep
                if r["meta"].get("dataset") == ds
                and r["meta"].get("optimizee_model") == mdl]
    print(f"Report scope: experiment={cli.experiment or '-'}  "
          f"dataset={ds}  model={mdl}  stage={cli.stage}")

    comp = [r for r in pool if not is_cross(r)]
    by_label = pick_per_label(comp)
    labels = order_labels(list(by_label))
    if cli.comparison_set:
        keep = set(COMPARISON_SETS[cli.comparison_set])
        labels = [l for l in labels if l in keep]
        if not labels:
            print(f"  (no runs for comparison set '{cli.comparison_set}': "
                  f"expected {sorted(keep)})")
            return
        print(f"  comparison set '{cli.comparison_set}': {labels}")
    # ---- completeness check (fact-check: NoReg LO and GAM LO were
    # trained but never evaluated, and the report silently omitted them —
    # the whole ablation). Warn loudly.
    _base8 = ["AdamW", "AdamW + SAM", "AdamW + GAM", "AdamW + GSAM",
              "No-Reg LO", "SAM LO", "GAM LO", "GSAM LO"]
    if cli.comparison_set:
        _expected = list(COMPARISON_SETS[cli.comparison_set])
    elif cli.experiment and cli.experiment.startswith("exp6"):
        _expected = _base8 + ["SGD"]
    else:
        _expected = _base8
    missing_methods = [m for m in _expected if m not in labels]
    if missing_methods:
        print("  WARNING: expected methods MISSING from this report: "
              + str(missing_methods) + " — run their eval jobs, or every "
              "comparison here is incomplete.")

    cmap = colors_for(labels)
    P = {l: by_label[l]["payload"] for l in labels}
    R = {l: by_label[l] for l in labels}

    if labels:
        # ---- 01 / 02 final accuracy & loss ----
        acc = {l: final_stat(P[l], "acc_ts") for l in labels}
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 5))
        bar(ax, labels, [acc[l][0] for l in labels],
            [acc[l][1] for l in labels], cmap, "Test accuracy (final)",
            f"Final test accuracy — {mdl} / {ds}")
        save(fig, out_dir, "01_test_accuracy.png")

        lss = {l: final_stat(P[l], "loss_ts") for l in labels}
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 5))
        bar(ax, labels, [lss[l][0] for l in labels],
            [lss[l][1] for l in labels], cmap, "Test loss (final)",
            f"Final test loss — {mdl} / {ds}")
        save(fig, out_dir, "02_test_loss.png")

        # ---- 03 / 04 test curves ----
        for key, ylab, fname in [("acc_ts", "Test accuracy",
                                  "03_test_accuracy_curves.png"),
                                 ("loss_ts", "Test loss",
                                  "04_test_loss_curves.png")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for l in labels:
                m, s = mean_curve(P[l], key)
                if m.size:
                    x = np.arange(m.size)
                    ax.plot(x, m, label=l, color=cmap[l])
                    ax.fill_between(x, m - s, m + s, color=cmap[l], alpha=0.15)
            ax.set_xlabel("Optimization step"); ax.set_ylabel(ylab)
            ax.set_title(f"{ylab} vs step — {mdl} / {ds}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            save(fig, out_dir, fname)

        # ---- 05 validation curves ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axi, key, ylab in [(axes[0], "loss_val", "Validation loss"),
                               (axes[1], "acc_val", "Validation accuracy")]:
            for l in labels:
                m, s = mean_curve(P[l], key, ffill=True)
                if m.size and np.isfinite(m).any():
                    x = np.arange(m.size)
                    axi.plot(x, m, label=l, color=cmap[l])
                    axi.fill_between(x, m - s, m + s, color=cmap[l], alpha=0.15)
            axi.set_xlabel("Optimization step"); axi.set_ylabel(ylab)
            axi.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Validation metrics — {mdl} / {ds}")
        save(fig, out_dir, "05_validation_curves.png")

        # ---- 06 / 07 radius panels (two-stage aware) ----
        ml = {l: sharpness_by_eps(R[l], "max_loss", cli.stage) for l in labels}
        gn = {l: sharpness_by_eps(R[l], "max_grad_norm", cli.stage)
              for l in labels}
        fig, ax = plt.subplots(figsize=(9, 5))
        grouped_eps(ax, labels, ml, cmap, "Max loss",
                    fr"Sharpness (max loss in $\varepsilon$-nghb) — "
                    fr"{cli.stage.replace('_', ' ')}")
        save(fig, out_dir, "06_sharpness_maxloss.png")
        fig, ax = plt.subplots(figsize=(9, 5))
        grouped_eps(ax, labels, gn, cmap, "Max grad norm",
                    fr"Max $\|\nabla L\|$ in $\varepsilon$-nghb — "
                    fr"{cli.stage.replace('_', ' ')}")
        save(fig, out_dir, "07_maxgradnorm.png")

        # ---- 06b/07b: loss-landscape metrics vs radius (log-x LINES,
        # poster 'Loss Landscape Analysis' panel style) ----
        for key, fname, ylab in [
                ("max_loss", "06b_sharpness_vs_eps_lines.png",
                 r"Max loss in $\varepsilon$-nghb"),
                ("max_grad_norm", "07b_gradnorm_vs_eps_lines.png",
                 r"Max $\|\nabla L\|$ in $\varepsilon$-nghb")]:
            data = {l: sharpness_by_eps(by_label[l], key, stage=cli.stage)
                    for l in labels}
            if any(data[l] for l in labels):
                fig, ax = plt.subplots(figsize=(6.2, 4.4))
                for l in labels:
                    d = data[l]
                    if not d:
                        continue
                    es = sorted(d)
                    mu = [d[e][0] for e in es]
                    sd = [d[e][1] for e in es]
                    ax.errorbar(es, mu, yerr=sd, marker="o", capsize=3,
                                label=l, color=cmap[l])
                ax.set_xscale("log")
                ax.set_xlabel(r"Perturbation radius $\varepsilon$ (log)")
                ax.set_ylabel(ylab)
                ax.set_title(f"{ylab} — {mdl} / {ds}")
                ax.legend()
                ax.grid(alpha=0.3)
                save(fig, out_dir, fname)

        # ---- 08 Hessian ----
        hs = {l: hessian_lmax(P[l]) for l in labels}
        have = [l for l in labels if hs[l]]
        if have:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            bar(axes[0], have, [hs[l][0] for l in have],
                [hs[l][1] for l in have], cmap, r"$\lambda_{max}$",
                "Largest Hessian eigenvalue", fmt="{:.3g}")
            for l in have:
                spec = hs[l][2][0]
                axes[1].plot(range(1, len(spec) + 1), spec, "o-", label=l,
                             color=cmap[l])
            axes[1].set_xlabel("eigenvalue index"); axes[1].set_yscale("log")
            axes[1].set_title("Hessian spectrum (instance 0)")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
            save(fig, out_dir, "08_hessian_spectrum.png")
        else:
            print("  (08 skipped: no eigen_values — run with calc_hessian=True)")

        # ---- 09 efficiency: eval-side + benchmark rows (STEP 11) ----
        bench = load_bench_rows(exp=cli.experiment, ds=ds, mdl=mdl) or \
            load_bench_rows(ds=ds, mdl=mdl)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ts = {l: opt_seconds(P[l]) for l in labels}
        bar(axes[0], labels, [ts[l][0] for l in labels],
            [ts[l][1] for l in labels], cmap,
            "Optimization time (s, mean per run)",
            r"Wall-clock ($\Sigma$ step times)", fmt="{:.1f}")
        if bench:
            blabels = order_labels([r["label"] for r in bench])
            bmap = colors_for(blabels)
            row = {r["label"]: r for r in bench}
            bar(axes[1], blabels,
                [row[l]["median_step_ms"] for l in blabels],
                [0] * len(blabels), bmap, "median step (ms)",
                "Per-step time (benchmark)", fmt="{:.2f}")
            bar(axes[2], blabels,
                [(row[l].get("total_flops_per_step") or 0) / 1e6
                 for l in blabels], [0] * len(blabels), bmap,
                "MFLOPs / step (incl. LSTM)",
                "Compute per step (benchmark)", fmt="{:.0f}")
        else:
            for a in axes[1:]:
                a.text(0.5, 0.5, "no benchmark json\n(run benchmark_compute.py)",
                       ha="center", transform=a.transAxes)
        fig.suptitle(f"Efficiency — {mdl} / {ds}")
        save(fig, out_dir, "09_efficiency.png")

        # ---- 24-27 EFFICIENCY ANALYSIS (poster panel 6, 2026-07-31) ----
        # Per-method: Time/Step, Steps/s, Samples/s, FLOPs/Step split into
        # optimizee vs LSTM, Fwd/Bwd passes, Peak GPU memory, optimizee &
        # LSTM parameter counts, one-time META-TRAIN FLOPs — sourced from
        # each eval JSON's `efficiency` dict (single source of truth with
        # benchmark_compute via flops_utils), bench rows as fallback.
        efficiency_flops_report(P, R, labels, cmap, out_dir, bench)

        # ---- 12 summary panel (the reviewer-image layout) ----
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
        bar(axes[0], labels, [acc[l][0] for l in labels],
            [acc[l][1] for l in labels], cmap, "Test accuracy",
            "Final test accuracy")
        grouped_eps(axes[1], labels, ml, cmap, "Max loss",
                    r"Sharpness (max loss in $\varepsilon$-nghb)")
        grouped_eps(axes[2], labels, gn, cmap, "Max grad norm",
                    r"Max $\|\nabla L\|$ in $\varepsilon$-nghb")
        fig.suptitle(f"Method comparison — {mdl} / {ds} "
                     f"({cli.stage.replace('_', ' ')})",
                     fontsize=14, fontweight="bold")
        save(fig, out_dir, "12_summary_panel.png")

        # ---- 13 convergence (train loss = the optimization trajectory) ----
        # fact-check: this figure silently vanished from the production
        # exp2 report. It must either render or say WHY it did not.
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            _n13 = 0
            for l in labels:
                m, s_ = mean_curve(P[l], "loss_tr")
                if m.size:
                    x = np.arange(m.size)
                    ax.plot(x, m, label=l, color=cmap[l])
                    ax.fill_between(x, m - s_, m + s_, color=cmap[l],
                                    alpha=0.15)
                    _n13 += 1
            if _n13 == 0:
                plt.close(fig)
                print("  (13_convergence SKIPPED: no train-loss "
                      "(instances_loss_tr) curves in any record — check "
                      "that eval jobs record loss_tr)")
            else:
                ax.set_xlabel("Optimization step")
                ax.set_ylabel("Training loss")
                ax.set_yscale("log")
                ax.set_title(f"Convergence — {mdl} / {ds}")
                ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
                save(fig, out_dir, "13_convergence.png")
        except Exception as _e13:
            print("  (13_convergence FAILED: "
                  + type(_e13).__name__ + ": " + str(_e13) + ")")

        # ---- 16 CIFAR-10-C robustness ----
        rob = {}
        for l in labels:
            c10 = [c for c in (P[l].get("cifar10c_list") or []) if c]
            if c10:
                rob[l] = float(np.mean([c["mean_corruption_accuracy"]
                                        for c in c10]))
        if rob:
            fig, ax = plt.subplots(figsize=(max(6, len(rob) * 1.8), 5))
            ls = order_labels(list(rob))
            x = np.arange(len(ls)); w = 0.38
            ax.bar(x - w / 2, [acc[l][0] for l in ls], w,
                   label="clean (CIFAR-10)", color=[cmap[l] for l in ls],
                   edgecolor="black")
            ax.bar(x + w / 2, [rob[l] for l in ls], w,
                   label="CIFAR-10-C (mean corruption)",
                   color=[cmap[l] for l in ls], alpha=0.55,
                   edgecolor="black", hatch="//")
            ax.set_xticks(x); ax.set_xticklabels(ls, rotation=20, ha="right",
                                                 fontsize=8, fontweight="bold")
            ax.set_ylabel("Accuracy")
            ax.set_title("Robustness: clean vs CIFAR-10-C")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            save(fig, out_dir, "16_cifar10c_robustness.png")

        # ---- 15 summary table (STEP: reviewer summary) ----
        eps_all = sorted({e for l in labels for e in ml[l]})
        emax = eps_all[-1] if eps_all else None
        rows = []
        bench_row = {r["label"]: r for r in (bench or [])}
        for l in labels:
            hl = hs.get(l); gg = gen_gap(P[l]); pm = peak_mem(P[l])
            rows.append({
                "method": l,
                "accuracy": round(acc[l][0], 4),
                "acc_std": round(acc[l][1], 4),
                "test_loss": round(lss[l][0], 4),
                f"pgd_max_loss@{emax}": round(ml[l].get(emax, (np.nan,))[0], 4)
                if emax else None,
                f"max_grad_norm@{emax}": round(gn[l].get(emax, (np.nan,))[0], 4)
                if emax else None,
                "hessian_lmax": round(hl[0], 4) if hl else None,
                "gen_gap_acc": round(gg[0], 4) if gg else None,
                "opt_time_s": round(ts[l][0], 2),
                "peak_mem_mb": round(pm[0], 1) if np.isfinite(pm[0]) else None,
                # FLOPs measured IN the experiment (efficiency dict of the
                # eval JSON itself, via flops_utils) — bench_* stays as the
                # independent cross-check:
                "eval_total_MFLOPs":
                    round(((P[l].get("efficiency") or {})
                           .get("total_flops_per_step") or 0) / 1e6, 1) or None,
                "eval_passes": (P[l].get("efficiency") or {}).get("passes_per_step"),
                # paper metric: meta-train FLOPs (one-time, 0 for AdamW
                # baselines) and the honest TOTAL = meta-train + meta-test
                "meta_train_TFLOPs": (lambda v: round(v / 1e12, 3) if v else
                                      (0.0 if v == 0 else None))(
                    (P[l].get("efficiency") or {}).get("meta_train_flops")),
                "bench_step_ms": bench_row.get(l, {}).get("median_step_ms"),
                "bench_total_MFLOPs":
                    round((bench_row.get(l, {}).get("total_flops_per_step")
                           or 0) / 1e6, 1) or None,
            })
        with open(out_dir / "15_summary_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        hdr = list(rows[0].keys())
        md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for r in rows:
            md.append("| " + " | ".join(str(r[h]) for h in hdr) + " |")
        (out_dir / "15_summary_table.md").write_text("\n".join(md))
        print(f"  wrote {out_dir/'15_summary_table.csv'} (+.md)")

        # ---- 15b: significance vs the matched AdamW baseline (Welch t) ----
        import math as _m

        def _welch(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            if len(a) < 2 or len(b) < 2:
                return None, None
            va, vb = a.var(ddof=1), b.var(ddof=1)
            t = (a.mean() - b.mean()) / _m.sqrt(va / len(a) + vb / len(b) + 1e-24)
            p = _m.erfc(abs(t) / _m.sqrt(2))          # normal approx
            return round(float(t), 3), round(float(p), 4)

        def _finals(payload):
            inst = series(payload, "acc_ts") or []
            return [r[-1] for r in inst if r]

        base_lbl = next((l for l in labels if l.startswith("AdamW")), None)
        if base_lbl:
            rows15b = [("method", "vs", "t_stat", "p_approx", "n")]
            base_fin = _finals(P[base_lbl])
            for l in labels:
                if l == base_lbl:
                    continue
                fin = _finals(P[l])
                if len(fin) < 2 or len(base_fin) < 2:
                    # a t-test on n<2 is undefined; the old code emitted the
                    # impossible p_approx=0.0 here (fact-check finding).
                    t, pv = float("nan"), float("nan")
                else:
                    t, pv = _welch(fin, base_fin)
                    if pv is not None and pv <= 0.0:
                        pv = 2.2e-16      # floored, never exactly 0
                rows15b.append((l, base_lbl, t, pv, f"{len(fin)}v{len(base_fin)}"))
            with open(out_dir / "15b_significance.csv", "w", newline="") as f:
                csv.writer(f).writerows(rows15b)
            print(f"  wrote {out_dir/'15b_significance.csv'} "
                  f"(final-accuracy Welch t vs {base_lbl}; needs n>=2 seeds)")

        # ---- 17: sweep convergence overlay (one panel per regularizer) ----
        sw = [r for r in records if is_sweep(r)]
        if sw:
            by_reg = defaultdict(list)
            for r in sw:
                tag = r["meta"].get("eval_tag") or ""
                for rg in ("gsam", "gam", "sam"):
                    if f"sw_{rg}_" in tag:
                        by_reg[rg].append(r)
                        break
            regs = [rg for rg in ("sam", "gam", "gsam") if by_reg.get(rg)]
            if regs:
                fig, axes = plt.subplots(1, len(regs),
                                         figsize=(5.2 * len(regs), 4.2),
                                         squeeze=False)
                for ax, rg in zip(axes[0], regs):
                    for r in sorted(by_reg[rg],
                                    key=lambda r: r["meta"].get("eval_tag") or ""):
                        mu, _sd = mean_curve(r["payload"], "loss_ts")
                        if mu.size:
                            ax.plot(mu, lw=1.2,
                                    label=(r["meta"].get("eval_tag") or "")
                                    .replace(f"sw_{rg}_", ""))
                    ax.set_yscale("log")
                    ax.set_xlabel("Optimization step")
                    ax.set_ylabel("Test loss (log)")
                    ax.set_title(f"{rg.upper()} sweep — convergence per "
                                 r"$\lambda$ cell")
                    ax.legend(fontsize=6)
                    ax.grid(alpha=0.3)
                save(fig, out_dir, "17_sweep_convergence.png")

        # ---- 18: EXPERIMENT 12 — convergence under different epsilon ----
        # Train-loss vs iteration for each smoothing radius the LO was
        # meta-trained with (sw_*_se<eps> cells); No-Reg LO = reference.
        import re as _re
        _dec = lambda t: float(t.replace("p", ".").replace("m", "-"))
        se_rows = defaultdict(lambda: defaultdict(list))   # reg -> eps -> recs
        for r in records:
            tag = (r["meta"].get("eval_tag") or "")
            mm = _re.search(r"sw_(sam|gam|gsam)_.*_se([0-9pm]+)", tag)
            if mm:
                se_rows[mm.group(1)][_dec(mm.group(2))].append(r)
        if se_rows:
            regs = [rg for rg in ("sam", "gam", "gsam") if se_rows.get(rg)]
            fig, axes = plt.subplots(1, len(regs),
                                     figsize=(5.4 * len(regs), 4.4),
                                     squeeze=False)
            noreg = by_label.get("No-Reg LO")
            for ax, rg in zip(axes[0], regs):
                for e in sorted(se_rows[rg]):
                    rec = sorted(se_rows[rg][e],
                                 key=lambda r: r["meta"].get("timestamp") or "")[-1]
                    mu, _ = mean_curve(rec["payload"], "loss_tr")
                    if mu.size:
                        ax.plot(mu, lw=1.4, label=rf"$\varepsilon$={e}")
                if noreg is not None:
                    mu, _ = mean_curve(noreg["payload"], "loss_tr")
                    if mu.size:
                        ax.plot(mu, lw=1.6, ls="--", color="k",
                                label="No-Reg LO (ref)")
                ax.set_yscale("log")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Train loss (log)")
                ax.set_title(f"{rg.upper()} LO — convergence vs "
                             r"smoothing radius $\varepsilon$")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            save(fig, out_dir, "18_convergence_vs_eps.png")

        # ---- 19: accuracy + sharpness-vs-epoch at multiple epsilon ------
        # (reviewer request: Accuracy / Sharpness / Generalization together)
        # scan ALL cell records for tracked payloads (the headline record
        # per label may be an untracked run — e.g. the hyper-eps one):
        tracked, _tsrc = {}, {}
        for r in records:
            m_ = r["meta"]
            if m_.get("dataset") != ds or m_.get("optimizee_model") != mdl \
                    or is_sweep(r):
                continue
            tt = (r["payload"].get("instances_sharpness_track") or [])
            if tt and tt[0] and tt[0].get("steps"):
                l_ = label_of(r)
                ts_ = m_.get("timestamp") or ""
                if l_ in labels and ts_ >= _tsrc.get(l_, ""):
                    tracked[l_], _tsrc[l_] = tt[0], ts_
        if tracked:
            fig, (axA, axS) = plt.subplots(2, 1, figsize=(7.0, 7.6),
                                           sharex=True)
            for l, tr in tracked.items():
                mu, _sd = mean_curve(P[l], "acc_ts")
                if mu.size:
                    axA.plot(mu, lw=1.4, label=l, color=cmap[l])
                steps = tr["steps"]
                for i, (e, vals) in enumerate(sorted(tr["eps"].items(),
                                                     key=lambda kv: float(kv[0]))):
                    axS.plot(steps, vals, lw=1.2, color=cmap[l],
                             alpha=1.0 - 0.15 * i, marker="o", ms=3,
                             label=rf"{l}  $\varepsilon$={e}")
            axA.set_ylabel("Test accuracy")
            axA.set_title(f"Accuracy and sharpness over training — {mdl}/{ds}")
            axA.legend(fontsize=8); axA.grid(alpha=0.3)
            axS.set_yscale("log")
            axS.set_xlabel("Iteration")
            axS.set_ylabel(r"Sharpness: max loss in $\varepsilon$-nghb (log)")
            axS.legend(fontsize=7, ncol=2); axS.grid(alpha=0.3)
            save(fig, out_dir, "19_sharpness_vs_epoch.png")

        # ---- per-experiment index for the paper folder ----
        idx = [f"# Report: {cli.experiment or f'{ds}/{mdl}'}",
               f"dataset={ds} model={mdl} stage={cli.stage}",
               f"methods: {', '.join(labels)}", "", "## Figures"]
        for f_ in sorted(out_dir.glob("*.png")):
            idx.append(f"- {f_.name}")
        (out_dir / "index.md").write_text("\n".join(idx))
        print(f"  wrote {out_dir/'index.md'}")

    # ---- 10 cross-DATASET heatmaps ----
    _cross_heatmaps(records, "train_dataset", "test_dataset",
                    "10_crossdataset_heatmap.png", out_dir,
                    "Cross-dataset accuracy (train → test)")
    # ---- 14 cross-ARCHITECTURE heatmaps (STEP 9) ----
    _cross_heatmaps(records, "train_architecture", "test_architecture",
                    "14_cross_architecture.png", out_dir,
                    "Cross-architecture accuracy (train → test)")

    # ---- 11 sweep heatmap (unchanged consumer of sw_ tags) ----
    _sweep_heatmap(records, out_dir)

    if cli.experiment and cli.experiment.startswith("exp6"):
        pacs_generalization_table(records, out_dir)

    # ---- paper-style two-stage metric bars (user's fig-4/5 layout) ------
    _meta_eps = None
    for l in labels:
        _meta_eps = _meta_eps or (R[l]["meta"].get("eps"))
    _set_metric = {"sam":  ("max_loss", "20_maxloss_conv_vs_completion",
                            "Maximum Loss in a neighbourhood"),
                   "gam":  ("max_grad_norm", "21_gradnorm_conv_vs_completion",
                            "Maximum norm of Gradient in a neighbourhood"),
                   "gsam": ("surrogate_gap", "22_surrogate_gap_conv_vs_completion",
                            "Surrogate Gap")}
    # fact-check: raw max-loss conflates the centre-loss level with
    # sharpness (subtracting it flips the SAM ranking). SAM sets therefore
    # ALSO get the normalised gap = max-loss minus stage loss.
    _gap_extra = ("surrogate_gap", "20b_sharpness_gap_conv_vs_completion",
                  "Sharpness gap (max loss - loss)")
    _wanted = (([_set_metric[cli.comparison_set]]
                + ([_gap_extra] if cli.comparison_set == "sam" else []))
               if cli.comparison_set else
               list(_set_metric.values()) + [_gap_extra])
    for _which, _fn, _yl in _wanted:
        paper_metric_bars(R, labels, cmap, out_dir, _which, _fn, _yl, _meta_eps)

    # ---- paper mode: prune to the reviewer-essential set ----------------
    if cli.figures == "paper":
        removed = 0
        for f in sorted(out_dir.iterdir()):
            if f.is_file() and not f.name.startswith(PAPER_KEEP_PREFIXES):
                f.unlink()
                removed += 1
        (out_dir / "index.md").write_text(
            "# Reviewer-essential figure set (--figures paper)\n\n" +
            "\n".join(f"- {f.name}" for f in sorted(out_dir.iterdir())
                       if f.suffix == ".png") + "\n")
        print(f"  (--figures paper: pruned {removed} legacy files; "
              f"use --figures all to restore)")

    print(f"\nAll figures in: {out_dir}")


def _cross_heatmaps(records, row_key, col_key, fname, out_dir, title):
    per_method = defaultdict(dict)   # label -> {(row,col): [accs]}
    for r in records:
        m = r["meta"]
        tr, te = m.get(row_key), m.get(col_key)
        if not tr or not te or is_sweep(r):
            continue
        a, _ = final_stat(r["payload"], "acc_ts")
        if np.isfinite(a):
            per_method[label_of(r)].setdefault((tr, te), []).append(a)
    per_method = {l: d for l, d in per_method.items()
                  if len({k[0] for k in d}) > 1 or len({k[1] for k in d}) > 1}
    if not per_method:
        print(f"  ({fname} skipped: no cross data for {row_key}->{col_key})")
        return
    ls = order_labels(list(per_method))
    fig, axes = plt.subplots(1, len(ls), figsize=(5.4 * len(ls), 4.6),
                             squeeze=False)
    for ax, l in zip(axes[0], ls):
        d = per_method[l]
        rows = sorted({k[0] for k in d}); cols = sorted({k[1] for k in d})
        mat = np.full((len(rows), len(cols)), np.nan)
        for (tr, te), accs in d.items():
            mat[rows.index(tr), cols.index(te)] = float(np.mean(accs))
        heat(ax, mat, rows, cols, l)
        ax.set_xlabel("test"); ax.set_ylabel("train")
    fig.suptitle(title)
    save(fig, out_dir, fname)


def _sweep_heatmap(records, out_dir):
    import re
    pat = re.compile(r"sw_(?P<reg>[a-z0-9_]+?)_sl(?P<sl>[0-9pm]+)_"
                     r"(?P<key>[a-z_]+?)(?P<rv>[0-9pm]+)"
                     r"(?:_lr(?P<lr>[0-9pm]+))?$")
    cells = defaultdict(dict)
    for r in records:
        tag = r["meta"].get("eval_tag") or ""
        mm = pat.match(tag)
        if not mm:
            continue
        conv = lambda t: float(t.replace("p", ".").replace("m", "-"))
        sl, rv = conv(mm["sl"]), conv(mm["rv"])
        a, _ = final_stat(r["payload"], "acc_ts")
        cells[mm["reg"]][(sl, rv)] = a
    for reg, d in cells.items():
        sls = sorted({k[0] for k in d}); rvs = sorted({k[1] for k in d})
        if len(sls) < 2 and len(rvs) < 2:
            continue
        mat = np.full((len(sls), len(rvs)), np.nan)
        for (sl, rv), a in d.items():
            mat[sls.index(sl), rvs.index(rv)] = a
        fig, ax = plt.subplots(figsize=(1.4 * len(rvs) + 3, 1.1 * len(sls) + 2))
        heat(ax, mat, [str(s) for s in sls], [str(v) for v in rvs],
             f"Sweep: {reg} — accuracy")
        ax.set_xlabel("strength / rho"); ax.set_ylabel("lambda_smooth")
        save(fig, out_dir, f"11_sweep_heatmap_{reg}.png")


if __name__ == "__main__":
    main()
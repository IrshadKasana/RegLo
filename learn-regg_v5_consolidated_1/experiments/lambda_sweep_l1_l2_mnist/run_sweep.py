#!/usr/bin/env python
"""
run_sweep.py — lambda sweep for L1 / L2 on MNIST over the four methods.

    1. MLP + AdamW                      (lr x wd grid; the lambda = 0 reference)
    2. MLP + AdamW + lambda*R           (lambda x lr x wd grid)
    3. MLP + LO                         (lambda = 0 reference)
    4. MLP + LO meta-trained with lambda*R (+ smoothing)   (lambda grid, several meta-seeds)

Every configuration is TRAINED and then EVALUATED on the held-out validation
split; the best lambda per (method, regularizer) is the arg-max of validation
accuracy.  Test metrics are stored but never used for selection.

    python run_sweep.py all [--smoke] [--workers 3] [--meta-seeds 0 1]
    python run_sweep.py report            # re-aggregate whatever has finished

Resumable: a job whose result JSON exists is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
EVAL_SEED0 = 50_000          # common random numbers: every config sees the same inits/batches


def lamstr(lam):
    return f"{lam:g}"


def dirs(smoke):
    root = HERE / "results" / ("smoke" if smoke else "full")
    d = {k: root / k for k in ("adamw", "lo", "ckpt", "logs")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


def make_cfg(smoke):
    import common as C
    return C.Cfg(**C.SMOKE) if smoke else C.Cfg()


# =============================================================================
# jobs (each in its own process)
# =============================================================================
def job_adamw(a):
    import common as C
    cfg, D = make_cfg(a.smoke), dirs(a.smoke)
    data = C.load_mnist()
    seeds = [EVAL_SEED0 + i for i in range(cfg.adamw_inits)]
    lams = (0.0,) if a.reg == "none" else cfg.lams(a.reg)
    wds = tuple(a.wds) if a.wds else cfg.adamw_wds
    suffix = "_x" if a.wds else ""          # extra weight-decay values -> separate file
    rows, t0 = [], time.time()
    for lr in cfg.adamw_lrs:
        for wd in wds:
            for lam in lams:
                ev = [C.run_adamw(s, cfg, data, lr, wd, a.reg, lam) for s in seeds]
                rows.append(dict(reg=a.reg, lr=lr, wd=wd, lam=lam, evals=ev))
                m = np.mean([e["val_acc"] for e in ev])
                print(f"adamw reg={a.reg} lr={lr:g} wd={wd:g} lam={lam:g} val_acc={m:.4f}", flush=True)
    json.dump(dict(reg=a.reg, eval_seeds=seeds, rows=rows, seconds=time.time() - t0, cfg=asdict(cfg)),
              open(D["adamw"] / f"adamw_{a.reg}{suffix}.json", "w"))


def job_lo(a):
    import common as C
    cfg, D = make_cfg(a.smoke), dirs(a.smoke)
    data = C.load_mnist()
    tag = f"lo_{a.kind}_lam{lamstr(a.lam)}_seed{a.seed}"
    ckpt = D["ckpt"] / f"{tag}.pt"
    lo, info = C.meta_train_lo(a.kind, a.lam, a.seed, cfg, data, ckpt,
                               log=lambda s: print(s, flush=True))
    seeds = [EVAL_SEED0 + i for i in range(cfg.lo_inits)]
    ev = [C.run_lo(lo, s, cfg, data) for s in seeds]
    json.dump(dict(kind=a.kind, lam=a.lam, seed=a.seed, eval_seeds=seeds, evals=ev,
                   meta_train=info, cfg=asdict(cfg), device=str(C.device)),
              open(D["lo"] / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: val_acc={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)


# =============================================================================
# orchestration
# =============================================================================
def run_jobs(jobs, workers, logdir):
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(f"[orchestrator] {len(jobs) - len(todo)} done, {len(todo)} to run, {workers} workers", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def go(j):
        name, argv, _ = j
        t = time.time()
        with open(logdir / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, str(HERE / "run_sweep.py"), *argv],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[orchestrator] {'ok  ' if rc == 0 else 'FAIL'} {name} ({time.time() - t:.0f}s)", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rcs = list(ex.map(go, todo))
    if any(rcs):
        print("[orchestrator] some jobs failed - see results/*/logs", flush=True)


def cmd_all(a):
    cfg, D = make_cfg(a.smoke), dirs(a.smoke)
    sm = ["--smoke"] if a.smoke else []
    jobs = []
    for reg in ("none", "l1", "l2"):
        jobs.append((f"adamw_{reg}", ["job-adamw", "--reg", reg, *sm], D["adamw"] / f"adamw_{reg}.json"))
    for s in a.meta_seeds:                        # seed-major: a usable sweep appears early
        specs = [("none", 0.0), ("smooth", 0.0)]
        for reg in ("l1", "l2"):
            specs += [(reg, lam) for lam in cfg.lams(reg)]
        for kind, lam in specs:
            tag = f"lo_{kind}_lam{lamstr(lam)}_seed{s}"
            jobs.append((tag, ["job-lo", "--kind", kind, "--lam", str(lam), "--seed", str(s), *sm],
                         D["lo"] / f"{tag}.json"))
    t0 = time.time()
    run_jobs(jobs, a.workers, D["logs"])
    cmd_report(a)
    print(f"[orchestrator] total {(time.time() - t0) / 60:.1f} min", flush=True)


STAGE1 = (0, 1)               # meta-seeds of the full-grid sweep


def cmd_confirm(a):
    """Stage 2: top-k lambdas per regularizer (ranked on stage-1 seeds only) + the
    lambda=0 references, re-trained on FRESH meta-seeds so the winner's-curse of a
    noisy stage-1 ranking cannot leak into the final choice."""
    D = dirs(a.smoke)
    fin_path = D["root"] / "finalists.json"
    if fin_path.exists():
        fin = json.load(open(fin_path))
    else:
        by = {}
        for f in D["lo"].glob("lo_*_seed*.json"):
            j = json.load(open(f))
            if j["seed"] in STAGE1 and j["kind"] in ("l1", "l2"):
                by.setdefault((j["kind"], j["lam"]), []).append(j)
        fin = {}
        for reg in ("l1", "l2"):
            rows = [(np.mean([e["val_acc"] for j in v for e in j["evals"]]),
                     -np.mean([e["val_loss"] for j in v for e in j["evals"]]), lam)
                    for (k, lam), v in by.items() if k == reg]
            fin[reg] = [lam for _, _, lam in sorted(rows, reverse=True)[:a.topk]]
        json.dump(fin, open(fin_path, "w"), indent=1)
    print("[confirm] finalists:", fin, flush=True)
    jobs = []
    for s_ in a.confirm_seeds:
        specs = [("none", 0.0), ("smooth", 0.0)] + [(r, l) for r in ("l1", "l2") for l in fin[r]]
        for kind, lam in specs:
            tag = f"lo_{kind}_lam{lamstr(lam)}_seed{s_}"
            jobs.append((tag, ["job-lo", "--kind", kind, "--lam", str(lam), "--seed", str(s_)],
                         D["lo"] / f"{tag}.json"))
    t0 = time.time()
    run_jobs(jobs, a.workers, D["logs"])
    cmd_report(a)
    print(f"[confirm] total {(time.time() - t0) / 60:.1f} min", flush=True)


# =============================================================================
# report
# =============================================================================
def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summ(evals):
    va = np.array([e["val_acc"] for e in evals])
    return dict(n=len(va), val_acc=float(va.mean()), val_acc_se=se(va),
                val_loss=float(np.mean([e["val_loss"] for e in evals])),
                test_acc=float(np.mean([e["test_acc"] for e in evals])),
                train_acc=float(np.mean([e["train_acc"] for e in evals])),
                l1_norm=float(np.mean([e["l1_norm"] for e in evals])),
                frac_small=float(np.mean([e["frac_small"] for e in evals])),
                per_run=va.tolist(), keys=list(range(len(va))), kind="init")


def lo_summ(per_seed):
    """Summarise a learned optimizer over META-seeds.  Each meta-seed is one
    independent LO (its 3 optimizee inits share that LO), so the SE is taken across
    meta-seed means, not across inits."""
    seeds = sorted(per_seed)
    va = np.array([per_seed[k]["val_acc"] for k in seeds])
    g = lambda m: float(np.mean([per_seed[k][m] for k in seeds]))
    return dict(n=len(seeds), n_meta_seeds=len(seeds), val_acc=float(va.mean()), val_acc_se=se(va),
                val_loss=g("val_loss"), test_acc=g("test_acc"), train_acc=g("train_acc"),
                l1_norm=g("l1_norm"), frac_small=g("frac_small"),
                per_run=va.tolist(), keys=seeds, kind="meta")


def better(s, t):             # s better than t: val acc, tie-break val loss
    return (s["val_acc"], -s["val_loss"]) > (t["val_acc"], -t["val_loss"])


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}"


def cmd_report(a):
    D = dirs(a.smoke)
    # ---- AdamW arms ---------------------------------------------------------
    ad = {}
    for reg in ("none", "l1", "l2"):
        rows = []
        for f in (D["adamw"] / f"adamw_{reg}.json", D["adamw"] / f"adamw_{reg}_x.json"):
            if f.exists():
                rows += [dict(r, **summ(r["evals"])) for r in json.load(open(f))["rows"]]
        if rows:
            ad[reg] = rows
    # ---- LO arms (aggregated over META-seeds) ---------------------------------
    lo_ms = {}                 # (kind, lam) -> {meta_seed: mean metrics over the 3 inits}
    for f in sorted(D["lo"].glob("lo_*_seed*.json")):
        j = json.load(open(f))
        lo_ms.setdefault((j["kind"], j["lam"]), {})[j["seed"]] = {
            m: float(np.mean([e[m] for e in j["evals"]]))
            for m in ("val_acc", "val_loss", "test_acc", "train_acc", "l1_norm", "frac_small")}
    lo = {k: lo_summ(v) for k, v in lo_ms.items()}                       # every meta-seed
    fresh = {k: lo_summ({s_: m for s_, m in v.items() if s_ not in STAGE1})
             for k, v in lo_ms.items() if any(s_ not in STAGE1 for s_ in v)}   # stage-2 only
    use = lambda k: fresh.get(k) or lo.get(k)       # references on the same seeds as the finalists

    out = dict(adamw_best_overall={}, lo_best={}, curves={})
    md = ["# λ sweep — MNIST, MLP 784-20-10, L1 and L2 regularizers\n"]
    md.append("Selection criterion: **mean validation accuracy** (3 000 held-out images, never trained on); "
              "ties broken by validation NLL. AdamW arms: `±` is the SE over 6 optimizee inits. "
              "Learned-optimizer arms: `±` is the SE **across meta-training seeds** (each seed is an independent "
              "LO; its 3 inits share it). Method-4 finalists are re-trained on fresh meta-seeds and the "
              "reported best λ is decided on those fresh seeds only. "
              "Test accuracy is listed for the selected configuration only, as a sanity check.\n")

    base1 = None
    if "none" in ad:
        base1 = max(ad["none"], key=lambda r: (r["val_acc"], -r["val_loss"]))
        out["adamw_best_overall"]["none"] = {k: base1[k] for k in ("lr", "wd", "val_acc", "val_acc_se", "val_loss", "test_acc")}
    base3 = use(("none", 0.0))
    smooth0 = use(("smooth", 0.0))

    for reg in ("l1", "l2"):
        R = reg.upper()
        md.append(f"## {R}\n")
        # method 2: per-lambda profile (best lr, wd for that lambda)
        prof2, full2 = {}, ad.get(reg, [])
        for r in full2:
            if r["lam"] not in prof2 or better(r, prof2[r["lam"]]):
                prof2[r["lam"]] = r
        lams2 = sorted(prof2)
        # method 4: per-lambda
        lams4 = sorted(l for (k, l) in lo if k == reg)
        curves = dict(adamw=[dict(lam=l, **{k: prof2[l][k] for k in ("lr", "wd", "val_acc", "val_acc_se", "val_loss", "test_acc", "n")}) for l in lams2],
                      lo=[dict(lam=l, **{k: lo[(reg, l)][k] for k in ("val_acc", "val_acc_se", "val_loss", "test_acc", "n", "n_meta_seeds")}) for l in lams4],
                      lo_fresh=[dict(lam=l, **{k: fresh[(reg, l)][k] for k in ("val_acc", "val_acc_se", "val_loss", "test_acc", "n_meta_seeds")}) for l in lams4 if (reg, l) in fresh])
        out["curves"][reg] = curves

        def best_of(rows):
            return max(rows, key=lambda r: (r["val_acc"], -r["val_loss"])) if rows else None

        b2 = best_of(full2)
        cand4 = [dict(lam=l, **fresh[(reg, l)]) for l in lams4 if (reg, l) in fresh]
        basis4 = "fresh meta-seeds" if cand4 else "stage-1 meta-seeds (NOT confirmed)"
        b4 = best_of(cand4 or [dict(lam=l, **lo[(reg, l)]) for l in lams4])
        if b2:
            out["adamw_best_overall"][reg] = {k: b2[k] for k in ("lr", "wd", "lam", "val_acc", "val_acc_se", "val_loss", "test_acc")}
        if b4:
            out["lo_best"][reg] = dict({k: b4[k] for k in ("lam", "val_acc", "val_acc_se", "val_loss", "test_acc", "n_meta_seeds")},
                                       basis=basis4)

        md.append(f"### Method 2 — AdamW + λ·{R}  (best lr, wd per λ; {len(full2)} configs)\n")
        if full2:
            md.append("| λ | best lr | best wd | val acc % | ± SE | val NLL | ‖θ‖₁ | frac(|θ|<1e-3) |")
            md.append("|---|---|---|---|---|---|---|---|")
            for l in lams2:
                r = prof2[l]
                star = " ◀ best" if b2 and l == b2["lam"] else ""
                md.append(f"| {l:g} | {r['lr']:g} | {r['wd']:g} | {pct(r['val_acc'])} | {pct(r['val_acc_se'])} | "
                          f"{r['val_loss']:.4f} | {r['l1_norm']:.0f} | {r['frac_small']:.3f} |{star}")
            md.append("")
        md.append(f"### Method 4 — LO + λ·{R} (+ smoothing)\n")
        if lams4:
            md.append("| λ | meta-seeds | val acc % (all) | ± SE | fresh seeds | val acc % (fresh) | ± SE | val NLL (all) | ‖θ‖₁ |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for l in lams4:
                r = lo[(reg, l)]
                fr = fresh.get((reg, l))
                fcols = (f"{fr['n_meta_seeds']} | {pct(fr['val_acc'])} | {pct(fr['val_acc_se'])}" if fr else "– | – | –")
                star = " ◀ best" if b4 and l == b4["lam"] else ""
                md.append(f"| {l:g} | {r['n_meta_seeds']} | {pct(r['val_acc'])} | {pct(r['val_acc_se'])} | {fcols} | "
                          f"{r['val_loss']:.4f} | {r['l1_norm']:.0f} |{star}")
            md.append("")

        # plateau: lambdas statistically indistinguishable from the best (within 1 SE)
        def plateau(rows, best):
            if not best:
                return []
            thr = best["val_acc"] - (best["val_acc_se"] if best["val_acc_se"] == best["val_acc_se"] else 0)
            return sorted(r["lam"] for r in rows if r["val_acc"] >= thr)

        md.append("**Best λ**\n")
        if b2:
            md.append(f"- Method 2: **λ = {b2['lam']:g}** (lr={b2['lr']:g}, wd={b2['wd']:g}) — val {pct(b2['val_acc'])} ± "
                      f"{pct(b2['val_acc_se'])} %; within 1 SE: {', '.join(f'{x:g}' for x in plateau([prof2[l] for l in lams2], b2))}")
            edge = b2["lam"] in (lams2[0], lams2[-1])
            if edge:
                md.append(f"  - ⚠ best λ is at the **edge** of the grid — extend the grid before trusting it.")
        if b4:
            md.append(f"- Method 4: **λ = {b4['lam']:g}** — val {pct(b4['val_acc'])} ± {pct(b4['val_acc_se'])} % "
                      f"({basis4}); within 1 SE: {', '.join(f'{x:g}' for x in plateau(cand4 or [dict(lam=l, **lo[(reg, l)]) for l in lams4], b4))}")
            if b4["lam"] in (lams4[0], lams4[-1]):
                md.append(f"  - ⚠ best λ is at the **edge** of the grid — extend the grid before trusting it.")
        md.append("")

    # ---- paired differences at the selected lambdas (same inits / batch order) ----
    def paired(x, y):
        if not x or not y:
            return "n/a"
        if x.get("kind") == y.get("kind") and x.get("keys") == y.get("keys"):
            d = (np.array(x["per_run"]) - np.array(y["per_run"])) * 100
            return f"{d.mean():+.2f} ± {se(d):.2f} pp  ({int((d > 0).sum())}/{len(d)} paired {'meta-seeds' if x['kind'] == 'meta' else 'runs'} better)"
        d = (x["val_acc"] - y["val_acc"]) * 100
        sd = 100 * np.hypot(*(v if v == v else 0.0 for v in (x["val_acc_se"], y["val_acc_se"])))
        return f"{d:+.2f} ± {sd:.2f} pp  (unpaired: different seed sets)"

    md.append("## Paired validation-accuracy differences at the selected λ\n")
    md.append("AdamW rows: same optimizee inits and batch order on both sides. LO rows: paired by meta-seed (fresh seeds when "
              "available). ± is the SE of the paired differences.\n")
    for reg in ("l1", "l2"):
        R = reg.upper()
        b2 = out["adamw_best_overall"].get(reg)
        b4 = out["lo_best"].get(reg)
        r2 = None
        if b2 and reg in ad:
            r2 = next((r for r in ad[reg] if (r["lr"], r["wd"], r["lam"]) == (b2["lr"], b2["wd"], b2["lam"])), None)
        r4 = use((reg, b4["lam"])) if b4 else None
        md.append(f"- **{R}**: (2) AdamW+{R} − (1) AdamW = {paired(r2, base1)}")
        md.append(f"  - (4) LO+{R} − (3) LO = {paired(r4, base3)}")
        md.append(f"  - (4) LO+{R} − LO+smoothing-only = {paired(r4, smooth0)}")
        md.append(f"  - (4) LO+{R} − (2) AdamW+{R} = {paired(r4, r2)}")
    md.append("")

    md.append("## λ = 0 references (no explicit regularizer)\n")
    md.append("| Method | config | val acc % | ± SE | val NLL | test acc % |")
    md.append("|---|---|---|---|---|---|")
    if base1:
        md.append(f"| 1. AdamW | lr={base1['lr']:g}, wd={base1['wd']:g} | {pct(base1['val_acc'])} | "
                  f"{pct(base1['val_acc_se'])} | {base1['val_loss']:.4f} | {pct(base1['test_acc'])} |")
    if base3:
        md.append(f"| 3. LO | {base3['n_meta_seeds']} meta-seeds | {pct(base3['val_acc'])} | "
                  f"{pct(base3['val_acc_se'])} | {base3['val_loss']:.4f} | {pct(base3['test_acc'])} |")
    if smooth0:
        md.append(f"| (diagnostic) LO + smoothing only | {smooth0['n_meta_seeds']} meta-seeds | {pct(smooth0['val_acc'])} | "
                  f"{pct(smooth0['val_acc_se'])} | {smooth0['val_loss']:.4f} | {pct(smooth0['test_acc'])} |")
    md.append("")
    md.append("Method 4 contains the smoothing term (as the proposed model does); the diagnostic row "
              "is its λ = 0 limit, so *method 4 vs diagnostic* isolates R.\n")

    out["refs"] = dict(adamw_none=out["adamw_best_overall"].get("none"),
                       lo_none=base3 and {k: base3[k] for k in ("val_acc", "val_acc_se", "val_loss", "test_acc")},
                       lo_smooth=smooth0 and {k: smooth0[k] for k in ("val_acc", "val_acc_se", "val_loss", "test_acc")})
    (D["root"] / "RESULTS.md").write_text("\n".join(md))
    json.dump(out, open(D["root"] / "best_lambda.json", "w"), indent=1)
    try:
        _plot(D, out)
    except Exception as e:                           # plotting must never lose the numbers
        print("plot failed:", repr(e))
    print("\n".join(md))


# palette: reference instance slots 1 (blue) / 2 (orange); light surface
BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _plot(D, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    refs = out["refs"]
    for ax, reg in zip(axes, ("l1", "l2")):
        ax.set_facecolor(SURF)
        cv = out["curves"].get(reg, dict(adamw=[], lo=[]))
        for key, color, label in (("adamw", BLUE, f"2. AdamW + {reg.upper()}"),
                                  ("lo", ORANGE, f"4. LO + {reg.upper()}")):
            pts = [p for p in cv[key] if p["lam"] > 0]
            if not pts:
                continue
            x = [p["lam"] for p in pts]
            y = [100 * p["val_acc"] for p in pts]
            e = [100 * (p["val_acc_se"] if p["val_acc_se"] == p["val_acc_se"] else 0) for p in pts]
            ax.errorbar(x, y, yerr=e, color=color, lw=2, marker="o", ms=6, mfc=color, mec=SURF,
                        mew=1.5, capsize=3, label=label, zorder=3)
            if key == "lo" and cv.get("lo_fresh"):
                fx = [p["lam"] for p in cv["lo_fresh"]]
                fy = [100 * p["val_acc"] for p in cv["lo_fresh"]]
                fe = [100 * (p["val_acc_se"] if p["val_acc_se"] == p["val_acc_se"] else 0) for p in cv["lo_fresh"]]
                ax.errorbar(fx, fy, yerr=fe, color=color, lw=0, marker="D", ms=8, mfc=SURF, mec=color,
                            mew=2, capsize=3, elinewidth=1.5, zorder=4, label="4. finalists, fresh meta-seeds")
                i = int(np.argmax(fy))
                ax.annotate(f"λ={fx[i]:g}\n{fy[i]:.2f}%", (fx[i], fy[i]), xytext=(0, 12),
                            textcoords="offset points", ha="center", fontsize=9, color=INK)
            else:
                i = int(np.argmax(y))
                ax.annotate(f"λ={x[i]:g}\n{y[i]:.2f}%", (x[i], y[i]), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=9, color=INK)
        for key, color, label in (("adamw_none", BLUE, "1. AdamW"), ("lo_none", ORANGE, "3. LO")):
            r = refs.get(key)
            if r:
                ax.axhline(100 * r["val_acc"], color=color, lw=1.5, ls=(0, (5, 3)), zorder=2, label=f"{label} (λ=0)")
        ax.set_xscale("log")
        ax.margins(x=0.08, y=0.22)
        ax.set_xlabel(f"λ  (penalty λ·{reg.upper()}(θ))", color=INK2)
        ax.set_title(f"{reg.upper()} regularizer", color=INK, loc="left", fontsize=12, pad=12)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    axes[0].set_ylabel("validation accuracy (%)", color=INK2)
    fig.tight_layout()
    fig.savefig(D["root"] / "lambda_sweep.png", dpi=150, facecolor=SURF)


# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "report", "confirm", "job-adamw", "job-lo"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--meta-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--confirm-seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--wds", type=float, nargs="+", default=None)
    p.add_argument("--reg", default="none", choices=["none", "l1", "l2"])
    p.add_argument("--kind", default="none", choices=["none", "smooth", "l1", "l2"])
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, str(HERE))
    {"all": cmd_all, "report": cmd_report, "confirm": cmd_confirm, "job-adamw": job_adamw, "job-lo": job_lo}[a.cmd](a)

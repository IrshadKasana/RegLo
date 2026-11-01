#!/usr/bin/env python
"""
run_experiment.py — stand-alone L1 / L2 comparison on MNIST (4 methods).

    1. MLP + AdamW
    2. MLP + AdamW + R(theta)                R in {L1, L2}
    3. MLP + learned optimizer (LO)
    4. MLP + LO meta-trained with R(theta) (+ smoothing term, like the proposed model)

Everything is trained and then evaluated by this script.  Phases
  tune  : lr (AdamW arms) and lambda (arms 2 & 4) are chosen on the VALIDATION split
  final : multi-seed training with the chosen hyper-parameters; TEST accuracy reported
  report: tables (RESULTS.md / results.json) and a bar chart

    python run_experiment.py all                  # whole pipeline
    python run_experiment.py all --smoke          # tiny pipeline check (~1 min)
    python run_experiment.py report               # re-aggregate existing results

Re-running skips jobs whose result JSON already exists (safe to resume).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable


def lamstr(lam):
    return f"{lam:g}"


def out_dirs(smoke):
    root = HERE / "results" / ("smoke" if smoke else "full")
    d = {k: root / k for k in ("tune", "final", "ckpt", "logs")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


def make_cfg(smoke):
    import common as C
    return C.Cfg(**C.SMOKE) if smoke else C.Cfg()


def tune_eval_seeds(cfg):
    return [50_000 + i for i in range(cfg.n_inits)]


def final_eval_seeds(cfg, seed):
    return [10_000 + 100 * seed + i for i in range(cfg.n_inits)]


def mean_dict(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# =============================================================================
# jobs (each runs in its own process)
# =============================================================================
def job_adamw_tune(a):
    import common as C
    cfg, D = make_cfg(a.smoke), out_dirs(a.smoke)
    data = C.load_mnist()
    grid = []
    t0 = time.time()
    for reg in ("none", "l1", "l2"):
        lams = (0.0,) if reg == "none" else cfg.lams(reg)
        for lr in cfg.adamw_lrs:
            for lam in lams:
                rows = [C.run_adamw(s, cfg, data, lr, reg, lam) for s in tune_eval_seeds(cfg)]
                m = mean_dict(rows)
                grid.append(dict(reg=reg, lr=lr, lam=lam, **m))
                print(f"adamw reg={reg:4s} lr={lr:g} lam={lam:g}  val_acc={m['val_acc']:.4f} "
                      f"val_loss={m['val_loss']:.4f}", flush=True)
    json.dump(dict(grid=grid, seconds=time.time() - t0), open(D["tune"] / "adamw.json", "w"), indent=1)


def job_lo(a):
    import common as C
    cfg, D = make_cfg(a.smoke), out_dirs(a.smoke)
    data = C.load_mnist()
    phase_dir = D["tune"] if a.phase == "tune" else D["final"]
    tag = (f"lo_{a.kind}_lam{lamstr(a.lam)}_seed{a.seed}" if a.phase == "tune"
           else f"lo_{a.kind}_seed{a.seed}")
    ckpt = D["ckpt"] / f"{a.phase}_{tag}.pt"
    lo, info = C.meta_train_lo(a.kind, a.lam, a.seed, cfg, data, ckpt,
                               log=lambda s: print(s, flush=True))
    seeds = tune_eval_seeds(cfg) if a.phase == "tune" else final_eval_seeds(cfg, a.seed)
    rows = [C.run_lo(lo, s, cfg, data) for s in seeds]
    res = dict(kind=a.kind, lam=a.lam, seed=a.seed, phase=a.phase, eval_seeds=seeds,
               evals=rows, mean=mean_dict(rows), meta_train=info, cfg=asdict(cfg), device=str(C.device))
    json.dump(res, open(phase_dir / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: val_acc={res['mean']['val_acc']:.4f} test_acc={res['mean']['test_acc']:.4f}")


def job_reeval(a):
    """Seed-0 reuse: evaluate the checkpoint meta-trained in the tune phase on the
    FINAL eval seeds (selection used validation accuracy only; test untouched)."""
    import torch
    import common as C
    cfg, D = make_cfg(a.smoke), out_dirs(a.smoke)
    data = C.load_mnist()
    tag = f"lo_{a.kind}_lam{lamstr(a.lam)}_seed{a.seed}"
    src = json.load(open(D["tune"] / f"{tag}.json"))
    lo = C.make_lo(cfg)
    lo.load_state_dict(torch.load(D["ckpt"] / f"tune_{tag}.pt", map_location=C.device))
    seeds = final_eval_seeds(cfg, a.seed)
    rows = [C.run_lo(lo, s, cfg, data) for s in seeds]
    res = dict(kind=a.kind, lam=a.lam, seed=a.seed, phase="final (tune checkpoint reused)",
               eval_seeds=seeds, evals=rows, mean=mean_dict(rows),
               meta_train=src["meta_train"], cfg=src["cfg"], device=str(C.device))
    json.dump(res, open(D["final"] / f"lo_{a.kind}_seed{a.seed}.json", "w"), indent=1)
    print(f"{tag} re-evaluated: test_acc={res['mean']['test_acc']:.4f}")


def job_adamw_final(a):
    import common as C
    cfg, D = make_cfg(a.smoke), out_dirs(a.smoke)
    sel = json.load(open(D["root"] / "selection.json"))
    data = C.load_mnist()
    seeds = final_eval_seeds(cfg, a.seed)
    res = dict(seed=a.seed, eval_seeds=seeds)
    for reg in ("none", "l1", "l2"):
        s = sel[f"adamw_{reg}"]
        rows = [C.run_adamw(sd, cfg, data, s["lr"], reg, s.get("lam", 0.0)) for sd in seeds]
        res[reg] = dict(lr=s["lr"], lam=s.get("lam", 0.0), evals=rows, mean=mean_dict(rows))
    json.dump(res, open(D["final"] / f"adamw_seed{a.seed}.json", "w"), indent=1)
    print("adamw seed", a.seed, {r: round(res[r]["mean"]["test_acc"], 4) for r in ("none", "l1", "l2")})


# =============================================================================
# orchestration
# =============================================================================
def run_jobs(jobs, gpu_lanes, cpu_lanes, cpu_threads, logdir):
    """jobs: list of (name, argv, result_path). Skips finished ones.

    Throughput note (measured): several processes sharing the single GPU just
    time-slice (N procs => N x slower each, no gain; CUDA MPS is unavailable
    here), while these tiny-kernel jobs run ~5x slower on a CPU core group but
    scale across cores.  So: `gpu_lanes` workers see the GPU, `cpu_lanes`
    workers run with CUDA hidden, all pulling from one shared queue.
    """
    todo = [(n, argv, rp) for n, argv, rp in jobs if not Path(rp).exists()]
    print(f"[orchestrator] {len(jobs) - len(todo)} done, {len(todo)} to run "
          f"({gpu_lanes} gpu lane(s), {cpu_lanes} cpu lane(s) x {cpu_threads} threads)", flush=True)
    q = queue.Queue()
    for j in todo:
        q.put(j)
    fails = []

    def lane(dev):
        env = dict(os.environ)
        if dev == "cpu":
            env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(cpu_threads),
                       MKL_NUM_THREADS=str(cpu_threads))
        else:
            env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        while True:
            try:
                name, argv, rp = q.get_nowait()
            except queue.Empty:
                return
            t = time.time()
            with open(logdir / f"{name}.log", "w") as fh:
                rc = subprocess.call([PY, str(HERE / "run_experiment.py"), *argv],
                                     stdout=fh, stderr=subprocess.STDOUT, env=env)
            if rc:
                fails.append(name)
            print(f"[orchestrator] {'ok  ' if rc == 0 else 'FAIL'} {name} on {dev} "
                  f"({time.time() - t:.0f}s)", flush=True)

    lanes = ["gpu"] * gpu_lanes + ["cpu"] * cpu_lanes
    with ThreadPoolExecutor(max_workers=len(lanes)) as ex:
        list(ex.map(lane, lanes))
    if fails:
        raise SystemExit(f"jobs failed: {fails} — see results/*/logs")


def select(D, cfg):
    """Pick hyper-parameters by mean VALIDATION accuracy (tie-break: val loss)."""
    key = lambda r: (-r["val_acc"], r["val_loss"])
    grid = json.load(open(D["tune"] / "adamw.json"))["grid"]
    sel = {}
    for reg in ("none", "l1", "l2"):
        b = min([g for g in grid if g["reg"] == reg], key=key)
        sel[f"adamw_{reg}"] = dict(lr=b["lr"], lam=b["lam"], val_acc=b["val_acc"])
    for reg in ("l1", "l2"):
        rows = [json.load(open(D["tune"] / f"lo_{reg}_lam{lamstr(l)}_seed{TUNE_SEED}.json"))
                for l in cfg.lams(reg)]
        b = min(rows, key=lambda r: key(r["mean"]))
        sel[f"lo_{reg}"] = dict(lam=b["lam"], val_acc=b["mean"]["val_acc"])
        sel[f"lo_{reg}_grid"] = {lamstr(r["lam"]): round(r["mean"]["val_acc"], 4) for r in rows}
    json.dump(sel, open(D["root"] / "selection.json", "w"), indent=1)
    return sel


TUNE_SEED = 0      # seed-0 tuning checkpoints are re-used as final seed 0


def cmd_all(a):
    cfg, D = make_cfg(a.smoke), out_dirs(a.smoke)
    sm = ["--smoke"] if a.smoke else []
    t0 = time.time()

    # ---- phase 1: tuning ----
    jobs = [("tune_adamw", ["job-adamw-tune", *sm], D["tune"] / "adamw.json")]
    for kind in ("none", "smooth"):          # lambda-free arms: trained once, seed TUNE_SEED
        jobs.append((f"tune_lo_{kind}", ["job-lo", "--kind", kind, "--lam", "0.0",
                                         "--seed", str(TUNE_SEED), "--phase", "tune", *sm],
                     D["tune"] / f"lo_{kind}_lam0_seed{TUNE_SEED}.json"))
    for reg in ("l1", "l2"):
        for lam in cfg.lams(reg):
            jobs.append((f"tune_lo_{reg}_{lamstr(lam)}",
                         ["job-lo", "--kind", reg, "--lam", str(lam), "--seed", str(TUNE_SEED),
                          "--phase", "tune", *sm],
                         D["tune"] / f"lo_{reg}_lam{lamstr(lam)}_seed{TUNE_SEED}.json"))
    run_jobs(jobs, a.gpu_lanes, a.cpu_lanes, a.cpu_threads, D["logs"])
    sel = select(D, cfg)
    print("[orchestrator] selection:\n" + json.dumps(sel, indent=1), flush=True)

    # ---- phase 2: final multi-seed ----
    jobs = []
    for s in a.seeds:
        jobs.append((f"final_adamw_s{s}", ["job-adamw-final", "--seed", str(s), *sm],
                     D["final"] / f"adamw_seed{s}.json"))
        for kind, lam in (("none", 0.0), ("smooth", 0.0),
                          ("l1", sel["lo_l1"]["lam"]), ("l2", sel["lo_l2"]["lam"])):
            if s == TUNE_SEED:
                jobs.append((f"final_reeval_{kind}_s{s}",
                             ["job-reeval", "--kind", kind, "--lam", str(lam), "--seed", str(s), *sm],
                             D["final"] / f"lo_{kind}_seed{s}.json"))
            else:
                jobs.append((f"final_lo_{kind}_s{s}",
                             ["job-lo", "--kind", kind, "--lam", str(lam), "--seed", str(s),
                              "--phase", "final", *sm],
                             D["final"] / f"lo_{kind}_seed{s}.json"))
    run_jobs(jobs, a.gpu_lanes, a.cpu_lanes, a.cpu_threads, D["logs"])
    cmd_report(a)
    print(f"[orchestrator] total {(time.time() - t0) / 60:.1f} min")


# =============================================================================
# report
# =============================================================================
def _ms(x, pct=True, nd=2):
    x = np.asarray(x, float) * (100 if pct else 1)
    sd = x.std(ddof=1) if len(x) > 1 else float("nan")
    return f"{x.mean():.{nd}f} ± {sd:.{nd}f}"


def cmd_report(a):
    D = out_dirs(a.smoke)
    sel = json.load(open(D["root"] / "selection.json"))
    seeds = sorted(int(p.stem.split("seed")[1]) for p in D["final"].glob("adamw_seed*.json"))
    ad = {s: json.load(open(D["final"] / f"adamw_seed{s}.json")) for s in seeds}
    lo = {(k, s): json.load(open(D["final"] / f"lo_{k}_seed{s}.json"))
          for k in ("none", "smooth", "l1", "l2") for s in seeds
          if (D["final"] / f"lo_{k}_seed{s}.json").exists()}

    def col(getter, field):
        return np.array([getter(s)[field] for s in seeds])

    arms = {}   # name -> {metric: per-seed array (mean over optimizee inits)}
    for reg in ("none", "l1", "l2"):
        arms[f"adamw_{reg}"] = {m: col(lambda s, r=reg: ad[s][r]["mean"], m)
                                for m in ad[seeds[0]]["none"]["mean"]}
    for k in ("none", "smooth", "l1", "l2"):
        arms[f"lo_{k}"] = {m: col(lambda s, kk=k: lo[(kk, s)]["mean"], m)
                           for m in lo[("none", seeds[0])]["mean"]}

    def table(reg):
        R = reg.upper()
        lam_ad, lr_ad = sel[f"adamw_{reg}"]["lam"], sel[f"adamw_{reg}"]["lr"]
        lam_lo = sel[f"lo_{reg}"]["lam"]
        rows = [
            ("1. MLP + AdamW", f"lr={sel['adamw_none']['lr']:g}, wd={0.01}", "adamw_none"),
            (f"2. MLP + AdamW + {R}", f"lr={lr_ad:g}, λ={lam_ad:g}", f"adamw_{reg}"),
            ("3. MLP + LO", "meta-lr=1e-3", "lo_none"),
            (f"4. MLP + LO + {R}", f"λ={lam_lo:g}, + smoothing", f"lo_{reg}"),
        ]
        out = [f"| Method | hyper-params | Test acc (%) | Test NLL | Val acc (%) | Train acc (%) | ‖θ‖₁ | frac(abs θ < 1e-3) |",
               "|---|---|---|---|---|---|---|---|"]
        for name, hp, key in rows:
            m = arms[key]
            out.append(f"| {name} | {hp} | **{_ms(m['test_acc'])}** | {_ms(m['test_loss'], False, 4)} | "
                       f"{_ms(m['val_acc'])} | {_ms(m['train_acc'])} | {m['l1_norm'].mean():.0f} | "
                       f"{m['frac_small'].mean():.3f} |")
        return "\n".join(out)

    def paired(a_key, b_key):
        d = (arms[a_key]["test_acc"] - arms[b_key]["test_acc"]) * 100
        wins = int((d > 0).sum())
        return f"{d.mean():+.2f} ± {d.std(ddof=1) if len(d) > 1 else float('nan'):.2f} pp ({wins}/{len(d)} seeds better)"

    md = []
    md.append(f"# MNIST: MLP + {{AdamW, learned optimizer}} × {{none, L1, L2}}\n")
    md.append(f"Seeds: {seeds} (each = fresh LO meta-training + {a_n_inits(D)} optimizee inits). "
              f"Numbers are mean ± std over seeds; **test accuracy is measured once, after training**, "
              f"at the final of {a_steps(D)} optimizee steps. Hyper-parameters were chosen on a held-out "
              f"validation split, never on test.\n")
    for reg in ("l1", "l2"):
        md.append(f"## Experiment: {reg.upper()} regularizer\n")
        md.append(table(reg) + "\n")
        md.append("Paired test-accuracy differences (same optimizee init + batch order):\n")
        md.append(f"- (2) AdamW+{reg.upper()} − (1) AdamW : {paired(f'adamw_{reg}', 'adamw_none')}")
        md.append(f"- (4) LO+{reg.upper()} − (3) LO       : {paired(f'lo_{reg}', 'lo_none')}")
        md.append(f"- (3) LO − (1) AdamW                  : {paired('lo_none', 'adamw_none')}")
        md.append(f"- (4) LO+{reg.upper()} − (2) AdamW+{reg.upper()} : {paired(f'lo_{reg}', f'adamw_{reg}')}")
        md.append(f"- (4) LO+{reg.upper()} − diagnostic LO+smoothing-only : {paired(f'lo_{reg}', 'lo_smooth')}\n")
    m = arms["lo_smooth"]
    md.append("## Diagnostic (not one of the 4 methods)\n")
    md.append(f"LO meta-trained with the smoothing term only (no L1/L2): test acc "
              f"**{_ms(m['test_acc'])}** %. Method 4 contains the smoothing term (as the proposed "
              f"model does), so method 4 vs method 3 mixes the effect of smoothing and of R; "
              f"compare against this row to isolate R.\n")
    md.append("## Selected hyper-parameters (validation accuracy)\n")
    md.append("```json\n" + json.dumps(sel, indent=1) + "\n```\n")
    (D["root"] / "RESULTS.md").write_text("\n".join(md))

    summary = {k: {m: dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else None,
                           per_seed=v.tolist()) for m, v in d.items()} for k, d in arms.items()}
    json.dump(dict(seeds=seeds, selection=sel, arms=summary),
              open(D["root"] / "results.json", "w"), indent=1)
    _plot(D, arms, seeds)
    print("\n".join(md))


def a_n_inits(D):
    f = next(iter(D["final"].glob("lo_none_seed*.json")))
    return len(json.load(open(f))["eval_seeds"])


def a_steps(D):
    f = next(iter(D["final"].glob("lo_none_seed*.json")))
    return json.load(open(f))["cfg"]["eval_steps"]


def _plot(D, arms, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = ["#8a8f98", "#4c78a8", "#d9a441", "#c0504d"]
    for ax, reg in zip(axes, ("l1", "l2")):
        keys = ["adamw_none", f"adamw_{reg}", "lo_none", f"lo_{reg}"]
        labels = ["1. AdamW", f"2. AdamW+{reg.upper()}", "3. LO", f"4. LO+{reg.upper()}"]
        mu = [arms[k]["test_acc"].mean() * 100 for k in keys]
        sd = [arms[k]["test_acc"].std(ddof=1) * 100 if len(seeds) > 1 else 0 for k in keys]
        ax.bar(range(4), mu, yerr=sd, color=colors, capsize=4)
        for i, k in enumerate(keys):
            ax.scatter([i] * len(seeds), arms[k]["test_acc"] * 100, s=12, color="k", zorder=3)
            ax.text(i, mu[i] + sd[i] + 0.05, f"{mu[i]:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(f"{reg.upper()} regularizer")
        ax.set_ylim(min(mu) - 1.5, max(mu) + 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MNIST test accuracy (%)")
    fig.tight_layout()
    fig.savefig(D["root"] / "test_accuracy.png", dpi=150)


# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "report", "job-adamw-tune", "job-adamw-final", "job-lo", "job-reeval"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gpu_lanes", type=int, default=1)
    p.add_argument("--cpu_lanes", type=int, default=0)
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--kind", default="none", choices=["none", "smooth", "l1", "l2"])
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="final", choices=["tune", "final"])
    a = p.parse_args()
    sys.path.insert(0, str(HERE))
    {"all": cmd_all, "report": cmd_report, "job-adamw-tune": job_adamw_tune,
     "job-adamw-final": job_adamw_final, "job-lo": job_lo,
     "job-reeval": job_reeval}[a.cmd](a)

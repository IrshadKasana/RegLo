#!/usr/bin/env python
"""
run_sam.py — the 4-method comparison with SAM as the regularizer (MNIST, MLP optimizee).

    1. MLP + AdamW                 (re-used from results/<base>/ : same seeds / eval inits)
    2. MLP + AdamW + SAM           lr x rho tuned on the validation split
    3. MLP + learned optimizer     (re-used from results/<base>/)
    4. MLP + LO + SAM              proposed-model objective (penalty mode), lambda_r tuned on val

    python run_sam.py all [--smoke] [--base full] [--threads 3]
    python run_sam.py report

New artefacts go to results/sam/{tune,final,ckpt,logs}; nothing from the L1/L2 run is modified.
"""
from __future__ import annotations
import argparse, json, os, queue, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
SEEDS = (0, 1, 2)
RHOS = (0.01, 0.05, 0.2, 1.0)       # AdamW+SAM neighbourhood radius (adaptive SAM)
LAMS_R = (0.01, 0.1, 1.0)           # LO-SAM penalty weight; rho_phi fixed at the repo default 0.05
SMOKE_RHOS, SMOKE_LAMS = (0.05,), (0.1,)


def dirs(smoke):
    root = HERE / "results" / ("sam_smoke" if smoke else "sam")
    d = {k: root / k for k in ("tune", "final", "ckpt", "logs")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


def cfg_of(smoke):
    import common as C
    return C.Cfg(**C.SMOKE) if smoke else C.Cfg()


def rhos(smoke): return SMOKE_RHOS if smoke else RHOS
def lams(smoke): return SMOKE_LAMS if smoke else LAMS_R
def ls(x): return f"{x:g}"
def tune_seeds(cfg): return [50_000 + i for i in range(cfg.n_inits)]
def final_seeds(cfg, s): return [10_000 + 100 * s + i for i in range(cfg.n_inits)]
def mean_dict(rows): return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# --------------------------------- jobs ---------------------------------------
def job_adamw_tune(a):
    import common as C, sam_extension as S
    cfg, D = cfg_of(a.smoke), dirs(a.smoke)
    data = C.load_mnist()
    grid = []
    for lr in cfg.adamw_lrs:
        for rho in rhos(a.smoke):
            rows = [S.run_adamw_sam(s, cfg, data, lr, rho) for s in tune_seeds(cfg)]
            m = mean_dict(rows)
            grid.append(dict(lr=lr, rho=rho, **m))
            print(f"adamw+sam lr={lr:g} rho={rho:g} val_acc={m['val_acc']:.4f}", flush=True)
    json.dump(dict(grid=grid), open(D["tune"] / "adamw_sam.json", "w"), indent=1)


def job_adamw_final(a):
    import common as C, sam_extension as S
    cfg, D = cfg_of(a.smoke), dirs(a.smoke)
    grid = json.load(open(D["tune"] / "adamw_sam.json"))["grid"]
    b = min(grid, key=lambda r: (-r["val_acc"], r["val_loss"]))
    data = C.load_mnist()
    seeds = final_seeds(cfg, a.seed)
    rows = [S.run_adamw_sam(s, cfg, data, b["lr"], b["rho"]) for s in seeds]
    json.dump(dict(seed=a.seed, lr=b["lr"], rho=b["rho"], eval_seeds=seeds, evals=rows,
                   mean=mean_dict(rows)), open(D["final"] / f"adamw_sam_seed{a.seed}.json", "w"), indent=1)


def job_lo_sam(a):
    import common as C, sam_extension as S
    cfg, D = cfg_of(a.smoke), dirs(a.smoke)
    data = C.load_mnist()
    tag = f"lo_sam_lam{ls(a.lam)}_seed{a.seed}"
    lo, info = S.meta_train_lo_sam(a.lam, a.seed, cfg, data, D["ckpt"] / f"{tag}.pt",
                                   log=lambda s: print(s, flush=True))
    seeds = final_seeds(cfg, a.seed)
    rows = [C.run_lo(lo, s, cfg, data) for s in seeds]
    json.dump(dict(lam_r=a.lam, rho=0.05, seed=a.seed, eval_seeds=seeds, evals=rows,
                   mean=mean_dict(rows), meta_train=info, cfg=asdict(cfg), device=str(C.device)),
              open(D["final"] / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: val={rows and mean_dict(rows)['val_acc']:.4f} test={mean_dict(rows)['test_acc']:.4f}")


# ------------------------------ orchestration ----------------------------------
def run_jobs(jobs, lanes, threads, logdir):
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(f"[sam] {len(jobs) - len(todo)} done, {len(todo)} to run, {lanes} lanes x {threads} threads", flush=True)
    q = queue.Queue()
    for j in todo: q.put(j)
    fails = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if a_cpu else os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def lane(_):
        while True:
            try: name, argv, rp = q.get_nowait()
            except queue.Empty: return
            t = time.time()
            with open(logdir / f"{name}.log", "w") as fh:
                rc = subprocess.call([PY, str(HERE / "run_sam.py"), *argv], stdout=fh,
                                     stderr=subprocess.STDOUT, env=env)
            if rc: fails.append(name)
            print(f"[sam] {'ok  ' if rc == 0 else 'FAIL'} {name} ({time.time() - t:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=lanes) as ex:
        list(ex.map(lane, range(lanes)))
    if fails: raise SystemExit(f"failed: {fails}")


a_cpu = True


def cmd_all(a):
    D = dirs(a.smoke); sm = ["--smoke"] if a.smoke else []
    t0 = time.time()
    jobs = []
    for lam in lams(a.smoke):
        for s in SEEDS:
            jobs.append((f"lo_sam_{ls(lam)}_s{s}", ["job-lo-sam", "--lam", str(lam), "--seed", str(s), *sm],
                         D["final"] / f"lo_sam_lam{ls(lam)}_seed{s}.json"))
    jobs.append(("adamw_sam_tune", ["job-adamw-tune", *sm], D["tune"] / "adamw_sam.json"))
    run_jobs(jobs, a.lanes, a.threads, D["logs"])
    run_jobs([(f"adamw_sam_final_s{s}", ["job-adamw-final", "--seed", str(s), *sm],
               D["final"] / f"adamw_sam_seed{s}.json") for s in SEEDS], len(SEEDS), 1, D["logs"])
    cmd_report(a)
    print(f"[sam] total {(time.time() - t0) / 60:.1f} min")


# --------------------------------- report ---------------------------------------
def cmd_report(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = dirs(a.smoke); base = HERE / "results" / a.base
    def load(p): return json.load(open(p))
    # method 2 / 4 selection (validation accuracy, mean over seeds)
    ad = [load(D["final"] / f"adamw_sam_seed{s}.json") for s in SEEDS]
    cand = {}
    for lam in lams(a.smoke):
        rs = [load(D["final"] / f"lo_sam_lam{ls(lam)}_seed{s}.json") for s in SEEDS]
        cand[lam] = rs
    key = lambda lam: (-np.mean([r["mean"]["val_acc"] for r in cand[lam]]),
                       np.mean([r["mean"]["val_loss"] for r in cand[lam]]))
    best = min(cand, key=key)
    arms = {
        "1. MLP + AdamW": [load(base / "final" / f"adamw_seed{s}.json")["none"]["mean"] for s in SEEDS],
        "2. MLP + AdamW + SAM": [r["mean"] for r in ad],
        "3. MLP + learned optimizer": [load(base / "final" / f"lo_none_seed{s}.json")["mean"] for s in SEEDS],
        "4. MLP + LO + SAM": [r["mean"] for r in cand[best]],
    }
    diag = [load(base / "final" / f"lo_smooth_seed{s}.json")["mean"] for s in SEEDS]
    hp = {"1. MLP + AdamW": f"lr={load(base / 'selection.json')['adamw_none']['lr']:g}",
          "2. MLP + AdamW + SAM": f"lr={ad[0]['lr']:g}, ρ={ad[0]['rho']:g} (adaptive)",
          "3. MLP + learned optimizer": "meta-lr=1e-3",
          "4. MLP + LO + SAM": f"λ_r={best:g}, ρ=0.05, + smoothing"}
    g = lambda rows, k: np.array([r[k] for r in rows])
    ms = lambda v, p=100, nd=2: f"{(v * p).mean():.{nd}f} ± {(v * p).std(ddof=1) * 1 if len(v) > 1 else float('nan'):.{nd}f}"
    ms = lambda v, p=100, nd=2: f"{(v * p).mean():.{nd}f} ± {(v * p).std(ddof=1):.{nd}f}"
    md = ["# MNIST, MLP optimizee: SAM as the regularizer\n",
          f"3 seeds (each = fresh LO meta-training and {cfg_of(a.smoke).n_inits} optimizee inits); mean ± std over seeds; "
          "test accuracy after 1000 optimizee steps. Methods 1 and 3 are re-used from the L1/L2 run "
          f"(`results/{a.base}`, same seeds and eval inits).\n",
          "| Method | hyper-params | Test acc (%) | Test NLL | Val acc (%) | Train acc (%) | per-seed test acc |",
          "|---|---|---|---|---|---|---|"]
    for n, rows in arms.items():
        t = g(rows, "test_acc")
        md.append(f"| {n} | {hp[n]} | **{ms(t)}** | {ms(g(rows, 'test_loss'), 1, 4)} | {ms(g(rows, 'val_acc'))} | "
                  f"{ms(g(rows, 'train_acc'))} | {', '.join(f'{v * 100:.2f}' for v in t)} |")
    pd = lambda x, y: (lambda d: f"{d.mean():+.2f} ± {d.std(ddof=1):.2f} pp ({int((d > 0).sum())}/{len(d)} seeds better)")(
        (g(arms[x], "test_acc") - g(arms[y], "test_acc")) * 100)
    md += ["\nPaired test-accuracy differences (same optimizee init + batch order):\n",
           f"- (2) AdamW+SAM − (1) AdamW : {pd('2. MLP + AdamW + SAM', '1. MLP + AdamW')}",
           f"- (4) LO+SAM − (3) LO       : {pd('4. MLP + LO + SAM', '3. MLP + learned optimizer')}",
           f"- (3) LO − (1) AdamW        : {pd('3. MLP + learned optimizer', '1. MLP + AdamW')}",
           f"- (4) LO+SAM − (2) AdamW+SAM: {pd('4. MLP + LO + SAM', '2. MLP + AdamW + SAM')}",
           f"\nDiagnostic (LO with smoothing only, no SAM): {ms(g(diag, 'test_acc'))} %.\n",
           "λ_r selection (mean validation accuracy over seeds): " +
           ", ".join(f"{l:g}: {np.mean([r['mean']['val_acc'] for r in cand[l]]) * 100:.2f}" for l in cand)]
    (D["root"] / "RESULTS_SAM.md").write_text("\n".join(md) + "\n")
    # image table + csv
    cells = [[n, hp[n], ms(g(r, "test_acc"))] for n, r in arms.items()]
    fig, ax = plt.subplots(figsize=(9.2, 3.0)); ax.axis("off")
    t = ax.table(cellText=cells, colLabels=["Method", "hyper-params", "SAM experiment\ntest acc (%)"],
                 cellLoc="center", loc="center", colWidths=[0.40, 0.34, 0.26]); t.auto_set_font_size(False)
    t.set_fontsize(10.5); t.scale(1, 2.3)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#b8bcc4")
        if i == 0: c.set_facecolor("#e8eaee"); c.set_text_props(weight="bold")
        elif j == 2: c.set_text_props(weight="bold")
    ax.set_title("MNIST test accuracy, SAM regularizer, mean ± std over 3 seeds", fontsize=11, pad=8)
    fig.tight_layout(); fig.savefig(D["root"] / "sam_table.png", dpi=170, facecolor="white")
    with open(D["root"] / "sam_table.csv", "w") as f:
        f.write("method,hyperparams,test_acc_mean,test_acc_std,seed0,seed1,seed2\n")
        for n, r in arms.items():
            t = g(r, "test_acc") * 100
            f.write(f'"{n}","{hp[n]}",{t.mean():.2f},{t.std(ddof=1):.2f},' + ",".join(f"{v:.2f}" for v in t) + "\n")
    json.dump(dict(best_lam_r=best, arms={n: {k: g(r, k).tolist() for k in r[0]} for n, r in arms.items()}),
              open(D["root"] / "results_sam.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "report", "job-adamw-tune", "job-adamw-final", "job-lo-sam"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--base", default="full", help="results/<base> holding methods 1 & 3")
    p.add_argument("--lanes", type=int, default=10)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, str(HERE))
    {"all": cmd_all, "report": cmd_report, "job-adamw-tune": job_adamw_tune,
     "job-adamw-final": job_adamw_final, "job-lo-sam": job_lo_sam}[a.cmd](a)

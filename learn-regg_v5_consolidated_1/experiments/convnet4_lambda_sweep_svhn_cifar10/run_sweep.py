#!/usr/bin/env python
"""
run_sweep.py — lambda sweep of the four methods on ConvNet-4, SVHN + CIFAR-10.

    M1  AdamW                                    (lr picked on validation)
    M2  AdamW + lambda * R(theta)                lambda over GRID, R in {l1,l2,sam,gsam,gam}
    M3  learned optimizer                        (= LO cell lambda_smooth=0, lambda_reg=0)
    M4  LO + lambda_smooth*L_smooth + lambda_reg*R   (lambda_smooth, lambda_reg) over GRID x GRID

GRID = {0, .001..009, .01..09, .1..1}  (29 values, user-specified; 0 switches the term off)

M4 is a 2-D grid of meta-trainings, so it is searched in stages that only use GRID values:
  axes   : lambda_smooth over GRID (lambda_reg=0; shared by all R)  and
           lambda_reg over GRID (lambda_smooth=0) for every R
  local  : full 2-D neighbourhood (+-radius steps in GRID) of the axis winners, per (dataset,R)
  full   : every GRID x GRID cell (optional; finished cells are skipped, so resumable)
Selection is on the VALIDATION split; test accuracy is stored, never used to select.

    python run_sweep.py all   --preset fast --stage both          # M1, M2, M4 axes+local
    python run_sweep.py report --preset fast
    python run_sweep.py all   --preset smoke --datasets cifar10 --regs l2 sam --workers 2
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
EVAL_SEED0 = 50_000

PRESETS = {
    "smoke": None,   # common.SMOKE
    "fast": dict(width=32, stages=(100, 200, 400), train_period=1, eval_steps=500,
                 lo_inits=2, adamw_inits=2, adamw_lrs=(1e-3, 3e-3)),
    "full": dict(),
}


_DATA = {}


def get_data(name):
    import common as C
    if name not in _DATA:
        _DATA[name] = C.load_data(name)
    return _DATA[name]


def g(x):
    return f"{x:g}"


def make_cfg(preset):
    import common as C
    over = C.SMOKE if preset == "smoke" else PRESETS[preset]
    return C.Cfg(**over)


def dirs(preset):
    root = HERE / "results" / preset
    d = {k: root / k for k in ("m1", "m2", "lo", "ckpt", "logs")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


# ----------------------------------------------------------------- jobs ------
def job_m1(a):
    import common as C
    cfg, D = make_cfg(a.preset), dirs(a.preset)
    data = get_data(a.dataset)
    rows = []
    for lr in cfg.adamw_lrs:
        ev = [C.run_adamw(EVAL_SEED0 + i, cfg, data, lr) for i in range(cfg.adamw_inits)]
        rows.append(dict(lr=lr, evals=ev))
        print(f"M1 {a.dataset} lr={lr:g} val={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)
    json.dump(dict(dataset=a.dataset, rows=rows, cfg=asdict(cfg)),
              open(D["m1"] / f"{a.dataset}.json", "w"), indent=1)


def job_m2(a):
    import common as C
    cfg, D = make_cfg(a.preset), dirs(a.preset)
    data = get_data(a.dataset)
    kind = "none" if a.lam == 0 else a.reg
    ev = [C.run_adamw(EVAL_SEED0 + i, cfg, data, a.adamw_lr, kind, a.lam) for i in range(cfg.adamw_inits)]
    json.dump(dict(dataset=a.dataset, reg=kind, lam=a.lam, adamw_lr=a.adamw_lr, evals=ev),
              open(D["m2"] / f"{a.dataset}_{kind}_lam{g(a.lam)}.json", "w"), indent=1)
    print(f"M2 {a.dataset} {kind} lam={a.lam:g} val={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)


def lo_tag(dataset, reg, ls, lr, seed):
    reg = "none" if lr == 0 else reg
    return f"{dataset}_{reg}_ls{g(ls)}_lr{g(lr)}_seed{seed}"


def job_lo(a):
    import common as C
    cfg, D = make_cfg(a.preset), dirs(a.preset)
    data = get_data(a.dataset)
    kind = "none" if a.lr == 0 else a.reg
    tag = lo_tag(a.dataset, a.reg, a.ls, a.lr, a.seed)
    lo, info = C.meta_train(kind, a.ls, a.lr, a.seed, cfg, data, D["ckpt"] / f"{tag}.pt",
                            log=lambda s: print(s, flush=True))
    ev = [C.run_lo(lo, EVAL_SEED0 + i, cfg, data) for i in range(cfg.lo_inits)]
    json.dump(dict(dataset=a.dataset, reg=kind, ls=a.ls, lr=a.lr, seed=a.seed, evals=ev,
                   meta_train=info, cfg=asdict(cfg), device=str(C.device)),
              open(D["lo"] / f"{tag}.json", "w"), indent=1)
    print(f"M3/M4 {tag}: val_acc={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)


# --------------------------------------------------------- orchestration -----
def best_lr(D, dataset):
    rows = json.load(open(D["m1"] / f"{dataset}.json"))["rows"]
    return max(rows, key=lambda r: np.mean([e["val_acc"] for e in r["evals"]]))["lr"]


def cells_axes(G, regs):
    """-> {reg: [(ls, lr)]}; the (ls, 0) axis is reg-independent (stored under 'none')."""
    out = {"none": [(x, 0.0) for x in G]}
    for r in regs:
        out[r] = [(0.0, x) for x in G if x != 0.0]
    return out


def cells_local(G, center, radius):
    nz = [x for x in G if x != 0.0]

    def rng(c):
        i = nz.index(min(nz, key=lambda v: abs(v - c)))
        return nz[max(0, i - radius): i + radius + 1]
    return [(a_, b_) for a_ in rng(center[0]) for b_ in rng(center[1])]


def lo_jobs(dataset, reg, cells, seeds, D):
    out = []
    for s in seeds:
        for ls, lr in cells:
            tag = lo_tag(dataset, reg, ls, lr, s)
            out.append((tag, ["job-lo", "--dataset", dataset, "--reg", reg if lr else "none",
                              "--ls", str(ls), "--lr", str(lr), "--seed", str(s)],
                        D["lo"] / f"{tag}.json"))
    return out


PHASES = ("m1", "m2", "axes", "local", "full")


def phase_jobs(a, phase, D):
    """-> [(name, argv, result_path)] for one phase. 'm2' needs m1; 'local' needs axes."""
    import common as C
    G = list(C.GRID)
    jobs = []
    if phase == "m1":
        jobs = [(f"m1_{d}", ["job-m1", "--dataset", d], D["m1"] / f"{d}.json") for d in a.datasets]
    elif phase == "m2":
        for d in a.datasets:
            lr = best_lr(D, d)
            for lam in G:
                for r in (["none"] if lam == 0 else a.regs):
                    kind = "none" if lam == 0 else r
                    jobs.append((f"m2_{d}_{kind}_lam{g(lam)}",
                                 ["job-m2", "--dataset", d, "--reg", r, "--lam", str(lam),
                                  "--adamw-lr", str(lr)], D["m2"] / f"{d}_{kind}_lam{g(lam)}.json"))
    elif phase == "axes":
        for d in a.datasets:
            for r, cells in cells_axes(G, a.regs).items():
                jobs += lo_jobs(d, r, cells, a.meta_seeds, D)
    elif phase in ("local", "full"):
        best = cmd_report(a, quiet=True)["m4_axes"]
        for d in a.datasets:
            for r in a.regs:
                if phase == "full":
                    cells = [(x, y) for x in G for y in G if y != 0.0]
                else:
                    c = best[d][r]
                    cells = cells_local(G, (c["ls"], c["lr"]), a.radius)
                jobs += lo_jobs(d, r, cells, a.meta_seeds, D)
    else:
        raise ValueError(phase)
    return jobs


def run_jobs(jobs, a, D):
    """Local orchestrator (subprocess per job; CPU box or one node with several GPUs)."""
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(f"[orchestrator] {len(jobs) - len(todo)} done, {len(todo)} to run, "
          f"{a.workers} workers x {a.threads} threads", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))

    def go(j):
        name, argv, _ = j
        t = time.time()
        with open(D["logs"] / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, str(HERE / "run_sweep.py"), *argv, "--preset", a.preset],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[orchestrator] {'ok  ' if rc == 0 else 'FAIL'} {name} ({time.time() - t:.0f}s)", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rcs = list(ex.map(go, todo))
    if any(rcs):
        print("[orchestrator] some jobs failed - see results/*/logs", flush=True)


def cmd_all(a):
    D = dirs(a.preset)
    t0 = time.time()
    phases = ["m1"] + ([] if a.skip_m2 else ["m2"])
    if a.stage in ("both", "axes"):
        phases.append("axes")
    if a.stage in ("both", "local"):
        phases.append("local")
    if a.stage == "full":
        phases.append("full")
    for ph in phases:
        run_jobs(phase_jobs(a, ph, D), a, D)
    cmd_report(a)
    print(f"[orchestrator] total {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------ HPC shard worker -----
def check_environment(a):
    import torch
    import common as C
    if a.require_gpu and not torch.cuda.is_available():
        sys.exit("CUDA not visible - torch/CUDA module mismatch. Fix the env on the LOGIN node.")
    need = {"cifar10": "datasets/cifar-10-batches-py/data_batch_1",
            "svhn": "datasets/train_32x32.mat"}
    root = C.REPO
    for d in a.datasets:
        f = root / need[d]
        if not f.exists():          # compute nodes are OFFLINE: stage data from the login node
            sys.exit(f"missing {f} - upload datasets/ (cifar-10-batches-py, train_32x32.mat, "
                     f"test_32x32.mat) before submitting; compute nodes have no internet.")
    print(f"[worker] torch {torch.__version__} device={C.device} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)


def cmd_worker(a):
    """Run shard `--shard` of `--nshards` of one phase, in-process, on this node's GPU.
    Finished jobs are skipped, a failing job never kills the shard, and no job is
    started that is not expected to finish inside --max-hours (resubmit to resume)."""
    check_environment(a)
    D = dirs(a.preset)
    # barriers: m2 needs every M1 result (lr), local/full need every axes result (centres).
    # Other lanes/processes produce them, so wait on the result files, not on PBS.
    need = {"m2": ["m1"], "local": ["axes"], "full": ["axes"]}.get(a.phase, [])
    for ph in need:
        deps = [j[2] for j in phase_jobs(a, ph, D)]
        t0 = time.time()
        while any(not Path(x).exists() for x in deps):
            if time.time() - t0 > a.wait_hours * 3600:
                print(f"[worker] barrier '{ph}' timed out; continuing with "
                      f"{sum(Path(x).exists() for x in deps)}/{len(deps)} results", flush=True)
                break
            time.sleep(30)
    jobs = sorted(phase_jobs(a, a.phase, D), key=lambda j: j[0])
    mine = [j for k, j in enumerate(jobs) if k % a.nshards == a.shard]
    todo = [j for j in mine if not Path(j[2]).exists()]
    print(f"[worker] phase={a.phase} shard {a.shard}/{a.nshards}: {len(mine)} jobs, "
          f"{len(mine) - len(todo)} done, {len(todo)} to run", flush=True)
    parser = build_parser()
    t_start, longest, failed = time.time(), 0.0, []
    for name, argv, path in todo:
        elapsed = time.time() - t_start
        if elapsed + 1.25 * longest > a.max_hours * 3600:
            print(f"WALLTIME_EXIT after {elapsed / 3600:.2f} h - resubmit this script to resume", flush=True)
            sys.exit(0)
        t = time.time()
        try:
            ja = parser.parse_args(argv + ["--preset", a.preset])
            {"job-m1": job_m1, "job-m2": job_m2, "job-lo": job_lo}[ja.cmd](ja)
            status = "ok"
        except Exception as e:                       # noqa: BLE001 - keep the shard alive
            import traceback
            traceback.print_exc()
            failed.append(name)
            status = f"FAIL ({type(e).__name__}: {e})"
        longest = max(longest, time.time() - t)
        print(f"[worker] {status} {name} ({time.time() - t:.0f}s)", flush=True)
    print(f"[worker] shard finished; {len(failed)} failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


GB_PER_PROC = 15.5        # measured: one GAM cell peaks at ~13 GB allocated / ~15 GB reserved (V100)


def max_procs_per_gpu() -> int:
    """Worker processes one GPU can hold: floor(memory / GB_PER_PROC) -> 2 on a 32 GB V100, 5 on an
    80 GB A100.  C4_MAX_PROCS overrides.  Every lane of a sweep must see the same GPU type, because the
    shard layout (lanes*procs) is derived from this number."""
    env = os.environ.get("C4_MAX_PROCS")
    if env:
        return max(int(env), 1)
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            return max(int(gb // GB_PER_PROC), 1)
    except Exception:
        pass
    return 2


def cmd_node(a):
    """One batch job = one node = one GPU. Run every phase with `--procs` worker processes
    sharing the GPU. Global shard id = lane*procs + p of lanes*procs, so the concurrent jobs
    ("lanes") split the work with no coordination."""
    t0 = time.time()
    cap = max_procs_per_gpu()
    if a.procs > cap:
        print(f"[node {a.lane}] --procs {a.procs} capped to {cap} (GPU memory)", flush=True)
        a.procs = cap                     # all lanes run this same code on the same GPU type -> consistent shards
    n = a.lanes * a.procs
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    phases = ["m1"] + ([] if a.skip_m2 else ["m2"]) + ["axes", "local"]
    base = ["--preset", a.preset, "--datasets", *a.datasets, "--regs", *a.regs,
            "--meta-seeds", *map(str, a.meta_seeds), "--radius", str(a.radius),
            "--max-hours", str(a.max_hours - (time.time() - t0) / 3600), "--nshards", str(n),
            "--wait-hours", str(a.wait_hours)]
    for ph in phases:
        left = a.max_hours - (time.time() - t0) / 3600
        if left <= 0.05:
            print("WALLTIME_EXIT before phase", ph, flush=True)
            return
        base[base.index("--max-hours") + 1] = f"{left:.3f}"
        procs = []
        for k in range(a.procs):
            shard = a.lane * a.procs + k
            if ph == "m1" and shard != 0:                       # m1 is tiny: shard 0 only
                continue
            cmd = [PY, str(HERE / "run_sweep.py"), "worker", "--phase", ph, "--shard", str(shard),
                   *base, *(["--require-gpu"] if a.require_gpu else [])]
            if ph == "m1":
                cmd[cmd.index("--nshards") + 1] = "1"
            log = open(dirs(a.preset)["logs"] / f"node{a.lane}_{ph}_p{k}.log", "a")
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        print(f"[node {a.lane}] phase {ph}: {len(procs)} workers", flush=True)
        rcs = [pr.wait() for pr in procs]
        print(f"[node {a.lane}] phase {ph} done rc={rcs} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    print(f"[node {a.lane}] all phases finished", flush=True)


# ------------------------------------------------------------- PBS gen -----
PBS = """#!/bin/bash
#==========================
# PBS Configuration  (Agastya, PBS Pro)
#==========================
#PBS -N {short}
#PBS -q {queue}
#PBS -l select=1:ncpus={ncpus}:mem={mem}
#PBS -l walltime={walltime}
#PBS -o {logdir}/{name}.out
#PBS -e {logdir}/{name}.err
#PBS -V

cd "$PBS_O_WORKDIR"
mkdir -p {logdir}

module purge
module load {cuda_module}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS={ncpus}
PY="{python}"

echo "Job {name} started $(date) on $(hostname)"
nvidia-smi || echo "WARNING : nvidia-smi not found"

cd {workdir}
{command}
rc=$?
echo "Finished $(date)  rc=$rc"
echo "NOTE: if the log ends with WALLTIME_EXIT, qsub this SAME script again (or use the chain)."
exit $rc
"""


def cmd_pbs(a):
    """Write lane PBS scripts + submit_agastya.sh into ./pbs/<preset>/.
    Agastya gpu-ext (HPC manual 4.3): >=40 cores/node, 1 V100/node, max 2 running jobs/user,
    walltime <= 7 days  ->  `--lanes` (<=2) whole-node jobs, each running `--procs` workers."""
    out = HERE / "pbs" / a.preset
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    rel = HERE.relative_to(HERE.parents[1])
    common = (f"--preset {a.preset} --datasets {' '.join(a.datasets)} --regs {' '.join(a.regs)} "
              f"--meta-seeds {' '.join(map(str, a.meta_seeds))} --radius {a.radius}")
    threads = max(a.ncpus // a.procs, 1)
    cfg_ = dict(queue=a.queue, ncpus=a.ncpus, mem=a.mem, walltime=a.walltime, cuda_module=a.cuda_module,
                python=a.python, logdir="logs", workdir=str(rel))
    # leave ~0.5 h of the walltime for the clean WALLTIME_EXIT + final report
    hh, mm, _ = (int(x) for x in a.walltime.split(":"))
    max_hours = max(hh + mm / 60 - 0.5, 0.5)
    for lane in range(a.lanes):
        cmd = (f'"$PY" run_sweep.py node --lane {lane} --lanes {a.lanes} --procs {a.procs} '
               f'--threads {threads} --max-hours {max_hours:g} --require-gpu {common}')
        (out / f"lane{lane}.pbs").write_text(PBS.format(
            name=f"lane{lane}", short=f"c4lane{lane}", command=cmd, **{**cfg_, "ncpus": a.ncpus}))
    (out / "report.pbs").write_text(PBS.format(
        name="report", short="c4rep", **{**cfg_, "ncpus": 4, "mem": "8gb", "walltime": "00:30:00"},
        command=f'"$PY" run_sweep.py report {common}'))
    sh = f"""#!/bin/bash
# Submit the ConvNet-4 sweep on Agastya.  Usage:  bash submit_agastya.sh [COPIES]
# gpu-ext allows 2 running jobs per user -> {a.lanes} lane job(s), one node/V100 each.
# COPIES = chain length per lane (each copy depends afterany on the previous). A lane stops
# cleanly before the walltime (WALLTIME_EXIT); the next copy resumes, finished cells are skipped,
# and surplus copies exit within seconds, so over-long chains are harmless.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"     # repo root: PBS_O_WORKDIR must be the root (scripts cd into experiments/...)
cd "$ROOT" && mkdir -p logs
COPIES="${{1:-1}}"
chain () {{            # $1 = script -> echoes id of the last link
    local dep="" id last=""
    for i in $(seq 1 "$COPIES"); do
        if [ -n "$dep" ]; then id=$(qsub -W depend=afterany:"$dep" "$HERE/$1"); else id=$(qsub "$HERE/$1"); fi
        dep="$id"; last="$id"
    done
    echo "$last"
}}
ids=""
for f in "$HERE"/lane*.pbs; do f=$(basename "$f"); id=$(chain "$f"); ids="${{ids:+$ids:}}$id"; echo "$f -> last job $id"; done
qsub -W depend=afterany:"$ids" "$HERE/report.pbs"
echo "queued. watch: qstat -u $USER ;  results: results/{a.preset}/summary.md"
"""
    (out / "submit_agastya.sh").write_text(sh)
    print(f"wrote {a.lanes} lane scripts + report.pbs + submit_agastya.sh to {out}")


# --------------------------------------------------------------- report ------
def _stat(vs):
    vs = np.asarray(vs, float)
    return float(vs.mean()), (float(vs.std(ddof=1) / np.sqrt(len(vs))) if len(vs) > 1 else float("nan"))


def cmd_report(a, quiet=False):
    import common as C
    D = dirs(a.preset)
    out = dict(m1={}, m2={}, m3={}, m4={}, m4_axes={}, curves={})
    for d in a.datasets:
        # ---- M1
        p = D["m1"] / f"{d}.json"
        if p.exists():
            rows = json.load(open(p))["rows"]
            r = max(rows, key=lambda r: np.mean([e["val_acc"] for e in r["evals"]]))
            out["m1"][d] = dict(lr=r["lr"], val=_stat([e["val_acc"] for e in r["evals"]]),
                                test=_stat([e["test_acc"] for e in r["evals"]]))
        # ---- M2
        out["m2"][d] = {}
        base = D["m2"] / f"{d}_none_lam0.json"
        for r in a.regs:
            rec = []
            for f in D["m2"].glob(f"{d}_{r}_lam*.json"):
                j = json.load(open(f)); rec.append(j)
            if base.exists():
                rec.append(json.load(open(base)))
            if not rec:
                continue
            rec.sort(key=lambda j: j["lam"])
            out["curves"][f"m2_{d}_{r}"] = [(j["lam"], _stat([e["val_acc"] for e in j["evals"]])[0]) for j in rec]
            b = max(rec, key=lambda j: np.mean([e["val_acc"] for e in j["evals"]]))
            out["m2"][d][r] = dict(lam=b["lam"], val=_stat([e["val_acc"] for e in b["evals"]]),
                                   test=_stat([e["test_acc"] for e in b["evals"]]), n_cells=len(rec))
        # ---- M3 / M4 : aggregate LO cells over meta-seeds
        cells = {}
        for f in D["lo"].glob(f"{d}_*.json"):
            j = json.load(open(f))
            cells.setdefault((j["reg"], j["ls"], j["lr"]), []).append(j)

        def agg(js):
            return dict(val=_stat([np.mean([e["val_acc"] for e in j["evals"]]) for j in js]),
                        test=_stat([np.mean([e["test_acc"] for e in j["evals"]]) for j in js]),
                        n_seeds=len(js))
        if ("none", 0.0, 0.0) in cells:
            out["m3"][d] = agg(cells[("none", 0.0, 0.0)])
        ls_axis = {k[1]: agg(v)["val"][0] for k, v in cells.items() if k[0] == "none" and k[2] == 0.0}
        ls_best = max(ls_axis, key=ls_axis.get) if ls_axis else 0.0
        out["curves"][f"m4_{d}_lambda_smooth_axis"] = sorted(ls_axis.items())
        out["m4"][d], out["m4_axes"][d] = {}, {}
        for r in a.regs:
            lr_axis = {k[2]: agg(v)["val"][0] for k, v in cells.items()
                       if k[0] == r and k[1] == 0.0}
            if (0.0 in ls_axis) and ("none", 0.0, 0.0) in cells:
                lr_axis[0.0] = agg(cells[("none", 0.0, 0.0)])["val"][0]
            if not lr_axis:
                continue
            out["curves"][f"m4_{d}_{r}_lambda_reg_axis"] = sorted(lr_axis.items())
            lr_best = max(lr_axis, key=lr_axis.get)
            out["m4_axes"][d][r] = dict(ls=ls_best, lr=lr_best)
            pool = {(k[1], k[2]): agg(v) for k, v in cells.items()
                    if k[0] == r or (k[0] == "none" and k[2] == 0.0)}
            (bls, blr), b = max(pool.items(), key=lambda kv: kv[1]["val"][0])
            out["m4"][d][r] = dict(ls=bls, lr=blr, n_cells=len(pool), **b)
    json.dump(out, open(D["root"] / "best_lambdas.json", "w"), indent=1)
    md = ["# ConvNet-4 lambda sweep — selected lambdas (selection on validation accuracy)\n",
          f"preset: `{a.preset}`   grid: {len(C.GRID)} values\n"]
    pm = lambda s: f"{100 * s[0]:.2f} ± {100 * s[1]:.2f}" if np.isfinite(s[1]) else f"{100 * s[0]:.2f}"
    for d in a.datasets:
        md += [f"\n## {d}\n", "| method | R | lambda_smooth | lambda_reg / lambda | val acc % | test acc % | cells |",
               "|---|---|---|---|---|---|---|"]
        if d in out["m1"]:
            m = out["m1"][d]
            md.append(f"| M1 AdamW (lr={m['lr']:g}) | – | – | – | {pm(m['val'])} | {pm(m['test'])} | – |")
        for r in a.regs:
            if r in out["m2"].get(d, {}):
                m = out["m2"][d][r]
                md.append(f"| M2 AdamW+R | {r} | – | {m['lam']:g} | {pm(m['val'])} | {pm(m['test'])} | {m['n_cells']} |")
        if d in out["m3"]:
            m = out["m3"][d]
            md.append(f"| M3 LO | – | 0 | 0 | {pm(m['val'])} | {pm(m['test'])} | – |")
        for r in a.regs:
            if r in out["m4"].get(d, {}):
                m = out["m4"][d][r]
                md.append(f"| M4 LO+smooth+R | {r} | {m['ls']:g} | {m['lr']:g} | {pm(m['val'])} | {pm(m['test'])} | {m['n_cells']} |")
    (D["root"] / "summary.md").write_text("\n".join(md) + "\n")
    if not quiet:
        print("\n".join(md))
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "report", "worker", "node", "pbs", "job-m1", "job-m2", "job-lo"])
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--datasets", nargs="+", default=["svhn", "cifar10"])
    ap.add_argument("--regs", nargs="+", default=["l1", "l2", "sam", "gsam", "gam"])
    ap.add_argument("--stage", default="both", choices=["both", "axes", "local", "full"])
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--meta-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-m2", action="store_true")
    # HPC shard worker
    ap.add_argument("--phase", choices=PHASES, default="axes")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=23.0)
    ap.add_argument("--require-gpu", action="store_true")
    # PBS generation (defaults follow pbs_scripts/sweep/*.pbs)
    ap.add_argument("--queue", default="gpu-ext")
    ap.add_argument("--ncpus", type=int, default=40, help="gpu-ext minimum is 40 cores/node")
    ap.add_argument("--mem", default="120gb")
    ap.add_argument("--lanes", type=int, default=2, help="concurrent jobs per user on gpu-ext = 2")
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--procs", type=int, default=5, help="worker processes sharing one GPU (capped by GPU memory, see max_procs_per_gpu)")
    ap.add_argument("--wait-hours", type=float, default=6.0, help="barrier timeout in a worker")
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--cuda-module", default="compilers/cuda/11.1")
    ap.add_argument("--python", default="$HOME/.conda/envs/learn_reg/bin/python")
    # single jobs
    ap.add_argument("--dataset"); ap.add_argument("--reg", default="none")
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--adamw-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=0.0, help="lambda_reg (job-lo)")
    ap.add_argument("--ls", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    a = build_parser().parse_args()
    if a.cmd in ("job-m1", "job-m2", "job-lo"):
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", a.threads)))
    {"all": cmd_all, "report": cmd_report, "worker": cmd_worker, "node": cmd_node, "pbs": cmd_pbs,
     "job-m1": job_m1, "job-m2": job_m2, "job-lo": job_lo}[a.cmd](a)


if __name__ == "__main__":
    main()

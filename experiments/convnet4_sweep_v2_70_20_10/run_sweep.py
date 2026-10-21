#!/usr/bin/env python
"""
run_sweep.py — five-method sweep on ConvNet-4, SVHN + CIFAR-10, 70:20:10 split, ID test + cross-dataset OOD.

    M1  AdamW                                         sweep lr
    M2  AdamW + lambda*R(theta)                       sweep lambda (GRID, >0) and rho (SAM/GSAM/GAM); R in {l1,l2,sam,gsam,gam}
    M3  learned optimizer (LO)                        sweep meta_lr (Adam rate that trains the LO)
    M4a LO + lambda_smooth*L_smooth                   sweep lambda_smooth (GRID); meta_lr = M3's pick
    M5  LO + lambda_smooth*L_smooth + lambda_reg*R    lambda_reg axis at M4a's lambda_smooth, then a joint
                                                      (2*radius+1)^2 block around (M4a's lambda_smooth, R's lambda_reg winner)

Every sweep grid holds strictly positive values only (no 0 in any grid); M3 (no smooth / no R term) is its own method.

All sweep values live in sweep_config.py.  SELECTION uses ONLY the source dataset's 20 % validation split of the seed-0
runs.  The 10 % test split (ID) and the target dataset's 20 % partition (OOD, source -> target) are stored for every cell
but are never read by any selection function.  After selection, the chosen M3 / M4a / M5 cells are re-meta-trained with
extra meta-seeds (`final` phase) so that test / OOD numbers carry a standard error.

Phases (each resumable: a finished result JSON is skipped):
    m1 -> m2 ;  m3 -> m4a -> axes -> local -> final ;  report

    python run_sweep.py all    --preset smoke --datasets cifar10 --regs l2 sam --workers 2     # CPU sanity check
    python run_sweep.py report --preset full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

import sweep_config as SC

HERE = Path(__file__).resolve().parent
PY = sys.executable
EVAL_SEED0 = 50_000                     # optimizee-initialisation seeds EVAL_SEED0 + i, identical for every method

PRESETS = {
    "smoke": None,   # common.SMOKE
    "fast": dict(width=32, stages=(100, 200, 400), train_period=1, eval_steps=500, lo_inits=2, adamw_inits=2,
                 adamw_lrs=(1e-3, 3e-3), meta_lrs=(1e-3, 3e-3), radius=1),
    "full": dict(),
}

_DATA = {}


def get_data(name):
    import common as C
    if name not in _DATA:
        _DATA[name] = C.load_data(name, ood_names=[o for o in SC.DATASETS if o != name])
    return _DATA[name]


def g(x):
    return f"{x:g}"


def make_cfg(preset):
    import common as C
    over = C.SMOKE if preset == "smoke" else PRESETS[preset]
    return C.Cfg(**over)


def rho_grid(cfg):
    return {k: tuple(v) for k, v in cfg.rho_grid}


def dirs(preset):
    root = HERE / "results" / preset
    d = {k: root / k for k in ("m1", "m2", "lo", "ckpt", "logs", "splits")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


def write_splits(D, name, data):
    """Document the split indices (identical wherever the dataset is used as source or as OOD target)."""
    import common as C
    p = D["splits"] / f"{name}.npz"
    if not p.exists():
        n = data.Xb.shape[0] + data.Xv.shape[0] + data.Xt.shape[0]
        tr, va, te = C.split_indices(n)
        tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, train=tr, val=va, test=te, split=np.array(SC.SPLIT), seed=SC.SPLIT_SEED)
        os.replace(tmp, p)


# ----------------------------------------------------------------- jobs ------
def job_m1(a):
    import common as C
    cfg, D = make_cfg(a.preset), dirs(a.preset)
    data = get_data(a.dataset)
    write_splits(D, a.dataset, data)
    rows = []
    for lr in cfg.adamw_lrs:
        ev = [C.run_adamw(EVAL_SEED0 + i, cfg, data, lr) for i in range(cfg.adamw_inits)]
        assert len(ev) == cfg.adamw_inits
        rows.append(dict(lr=lr, evals=ev))
        print(f"M1 {a.dataset} lr={lr:g} val={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)
    json.dump(dict(dataset=a.dataset, rows=rows, cfg=asdict(cfg)),
              open(D["m1"] / f"{a.dataset}.json", "w"), indent=1)


def m2_tag(dataset, reg, lam, rho):
    return f"{dataset}_{reg}_lam{g(lam)}" + (f"_rho{g(rho)}" if rho else "")


def job_m2(a):
    import common as C
    cfg, D = make_cfg(a.preset), dirs(a.preset)
    data = get_data(a.dataset)
    rho = a.rho if a.rho else None
    ev = [C.run_adamw(EVAL_SEED0 + i, cfg, data, a.adamw_lr, a.reg, a.lam, rho) for i in range(cfg.adamw_inits)]
    assert len(ev) == cfg.adamw_inits
    json.dump(dict(dataset=a.dataset, reg=a.reg, lam=a.lam, rho=rho, adamw_lr=a.adamw_lr, evals=ev),
              open(D["m2"] / f"{m2_tag(a.dataset, a.reg, a.lam, rho)}.json", "w"), indent=1)
    print(f"M2 {m2_tag(a.dataset, a.reg, a.lam, rho)} val={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)


def lo_tag(dataset, reg, ls, lr, mlr, seed):
    reg = "none" if lr == 0 else reg
    return f"{dataset}_{reg}_ls{g(ls)}_lr{g(lr)}_mlr{g(mlr)}_seed{seed}"


def job_lo(a):
    import common as C
    cfg, D = make_cfg(a.preset), dirs(a.preset)
    data = get_data(a.dataset)
    write_splits(D, a.dataset, data)
    kind = "none" if a.lr == 0 else a.reg
    tag = lo_tag(a.dataset, a.reg, a.ls, a.lr, a.mlr, a.seed)
    lo, info = C.meta_train(kind, a.ls, a.lr, a.seed, cfg, data, D["ckpt"] / f"{tag}.pt", meta_lr=a.mlr,
                            log=lambda s: print(s, flush=True))
    ev = [C.run_lo(lo, EVAL_SEED0 + i, cfg, data) for i in range(cfg.lo_inits)]
    assert len(ev) == cfg.lo_inits
    json.dump(dict(dataset=a.dataset, reg=kind, ls=a.ls, lr=a.lr, mlr=a.mlr, seed=a.seed, evals=ev,
                   meta_train=info, cfg=asdict(cfg), device=str(C.device)),
              open(D["lo"] / f"{tag}.json", "w"), indent=1)
    print(f"LO {tag}: val_acc={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)


# ------------------------------------------------- selection (VALIDATION ONLY) -----
# Every function below reads ONLY `val_acc`.  test_acc / ood are never touched here.
def _val(evals):
    return float(np.mean([e["val_acc"] for e in evals]))


def best_lr(D, dataset):
    rows = json.load(open(D["m1"] / f"{dataset}.json"))["rows"]
    return max(rows, key=lambda r: _val(r["evals"]))["lr"]


def lo_cells(D, d, seed=None):
    """{(reg, ls, lr, mlr): {seed: (json, path)}} for dataset d (optionally one seed only)."""
    out = {}
    for f in D["lo"].glob(f"{d}_*.json"):
        j = json.load(open(f))
        if seed is None or j["seed"] == seed:
            out.setdefault((j["reg"], j["ls"], j["lr"], j["mlr"]), {})[j["seed"]] = (j, f)
    return out


def _argmax(cells, pred):
    c = {k: _val(v[SC.SELECT_SEED][0]["evals"]) for k, v in cells.items() if pred(k) and SC.SELECT_SEED in v}
    return max(c, key=c.get) if c else None


def best_mlr(D, d):
    """M3 selection: best meta_lr among the (lambda_smooth, lambda_reg) = (0, 0) cells."""
    k = _argmax(lo_cells(D, d, SC.SELECT_SEED), lambda k: k[0] == "none" and k[1] == 0.0 and k[2] == 0.0)
    return k[3] if k else SC.META_LR_DEFAULT


def select_lo(D, d, regs):
    """Stage-wise selection of the learned-optimizer methods (seed-0 validation accuracy only)."""
    cells = lo_cells(D, d, SC.SELECT_SEED)
    mlr = best_mlr(D, d)
    sel = dict(mlr=mlr, m3=("none", 0.0, 0.0, mlr))
    k = _argmax(cells, lambda k: k[0] == "none" and k[1] > 0.0 and k[2] == 0.0 and k[3] == mlr)
    sel["m4a"] = k                                                # lambda_smooth > 0, no R
    sel["axis"], sel["m5"] = {}, {}
    ls_star = k[1] if k else None                                 # M5 lambda_reg axis runs at M4a's lambda_smooth
    for r in regs:
        sel["axis"][r] = _argmax(cells, lambda k, r=r: k[0] == r and k[1] == ls_star and k[2] > 0.0 and k[3] == mlr)
        sel["m5"][r] = _argmax(cells, lambda k, r=r: k[0] == r and k[1] > 0.0 and k[2] > 0.0 and k[3] == mlr)
    return sel


# --------------------------------------------------------- orchestration -----
def cells_local(G, center, radius):
    nz = [x for x in G if x != 0.0]

    def rng(c):
        i = nz.index(min(nz, key=lambda v: abs(v - c)))
        return nz[max(0, i - radius): i + radius + 1]
    return [(a_, b_) for a_ in rng(center[0]) for b_ in rng(center[1])]


def lo_job(dataset, reg, ls, lr, mlr, seed, D):
    tag = lo_tag(dataset, reg, ls, lr, mlr, seed)
    return (tag, ["job-lo", "--dataset", dataset, "--reg", reg if lr else "none", "--ls", str(ls), "--lr", str(lr),
                  "--mlr", str(mlr), "--seed", str(seed)], D["lo"] / f"{tag}.json")


PHASES = ("m1", "m2", "m3", "m4a", "axes", "local", "final", "full")
NEEDS = {"m2": ["m1"], "m4a": ["m3"], "axes": ["m4a"], "local": ["axes"], "final": ["local"], "full": ["axes"]}


def phase_jobs(a, phase, D):
    """-> [(name, argv, result_path)] for one phase."""
    cfg = make_cfg(a.preset)
    G = list(cfg.grid)
    radius = a.radius if a.radius is not None else cfg.radius
    S = SC.SELECT_SEED
    jobs = []
    if phase == "m1":
        jobs = [(f"m1_{d}", ["job-m1", "--dataset", d], D["m1"] / f"{d}.json") for d in a.datasets]
    elif phase == "m2":
        rg = rho_grid(cfg)
        for d in a.datasets:
            lr = best_lr(D, d)
            for r in a.regs:
                for lam in G:
                    for rho in (rg[r] if r in rg else (None,)):
                        tag = m2_tag(d, r, lam, rho)
                        argv = ["job-m2", "--dataset", d, "--reg", r, "--lam", str(lam), "--adamw-lr", str(lr)]
                        jobs.append((f"m2_{tag}", argv + (["--rho", str(rho)] if rho else []),
                                     D["m2"] / f"{tag}.json"))
    elif phase == "m3":
        for d in a.datasets:
            jobs += [lo_job(d, "none", 0.0, 0.0, mlr, S, D) for mlr in cfg.meta_lrs]
    elif phase == "m4a":
        for d in a.datasets:
            mlr = best_mlr(D, d)
            jobs += [lo_job(d, "none", ls, 0.0, mlr, S, D) for ls in G]                           # M4a: lambda_smooth axis
    elif phase == "axes":
        for d in a.datasets:                                                                      # M5: lambda_reg axis at M4a's lambda_smooth
            sel = select_lo(D, d, a.regs)
            if sel["m4a"] is None:
                continue
            for r in a.regs:
                jobs += [lo_job(d, r, sel["m4a"][1], lr, sel["mlr"], S, D) for lr in G]
    elif phase in ("local", "full"):
        for d in a.datasets:
            sel = select_lo(D, d, a.regs)
            mlr = sel["mlr"]
            for r in a.regs:
                if phase == "full":
                    cells = [(x, y) for x in G for y in G]
                else:
                    if sel["m4a"] is None or sel["axis"][r] is None:
                        continue
                    cells = cells_local(G, (sel["m4a"][1], sel["axis"][r][2]), radius)
                jobs += [lo_job(d, r, ls, lr, mlr, S, D) for ls, lr in cells]
    elif phase == "final":
        seen = set()
        for d in a.datasets:
            sel = select_lo(D, d, a.regs)
            picks = [sel["m3"], sel["m4a"]] + [sel["m5"][r] for r in a.regs]
            for k in picks:
                if k is None:
                    continue
                for s in a.final_seeds:
                    j = lo_job(d, k[0], k[1], k[2], k[3], s, D)
                    if j[0] not in seen:
                        seen.add(j[0])
                        jobs.append(j)
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
    phases = ["m1"] + ([] if a.skip_m2 else ["m2"]) + ["m3", "m4a", "axes", "local", "final"]
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
    for d in SC.DATASETS:                       # every dataset is a source AND an OOD target
        for f in (root / need[d], root / "datasets/test_32x32.mat" if d == "svhn" else root / need[d]):
            if not f.exists():          # compute nodes are OFFLINE: stage data from the login node
                sys.exit(f"missing {f} - upload datasets/ (cifar-10-batches-py, train_32x32.mat, "
                         f"test_32x32.mat) before submitting; compute nodes have no internet.")
    print(f"[worker] torch {torch.__version__} device={C.device} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)


def _run_with_oom_retry(fn, ja, tries=4):
    """Two workers share one V100; a GAM cell peaks at ~13 GB.  On CUDA OOM wait for the neighbour and retry."""
    import gc
    import torch
    for k in range(tries):
        try:
            return fn(ja)
        except RuntimeError as e:
            if "out of memory" not in str(e) or k == tries - 1:
                raise
            print(f"[worker] CUDA OOM, retry {k + 1}/{tries - 1} after cool-down", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(45 * (k + 1))


def cmd_worker(a):
    """Run shard `--shard` of `--nshards` of one phase, in-process, on this node's GPU.
    Finished jobs are skipped, a failing job never kills the shard, and no job is
    started that is not expected to finish inside --max-hours (resubmit to resume)."""
    check_environment(a)
    D = dirs(a.preset)
    # barriers: other lanes/processes produce the dependencies, so wait on the result files, not on PBS.
    for ph in NEEDS.get(a.phase, []):
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
            _run_with_oom_retry({"job-m1": job_m1, "job-m2": job_m2, "job-lo": job_lo}[ja.cmd], ja)
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


MAX_PROCS_PER_GPU = 2


def cmd_node(a):
    """One PBS job = one node = one V100. Run every phase with `--procs` worker processes
    sharing the GPU. Global shard id = lane*procs + p of lanes*procs, so the (at most 2)
    concurrent gpu-ext jobs ("lanes") split the work with no coordination."""
    t0 = time.time()
    if a.procs > MAX_PROCS_PER_GPU:       # measured on V100-32GB: one GAM cell peaks at ~13 GB (15 GB reserved)
        print(f"[node {a.lane}] --procs {a.procs} capped to {MAX_PROCS_PER_GPU} (GPU memory)", flush=True)
        a.procs = MAX_PROCS_PER_GPU       # both lanes run this same code -> shard layout stays consistent
    n = a.lanes * a.procs
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    phases = ["m1"] + ([] if a.skip_m2 else ["m2"]) + ["m3", "m4a", "axes", "local", "final"]
    base = ["--preset", a.preset, "--datasets", *a.datasets, "--regs", *a.regs,
            "--final-seeds", *map(str, a.final_seeds),
            "--max-hours", str(a.max_hours - (time.time() - t0) / 3600), "--nshards", str(n)]
    if a.radius is not None:
        base += ["--radius", str(a.radius)]
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
              f"--final-seeds {' '.join(map(str, a.final_seeds))}" + (f" --radius {a.radius}" if a.radius is not None else ""))
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
            name=f"lane{lane}", short=f"v2lane{lane}", command=cmd, **{**cfg_, "ncpus": a.ncpus}))
    (out / "report.pbs").write_text(PBS.format(
        name="report", short="v2rep", **{**cfg_, "ncpus": 4, "mem": "8gb", "walltime": "00:30:00"},
        command=f'"$PY" run_sweep.py report {common}'))
    sh = f"""#!/bin/bash
# Submit the five-method sweep on Agastya.  Usage:  bash submit_agastya.sh [COPIES]
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


METRICS = ("train_acc", "test_acc")


def _metrics(samples, targets):
    """samples = list of per-run metric dicts (evals, or per-seed means) -> {metric: (mean, SE)}."""
    out = {m: _stat([s[m] for s in samples]) for m in METRICS}
    out["ood"] = {t: dict(acc=_stat([s["ood"][t]["acc"] for s in samples]),
                          auroc_msp=_stat([s["ood"][t]["auroc_msp"] for s in samples])) for t in targets}
    return out


def _seed_means(per_seed, targets):
    """{seed: (json, path)} -> one metric dict per meta-seed (mean over that seed's optimizee inits)."""
    rows = []
    for s in sorted(per_seed):
        ev = per_seed[s][0]["evals"]
        rows.append(dict(train_acc=np.mean([e["train_acc"] for e in ev]), test_acc=np.mean([e["test_acc"] for e in ev]),
                         ood={t: dict(acc=np.mean([e["ood"][t]["acc"] for e in ev]),
                                      auroc_msp=np.mean([e["ood"][t]["auroc_msp"] for e in ev])) for t in targets}))
    return rows


def _mean_curve(evals):
    keys = ("train_loss", "val_loss", "val_acc")
    n = min(len(e.get("curve", [])) for e in evals)
    if n == 0:
        return []
    return [dict(step=evals[0]["curve"][i]["step"], **{k: float(np.mean([e["curve"][i][k] for e in evals])) for k in keys})
            for i in range(n)]


def cmd_report(a, quiet=False):
    D = dirs(a.preset)
    cfg = make_cfg(a.preset)
    S = SC.SELECT_SEED
    res = {"protocol": dict(split=SC.SPLIT, split_seed=SC.SPLIT_SEED, selection="validation split of the source dataset, "
                            f"meta-seed {S} (M1/M2: mean over {cfg.adamw_inits} inits)",
                            id_test="source dataset 10 % test split", ood="target dataset 20 % partition (source -> target)",
                            plus_minus="std. error: over optimizee inits for M1/M2, over meta-seeds for M3/M4a/M5",
                            preset=a.preset, cfg=asdict(cfg)), "results": {}}
    rows = []                                                             # all_cells.csv
    for d in a.datasets:
        targets = [t for t in SC.DATASETS if t != d]
        R = res["results"][d] = {}

        def add_row(method, reg, lr_adamw, lam, rho, ls, lamr, mlr, seed, evals):
            row = dict(method=method, source=d, reg=reg, adamw_lr=lr_adamw, lam=lam, rho=rho, ls=ls, lam_reg=lamr,
                       meta_lr=mlr, seed=seed, n_inits=len(evals), val_acc=_val(evals),
                       **{f"val_acc_init{i}": e["val_acc"] for i, e in enumerate(evals)}, train_acc=np.mean([e["train_acc"] for e in evals]),
                       test_acc=np.mean([e["test_acc"] for e in evals]),
                       seconds=np.mean([e.get("seconds", np.nan) for e in evals]))
            for t in targets:
                row[f"ood_{t}_acc"] = np.mean([e["ood"][t]["acc"] for e in evals])
                row[f"ood_{t}_auroc"] = np.mean([e["ood"][t]["auroc_msp"] for e in evals])
            rows.append(row)

        # ---- M1
        p = D["m1"] / f"{d}.json"
        if p.exists():
            m1rows = json.load(open(p))["rows"]
            for r in m1rows:
                add_row("M1", "-", r["lr"], None, None, None, None, None, "-", r["evals"])
            r = max(m1rows, key=lambda r: _val(r["evals"]))
            R["M1"] = dict(config=dict(adamw_lr=r["lr"]), val=_val(r["evals"]), n=len(r["evals"]), n_kind="inits",
                           **_metrics(r["evals"], targets), curve=_mean_curve(r["evals"]))
        # ---- M2
        for r in a.regs:
            recs = [json.load(open(f)) for f in D["m2"].glob(f"{d}_{r}_lam*.json")]
            rg = rho_grid(cfg).get(r)
            if rg:                                 # ignore cells from an earlier, different rho grid
                recs = [j for j in recs if j["rho"] is not None and any(abs(j["rho"] - x) < 1e-12 for x in rg)]
            for j in recs:
                add_row("M2", r, j["adamw_lr"], j["lam"], j["rho"], None, None, None, "-", j["evals"])
            if recs:
                b = max(recs, key=lambda j: _val(j["evals"]))
                R[f"M2:{r}"] = dict(config=dict(reg=r, lam=b["lam"], rho=b["rho"], adamw_lr=b["adamw_lr"]),
                                    val=_val(b["evals"]), n=len(b["evals"]), n_kind="inits", n_cells=len(recs),
                                    **_metrics(b["evals"], targets), curve=_mean_curve(b["evals"]))
        # ---- learned optimizer methods
        allc = lo_cells(D, d)
        for k, per in allc.items():
            for s, (j, f) in sorted(per.items()):
                method = "M3" if (k[1] == 0.0 and k[2] == 0.0) else "M4a" if k[2] == 0.0 else "M5"
                add_row(method, k[0], None, None, None, k[1], k[2], k[3], s, j["evals"])
        sel = select_lo(D, d, a.regs)

        def lo_entry(key, method, reg):
            if key is None or key not in allc or S not in allc[key]:
                return
            per = allc[key]
            samples = _seed_means(per, targets)
            n_cells = sum(1 for k in allc if (method == "M3" and k[1] == 0.0 and k[2] == 0.0) or
                          (method == "M4a" and k[0] == "none" and k[2] == 0.0 and k[1] > 0.0 and k[3] == sel["mlr"]) or
                          (method == "M5" and k[0] == reg and k[1] > 0.0 and k[2] > 0.0 and k[3] == sel["mlr"]))
            R[method if reg is None else f"{method}:{reg}"] = dict(
                config=dict(reg=reg, ls=key[1], lam_reg=key[2], meta_lr=key[3]),
                val=_val(per[S][0]["evals"]), n=len(per), n_kind="meta-seeds", n_cells=n_cells,
                **_metrics(samples, targets), curve=_mean_curve(per[S][0]["evals"]),
                meta_train_seconds=float(np.mean([v[0]["meta_train"]["meta_train_seconds"] for v in per.values()])),
                files=[str(v[1].relative_to(D["root"])) for v in per.values()])
        lo_entry(sel["m3"], "M3", None)
        lo_entry(sel["m4a"], "M4a", None)
        for r in a.regs:
            lo_entry(sel["m5"][r], "M5", r)
    json.dump(res, open(D["root"] / "best_config.json", "w"), indent=1)
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(D["root"] / "all_cells.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    md = write_summary(res, a, D)
    if not quiet:
        print(md)
    return res


LABEL = {"M1": "M1 AdamW", "M2": "M2 AdamW+R", "M3": "M3 LO", "M4a": "M4a LO+smooth", "M5": "M5 LO+smooth+R"}
NAME = {"svhn": "SVHN", "cifar10": "CIFAR-10"}


def write_summary(res, a, D):
    pm = lambda s: (f"{100 * s[0]:.2f} ± {100 * s[1]:.2f}" if np.isfinite(s[1]) else f"{100 * s[0]:.2f}")
    md = ["# Five-method sweep — ConvNet-4, 70:20:10 split\n",
          "Selection: source validation split (seed 0) only. ID test = source 10 % split. OOD = selected model, no "
          "retraining, on the TARGET dataset's 20 % partition. ± = std. error over inits (M1/M2) or meta-seeds (M3/M4a/M5). "
          "OOD acc is accuracy under the *target's own label ids* (the two label spaces are unrelated, so ≈ chance is "
          "expected); OOD AUROC = ID-test vs OOD separation by max-softmax confidence.\n"]
    for d in a.datasets:
        R = res["results"].get(d, {})
        targets = [t for t in SC.DATASETS if t != d]
        for t in targets:
            md += [f"\n## {NAME[d]} (source) → {NAME[t]} (OOD target)\n",
                   f"| method | R | config | val acc % (selection) | ID test acc % | OOD acc % → {NAME[t]} | OOD AUROC | cells | n |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for key, e in R.items():
                m, _, reg = key.partition(":")
                c = e["config"]
                if m == "M1":
                    cfg_s = f"lr={c['adamw_lr']:g}"
                elif m == "M2":
                    cfg_s = f"λ={c['lam']:g}" + (f", ρ={c['rho']:g}" if c["rho"] else "") + f" (lr={c['adamw_lr']:g})"
                elif m == "M3":
                    cfg_s = f"meta_lr={c['meta_lr']:g}"
                elif m == "M4a":
                    cfg_s = f"λ_smooth={c['ls']:g}, meta_lr={c['meta_lr']:g}"
                else:
                    cfg_s = f"λ_smooth={c['ls']:g}, λ_reg={c['lam_reg']:g}, meta_lr={c['meta_lr']:g}"
                o = e["ood"][t]
                md.append(f"| {LABEL[m]} | {reg or '-'} | {cfg_s} | {100 * e['val']:.2f} | {pm(e['test_acc'])} | "
                          f"{pm(o['acc'])} | {o['auroc_msp'][0]:.3f} | {e.get('n_cells', '-')} | {e['n']} {e['n_kind']} |")
    text = "\n".join(md) + "\n"
    (D["root"] / "summary.md").write_text(text)
    return text


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "report", "worker", "node", "pbs", "job-m1", "job-m2", "job-lo"])
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--datasets", nargs="+", default=list(SC.DATASETS))
    ap.add_argument("--regs", nargs="+", default=list(SC.REGS))
    ap.add_argument("--radius", type=int, default=None, help="M5 joint-block half-width (default: sweep_config.RADIUS)")
    ap.add_argument("--final-seeds", type=int, nargs="+", default=list(SC.FINAL_SEEDS),
                    help="extra meta-seeds for the SELECTED M3/M4a/M5 cells")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-m2", action="store_true")
    # HPC shard worker
    ap.add_argument("--phase", choices=PHASES, default="axes")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=23.0)
    ap.add_argument("--require-gpu", action="store_true")
    # PBS generation
    ap.add_argument("--queue", default="gpu-ext")
    ap.add_argument("--ncpus", type=int, default=40, help="gpu-ext minimum is 40 cores/node")
    ap.add_argument("--mem", default="120gb")
    ap.add_argument("--lanes", type=int, default=2, help="concurrent jobs per user on gpu-ext = 2")
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2, help="worker processes sharing one V100 (capped at MAX_PROCS_PER_GPU)")
    ap.add_argument("--wait-hours", type=float, default=6.0, help="barrier timeout in a worker")
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--cuda-module", default="compilers/cuda/11.1")
    ap.add_argument("--python", default="$HOME/.conda/envs/learn_reg/bin/python")
    # single jobs
    ap.add_argument("--dataset"); ap.add_argument("--reg", default="none")
    ap.add_argument("--lam", type=float, default=0.0, help="M2 lambda")
    ap.add_argument("--rho", type=float, default=None, help="M2 perturbation radius (default: per-regularizer default)")
    ap.add_argument("--adamw-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=0.0, help="lambda_reg (job-lo)")
    ap.add_argument("--ls", type=float, default=0.0, help="lambda_smooth (job-lo)")
    ap.add_argument("--mlr", type=float, default=SC.META_LR_DEFAULT, help="meta_lr: Adam rate of the LO's weights (job-lo)")
    ap.add_argument("--seed", type=int, default=SC.SELECT_SEED)
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

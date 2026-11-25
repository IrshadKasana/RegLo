"""PARAM Rudra driver for the full 6-method x 5-regularizer x {Adam, SGD} pipeline, three cases:

    mlp_mnist           meta-train (= seen) task: MLP       + MNIST
    convnet4_cifar10    meta-train (= seen) task: ConvNet-4 + CIFAR-10
    resnet18_imagenet   meta-train (= seen) task: ResNet-18 + ImageNet-32 (LO meta-trained on a narrower
                        ResNet-18 proxy, width R18_META_WIDTH, default 16; phases 2-3 use the full width-64 net)

Each case is a queue of 24 work units, one per run id of `run_pipeline.py --list` (M1_adam_noreg ...
M6_LO_gsam). A unit runs phases 1 -> 2 -> 3 for that run id in its own process. A "lane" is one SLURM job
(1 A100) that runs a pool of such processes, as many at once as fit in the GPU-memory budget. Lanes of the
same case share the queue through claim files, so any number of lanes (and resubmitted lanes) cooperate:

    artifacts/pipeline/<experiment>/rudra_queue/
        claims/<run_id>      held by a running process (mtime refreshed every minute; stale after 10 min)
        done/<run_id>        unit finished (phases 1-3 rc=0)
        failed/<run_id>      one line per failed attempt (a unit is skipped after MAX_FAILS attempts)
        logs/<run_id>.log    stdout+stderr of every attempt of that unit (appended)

Everything inside a unit is resumable: meta-training writes a resume file after every meta-trained
optimizee, optimizee trainings are cached, finished phases are skipped. So a lane stopped at its walltime
loses at most one optimizee unroll / one optimizee training, and the next lane continues.

Subcommands (all from the repository root):
    python experiments/pipeline/rudra/rudra_jobs.py list   <case>          run ids in scheduling order + GPU GB
    python experiments/pipeline/rudra/rudra_jobs.py args   <case>          run_pipeline.py arguments of the case
    python experiments/pipeline/rudra/rudra_jobs.py pool   <case> --procs N --max-hours H
    python experiments/pipeline/rudra/rudra_jobs.py status [case ...]      progress of every case
    python experiments/pipeline/rudra/rudra_jobs.py report <case>          (re)build the 3D tables of a case

Pool exit codes (lane.sbatch resubmits itself only on 10):
    0  every unit of the case is done and the report was built
    10 stopped at the walltime with work left            -> resubmit
    11 nothing left to start; remaining units are being run by other lanes
    12 the only units left have failed MAX_FAILS times    -> look at rudra_queue/logs, fix, `reset-failed`
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = [sys.executable, "-u", str(ROOT / "experiments" / "pipeline" / "run_pipeline.py")]

CASES = {
    "mlp_mnist": {"optimizee": "mlp", "dataset": "mnist"},
    "convnet4_cifar10": {"optimizee": "convnet4", "dataset": "cifar10"},
    "resnet18_imagenet": {"optimizee": "resnet18", "dataset": "imagenet"},
}

# Scheduling order: longest first (relative meta-training cost per step measured on the pipeline:
# M3 1, M5-L1/L2 1, M5-SAM/GSAM ~1.5, M5-GAM ~2.4, M4 ~4, M6-L1/L2 ~4, M6-SAM/GSAM ~6, M6-GAM ~14).
PRIORITY = {"M6_LO_gam": 14, "M6_LO_sam": 6, "M6_LO_gsam": 6, "M6_LO_l1": 4, "M6_LO_l2": 4, "M4_LO_noreg": 4,
            "M5_LO_gam": 2.4, "M5_LO_sam": 1.5, "M5_LO_gsam": 1.5, "M5_LO_l1": 1, "M5_LO_l2": 1, "M3_LO_noreg": 1}

CLAIM_STALE_S = float(os.environ.get("CLAIM_STALE_MIN", "10")) * 60
MAX_FAILS = int(os.environ.get("MAX_FAILS", "2"))


def case_args(case: str):
    """run_pipeline.py arguments that define the case (also used for the report)."""
    if case not in CASES:
        raise SystemExit(f"unknown case '{case}'; choose from {list(CASES)}")
    c = CASES[case]
    args = ["--meta-optimizee", c["optimizee"], "--meta-dataset", c["dataset"],
            "--set", f"experiment.name={experiment_name(case)}",
            "--set", "experiment.download=false",                       # compute nodes have no internet
            "--set", f"experiment.num_workers={int(os.environ.get('NUM_WORKERS', '3'))}",
            "--set", "optimizee_evaluation.skip_missing_datasets=false",  # a full run must not silently skip one
            "--device", "cuda:0"]
    if os.environ.get("DATA_ROOT"):
        args += ["--set", f"experiment.data_root={os.environ['DATA_ROOT']}"]
    if case == "resnet18_imagenet":
        w = int(os.environ.get("R18_META_WIDTH", "16"))
        args += ["--set", f"meta_training.models={{resnet18: {{width: {w}}}}}"]
    for extra in os.environ.get("EXTRA_SETS", "").split():   # e.g. EXTRA_SETS="meta_training.smooth_every=5"
        args += ["--set", extra]
    if os.environ.get("QUICK") == "1":                        # tiny preset: dry-run of the job machinery
        args += ["--quick"]
    return args


def experiment_name(case: str) -> str:
    return os.environ.get("EXP_PREFIX", "rudra") + "_" + case


def queue_dir(case: str) -> Path:
    return ROOT / "artifacts" / "pipeline" / experiment_name(case) / "rudra_queue"


def run_ids(case: str):
    out = subprocess.run(RUN + case_args(case) + ["--list"], cwd=ROOT, check=True, capture_output=True, text=True)
    ids = [l.strip() for l in out.stdout.splitlines() if l.strip()]

    # learned optimizers first (longest first), then M1, then M2 (M2's phase-2 trainings equal M1's once R is
    # stripped, so they are read from M1's cache)
    group = lambda r: 1 if r.startswith("M1_") else 2 if r.startswith("M2_") else 0
    return sorted(ids, key=lambda r: (group(r), -PRIORITY.get(r, 0), r))


def gpu_gb(case: str, rid: str) -> float:
    """Peak GPU memory estimate of one unit (GB): max over meta-training (methods 3-6) and phases 2-3, where a
    frozen LO trains the full ResNet-18 (LSTM state ~3.6 GB + activations at batch 512). MEM_SCALE rescales
    all estimates after the GPU check."""
    scale = float(os.environ.get("MEM_SCALE", "1.0"))
    eval_peak = 10.0
    if not rid.startswith(("M3", "M4", "M5", "M6")):
        return scale * (14.0 if case == "resnet18_imagenet" else eval_peak)
    weights = {"mlp_mnist": 61_670, "convnet4_cifar10": 30_186}.get(case)
    if weights is None:   # resnet18 proxy: weights of the width-w CIFAR ResNet-18 (10 classes ~ 1000 classes + fc)
        w = int(os.environ.get("R18_META_WIDTH", "16"))
        weights = {8: 0.24e6, 16: 0.83e6, 32: 3.05e6, 64: 11.7e6}.get(w, 11.7e6 * (w / 64) ** 2)
    meta = 1.4e3 * weights * 20 / 2 ** 30                        # README: ~1.4 KB per weight per unroll step
    if rid.startswith(("M4", "M6")):
        meta *= 2.0                                                # R_smooth (measured ~2x on MLP and ResNet-8)
    return scale * max(eval_peak, meta + 3.0)


# ----------------------------------------------------------------------------------------------- queue files

def _touch_new(path: Path, text: str) -> bool:
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, text.encode())
    os.close(fd)
    return True


def claim(q: Path, rid: str) -> bool:
    path = q / "claims" / rid
    tag = f"job {os.environ.get('SLURM_JOB_ID') or os.environ.get('PBS_JOBID', '-')} host {socket.gethostname()} pid {os.getpid()} {time.ctime()}\n"
    if _touch_new(path, tag):
        return True
    try:
        if time.time() - path.stat().st_mtime > CLAIM_STALE_S:   # its lane died without cleaning up
            path.unlink(missing_ok=True)
            return _touch_new(path, tag)
    except FileNotFoundError:
        return _touch_new(path, tag)
    return False


def is_claimed(q: Path, rid: str) -> bool:
    path = q / "claims" / rid
    try:
        return time.time() - path.stat().st_mtime <= CLAIM_STALE_S
    except FileNotFoundError:
        return False


def n_fails(q: Path, rid: str) -> int:
    f = q / "failed" / rid
    return len(f.read_text().splitlines()) if f.exists() else 0


def is_done(q: Path, rid: str) -> bool:
    return (q / "done" / rid).exists()


# ----------------------------------------------------------------------------------------------- pool

def pool(case: str, procs: int, max_hours: float, budget_gb: float) -> int:
    q = queue_dir(case)
    for sub in ("claims", "done", "failed", "logs"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    ids = run_ids(case)
    deadline = time.time() + max_hours * 3600
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID", "local")
    print(f"[pool] case {case} | {len(ids)} units | procs {procs} | GPU budget {budget_gb:.0f} GB | "
          f"stops at {time.ctime(deadline)} | job {job} | host {socket.gethostname()}", flush=True)
    running = {}           # rid -> (Popen, gb, start_time, log handle)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            for rid in list(running):
                try:
                    os.utime(q / "claims" / rid)
                except OSError:
                    pass
    threading.Thread(target=heartbeat, daemon=True).start()

    def start(rid, gb):
        log = open(q / "logs" / f"{rid}.log", "a")
        log.write(f"\n===== {time.ctime()} job {job} host {socket.gethostname()} start {rid} =====\n")
        log.flush()
        cmd = RUN + case_args(case) + ["--phases", "1", "2", "3", "--run-ids", rid]
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        running[rid] = (p, gb, time.time(), log)
        print(f"[pool] {time.strftime('%m-%d %H:%M')} start {rid} (~{gb:.0f} GB; "
              f"{sum(v[1] for v in running.values()):.0f}/{budget_gb:.0f} GB in use)", flush=True)

    def finish(rid, rc):
        p, gb, t0, log = running.pop(rid)
        hours = (time.time() - t0) / 3600
        log.write(f"===== {time.ctime()} {rid} exit {rc} after {hours:.2f} h =====\n")
        log.close()
        if rc == 0:
            (q / "done" / rid).write_text(f"{time.ctime()} job {job} {hours:.2f} h\n")
            print(f"[pool] {time.strftime('%m-%d %H:%M')} DONE {rid} ({hours:.2f} h)", flush=True)
        else:
            with open(q / "failed" / rid, "a") as fh:
                fh.write(f"{time.ctime()} job {job} rc {rc} after {hours:.2f} h\n")
            print(f"[pool] {time.strftime('%m-%d %H:%M')} FAILED {rid} rc {rc} (attempt {n_fails(q, rid)}/{MAX_FAILS};"
                  f" see {q / 'logs' / (rid + '.log')})", flush=True)
        (q / "claims" / rid).unlink(missing_ok=True)

    hit_deadline = False
    stop_now = threading.Event()   # SIGTERM (scancel, or SLURM at the walltime): stop like at the deadline
    signal.signal(signal.SIGTERM, lambda *_: stop_now.set())
    signal.signal(signal.SIGINT, lambda *_: stop_now.set())
    while True:
        for rid, (p, *_rest) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                finish(rid, rc)
        if time.time() >= deadline or stop_now.is_set():
            hit_deadline = True
            break
        # start what fits: longest first; skip a unit that does not fit now in favour of a smaller one
        if len(running) < procs:
            used = sum(v[1] for v in running.values())
            for rid in ids:
                if len(running) >= procs:
                    break
                if rid in running or is_done(q, rid) or n_fails(q, rid) >= MAX_FAILS or is_claimed(q, rid):
                    continue
                gb = gpu_gb(case, rid)
                if running and used + gb > budget_gb:
                    continue
                if claim(q, rid):
                    start(rid, gb)
                    used += gb
        if not running:
            break
        stop_now.wait(20)

    if hit_deadline and running:
        print(f"[pool] walltime margin reached: stopping {list(running)} (they resume in the next lane)", flush=True)
        for rid, (p, *_r) in running.items():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        t_end = time.time() + 90
        while time.time() < t_end and any(v[0].poll() is None for v in running.values()):
            time.sleep(3)
        for rid, (p, gb, t0, log) in list(running.items()):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            log.write(f"===== {time.ctime()} {rid} stopped at the walltime margin (will resume) =====\n")
            log.close()
            (q / "claims" / rid).unlink(missing_ok=True)   # free it at once for the next lane
            running.pop(rid)
    stop.set()

    left = [r for r in ids if not is_done(q, r)]
    if not left:
        return report(case)
    startable = [r for r in left if n_fails(q, r) < MAX_FAILS]
    if hit_deadline:
        print(f"[pool] RESULT deadline: {len(left)} units left -> resubmit", flush=True)
        return 10
    if any(is_claimed(q, r) for r in startable):
        print(f"[pool] RESULT idle: {len(left)} units left, all held by other lanes", flush=True)
        return 11
    if startable:   # unclaimed but nothing started (e.g. too big for the budget): treat as work left
        print(f"[pool] RESULT deadline-like: {startable} could not be started -> resubmit", flush=True)
        return 10
    print(f"[pool] RESULT failed: {left} failed {MAX_FAILS} times; see {q / 'logs'}", flush=True)
    return 12


def report(case: str) -> int:
    q = queue_dir(case)
    q.mkdir(parents=True, exist_ok=True)
    if not _touch_new(q / "report.lock", time.ctime()):
        print("[pool] another lane is building the report", flush=True)
        return 0
    try:
        rc = subprocess.run(RUN + case_args(case) + ["--phases", "report"], cwd=ROOT).returncode
    finally:
        (q / "report.lock").unlink(missing_ok=True)
    if rc == 0:
        (q / "REPORT_DONE").write_text(time.ctime() + "\n")
        tables = q.parent / "tables"
        print(f"[pool] RESULT all_done: report in {tables} (summary.md, breakdown.md, results_long.csv)", flush=True)
    return rc


def status(cases) -> int:
    for case in cases or list(CASES):
        q = queue_dir(case)
        try:
            ids = run_ids(case)
        except subprocess.CalledProcessError as exc:
            print(f"{case}: cannot list run ids: {exc.stderr[-500:]}")
            continue
        done = [r for r in ids if is_done(q, r)]
        run = [r for r in ids if not is_done(q, r) and is_claimed(q, r)]
        dead = [r for r in ids if not is_done(q, r) and n_fails(q, r) >= MAX_FAILS]
        rep = "report built" if (q / "REPORT_DONE").exists() else "report pending"
        print(f"== {case} ({experiment_name(case)}): {len(done)}/{len(ids)} done, {len(run)} running, "
              f"{len(dead)} failed-out, {rep}")
        ckdir = q.parent / "checkpoints" / "lo"
        for r in ids:
            if is_done(q, r):
                continue
            tag = "RUNNING" if r in run else ("FAILED-OUT" if r in dead else "waiting")
            extra = ""
            if (ckdir / f"{r}.resume.pt").exists():
                age = (time.time() - (ckdir / f"{r}.resume.pt").stat().st_mtime) / 60
                extra = f" meta-training resume file updated {age:.0f} min ago"
            elif (ckdir / f"{r}.json").exists():
                extra = " meta-training finished; in phase 2/3"
            fails = n_fails(q, r)
            print(f"   {tag:10s} {r:16s}{' fails=' + str(fails) if fails else ''}{extra}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "args", "report", "reset-failed"):
        sp = sub.add_parser(name)
        sp.add_argument("case")
    sp = sub.add_parser("pool")
    sp.add_argument("case")
    sp.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", "4")))
    sp.add_argument("--max-hours", type=float, default=float(os.environ.get("MAX_HOURS", "47.5")))
    sp.add_argument("--gpu-budget-gb", type=float, default=float(os.environ.get("GPU_BUDGET_GB", "72")))
    sp = sub.add_parser("status")
    sp.add_argument("cases", nargs="*")
    a = ap.parse_args()
    if a.cmd == "list":
        for rid in run_ids(a.case):
            print(f"{rid:16s} ~{gpu_gb(a.case, rid):5.1f} GB")
        return 0
    if a.cmd == "args":
        print(" ".join(f"'{x}'" if " " in x else x for x in case_args(a.case)))
        return 0
    if a.cmd == "reset-failed":
        for f in (queue_dir(a.case) / "failed").glob("*"):
            f.unlink()
        print("failure counters cleared")
        return 0
    if a.cmd == "pool":
        return pool(a.case, a.procs, a.max_hours, a.gpu_budget_gb)
    if a.cmd == "report":
        return report(a.case)
    return status(a.cases)


if __name__ == "__main__":
    sys.exit(main())

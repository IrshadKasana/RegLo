"""Run a whole study on one machine, stage by stage — a laptop, a workstation or a cloud GPU box.

    plan = plan_study("main", stages=(1, 2, 3, 4))       # the runs, with what each one waits for
    Runner(plan, slots=["0", "1"]).run()                 # two GPUs, one run per GPU at a time

Dependencies follow the study design: an M3 run waits for its round's m3_lstm arm of Stage 1, an
M4-r run for that round's m4_lstm_reg-r arm; M1 and M2 need nothing, so on several GPUs they fill
the time Stage 1 takes. Stage 4's timing waits for every Stage-1 arm (it times the trained φ); its audits are
planned once the training runs have finished (:func:`audit_tasks`). A run whose prerequisite
failed is skipped, the others carry on. Without slots everything runs in this process, in plan
order, with its output on the console (and breakpoints work in an IDE).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from reglo.config import ConfigError, compose_meta, compose_run, default_meta_tag, load_experiment, project_root
from reglo.manifest import expand, manifest_path, run_line, validate, write_manifest
from reglo.profiles import active_profile
from reglo.study import study_experiments
from reglo.utils.paths import diagnostics_dir_for, meta_run_dir, parse_method_tag, profile_artifacts, run_dir
from reglo.utils.runtime import EXIT_INTERRUPTED, EXIT_OK, describe_exit

__all__ = ["Runner", "Task", "audit_tasks", "finished", "plan_study"]

Line = List[str]
#: the --only keys each kind of line can be filtered by (others do not apply to it and are ignored)
_FIELDS = {
    "meta_train": {"meta_training", "regularizer", "optimizee", "dataset", "round", "seed"},
    "train": {"method", "regularizer", "optimizee", "dataset", "round", "seed"},
    "compute": {"optimizee", "dataset"},
    "diagnostics": {"method", "regularizer", "optimizee", "dataset", "round", "seed"},
}


@dataclass
class Task:
    key: str                      # unique: "<experiment>:<line number>"
    line: Line                    # script name and arguments (a manifest line)
    stage: str                    # "1", "2", "3", "2-3" (extended), "pilot", "4"
    experiment: str
    needs: Set[str] = field(default_factory=set)
    priority: Tuple[int, int] = (1, 0)
    status: str = "pending"       # pending · running · done · failed · interrupted · skipped
    code: Optional[int] = None
    seconds: float = 0.0
    log: Optional[Path] = None
    note: str = ""

    @property
    def title(self) -> str:
        return " ".join(a for a in self.line if not a.startswith(("study=", "profile=")))


def _arg(line: Sequence[str], key: str) -> Optional[str]:
    return next((a.split("=", 1)[1] for a in line if a.startswith(key + "=")), None)


def _stage_of(spec: Mapping) -> str:
    stage = spec.get("stage")
    if isinstance(stage, list):
        return "-".join(str(s) for s in stage)
    return str(stage) if stage is not None else str(spec.get("name"))


def _subset(only: Optional[Mapping[str, str]], kind: str) -> Dict[str, str]:
    return {k: v for k, v in (only or {}).items() if k in _FIELDS[kind]}


def _flag(line: Sequence[str], flag: str) -> Optional[str]:
    return line[line.index(flag) + 1] if flag in line and line.index(flag) + 1 < len(line) else None


def finished(line: Sequence[str]) -> bool:
    """Has this run already produced its result? (Then a restart skips it without starting a process.)"""
    script, args = line[0], list(line[1:])
    try:
        if script == "train_optimizee":
            cfg = compose_run(args)
            return not cfg.training.rerun and (run_dir(cfg) / "final.json").is_file()
        if script == "meta_train":
            cfg = compose_meta(args)
            return not cfg.meta_training.rerun and (meta_run_dir(cfg) / "summary.json").is_file()
        if script == "run_diagnostics":
            run = _flag(args, "--run")
            return run is not None and "--redo" not in args and (diagnostics_dir_for(run) / "diagnostics.json").is_file()
        if script == "profile_compute":
            spec = load_experiment(_flag(args, "--experiment") or "stage4_analysis", profile=_flag(args, "--profile"))
            optimizee, dataset = spec["compute"]["pairs"][int(_flag(args, "--index") or 0)]
            return (profile_artifacts(spec["profile"]) / "diagnostics" / "compute" / f"{optimizee}-{dataset}.json").is_file()
    except (ConfigError, ValueError, KeyError, IndexError, OSError):
        return False
    return False


def _keep(line: Sequence[str], regularizers: Optional[Sequence[str]]) -> bool:
    """With ``regularizers``, keep the cube blocks of those r: their M2/M4 runs, and M1/M3 (r = none)."""
    if not regularizers:
        return True
    reg = _arg(line, "regularizer") or "none"
    return reg == "none" or reg in regularizers


def plan_study(
    study: str = "main",
    stages: Sequence[int] = (1, 2, 3, 4),
    extended: bool = False,
    pilot: bool = False,
    only: Optional[Mapping[str, str]] = None,
    profile: Optional[str] = None,
    artifacts_root: Optional[Path] = None,
    write: bool = True,
    regularizers: Optional[Sequence[str]] = None,
    skip_finished: bool = True,
) -> List[Task]:
    """Every run of the study's experiment files, validated, with its dependencies.

    ``only`` filters like ``make_manifest.py --only`` (each key where it applies);
    ``regularizers`` keeps the cube blocks of those regularizers (M1 and M3 stay).
    Writes each experiment's manifest to artifacts/<profile>/manifests/ (``write``) as a record.
    Runs that have already finished are marked done (``skip_finished``) and are not started again.
    """
    profile = profile or active_profile()
    specs = list(study_experiments(study, profile=profile))
    if pilot:
        specs += [s for s in study_experiments("pilot", profile=profile) if s not in specs]
    wanted = set(int(s) for s in stages)
    tasks: List[Task] = []
    arms: Dict[Tuple[str, str, str], str] = {}  # (meta_training, regularizer, round) → Stage-1 task key
    default_round = default_meta_tag()
    order = 0

    def add(spec, kind: str, stage: str) -> List[Task]:
        nonlocal order
        lines = [line for line in expand(spec, [kind], _subset(only, kind), artifacts_root) if _keep(line, regularizers)]
        if not lines:
            return []
        validate(lines, spec)
        if write:
            write_manifest(lines, manifest_path(spec["name"], [kind], _subset(only, kind), artifacts_root, profile))
        made = []
        for number, line in enumerate(lines):
            order += 1
            task = Task(f"{spec['name']}:{kind}:{number}", line, stage, str(spec["name"]),
                        priority=(0 if stage == "1" else 1, order))
            if skip_finished and finished(line):
                task.status, task.note = "done", "finished earlier"
            made.append(task)
        tasks.extend(made)
        return made

    for spec in specs:  # Stage 1 first: everything else may wait for it
        if "meta_train" in spec and 1 in wanted:
            for task in add(spec, "meta_train", "1"):
                arms[(_arg(task.line, "meta_training"), _arg(task.line, "regularizer") or "none",
                      _arg(task.line, "meta_tag") or default_round)] = task.key
    for spec in specs:
        if "train" not in spec:
            continue
        stage = "pilot" if spec.get("study") == "pilot" else _stage_of(spec)
        stage_numbers = set(int(s) for s in stage.split("-")) if stage != "pilot" else {2}
        if spec.get("extended") and not extended:
            continue
        if not stage_numbers & wanted:
            continue
        for task in add(spec, "train", stage):
            method = _arg(task.line, "method")
            if method in ("m3_lstm", "m4_lstm_reg"):
                reg = "none" if method == "m3_lstm" else _arg(task.line, "regularizer")
                arm = arms.get((method, reg, _arg(task.line, "meta_tag") or default_round))
                if arm:
                    task.needs.add(arm)
    if 4 in wanted:
        for spec in specs:
            if "compute" in spec:
                for task in add(spec, "compute", "4"):
                    task.needs |= set(arms.values())
    return tasks


def audit_tasks(study: str = "main", profile: Optional[str] = None, artifacts_root: Optional[Path] = None,
                only: Optional[Mapping[str, str]] = None, write: bool = True,
                regularizers: Optional[Sequence[str]] = None) -> List[Task]:
    """Stage 4's audits: one per finished training run of the study (planned after training)."""
    profile = profile or active_profile()
    tasks: List[Task] = []
    for spec in study_experiments(study, profile=profile):
        if "diagnostics" not in spec:
            continue
        lines = expand(spec, ["diagnostics"], _subset(only, "diagnostics"), artifacts_root)
        lines = [line for line in lines if _keep_run(line, regularizers)]
        if not lines:
            continue
        validate(lines, spec)
        if write:
            write_manifest(lines, manifest_path(spec["name"], ["diagnostics"], _subset(only, "diagnostics"),
                                                artifacts_root, profile))
        for i, line in enumerate(lines):
            task = Task(f"{spec['name']}:diagnostics:{i}", line, "4", str(spec["name"]), priority=(2, i))
            if finished(line):
                task.status, task.note = "done", "finished earlier"
            tasks.append(task)
    return tasks


def _keep_run(line: Sequence[str], regularizers: Optional[Sequence[str]]) -> bool:
    """The diagnostics-line version of :func:`_keep` (the regularizer is in the run folder's name)."""
    if not regularizers:
        return True
    run = Path(line[line.index("--run") + 1])
    reg = parse_method_tag(run.parts[-2])["regularizer"]
    return reg == "none" or reg in regularizers


class Runner:
    """Executes tasks in dependency order: in this process (``slots`` empty) or as child processes,
    one per slot at a time. A slot is a GPU id (``CUDA_VISIBLE_DEVICES``) or ``"cpu"``."""

    def __init__(self, tasks: Sequence[Task], slots: Sequence[str] = (), log_dir: Optional[Path] = None,
                 threads_per_job: Optional[int] = None, echo: Callable[[str], None] = print,
                 poll_seconds: float = 0.5) -> None:
        self.tasks = list(tasks)
        self.by_key = {t.key: t for t in self.tasks}
        self.slots = list(slots)
        self.log_dir = log_dir or (profile_artifacts() / "logs")
        self.threads = threads_per_job
        self.echo = echo
        self.poll = poll_seconds
        unknown = {n for t in self.tasks for n in t.needs if n not in self.by_key}
        if unknown:
            raise ValueError(f"tasks wait for unknown task(s): {sorted(unknown)}")

    # ---------------------------------------------------------------- shared

    def _ready(self) -> List[Task]:
        ready = []
        for task in self.tasks:
            if task.status != "pending":
                continue
            states = [self.by_key[n].status for n in task.needs]
            if any(s in ("failed", "skipped", "interrupted") for s in states):
                blocked = [n for n in task.needs if self.by_key[n].status in ("failed", "skipped", "interrupted")]
                task.status, task.note = "skipped", f"waits for {', '.join(blocked)}, which did not finish"
                continue
            if all(s == "done" for s in states):
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority)

    def _finish(self, task: Task, code: int, seconds: float) -> None:
        task.code, task.seconds = code, seconds
        task.status = "done" if code == EXIT_OK else ("interrupted" if code == EXIT_INTERRUPTED else "failed")
        done = sum(1 for t in self.tasks if t.status in ("done", "failed", "interrupted", "skipped"))
        mark = "ok " if code == EXIT_OK else "!! "
        where = f"  [{task.log}]" if task.log and code != EXIT_OK else ""
        self.echo(f"{mark}[{done}/{len(self.tasks)}] stage {task.stage} · {task.title} — {describe_exit(code)} "
                  f"({seconds:.0f}s){where}")

    def run(self) -> List[Task]:
        if not self.slots:
            self._run_here()
        else:
            self._run_children()
        return self.tasks

    # ---------------------------------------------------------------- in this process

    def _run_here(self) -> None:
        while True:
            ready = self._ready()
            if not ready:
                break
            task = ready[0]
            task.status = "running"
            self.echo(f"\n=== stage {task.stage} · {task.title}")
            started = time.monotonic()
            code = run_line(task.line)
            self._finish(task, code, time.monotonic() - started)
        self._skip_rest()

    # ---------------------------------------------------------------- child processes

    def _command(self, task: Task) -> List[str]:
        script = project_root() / "scripts" / f"{task.line[0]}.py"
        return [sys.executable, str(script), *task.line[1:]]

    def _environment(self, slot: str) -> Dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "" if slot == "cpu" else slot
        if self.threads:
            env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(self.threads)
        return env

    def _log_path(self, task: Task) -> Path:
        """``logs/<experiment>/<kind>-<number>[-<what>].log`` — e.g. train-0013-m4_lstm_reg-convnet4-svhn-sam-42."""
        parts = task.key.split(":")
        label = "-".join(f"{int(p):04d}" if p.isdigit() else p for p in parts[1:]) or parts[0]
        name = "-".join(a.split("=", 1)[1] for a in task.line[1:]
                        if a.split("=", 1)[0] in ("meta_training", "method", "optimizee", "dataset", "regularizer",
                                                  "meta_tag", "seed"))
        return self.log_dir / task.experiment / f"{label}{'-' + name if name else ''}.log"

    def _run_children(self) -> None:
        free = list(self.slots)
        running: Dict[str, Tuple[Task, subprocess.Popen, float, str, object]] = {}
        try:
            while True:
                for task in self._ready():
                    if not free:
                        break
                    slot = free.pop(0)
                    task.status, task.log = "running", self._log_path(task)
                    task.log.parent.mkdir(parents=True, exist_ok=True)
                    handle = open(task.log, "w", encoding="utf-8")
                    handle.write(f"# {' '.join(task.line)}\n# slot {slot}\n")
                    handle.flush()
                    process = subprocess.Popen(self._command(task), stdout=handle, stderr=subprocess.STDOUT,
                                               env=self._environment(slot), cwd=str(project_root()))
                    running[task.key] = (task, process, time.monotonic(), slot, handle)
                    self.echo(f"-> stage {task.stage} · {task.title}  (slot {slot})")
                if not running:
                    break
                time.sleep(self.poll)
                for key, (task, process, started, slot, handle) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    del running[key]
                    free.append(slot)
                    self._finish(task, int(code), time.monotonic() - started)
        except KeyboardInterrupt:
            self.echo("interrupted: stopping the running jobs (they resume when started again)")
            for task, process, _, _, handle in running.values():
                process.terminate()
            for task, process, _, _, handle in running.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                handle.close()
                task.status, task.note = "interrupted", "stopped by Ctrl+C"
            raise
        self._skip_rest()

    def _skip_rest(self) -> None:
        for task in self.tasks:
            if task.status == "pending":
                task.status, task.note = "skipped", "a prerequisite did not finish"

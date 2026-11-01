"""Experiment grids → run lists ("manifests"): one line per run.

A manifest line is a script name followed by its arguments, shell-quoted, e.g.

    train_optimizee method=m4_lstm_reg optimizee=resnet18 dataset=svhn regularizer=sam meta_tag=B seed=42 study=main profile=production

(``meta_tag=<round>``: M3/M4 lines, once per round the experiment file lists; Stage-1 lines carry
the round's source pair too.)

Every line names its profile, so it runs the same on a laptop, a PBS or SLURM array element
(``scripts/run_manifest.py <manifest> <i>``) or under ``scripts/run_study.py``. Before anything
is written, every line is checked: each run's configuration is composed (compute profiles: all
twelve variants), duplicates are refused, and the Stage-1 arms of the experiment are checked
to differ in nothing but their objective — always on the whole grid, even when ``--only``
selects some of them.

An experiment file (configs/experiments/*.yaml) may hold four blocks, run in this order:

    meta_train   →  meta_train lines        (Stage 1: one per round × arm × seed)
    train        →  train_optimizee lines   (Stage 1's M2, Stages 2–3: one per pair × method × regularizer
                                             [× round for M3/M4] × seed; ``from_stage1`` methods are
                                             skipped on a round's source pair, where Stage 1 trains them)
    diagnostics  →  run_diagnostics lines   (Stage 4: one per finished run of the study)
    compute      →  profile_compute lines   (Stage 4: one per optimizee × dataset pair)

Its ``rounds`` list (configs/rounds.yaml) says which rounds Stage 1 trains and whose optimizers
M3/M4 use; the evaluation settings follow from the rounds' sources (reglo.study), not from here.
"""

from __future__ import annotations

import runpy
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from omegaconf import OmegaConf

from reglo.config import ConfigError, available, compose_meta, compose_run, config_dir, load_experiment, project_root
from reglo.profiles import active_profile
from reglo.study import LEARNED, Pair, pair_overrides, spec_rounds, training_pairs
from reglo.utils.paths import find_runs, parse_method_tag, profile_artifacts

__all__ = [
    "KINDS",
    "ManifestError",
    "check_meta_arms",
    "compute_overrides",
    "expand",
    "manifest_path",
    "meta_differences",
    "parse_only",
    "read_manifest",
    "run_line",
    "validate",
    "write_manifest",
]

KINDS = ("meta_train", "train", "diagnostics", "compute")
SCRIPTS = {"meta_train", "train_optimizee", "run_diagnostics", "profile_compute"}
ONLY_KEYS = ("meta_training", "method", "regularizer", "optimizee", "dataset", "round", "seed")
# What two Stage-1 arms may differ in: the objective (regularizer r, L_smooth) and bookkeeping …
META_FREE_KEYS = {"selected", "regularizer", "smooth"}
META_FREE_ENTRY_KEYS = {"name", "allowed_regularizers", "objective"}
# … and, between rounds, the source pair (and the per-dataset numbers that come with it)
ROUND_FREE_KEYS = {"optimizee", "dataset", "meta_tag", "training"}

Line = List[str]


class ManifestError(ValueError):
    """An experiment grid that would launch invalid or inconsistent runs."""


def parse_only(items: Optional[Iterable[str]]) -> Dict[str, str]:
    """``["optimizee=resnet18", "seed=42"]`` → ``{"optimizee": "resnet18", "seed": "42"}``."""
    only: Dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ManifestError(f"--only takes key=value, got '{item}'")
        if key not in ONLY_KEYS:  # a misspelt key would otherwise filter nothing and launch the whole grid
            raise ManifestError(f"--only key '{key}' is not one of {', '.join(ONLY_KEYS)}")
        if key in only:
            raise ManifestError(f"--only {key}= given twice; submit one value at a time")
        only[key] = value
    return only


def _method_optimizer(method: str) -> str:
    path = config_dir() / "methods" / f"{method}.yaml"
    if not path.is_file():
        raise ManifestError(f"Unknown method '{method}' in the grid. Choose one of: {', '.join(available('method'))}")
    return str(OmegaConf.load(path).defaults.optimizer)


def _with_reg(reg: str) -> List[str]:
    return [] if reg == "none" else [f"regularizer={reg}"]


def _run_identity(run: Path) -> Dict[str, str]:
    optimizee, _, dataset = run.parts[-3].partition("-")
    tag = parse_method_tag(run.parts[-2])
    identity = {"optimizee": optimizee, "dataset": dataset, "method": tag["method"],
                "regularizer": tag["regularizer"], "seed": run.parts[-1].replace("seed", "")}
    if tag["round"]:
        identity["round"] = tag["round"]
    return identity


def expand(
    spec: Dict[str, Any],
    kinds: Optional[Sequence[str]] = None,
    only: Optional[Dict[str, str]] = None,
    artifacts_root: Optional[Union[str, Path]] = None,
) -> List[Line]:
    """Every run an experiment file describes, as manifest lines (script name first).

    ``spec`` comes from :func:`reglo.config.load_experiment`, at the scale of its profile; every
    line carries that profile. ``artifacts_root`` is the root holding the profiles' folders
    (default ``$REGLO_ARTIFACTS`` or ``<repo>/artifacts``): diagnostics lines are made for the
    finished runs found there. ``only`` keeps the runs whose fields match; a key that none of the
    candidate runs has (``meta_training=…`` on a training grid) is an error rather than a filter
    that keeps everything.
    """
    only = only or {}
    unknown = set(kinds or []) - set(KINDS)
    if unknown:
        raise ManifestError(f"Unknown kind(s) {sorted(unknown)}; choose from {KINDS}")
    wanted = [k for k in KINDS if k in spec and (not kinds or k in kinds)]
    study = str(spec.get("study", "adhoc"))
    name = str(spec["name"])
    profile = str(spec.get("profile") or active_profile())
    candidates: List[tuple] = []  # (identity, line) before filtering

    rounds = spec_rounds(spec)

    if "meta_train" in wanted:
        block = spec["meta_train"]
        for rnd in rounds or [None]:  # no rounds listed: the default meta_tag, the base config's pair
            source = [] if rnd is None else [f"optimizee={rnd.optimizee}", f"dataset={rnd.dataset}", f"meta_tag={rnd.name}"]
            for entry, regs in block["runs"].items():
                for reg in regs:
                    for seed in block["seeds"]:
                        identity = {"meta_training": entry, "regularizer": reg, "seed": str(seed)}
                        if rnd is not None:
                            identity.update(round=rnd.name, optimizee=rnd.optimizee, dataset=rnd.dataset)
                        candidates.append((identity, [
                            "meta_train", f"meta_training={entry}", *source, *_with_reg(reg), f"seed={seed}",
                            f"study={study}", f"profile={profile}", *(block.get("overrides") or [])]))

    if "train" in wanted:
        block = spec["train"]
        per_optimizer = block.get("optimizer_overrides") or {}
        pairs = training_pairs(spec)
        # the round a pair belongs to when it is a round's source (Stage 1's M2: pairs: sources)
        owner = {Pair(r.optimizee, r.dataset): r.name for r in rounds} if block.get("pairs") == "sources" else {}
        # methods whose runs on a round's source pair Stage 1 trains (M2): reused there, not repeated
        reused = set(block.get("from_stage1") or [])
        sources = {Pair(r.optimizee, r.dataset) for r in rounds}
        for pair in pairs:
            for method, regs in block["methods"].items():
                if method in reused and pair in sources:
                    continue
                extra = [*pair_overrides(block, pair),
                         *((per_optimizer.get(_method_optimizer(method)) or {}).get(pair.optimizee, []))]
                # M3/M4 run once per round (whose optimizers they load); M1/M2 use none
                tags = [r.name for r in rounds] if method in LEARNED and rounds else [None]
                for tag in tags:
                    for reg in regs:
                        for seed in block["seeds"]:
                            identity = {"method": method, "regularizer": reg, "optimizee": pair.optimizee,
                                        "dataset": pair.dataset, "seed": str(seed)}
                            if tag is not None or pair in owner:
                                identity["round"] = tag if tag is not None else owner[pair]
                            candidates.append((identity, [
                                "train_optimizee", f"method={method}", f"optimizee={pair.optimizee}",
                                f"dataset={pair.dataset}", *_with_reg(reg), *([f"meta_tag={tag}"] if tag else []),
                                f"seed={seed}", f"study={study}", f"profile={profile}",
                                *(block.get("overrides") or []), *extra]))

    if "diagnostics" in wanted:
        root = profile_artifacts(profile, artifacts_root)
        for run in find_runs(root, str(spec["diagnostics"].get("study", study))):
            candidates.append((_run_identity(run), ["run_diagnostics", "--experiment", name, "--profile", profile,
                                                    "--run", run.as_posix()]))

    if "compute" in wanted:
        for index, (optimizee, dataset) in enumerate(spec["compute"]["pairs"]):
            candidates.append(({"optimizee": optimizee, "dataset": dataset},
                               ["profile_compute", "--experiment", name, "--profile", profile, "--index", str(index)]))

    if only and candidates:
        fields = set().union(*(identity.keys() for identity, _ in candidates))
        idle = sorted(set(only) - fields)
        if idle:
            raise ManifestError(
                f"--only {', '.join(idle)}: no run of '{name}' ({', '.join(wanted)}) has that field; "
                f"its runs can be filtered by {', '.join(k for k in ONLY_KEYS if k in fields)}"
            )
    kept = [(identity, line) for identity, line in candidates
            if all(identity[k] == v for k, v in only.items() if k in identity)]
    limit = (spec.get("diagnostics") or {}).get("per_method") if "diagnostics" in wanted else None
    if limit is not None:  # audit at most `limit` runs per pair and method (the smoke profile: one)
        taken: Dict[tuple, int] = {}
        sampled = []
        for identity, line in kept:
            if line[0] == "run_diagnostics":
                key = (identity.get("optimizee"), identity.get("dataset"), identity.get("method"))
                if taken.get(key, 0) >= int(limit):
                    continue
                taken[key] = taken.get(key, 0) + 1
            sampled.append((identity, line))
        kept = sampled
    return [line for _, line in kept]


# ───────────────────────────────────────────────────────────────── validation

def meta_differences(a: Mapping[str, Any], b: Mapping[str, Any], free: Iterable[str] = ()) -> List[str]:
    """Keys in which two composed Stage-1 configs differ beyond their objective (and ``free``)."""
    out = []
    skip = META_FREE_KEYS | set(free)
    for key in sorted(set(a) | set(b)):
        if key in skip:
            continue
        va, vb = a.get(key), b.get(key)
        if key == "meta_training" and isinstance(va, Mapping) and isinstance(vb, Mapping):
            out += [f"meta_training.{k}" for k in sorted(set(va) | set(vb))
                    if k not in META_FREE_ENTRY_KEYS and va.get(k) != vb.get(k)]
        elif va != vb:
            out.append(key)
    return out


def check_meta_arms(lines: Sequence[Line]) -> None:
    """M3's and every M4's φ must be meta-trained identically except for the objective (per seed and
    round); the rounds' arms may differ only in their source pair."""
    groups: Dict[Any, List[tuple]] = {}
    for number, line in enumerate(lines, start=1):
        cfg = OmegaConf.to_container(compose_meta(line[1:]), resolve=True)
        groups.setdefault((cfg.get("seed"), cfg.get("meta_tag")), []).append((number, line, cfg))

    def name(line: Line) -> str:
        return " ".join(a for a in line[1:] if a.startswith(("meta_training=", "regularizer=", "meta_tag=")))

    for arms in groups.values():
        _, first_line, first = arms[0]
        for _, line, cfg in arms[1:]:
            differing = meta_differences(first, cfg)
            if differing:
                raise ManifestError(
                    f"Stage-1 arms '{name(first_line)}' and '{name(line)}' differ in {', '.join(differing)}: "
                    f"M3 and M4 must be meta-trained identically except for the objective"
                )
    firsts: Dict[Any, tuple] = {}
    for (seed, _), arms in groups.items():
        _, line, cfg = arms[0]
        if seed in firsts:
            differing = meta_differences(firsts[seed][1], cfg, ROUND_FREE_KEYS)
            if differing:
                raise ManifestError(
                    f"Stage-1 arms '{name(firsts[seed][0])}' and '{name(line)}' differ in {', '.join(differing)}: "
                    f"the rounds must meta-train identically except for their source pair"
                )
        else:
            firsts[seed] = (line, cfg)


def _flag(args: Sequence[str], flag: str) -> Optional[str]:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def compute_overrides(spec: Mapping[str, Any], index: int) -> List[str]:
    """The overrides profile_compute composes the twelve variants of compute pair ``index`` with."""
    block = spec.get("compute")
    if not block:
        raise ManifestError(f"experiment '{spec['name']}' has no compute block")
    if not 0 <= index < len(block["pairs"]):
        raise ManifestError(f"--index {index} is outside the {len(block['pairs'])} compute pairs")
    optimizee, dataset = block["pairs"][index]
    return [f"optimizee={optimizee}", f"dataset={dataset}", f"study={spec.get('study', 'adhoc')}",
            f"profile={spec['profile']}", *(block.get("overrides") or [])]


def _check_compute(args: Sequence[str]) -> None:
    """Compose all twelve variants a profile_compute line will time."""
    from reglo.diagnostics.compute import PROTOCOL_VARIANTS  # torch import; only for compute lines

    spec = load_experiment(_flag(args, "--experiment") or "stage4_analysis", profile=_flag(args, "--profile"))
    base = compute_overrides(spec, int(_flag(args, "--index") or 0))
    for method, reg in PROTOCOL_VARIANTS:
        compose_run([f"method={method}", *base, *_with_reg(reg)])


def _check_diagnostics(args: Sequence[str]) -> None:
    spec = load_experiment(_flag(args, "--experiment") or "stage4_analysis", profile=_flag(args, "--profile"))
    if not spec.get("diagnostics"):
        raise ManifestError(f"experiment '{spec['name']}' has no diagnostics block")
    run = _flag(args, "--run")
    if run is not None and not (Path(run) / "final.json").is_file():
        raise ManifestError(f"{run} has no final.json")


def validate(lines: Sequence[Line], spec: Optional[Mapping[str, Any]] = None) -> None:
    """Check every line before anything runs (see the module docstring).

    With ``spec``, the Stage-1 sameness check covers the experiment's whole meta_train block,
    so a manifest of a subset of the arms is still compared against the others.
    """
    errors: List[str] = []
    seen: Set[tuple] = set()
    meta_lines: List[Line] = []
    for number, line in enumerate(lines, start=1):
        if tuple(line) in seen:
            errors.append(f"line {number}: duplicate of an earlier line ({' '.join(line)})")
            continue
        seen.add(tuple(line))
        if any("\n" in arg or "\r" in arg for arg in line):
            errors.append(f"line {number}: an argument contains a line break")
            continue
        script, args = line[0], line[1:]
        if script not in SCRIPTS:
            errors.append(f"line {number}: unknown script '{script}'")
            continue
        try:
            if script == "train_optimizee":
                compose_run(args)
            elif script == "meta_train":
                compose_meta(args)
                meta_lines.append(line)
            elif script == "profile_compute":
                _check_compute(args)
            else:
                _check_diagnostics(args)
        except (ConfigError, ManifestError) as err:
            errors.append(f"line {number}: {err}")
    if errors:
        raise ManifestError("Invalid runs:\n  " + "\n  ".join(errors))
    if meta_lines:
        full = expand(dict(spec), ["meta_train"]) if spec and "meta_train" in spec else meta_lines
        check_meta_arms(full)


# ───────────────────────────────────────────────────────────────── files and running

def manifest_path(
    experiment: str,
    kinds: Optional[Sequence[str]] = None,
    only: Optional[Dict[str, str]] = None,
    artifacts_root: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
) -> Path:
    """``artifacts/<profile>/manifests/<experiment>[__<kinds>][__<key>-<value>…].txt``"""
    parts = [experiment]
    if kinds:
        parts.append("+".join(kinds))
    parts += [f"{k}-{v}" for k, v in sorted((only or {}).items())]
    return profile_artifacts(profile, artifacts_root) / "manifests" / ("__".join(parts) + ".txt")


def write_manifest(lines: Sequence[Line], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(shlex.join(line) + "\n" for line in lines), encoding="utf-8")
    return path


def read_manifest(path: Union[str, Path]) -> List[Line]:
    """Non-empty, non-comment lines, split the way a POSIX shell would."""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#"):
            lines.append(shlex.split(raw))
    return lines


def run_line(line: Line) -> int:
    """Run one manifest line in this process; its exit code (a crash counts as 1, not as an exception)."""
    script, args = line[0], line[1:]
    if script not in SCRIPTS:
        raise ManifestError(f"unknown script '{script}'")
    try:
        main = runpy.run_path(str(project_root() / "scripts" / f"{script}.py"))["main"]
        return int(main(list(args)) or 0)
    except SystemExit as stop:  # argparse errors and explicit exits inside the script
        code = stop.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except Exception:  # noqa: BLE001 - one failed run must not stop the others
        traceback.print_exc(file=sys.stderr)
        return 1

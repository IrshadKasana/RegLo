"""Connectivity of a study: does every stage get what it needs from the stages before it?

    Stage 1 → 2/3   every M3/M4 run reads a φ that a Stage-1 arm writes (same round, same
                    regularizer); the arms differ only in their objective, the rounds only in
                    their source pair; meta-training and the optimizee runs split each source
                    dataset the same way, so the optimizee half was never seen by the optimizers;
                    M2 (AdamW + r) is trained on every round's source pair
    2/3 → 4         every round's source pair is evaluated (its seen settings exist); each
                    evaluation pair is trained by one grid; M2 and M4 use the same regularizers
                    (complete cube blocks); M2 (AdamW + r) is trained on every evaluation pair, on
                    a round's source pair by Stage 1 (reused), identically configured;
                    the Stage-4 audits and timings point at this study and its pairs
    everything      every run of every grid composes into a valid configuration

plus, read from ``artifacts/``, how far each stage has got. Used by scripts/check_study.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from reglo.config import ConfigError, compose_meta, compose_run, default_meta_tag, experiment_names, load_experiment
from reglo.manifest import ManifestError, check_meta_arms, expand
from reglo.profiles import active_profile
from reglo.rounds import ROLE_LABELS
from reglo.study import (
    LEARNED,
    REG_NAMES,
    SETTINGS,
    STAGES,
    Pair,
    is_evaluation,
    spec_rounds,
    study_experiments,
    study_rounds,
    study_settings,
    training_pairs,
)
from reglo.utils.paths import default_artifacts_root, diagnostics_dir_for, find_runs, meta_run_dir, run_dir

__all__ = ["Check", "StudyCheck", "check_study"]

PathLike = Union[str, Path]


@dataclass
class Check:
    ok: bool
    text: str
    warning: bool = False  # worth knowing, but does not fail the study

    @property
    def mark(self) -> str:
        return "[ok]  " if self.ok else ("[warn]" if self.warning else "[FAIL]")


@dataclass
class StudyCheck:
    study: str
    profile: str = ""
    folder: str = ""
    overview: List[str] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok or c.warning for c in self.checks)

    def text(self) -> str:
        head = f"Study '{self.study}' — profile {self.profile}" + (f" ({self.folder})" if self.folder else "")
        lines = [head, ""] + self.overview + ["", "Connectivity checks"]
        lines += [f"  {c.mark} {c.text}" for c in self.checks]
        failed = sum(1 for c in self.checks if not c.ok and not c.warning)
        lines += ["", "all checks passed" if failed == 0 else f"{failed} check(s) failed"]
        return "\n".join(lines)


def _arg(line: List[str], key: str) -> Optional[str]:
    return next((a.split("=", 1)[1] for a in line if a.startswith(key + "=")), None)


def _split(cfg: DictConfig) -> Tuple[Any, ...]:
    s = cfg.dataset.meta_split
    return bool(s.enabled), float(s.fraction), int(s.seed)


def check_study(study: str, artifacts_root: Optional[PathLike] = None, profile: Optional[str] = None) -> StudyCheck:
    """Static checks of the experiment files of ``study`` at the scale of ``profile`` (default: the
    active one), and its progress under ``<artifacts_root>/<profile>/``."""
    profile = profile or active_profile()
    base = Path(artifacts_root).expanduser() if artifacts_root else default_artifacts_root()
    root = base / profile
    at_root = [f"paths.artifacts_root={base.as_posix()}"]  # compose against the folder being inspected
    report = StudyCheck(study, profile, root.as_posix())
    add = report.checks.append
    specs = study_experiments(study, profile=profile)
    if not specs:
        add(Check(False, f"no experiment file declares study: {study}"))
        return report

    # every line composes; Stage 1's sameness check runs on each file's whole meta grid. Runs that
    # differ only in their seed compose identically, so one of them is composed per variant.
    metas: List[Tuple[List[str], DictConfig]] = []
    trains: List[Tuple[str, List[str], DictConfig]] = []
    trained_by: Dict[Pair, List[str]] = {}      # evaluation pairs → the grids training them
    benchmark: List[Tuple[str, List[str]]] = []  # Stage 1's M2 lines
    errors: List[str] = []
    for spec in specs:
        try:
            meta_lines = expand(spec, ["meta_train"]) if "meta_train" in spec else []
            if meta_lines:
                check_meta_arms(meta_lines)
            for line in meta_lines:
                metas.append((line, compose_meta(line[1:] + at_root)))
            variants: Dict[tuple, DictConfig] = {}
            for line in expand(spec, ["train"]) if "train" in spec else []:
                variant = tuple(a for a in line if not a.startswith("seed="))
                if variant not in variants:
                    variants[variant] = compose_run(line[1:] + at_root)
                trains.append((str(spec["name"]), line, variants[variant]))
                if not is_evaluation(spec):
                    benchmark.append((str(spec["name"]), line))
        except (ConfigError, ManifestError) as err:
            errors.append(f"{spec['name']}: {err}")
            continue
        if is_evaluation(spec):
            for pair in training_pairs(spec):
                trained_by.setdefault(pair, []).append(str(spec["name"]))
    if errors:
        for error in errors:
            add(Check(False, error))
    else:
        add(Check(True, f"all {len(metas)} Stage-1 meta-training runs and {len(trains)} optimizee runs compose; "
                        "the Stage-1 arms differ only in their objective, the rounds only in their source pair"))
    on_validation = sorted({name for name, _, cfg in trains if float(cfg.training.get("validation_fraction", 0) or 0) > 0})
    if on_validation:
        add(Check(False, f"{', '.join(on_validation)}: runs evaluate on a validation split "
                         "(training.validation_fraction > 0), so their 'test' numbers are not test-set numbers"))
    keys = [tuple(line) for _, line, _ in trains]
    if len(keys) != len(set(keys)):
        add(Check(False, f"{len(keys) - len(set(keys))} run(s) appear twice across the study's grids"))

    # Stage 1 → Stages 2/3 (a φ is found by its round, so another study's Stage 1 may provide it: the pilot)
    produced = {meta_run_dir(cfg) / "phi_best.pt": line for line, cfg in metas}
    elsewhere: Dict[Path, str] = {}
    for spec in (load_experiment(n, profile=profile) for n in experiment_names()):
        if "meta_train" in spec and str(spec.get("study")) != study:
            for line in expand(spec, ["meta_train"]):
                elsewhere[meta_run_dir(compose_meta(line[1:] + at_root)) / "phi_best.pt"] = str(spec["name"])
    splits = {str(cfg.dataset.name): _split(cfg) for _, cfg in metas}
    needed: Dict[Path, Set[str]] = {}
    mismatched: Set[str] = set()
    for name, _, cfg in trains:
        if cfg.method.get("checkpoint"):
            needed.setdefault(Path(str(cfg.method.checkpoint)), set()).add(name)
        dataset = str(cfg.dataset.name)
        if dataset in splits and _split(cfg) != splits[dataset]:
            mismatched.add(f"{name} on {dataset}")
    missing = sorted(p for p in needed if p not in produced and p not in elsewhere)
    if missing:
        add(Check(False, "M3/M4 runs read optimizers that no Stage-1 arm writes: "
                         + ", ".join(f"{p.parent.parent.name}/{p.parent.name}" for p in missing)))
    elif needed:
        tags = sorted({p.parent.parent.name for p in needed})
        others = sorted({elsewhere[p] for p in needed if p not in produced})
        add(Check(True, f"every M3/M4 run reads a φ that Stage 1 writes ({len(needed)} optimizer(s), "
                        f"round(s) {', '.join(tags)}" + (f"; from {', '.join(others)}" if others else "") + ")"))
    unused = sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in produced if p not in needed)
    if unused and trains:
        add(Check(False, f"Stage-1 optimizers that no run of the study uses: {', '.join(unused)}", warning=True))
    if mismatched:
        add(Check(False, "the meta split differs between Stage 1 and " + ", ".join(sorted(mismatched))
                         + ": the optimizee half may overlap the meta-training half"))
    elif splits:
        add(Check(True, "Stage 1 and the optimizee runs split " + ", ".join(sorted(splits))
                        + " identically: every optimizee run trains on the half the optimizers never saw"))

    rounds = study_rounds(study, profile=profile)
    m2_regs = {_arg(line, "regularizer") for _, line in benchmark if _arg(line, "method") == "m2_adamw_reg"}
    if benchmark:
        sources = {(r.optimizee, r.dataset) for spec in specs if not is_evaluation(spec) for r in spec_rounds(spec)}
        trained = {(_arg(line, "optimizee"), _arg(line, "dataset")) for _, line in benchmark}
        add(Check(trained == sources, "Stage 1 trains M2 (AdamW + r) on the source pair of every round"
                  + ("" if trained == sources else f" — trains {sorted(trained)}, sources {sorted(sources)}")))

    # Stages 2/3 → Stage 4
    settings = study_settings(study, profile=profile)
    if trained_by:
        unseen = [f"{r.name} ({Pair(r.optimizee, r.dataset).label})" for r in rounds
                  if r.name in settings and not settings[r.name].get("seen_optimizee")]
        add(Check(not unseen, "every round's source pair is evaluated (its seen settings exist)"
                  + (f" — not: {', '.join(unseen)}" if unseen else "")))
        twice = sorted(f"{p.label} ({', '.join(by)})" for p, by in trained_by.items() if len(by) > 1)
        add(Check(not twice, "no evaluation pair is trained by two grids" + (f" — twice: {'; '.join(twice)}" if twice else "")))
        m4_regs = {_arg(line, "regularizer") for name, line, _ in trains if _arg(line, "method") == "m4_lstm_reg"}
        arm_regs = {_arg(line, "regularizer") for line, _ in metas if _arg(line, "meta_training") == "m4_lstm_reg"}
        incomplete = []
        for pair in trained_by:
            for rnd in {_arg(line, "meta_tag") or default_meta_tag() for _, line, _ in trains
                        if _arg(line, "method") in LEARNED and (_arg(line, "optimizee"), _arg(line, "dataset"))
                        == (pair.optimizee, pair.dataset)}:
                regs = {_arg(line, "regularizer") for _, line, _ in trains
                        if (_arg(line, "optimizee"), _arg(line, "dataset"), _arg(line, "method"))
                        == (pair.optimizee, pair.dataset, "m4_lstm_reg") and (_arg(line, "meta_tag") or default_meta_tag()) == rnd}
                if regs != m4_regs:
                    incomplete.append(f"{pair.label} @{rnd}")
        blocks = incomplete or (m2_regs and m2_regs != m4_regs) or (arm_regs and arm_regs != m4_regs)
        add(Check(not blocks, "M2 (Stage 1) and M4 (every round, every pair) use the same regularizers "
                  "(complete cube blocks)" + (f" — not on {', '.join(incomplete)}" if incomplete else "")
                  + (f" — M2: {sorted(m2_regs)}, M4: {sorted(m4_regs)}" if m2_regs and m2_regs != m4_regs else "")))
        # M2 = AdamW + r on every evaluation pair: its own grid, or Stage 1 on a round's source pair
        m2_on: Dict[Tuple[str, str], Set[str]] = {}
        for _, line, _ in trains:
            if _arg(line, "method") == "m2_adamw_reg":
                m2_on.setdefault((_arg(line, "optimizee"), _arg(line, "dataset")), set()).add(_arg(line, "regularizer"))
        without = sorted(p.label for p in trained_by if m2_on.get((p.optimizee, p.dataset), set()) != m4_regs)
        add(Check(not without, "M2 (AdamW + r) is trained with every regularizer on every evaluation pair"
                  + (f" — not on {', '.join(without)}" if without else "")))
        # a reused Stage-1 run must be the run the evaluation grid would have trained (one config per
        # variant: the seed is the only thing that differs between its runs)
        def unseeded(cfg: DictConfig) -> Dict[str, Any]:
            return {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k != "seed"}

        variant_of = lambda line: (_arg(line, "optimizee"), _arg(line, "dataset"), _arg(line, "method"),  # noqa: E731
                                   _arg(line, "regularizer"))
        bench_lines = {tuple(line) for _, line in benchmark}
        stage1 = {variant_of(line): cfg for _, line, cfg in trains if tuple(line) in bench_lines}
        mismatch = []
        for spec in specs:
            reused = set((spec.get("train") or {}).get("from_stage1") or [])
            if not is_evaluation(spec) or not reused:
                continue
            probe = dict(spec, train={k: v for k, v in spec["train"].items() if k != "from_stage1"})
            for key in {variant_of(line): line for line in expand(probe, ["train"])}.items():
                (optimizee, dataset, method, reg), line = key
                if method not in reused or not any((r.optimizee, r.dataset) == (optimizee, dataset)
                                                   for r in spec_rounds(spec)):
                    continue
                if key[0] not in stage1 or unseeded(stage1[key[0]]) != unseeded(compose_run(line[1:] + at_root)):
                    mismatch.append(f"{Pair(optimizee, dataset).label} {method}-{reg}")
        add(Check(not mismatch, "the M2 runs reused from Stage 1 on the source pairs are configured exactly as the "
                  "evaluation grids would train them" + (f" — differ: {', '.join(sorted(set(mismatch)))}" if mismatch else "")))
    for spec in specs:
        block = spec.get("diagnostics")
        if block:
            audited = str(block.get("study", study))
            add(Check(audited == study, f"{spec['name']} audits the runs of study '{audited}'"))
        pairs = (spec.get("compute") or {}).get("pairs", [])
        if pairs:
            strangers = [f"{o}-{d}" for o, d in pairs if Pair(str(o), str(d)) not in trained_by]
            add(Check(not strangers, f"{spec['name']}: the timed pairs are pairs the study trains"
                      + (f" — not: {', '.join(strangers)}" if strangers else ""), warning=bool(strangers)))

    report.overview += _progress(study, root, specs, metas, trains, benchmark, settings, trained_by, rounds)
    return report


def _done(cfg: DictConfig, line: List[str]) -> bool:
    folder = run_dir(cfg).parent / f"seed{_arg(line, 'seed') or cfg.seed}"  # cfg was composed for one seed
    return (folder / "final.json").is_file()


def _progress(study, root, specs, metas, trains, benchmark, settings, trained_by, rounds) -> List[str]:
    """How far each stage has got, read from the artifacts folder."""
    out = [f"Stage 1 · {STAGES[1]}"]
    if not metas and not benchmark:
        out.append("  (no Stage-1 grid in this study)")
    by_round: Dict[str, List[Tuple[List[str], DictConfig]]] = {}
    for line, cfg in metas:
        by_round.setdefault(str(cfg.meta_tag), []).append((line, cfg))
    sources = {r.name: r for r in rounds}
    bench_lines = {tuple(line) for _, line in benchmark}
    for tag, arms in by_round.items():
        rnd = sources.get(tag)
        cfg0 = arms[0][1]
        out.append(f"  round {tag} — source {Pair(str(cfg0.selected.optimizee), str(cfg0.selected.dataset)).label}"
                   + (f" ({rnd.description})" if rnd and rnd.description else ""))
        for line, cfg in arms:
            folder = meta_run_dir(cfg)
            state = ("done" if (folder / "summary.json").is_file() else
                     "in progress" if (folder / "phi_last.pt").is_file() or (folder / "meta_log.jsonl").is_file() else
                     "not started")
            reg = _arg(line, "regularizer") or "none"
            role = "M3" if reg == "none" else f"M4-{REG_NAMES[reg]}"
            out.append(f"    {folder.parent.name}/{folder.name:<20} {role:<8} {state}")
        mine = [(line, cfg) for _, line, cfg in trains if tuple(line) in bench_lines
                and (_arg(line, "optimizee"), _arg(line, "dataset")) == (str(cfg0.selected.optimizee), str(cfg0.selected.dataset))]
        if mine:
            done = sum(_done(cfg, line) for line, cfg in mine)
            out.append(f"    M2 = AdamW + r on the source pair: {done:>3} / {len(mine):<3} runs finished")

    # (pair, round | "" for M1 | "M2") → [finished, total]; M2 on a source pair counts Stage 1's runs
    counts: Dict[Tuple[Pair, str], List[int]] = {}
    for name, line, cfg in trains:
        method = _arg(line, "method")
        if tuple(line) in bench_lines and method != "m2_adamw_reg":
            continue
        key = (Pair(_arg(line, "optimizee"), _arg(line, "dataset")),
               (_arg(line, "meta_tag") or default_meta_tag()) if method in LEARNED else
               "M2" if method == "m2_adamw_reg" else "")
        entry = counts.setdefault(key, [0, 0])
        entry[0] += int(_done(cfg, line))
        entry[1] += 1
    for stage in (2, 3):
        rows = []
        for tag, per_setting in settings.items():
            for key in (s.key for s in SETTINGS):
                for sp in per_setting.get(key, []):
                    if sp.setting.stage != stage:
                        continue
                    m1 = counts.get((sp.pair, ""), [0, 0])
                    m2 = counts.get((sp.pair, "M2"), [0, 0])
                    learned = counts.get((sp.pair, tag), [0, 0])
                    label = sp.pair.label + (" (extended)" if sp.extended else "")
                    role = ROLE_LABELS[sp.role] if sp.role in ("unseen_both",) else ""
                    rows.append(f"  {tag}  {sp.setting.label:<17} {label:<38} M1 {m1[0]:>2}/{m1[1]:<2} · M2 {m2[0]:>2}/{m2[1]:<2} · M3+M4 "
                                f"{learned[0]:>3}/{learned[1]:<3}{('  ' + role) if role else ''}")
        if rows:
            out += ["", f"Stage {stage} · {STAGES[stage]} (by round)"] + rows
    evaluated = {sp.pair for per in settings.values() for pairs in per.values() for sp in pairs}
    extra = [p for p in trained_by if p not in evaluated]
    if extra:
        out += ["", "Grids outside the evaluation settings"]
        out += [f"  {p.label:<38} M1 {counts.get((p, ''), [0, 0])[0]} / {counts.get((p, ''), [0, 0])[1]}" for p in extra]

    out += ["", f"Stage 4 · {STAGES[4]}"]
    if settings:
        columns = sum(len(v) for per in settings.values() for v in per.values())
        out.append(f"  3D tables per round: test accuracy, generalization gap, FLOPs — {columns} setting column(s) in "
                   f"{len(settings)} round(s); python scripts/make_report.py --profile {root.name} --studies {study}")
    runs = find_runs(root, study)
    audited = sum(1 for run in runs if (diagnostics_dir_for(run) / "diagnostics.json").is_file())
    limit = next(((s.get("diagnostics") or {}).get("per_method") for s in specs if s.get("diagnostics")), None)
    out.append(f"  audits: {audited} of {len(runs)} finished runs"
               + (f" (this profile audits at most {limit} per pair and method)" if limit is not None else ""))
    for spec in specs:
        pairs = (spec.get("compute") or {}).get("pairs", [])
        if pairs:
            profiled = sum(1 for o, d in pairs if (root / "diagnostics" / "compute" / f"{o}-{d}.json").is_file())
            out.append(f"  same-GPU timing ({spec['name']}): {profiled} / {len(pairs)} pairs profiled")
    return out

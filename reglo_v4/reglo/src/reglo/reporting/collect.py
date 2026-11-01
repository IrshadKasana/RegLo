"""Read every result of one profile (``artifacts/<profile>/``) into flat records and seed statistics.

Sources (all written by the scripts, never edited by hand)::

    runs/<study>/<optimizee>-<dataset>/<method>[-<r>][@<round>]/seed<k>/final.json     Stages 1 (M2) – 3
    runs/…/seed<k>/metrics.jsonl, weights.json                               curves, weight histogram
    diagnostics/<study>/…/seed<k>/diagnostics.json                           Stage 4 audits
    diagnostics/compute/<optimizee>-<dataset>.json                           Stage 4 timing
    meta_training/<round>/<arm>/summary.json, meta_log.jsonl                 Stage 1

A *record* is one run as a flat dict (identity + numbers). Seeds are pooled by
:func:`aggregate` into :class:`Stat` objects (mean, sample std, n), per (study, pair, method,
regularizer, round) — the round is that of the φ an M3/M4 run used, ``""`` for M1/M2.
:func:`round_view` turns that into the five-part keys the tables use for one round: M1, M2
(AdamW + r, trained with r on every pair — by Stage 1 on the source pairs), and the round's
M3/M4. Only runs whose status is ``completed`` enter the statistics;
diverged runs are counted separately so a table can say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from reglo.study import DATASET_NAMES, LABELS, LEARNED, METHODS, OPTIMIZEE_NAMES, REG_NAMES, REGULARIZERS, Pair
from reglo.utils.logging import read_json, read_jsonl
from reglo.utils.paths import diagnostics_dir_for, find_runs, parse_method_tag

__all__ = [
    "DATASET_NAMES",
    "LABELS",
    "METHODS",
    "OPTIMIZEE_NAMES",
    "REGULARIZERS",
    "REG_NAMES",
    "STUDY_ORDER",
    "Group",
    "GroupKey",
    "Stat",
    "aggregate",
    "collect_compute",
    "collect_meta",
    "collect_runs",
    "discover_meta_tags",
    "discover_studies",
    "epoch_curves",
    "pairs_of",
    "round_view",
    "run_record",
    "to_float",
]

STUDY_ORDER = ("main", "pilot")

PathLike = Union[str, Path]
GroupKey = Tuple[str, str, str, str, str]  # study, optimizee, dataset, method, regularizer (one round's view)
RoundKey = Tuple[str, str, str, str, str, str]  # … + round ("" for M1/M2)

_IDENTITY = ("optimizee", "dataset", "method", "label", "regularizer", "seed", "status", "phi_source", "role")
_FINAL_NUMBERS = (
    "test_top1", "test_top5", "test_loss", "train_top1", "train_loss", "gen_gap", "gen_gap_top1",
    "wall_clock_s", "seconds_per_epoch", "passes_per_step", "flops_per_step", "total_gflops",
    "epochs_completed", "steps", "converged_step",
)
SNAPSHOTS = ("converged", "final")


def to_float(value: Any) -> Optional[float]:
    """JSON number, or the strings 'nan'/'inf' our writer uses for non-finite values → float; else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _radius_entry(per_radius: Mapping[str, Any], radius: float) -> Mapping[str, Any]:
    """Sharpness results are keyed by str(radius); match numerically."""
    for key, value in per_radius.items():
        r = to_float(key)
        if r is not None and math.isclose(r, radius, rel_tol=1e-9, abs_tol=1e-15):
            return value
    return {}


def _latest_attempt(rows: Sequence[Mapping[str, Any]], key) -> List[Mapping[str, Any]]:
    """Rows of the attempt that counts: after a resume a trainer appends rows again from its last
    checkpoint, so whenever ``key`` goes backwards the earlier rows from that point on are dropped."""
    kept: List[Mapping[str, Any]] = []
    for row in rows:
        k = key(row)
        if kept and k <= key(kept[-1]):
            kept = [r for r in kept if key(r) < k]
        kept.append(row)
    return kept


# ───────────────────────────────────────────────────────────────── Stage 2–3 runs

def run_record(run: PathLike, radius: float = 0.01) -> Dict[str, Any]:
    """One finished run (final.json, plus its Stage-4 audit if present) as a flat record."""
    run = Path(run)
    final = read_json(run / "final.json")
    rec: Dict[str, Any] = {"study": final.get("study", run.parts[-4])}
    rec.update({k: final.get(k) for k in _IDENTITY})
    rec.update({k: to_float(final.get(k)) for k in _FINAL_NUMBERS})
    rec["seed"] = int(final.get("seed", -1))
    rec["round"] = str(final.get("round") or parse_method_tag(run.parts[-2])["round"] or "")
    rec["run"] = run.as_posix()
    weights = final.get("weights") or {}
    for k in ("num_params", "l1", "l2", "sparsity"):
        rec[k] = to_float(weights.get(k))
    steps, seconds, flops = rec["steps"], rec["wall_clock_s"], rec["flops_per_step"]
    rec["gflops_per_step"] = None if flops is None else flops / 1e9
    rec["train_minutes"] = None if seconds is None else seconds / 60.0
    rec["ms_per_step"] = None if not steps or seconds is None else 1e3 * seconds / steps

    rec["audited"] = False
    try:
        diag_path = diagnostics_dir_for(run) / "diagnostics.json"
    except (ValueError, IndexError):
        diag_path = None
    if diag_path is not None and diag_path.is_file():
        diag = read_json(diag_path)
        if diag.get("status") == "completed":
            rec["audited"] = True
            hessian = (diag.get("hessian") or {}).get("final") or {}
            rec["lambda_max"] = to_float(hessian.get("lambda_max"))
            rec["hessian_trace"] = to_float(hessian.get("trace"))
            for snap in SNAPSHOTS:
                entry = _radius_entry((diag.get("sharpness") or {}).get(snap) or {}, radius)
                for key in ("loss", "max_loss", "surrogate_gap", "max_grad_norm"):
                    rec[f"{key}_{snap}"] = to_float(entry.get(key))
                snap_weights = (diag.get("weights") or {}).get(snap) or {}
                for key in ("l1", "l2", "sparsity"):
                    rec[f"{key}_{snap}"] = to_float(snap_weights.get(key))
    return rec


def discover_studies(artifacts_root: PathLike) -> List[str]:
    """Studies with at least one finished run: main first, then pilot, others."""
    base = Path(artifacts_root) / "runs"
    found = {p.name for p in base.iterdir() if p.is_dir() and any(p.glob("*/*/seed*/final.json"))} if base.is_dir() else set()
    return sorted(found, key=lambda name: (STUDY_ORDER.index(name) if name in STUDY_ORDER else len(STUDY_ORDER), name))


def collect_runs(artifacts_root: PathLike, studies: Optional[Sequence[str]] = None, radius: float = 0.01) -> List[Dict[str, Any]]:
    """Records of every run of ``studies`` (default: all) that has written final.json."""
    studies = list(studies) if studies else discover_studies(artifacts_root)
    return [run_record(run, radius) for study in studies for run in find_runs(artifacts_root, study)]


def pairs_of(records: Iterable[Mapping[str, Any]], study: str) -> List[Pair]:
    """(optimizee, dataset) pairs with runs in ``study``, in the vocabulary's order."""
    found = {(r["optimizee"], r["dataset"]) for r in records if r["study"] == study}
    order_o, order_d = list(OPTIMIZEE_NAMES), list(DATASET_NAMES)
    ordered = sorted(found, key=lambda p: (order_d.index(p[1]) if p[1] in order_d else 99,
                                           order_o.index(p[0]) if p[0] in order_o else 99, p))
    return [Pair(o, d) for o, d in ordered]


def epoch_curves(runs: Sequence[str], key: str = "train_loss_running") -> Optional[Tuple[List[float], List[float]]]:
    """Seed-mean of a per-epoch quantity from the runs' metrics.jsonl: (epochs, values)."""
    per_epoch: Dict[int, List[float]] = {}
    for run in runs:
        path = Path(run) / "metrics.jsonl"
        if not path.is_file():
            continue
        rows = _latest_attempt([r for r in read_jsonl(path) if r.get("kind") == "epoch"], lambda r: int(r.get("epoch", 0)))
        for row in rows:
            value = to_float(row.get(key))
            if value is not None and math.isfinite(value):
                per_epoch.setdefault(int(row["epoch"]), []).append(value)
    if not per_epoch:
        return None
    epochs = sorted(per_epoch)
    return [float(e) for e in epochs], [sum(per_epoch[e]) / len(per_epoch[e]) for e in epochs]


# ───────────────────────────────────────────────────────────────── statistics

@dataclass
class Stat:
    """Seed statistics of one number: mean, sample standard deviation (n − 1), n."""

    values: List[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> Optional[float]:
        return sum(self.values) / self.n if self.values else None

    @property
    def std(self) -> Optional[float]:
        """None below two values: one seed says nothing about the spread."""
        if self.n < 2:
            return None
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (self.n - 1))


@dataclass
class Group:
    """All seeds of one (study, optimizee, dataset, method, regularizer, round)."""

    stats: Dict[str, Stat] = field(default_factory=dict)
    completed: int = 0
    diverged: int = 0
    seeds: List[int] = field(default_factory=list)
    runs: List[str] = field(default_factory=list)  # completed run folders

    def get(self, key: str) -> Stat:
        return self.stats.get(key, Stat())


def aggregate(records: Iterable[Mapping[str, Any]]) -> Dict[RoundKey, Group]:
    """Pool seeds. Non-finite values are left out of the statistics (and show up as a smaller n)."""
    groups: Dict[RoundKey, Group] = {}
    for rec in records:
        key = (rec["study"], rec["optimizee"], rec["dataset"], rec["method"], rec["regularizer"],
               (rec.get("round") or "") if rec["method"] in LEARNED else "")
        group = groups.setdefault(key, Group())
        group.seeds.append(int(rec["seed"]))
        if rec.get("status") != "completed":
            group.diverged += rec.get("status") == "diverged"
            continue
        group.completed += 1
        group.runs.append(rec["run"])
        for name, value in rec.items():
            if isinstance(value, float) and math.isfinite(value):
                group.stats.setdefault(name, Stat()).values.append(value)
    return groups


def round_view(groups: Mapping[RoundKey, Group], round_name: str) -> Dict[GroupKey, Group]:
    """The groups as one round sees them, keyed (study, optimizee, dataset, method, regularizer).

    M1 and M2 (AdamW + r, its own runs with r on) always — they use no learned optimizer; M3/M4
    with this round's φ.
    """
    view: Dict[GroupKey, Group] = {}
    for (study, o, d, m, r, rnd), group in groups.items():
        if m not in LEARNED or rnd == round_name:
            view[(study, o, d, m, r)] = group
    return view


# ───────────────────────────────────────────────────────────────── Stage 4 timing

def collect_compute(artifacts_root: PathLike) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``{(optimizee, dataset): profile}`` from diagnostics/compute/*.json."""
    folder = Path(artifacts_root) / "diagnostics" / "compute"
    profiles = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        profile = read_json(path)
        profiles[(profile["optimizee"], profile["dataset"])] = profile
    return profiles


# ───────────────────────────────────────────────────────────────── Stage 1

def discover_meta_tags(artifacts_root: PathLike) -> List[str]:
    """Round (meta_tag) folders that hold at least one arm with a summary.json or meta_log.jsonl."""
    base = Path(artifacts_root) / "meta_training"
    if not base.is_dir():
        return []
    tags = [p.name for p in base.iterdir() if p.is_dir() and (any(p.glob("*/summary.json")) or any(p.glob("*/meta_log.jsonl")))]
    return sorted(tags)


def collect_meta(artifacts_root: PathLike, meta_tag: str = "A") -> List[Dict[str, Any]]:
    """One entry per Stage-1 arm: its summary (if finished) and its full meta log.

    ``arm`` is the folder name (``m3_lstm``, ``m4_lstm_reg-sam``, …); ``regularizer`` is
    ``none`` for M3.
    """
    base = Path(artifacts_root) / "meta_training" / meta_tag
    arms = []
    for folder in sorted(base.iterdir()) if base.is_dir() else []:
        log_path, summary_path = folder / "meta_log.jsonl", folder / "summary.json"
        if not folder.is_dir() or not (log_path.is_file() or summary_path.is_file()):
            continue
        entry, _, reg = folder.name.partition("-")
        rows = read_jsonl(log_path) if log_path.is_file() else []
        arms.append({
            "arm": folder.name,
            "meta_training": entry,
            "regularizer": reg or "none",
            "summary": read_json(summary_path) if summary_path.is_file() else {"status": "running"},
            "steps": _latest_attempt([r for r in rows if r.get("kind") == "meta_step"],
                                     lambda r: (int(r.get("trajectory", 0)), int(r.get("truncation", 0)))),
            "validation": _latest_attempt([r for r in rows if r.get("kind") == "validation"],
                                          lambda r: int(r.get("trajectory", 0))),
        })
    order = {"none": 0, **{r: i + 1 for i, r in enumerate(REGULARIZERS)}}
    return sorted(arms, key=lambda a: (a["meta_training"] != "m3_lstm", order.get(a["regularizer"], 99)))

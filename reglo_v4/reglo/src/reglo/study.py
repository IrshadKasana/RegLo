"""The study design — the thesis figure — in code.

    Stage 1  Training: meta-train the optimizers; train M2    per round (configs/rounds.yaml), on its source pair:
                                                              φ for M3, φ*_r for M4 (one per regularizer r), and
                                                              M2 = AdamW + r (M1 without r is its reference)
    Stage 2  Optimizer evaluation                             seen optimizee  vs  unseen optimizee
    Stage 3  Optimizee evaluation                             seen dataset    vs  unseen datasets
    Stage 4  Evaluation metrics                               test accuracy ↑, generalization gap ↓, FLOPs ↓,
             one 3D table per round and metric: method × regularizer × evaluation setting,
             and the cross-evaluation of the rounds' optimizers on every pair

Regularizers in Stages 2–3: M2 = AdamW + r computes r at every step on every evaluation pair
(the explicit-regularization baseline); M3 and M4 run their frozen Stage-1 LSTM and compute no r.
On a round's source pair M2's runs are Stage 1's (``from_stage1`` in the evaluation files), so
no run is trained twice.

Rounds and settings. A round meta-trains its optimizers on one source pair (A: ConvNet-4 / SVHN,
B: MLP / FMNIST). Every evaluation pair is trained with M1 once and with M3/M4 of every round
listed in its experiment file (cross-evaluation). Its evaluation setting is relative to the
round's source and derived, not listed::

    seen optimizee + seen dataset   the source pair itself              Stage 2 and Stage 3
    unseen optimizee                another network, source dataset     Stage 2
    unseen dataset                  another dataset (same network, or   Stage 3
                                    another one too)

Experiment files (configs/experiments/) list what they train::

    study: main                         # run namespace: artifacts/<profile>/runs/main/
    rounds: [A, B]                      # whose optimizers M3/M4 use (Stage 1: which rounds to train)
    train:
      pairs: [[convnet4, svhn], [mlp, svhn], [resnet18, svhn]]   # Stage 1: pairs: sources

Each (optimizee, dataset) pair is trained by one evaluation file per study and serves every
setting it falls in, for every round.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from reglo.config import ConfigError, experiment_names, load_experiment
from reglo.rounds import ROLES, Round, load_rounds, require_rounds

__all__ = [
    "DATASET_NAMES",
    "EVALUATION_ROLES",
    "LABELS",
    "METHODS",
    "METHOD_NAMES",
    "METRICS",
    "OPTIMIZEE_NAMES",
    "REGULARIZERS",
    "REG_NAMES",
    "SETTINGS",
    "STAGES",
    "Metric",
    "Pair",
    "Setting",
    "SettingPair",
    "evaluation_specs",
    "is_evaluation",
    "pair_overrides",
    "spec_rounds",
    "study_experiments",
    "study_rounds",
    "study_settings",
    "training_pairs",
]

PathLike = Union[str, Path]

STAGES = {
    1: "Training: meta-train the optimizers, train M2 with its regularizer",
    2: "Optimizer evaluation",
    3: "Optimizee evaluation",
    4: "Evaluation metrics",
}

METHODS = ("m1_adamw", "m2_adamw_reg", "m3_lstm", "m4_lstm_reg")
LEARNED = ("m3_lstm", "m4_lstm_reg")
LABELS = {"m1_adamw": "M1", "m2_adamw_reg": "M2", "m3_lstm": "M3", "m4_lstm_reg": "M4"}
METHOD_NAMES = {"m1_adamw": "Optimizee + AdamW", "m2_adamw_reg": "Optimizee + AdamW + Reg",
                "m3_lstm": "Optimizee + Learned Optimizer",
                "m4_lstm_reg": "Optimizee + Learned Optimizer + Custom Regularizer (proposed)"}
REGULARIZERS = ("l1", "l2", "sam", "gam", "gsam")
REG_NAMES = {"none": "—", "l1": "L1", "l2": "L2", "sam": "SAM", "gam": "GAM", "gsam": "GSAM"}
OPTIMIZEE_NAMES = {"convnet4": "ConvNet-4", "mlp": "MLP", "resnet18": "ResNet-18", "wrn28": "WRN-28",
                   "vit_lite7": "ViT-Lite-7/4"}
DATASET_NAMES = {"svhn": "SVHN", "fmnist": "FMNIST", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
                 "synthetic": "synthetic"}


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    stage: int
    meaning: str


SETTINGS = (  # the X-axis of the result cubes, in this order
    Setting("seen_optimizee", "Seen optimizee", 2, "the round's source pair: the network its optimizers were meta-trained on"),
    Setting("unseen_optimizee", "Unseen optimizee", 2, "another network on the round's source dataset"),
    Setting("seen_dataset", "Seen dataset", 3, "the round's source pair: its dataset (the half held out from meta-training)"),
    Setting("unseen_dataset", "Unseen dataset", 3, "another dataset — with the source network, or another one too"),
)
_SETTING_KEYS = tuple(s.key for s in SETTINGS)
_BY_KEY = {s.key: s for s in SETTINGS}
#: the settings a pair serves, by its role for a round
EVALUATION_ROLES = {
    "seen": ("seen_optimizee", "seen_dataset"),
    "unseen_optimizee": ("unseen_optimizee",),
    "unseen_dataset": ("unseen_dataset",),
    "unseen_both": ("unseen_dataset",),
}


@dataclass(frozen=True)
class Metric:
    key: str             # file names: cube_<key>
    record: str          # field of a run record (final.json, flattened)
    label: str
    unit: str
    scale: float         # 100: fractions shown in %
    digits: int
    better: str          # "higher" | "lower"
    seeds_vary: bool = True   # False: identical for every seed (FLOPs), so no ± is shown

    @property
    def arrow(self) -> str:
        return "↑" if self.better == "higher" else "↓"


METRICS = (  # Stage 4, one 3D table each
    Metric("accuracy", "test_top1", "Test accuracy", "%", 100.0, 2, "higher"),
    Metric("gen_gap", "gen_gap_top1", "Generalization gap", "pp, train − test top-1", 100.0, 2, "lower"),
    Metric("flops", "gflops_per_step", "FLOPs", "GFLOPs per training step", 1.0, 3, "lower", seeds_vary=False),
)


@dataclass(frozen=True)
class Pair:
    optimizee: str
    dataset: str

    @property
    def tag(self) -> str:  # folder name under artifacts/<profile>/runs/<study>/
        return f"{self.optimizee}-{self.dataset}"

    @property
    def label(self) -> str:
        return f"{OPTIMIZEE_NAMES.get(self.optimizee, self.optimizee)} / {DATASET_NAMES.get(self.dataset, self.dataset)}"


@dataclass(frozen=True)
class SettingPair:
    setting: Setting
    pair: Pair
    extended: bool       # beyond the core design (WRN-28, ViT-Lite, CIFAR-100)
    experiment: str      # the experiment file that trains it
    round: str = ""      # the round whose optimizers M3/M4 use here
    role: str = "seen"   # what the pair is to that round (reglo.rounds.ROLES)


def _pairs(raw: Any, where: str) -> List[Pair]:
    pairs = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError(f"{where}: a pair is [optimizee, dataset], got {item!r}")
        pairs.append(Pair(str(item[0]), str(item[1])))
    return pairs


def spec_rounds(spec: Mapping[str, Any]) -> List[Round]:
    """The rounds an experiment file names (``rounds: [A, B]``); none listed: no rounds."""
    if "rounds" not in spec:
        return []
    return require_rounds(spec["rounds"] or [], f"experiment '{spec.get('name')}'")


def training_pairs(spec: Mapping[str, Any]) -> List[Pair]:
    """The (optimizee, dataset) pairs an experiment's ``train`` block trains, in file order.

    ``train.pairs: [[o, d], …]`` lists them; ``train.pairs: sources`` means the source pair of
    each round in ``rounds`` (Stage 1's M2); the older form ``optimizees`` × ``datasets``
    (every combination) is still accepted.
    """
    block = spec.get("train") or {}
    if block.get("pairs") == "sources":
        rounds = spec_rounds(spec)
        if not rounds:
            raise ConfigError(f"experiment '{spec.get('name')}': train.pairs: sources needs a rounds list")
        return [Pair(r.optimizee, r.dataset) for r in rounds]
    if "pairs" in block:
        return _pairs(block["pairs"], f"{spec.get('name')}.train.pairs")
    return [Pair(str(o), str(d)) for o in block.get("optimizees", []) for d in block.get("datasets", [])]


def is_evaluation(spec: Mapping[str, Any]) -> bool:
    """A file whose runs are evaluated in Stages 2–3: it trains optimizees, and not as Stage 1."""
    stage = spec.get("stage")
    stages = stage if isinstance(stage, list) else [stage]
    return "train" in spec and 1 not in stages


def _order(spec: Mapping[str, Any]) -> Tuple[int, int, str]:
    stage = spec.get("stage")
    first = min(stage) if isinstance(stage, list) and stage else stage if isinstance(stage, int) else 99
    return (1 if spec.get("extended") else 0, first, str(spec.get("name")))


def study_experiments(study: str, root: Optional[PathLike] = None, profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every experiment file of ``study`` at the scale of ``profile`` (default: the active one),
    core before extended, then by stage."""
    specs = [load_experiment(name, root, profile) for name in experiment_names(root)]
    return sorted((s for s in specs if str(s.get("study")) == study), key=_order)


def evaluation_specs(study: str, root: Optional[PathLike] = None, profile: Optional[str] = None) -> List[Dict[str, Any]]:
    return [s for s in study_experiments(study, root, profile) if is_evaluation(s)]


def study_rounds(study: str, root: Optional[PathLike] = None, profile: Optional[str] = None) -> List[Round]:
    """Every round some experiment file of ``study`` names, in the order of configs/rounds.yaml."""
    named = {r.name for spec in study_experiments(study, root, profile) for r in spec_rounds(spec)}
    return [r for name, r in load_rounds(root).items() if name in named]


def _rank(pair: Pair) -> Tuple[int, int]:
    datasets, optimizees = list(DATASET_NAMES), list(OPTIMIZEE_NAMES)
    return (datasets.index(pair.dataset) if pair.dataset in datasets else 99,
            optimizees.index(pair.optimizee) if pair.optimizee in optimizees else 99)


def study_settings(study: str, root: Optional[PathLike] = None,
                   profile: Optional[str] = None) -> Dict[str, Dict[str, List[SettingPair]]]:
    """``{round: {setting key: [SettingPair, …]}}`` — every evaluation pair of ``study``, placed in
    the settings its role for each round puts it in (see the module docstring).

    Within a setting: by role, core before extended, then dataset and network order. A pair
    trained by an evaluation file appears for the rounds that file lists.
    """
    for spec in study_experiments(study, root, profile):
        if "settings" in spec:
            raise ConfigError(f"experiment '{spec['name']}': evaluation settings are derived from configs/rounds.yaml "
                              "(seen / unseen relative to each round's source); remove its settings block")
    pool: Dict[str, List[Tuple[Pair, bool, str]]] = {}
    for spec in evaluation_specs(study, root, profile):
        for rnd in spec_rounds(spec):
            listed = pool.setdefault(rnd.name, [])
            for pair in training_pairs(spec):
                if all(p != pair for p, _, _ in listed):
                    listed.append((pair, bool(spec.get("extended")), str(spec["name"])))
    out: Dict[str, Dict[str, List[SettingPair]]] = {}
    for rnd in study_rounds(study, root, profile):
        entries = sorted(pool.get(rnd.name, []),
                         key=lambda e: (ROLES.index(rnd.role(e[0].optimizee, e[0].dataset)), e[1], _rank(e[0])))
        settings: Dict[str, List[SettingPair]] = {}
        for pair, extended, experiment in entries:
            role = rnd.role(pair.optimizee, pair.dataset)
            for key in EVALUATION_ROLES[role]:
                settings.setdefault(key, []).append(SettingPair(_BY_KEY[key], pair, extended, experiment, rnd.name, role))
        if settings:
            out[rnd.name] = {key: settings[key] for key in _SETTING_KEYS if key in settings}
    return out


def pair_overrides(block: Mapping[str, Any], pair: Pair) -> List[str]:
    """Overrides for one pair: ``pair_overrides`` keyed by ``<optimizee>-<dataset>`` or by ``<optimizee>``."""
    table = block.get("pair_overrides") or {}
    return [*(table.get(pair.optimizee) or []), *(table.get(pair.tag) or [])]


def describe(sequence: Sequence[Pair]) -> str:
    return ", ".join(p.label for p in sequence)
